"""
Fusion engine — v2 (post Operational Blueprint v2.0).

Per-thermal-frame pipeline:

    1. Grab latest thermal frame + auxiliary samples (approximate-time sync).
    2. Run blob detector on the frame -> target candidates.
    3. Feed candidates to the Tracker, which maintains per-track circular
       buffers of length 20. Tracker returns track_ids that are "ready"
       (buffer full).
    4. For each ready track, pull (patches, features) tensors and run the
       TransformerTrackClassifier. Update the track's classification.
    5. Publish one FusionResult per frame to the dashboard queue.

Spec instrumentation retained from v1:
    * Spec 1  — end-to-end capture->publish p95 < 300 ms
    * Spec 7  — no output gap > 5 s (availability)
    * Spec 9  — temporal sync skew p95 ≤ ±15 ms (tight sensors)
    * Spec 10 — dashboard update rate ≥ 5 Hz (inherent: fusion at ~30 Hz)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

from fusion.shared_types import SharedThermalFrame, ThermalMeta, age_ms, now_ts
from fusion.detector import detect
from fusion.tracker import Tracker, SEQ_LEN
from model.track_transformer import load_model

log = logging.getLogger("fusion")


# Non-blocking alignment policy (Spec 9).
MAX_AGES_MS = {
    "lidar": 50.0,
    "imu":   15.0,
    "gps":   2000.0,
    "rf":    300.0,
}
TIGHT_SYNC = {"lidar", "imu"}

# Confirmed-threat threshold. Tune to your trained model's ROC — the notebook
# analysis suggests 0.989 for Pd=0.81 / FAR=0.004.
THREAT_THRESHOLD = 0.5


@dataclass
class TrackResult:
    track_id: int
    cx: float
    cy: float
    w: int
    h: int
    prob: float
    is_threat: bool


@dataclass
class FusionResult:
    """What gets sent to the dashboard for each thermal frame."""
    frame_id: int
    fusion_timestamp: float
    thermal_frame: np.ndarray
    tracks: list[TrackResult] = field(default_factory=list)
    # Auxiliary snapshot (may contain None entries if stale/missing).
    imu_roll: float = 0.0
    imu_pitch: float = 0.0
    imu_yaw: float = 0.0
    gps_lat: float = 0.0
    gps_lon: float = 0.0
    gps_alt: float = 0.0
    gps_fix: int = 0
    rf_peak_mhz: float = 0.0
    rf_velocity_mps: float = 0.0
    lidar_range_m: float | None = None
    # Per-frame diagnostics for HUD + log.
    inference_ms: float = 0.0
    ages_ms: dict = field(default_factory=dict)


def run(thermal_shm_name: str,
        thermal_shape: tuple,
        latest_dict,
        detection_queue,
        shutdown_event,
        tensorrt_engine: str | None = None,
        pytorch_weights: str | None = None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    shm = SharedThermalFrame(shape=thermal_shape, name=thermal_shm_name, create=False)
    tracker = Tracker(frame_shape=thermal_shape)
    model = load_model(tensorrt_engine=tensorrt_engine, pytorch_weights=pytorch_weights)

    log.info("Fusion engine started (SEQ_LEN=%d)", SEQ_LEN)

    # --- Metrics (Specs 1, 7, 9) ---
    METRIC_WINDOW = 1000
    METRIC_LOG_EVERY = 60
    e2e_latencies_ms: list[float] = []
    sync_skews_ms: list[float] = []
    last_emit_monotonic = time.monotonic()
    longest_gap_s = 0.0

    def p95(xs):
        if not xs: return float("nan")
        s = sorted(xs); return s[int(0.95 * (len(s) - 1))]

    last_frame_id = -1
    last_frame_ts = 0.0
    snapshot_id = 0
    inferences_run = 0
    waited_for_first = False

    try:
        while not shutdown_event.is_set():
            meta: ThermalMeta | None = latest_dict.get("thermal")
            if meta is None or meta.frame_id == last_frame_id:
                if not waited_for_first:
                    log.info("Waiting for first thermal frame...")
                    waited_for_first = True
                time.sleep(0.005)
                continue
            last_frame_id = meta.frame_id

            # --- Grab auxiliary samples at the moment we grab the frame ---
            lidar_s = latest_dict.get("lidar")
            imu_s   = latest_dict.get("imu")
            gps_s   = latest_dict.get("gps")
            rf_s    = latest_dict.get("rf")

            ref_ts = now_ts()
            ages = {
                "thermal": age_ms(meta.timestamp, ref_ts),
                "lidar":   age_ms(lidar_s.timestamp, ref_ts) if lidar_s else float("inf"),
                "imu":     age_ms(imu_s.timestamp, ref_ts)   if imu_s   else float("inf"),
                "gps":     age_ms(gps_s.timestamp, ref_ts)   if gps_s   else float("inf"),
                "rf":      age_ms(rf_s.timestamp, ref_ts)    if rf_s    else float("inf"),
            }
            # Non-blocking: stale -> None. Inference still runs.
            def fresh(name, s): return s if (s is not None and ages[name] <= MAX_AGES_MS[name]) else None
            lidar_s = fresh("lidar", lidar_s)
            imu_s   = fresh("imu",   imu_s)
            gps_s   = fresh("gps",   gps_s)
            rf_s    = fresh("rf",    rf_s)

            # --- Load frame from shared memory ---
            frame = shm.read()

            # --- Detect candidates ---
            candidates = detect(frame)

            # --- Update tracker ---
            dt = (meta.timestamp - last_frame_ts) if last_frame_ts > 0 else 0.033
            last_frame_ts = meta.timestamp
            roll = imu_s.roll if imu_s else 0.0
            pitch = imu_s.pitch if imu_s else 0.0
            ready_tids = tracker.update(frame, candidates, roll, pitch, dt)

            # --- Run transformer on ready tracks ---
            total_infer_ms = 0.0
            for tid in ready_tids:
                patches, features = tracker.build_model_inputs(tid)
                out = model.infer(patches, features)
                prob = float(out.probs[0])
                tracker.set_classification(tid, prob)
                total_infer_ms += out.infer_ms
                inferences_run += 1

            # --- Build FusionResult ---
            tracks_out = []
            for tid, tr in tracker.tracks.items():
                tracks_out.append(TrackResult(
                    track_id=tid, cx=tr.last_cx, cy=tr.last_cy,
                    w=tr.last_w, h=tr.last_h,
                    prob=tr.last_prob,
                    is_threat=tr.confirmed and tr.last_prob >= THREAT_THRESHOLD,
                ))

            result = FusionResult(
                frame_id=meta.frame_id,
                fusion_timestamp=ref_ts,
                thermal_frame=frame,
                tracks=tracks_out,
                imu_roll=roll, imu_pitch=pitch,
                imu_yaw=(imu_s.yaw if imu_s else 0.0),
                gps_lat=(gps_s.latitude if gps_s else 0.0),
                gps_lon=(gps_s.longitude if gps_s else 0.0),
                gps_alt=(gps_s.altitude if gps_s else 0.0),
                gps_fix=(gps_s.fix_quality if gps_s else 0),
                rf_peak_mhz=(rf_s.peak_freq_mhz if rf_s else 0.0),
                rf_velocity_mps=(rf_s.radial_velocity_mps if rf_s else 0.0),
                lidar_range_m=(float(np.linalg.norm(lidar_s.points[0])) if (lidar_s and len(lidar_s.points) > 0) else None),
                inference_ms=total_infer_ms,
                ages_ms=ages,
            )
            snapshot_id += 1

            # --- Publish (non-blocking) ---
            try:
                detection_queue.put_nowait(result)
            except Exception:
                pass   # dashboard lagging — drop this frame

            # --- Metrics ---
            e2e_ms = (now_ts() - meta.timestamp) * 1000.0
            e2e_latencies_ms.append(e2e_ms)
            if len(e2e_latencies_ms) > METRIC_WINDOW:
                e2e_latencies_ms.pop(0)

            tight_ages = [ages[s] for s in TIGHT_SYNC if ages[s] != float("inf")]
            if tight_ages:
                sync_skews_ms.append(max(tight_ages))
                if len(sync_skews_ms) > METRIC_WINDOW:
                    sync_skews_ms.pop(0)

            now_mono = time.monotonic()
            gap = now_mono - last_emit_monotonic
            if gap > longest_gap_s:
                longest_gap_s = gap
            last_emit_monotonic = now_mono

            if snapshot_id % METRIC_LOG_EVERY == 0:
                e2e_p95 = p95(e2e_latencies_ms)
                sync_p95 = p95(sync_skews_ms)
                log.info(
                    "snap=%d tracks=%d infer_total=%d | e2e p95=%.1fms (<300) "
                    "sync p95=%.1fms (≤15) gap_max=%.2fs (<5) | per-snap infer=%.1fms",
                    snapshot_id, len(tracks_out), inferences_run,
                    e2e_p95, sync_p95, longest_gap_s, total_infer_ms,
                )
                if e2e_p95 > 300:
                    log.warning("SPEC1 VIOLATION: e2e p95 %.1fms > 300ms", e2e_p95)
                if sync_p95 > 15:
                    log.warning("SPEC9 VIOLATION: sync skew p95 %.1fms > 15ms", sync_p95)
                if longest_gap_s > 5:
                    log.warning("SPEC7 VIOLATION: output gap %.2fs > 5s", longest_gap_s)
    finally:
        shm.close()
        if e2e_latencies_ms:
            log.info("Final metrics: e2e p95=%.1fms sync p95=%.1fms "
                     "gap_max=%.2fs snaps=%d inferences=%d",
                     p95(e2e_latencies_ms), p95(sync_skews_ms),
                     longest_gap_s, snapshot_id, inferences_run)
        log.info("Fusion engine stopped")
