"""
OpenCV HUD dashboard — v2.

Displays:
  * Upscaled thermal stream with INFERNO colormap
  * Per-track bounding boxes with persistent Track IDs (T#N) and
    classification probability
  * IMU, GPS, RF Doppler, LiDAR range, and latency diagnostics
  * Global target coordinates (via gps_transform.local_to_global) for
    any confirmed threat — satisfies v2 deployment checklist item 3

Reads FusionResult objects from the detection_queue.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from fusion.gps_transform import local_to_global

log = logging.getLogger("dashboard")


DISPLAY_SCALE = 4
PANEL_W = 360


# Metres-per-pixel rough calibration for pixel coords -> relative XY.
# Replace with a proper intrinsic calibration once you've characterized
# the Lepton lens. For the HUD global-coord overlay this is good enough
# to see the feature working.
METRES_PER_PIXEL = 0.1


def _pixel_to_local_xy(cx: float, cy: float, frame_w: int, frame_h: int) -> tuple[float, float]:
    """Image (cx, cy) with origin at frame center -> (east_m, north_m) in sensor frame."""
    rel_x = (cx - frame_w / 2.0) * METRES_PER_PIXEL
    rel_y = (frame_h / 2.0 - cy) * METRES_PER_PIXEL   # image-y grows down
    return rel_x, rel_y


def _render(result) -> np.ndarray:
    """Render one FusionResult to an OpenCV BGR image."""
    th = (result.thermal_frame * 255.0).clip(0, 255).astype(np.uint8)
    th_color = cv2.applyColorMap(th, cv2.COLORMAP_INFERNO)
    th_color = cv2.resize(
        th_color, (th.shape[1] * DISPLAY_SCALE, th.shape[0] * DISPLAY_SCALE),
        interpolation=cv2.INTER_NEAREST,
    )
    H, W = th_color.shape[:2]
    frame_h, frame_w = result.thermal_frame.shape

    # Determine "any threat confirmed" for the top banner.
    any_threat = any(t.is_threat for t in result.tracks)

    # --- Draw tracks ---
    for t in result.tracks:
        cx, cy = t.cx * DISPLAY_SCALE, t.cy * DISPLAY_SCALE
        w2, h2 = max(8, t.w * DISPLAY_SCALE // 2), max(8, t.h * DISPLAY_SCALE // 2)
        x1, y1 = int(cx - w2), int(cy - h2)
        x2, y2 = int(cx + w2), int(cy + h2)
        color = (0, 0, 255) if t.is_threat else (0, 255, 255)
        cv2.rectangle(th_color, (x1, y1), (x2, y2), color, 2)
        label = f"T#{t.track_id} {t.prob:.2f}"
        cv2.putText(th_color, label, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # --- Right-side info panel ---
    panel = np.zeros((H, PANEL_W, 3), dtype=np.uint8)

    def line(text, y, color=(220, 220, 220), scale=0.5):
        cv2.putText(panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1, cv2.LINE_AA)

    # Threat banner.
    banner_color = (0, 0, 255) if any_threat else (0, 160, 0)
    cv2.rectangle(panel, (0, 0), (PANEL_W, 38), banner_color, -1)
    cv2.putText(panel, "THREAT" if any_threat else "CLEAR", (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, f"tracks={len(result.tracks)}", (PANEL_W - 110, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    y = 62
    line(f"frame #{result.frame_id}", y); y += 20
    line("--- Tracks ---", y, (180, 220, 255)); y += 18
    if result.tracks:
        for t in result.tracks[:6]:   # show at most 6 to avoid overflow
            tag = "THR" if t.is_threat else "   "
            line(f"T#{t.track_id:<2} p={t.prob:.3f} {tag}", y,
                 (0, 0, 255) if t.is_threat else (200, 200, 200))
            y += 16
    else:
        line("(no tracks)", y, (120, 120, 120)); y += 16

    y += 6
    line("--- IMU ---", y, (180, 180, 255)); y += 16
    line(f"R/P/Y: {result.imu_roll:+.1f} {result.imu_pitch:+.1f} {result.imu_yaw:+.1f}", y); y += 20

    # --- GPS + global coords for threats ---
    line("--- GPS ---", y, (180, 255, 180)); y += 16
    if result.gps_fix > 0:
        line(f"lat: {result.gps_lat:+.6f}", y); y += 15
        line(f"lon: {result.gps_lon:+.6f}", y); y += 15
        line(f"alt: {result.gps_alt:.1f} m", y); y += 18

        # Global target lat/lon for confirmed threats (v2 checklist).
        for t in result.tracks:
            if not t.is_threat:
                continue
            rel_x_m, rel_y_m = _pixel_to_local_xy(t.cx, t.cy, frame_w, frame_h)
            tgt_lat, tgt_lon = local_to_global(
                result.gps_lat, result.gps_lon, rel_x_m, rel_y_m,
                yaw_deg=result.imu_yaw,
            )
            line(f"T#{t.track_id} -> {tgt_lat:+.6f}, {tgt_lon:+.6f}", y,
                 (100, 100, 255)); y += 15
    else:
        line("(no fix)", y, (120, 120, 120)); y += 18

    y += 4
    line("--- RF (PCL) ---", y, (255, 220, 180)); y += 16
    line(f"peak: {result.rf_peak_mhz:.1f} MHz", y); y += 15
    line(f"v_rad: {result.rf_velocity_mps:+.2f} m/s", y); y += 18

    line("--- LiDAR ---", y, (200, 200, 255)); y += 16
    if result.lidar_range_m is not None:
        line(f"range: {result.lidar_range_m:.2f} m", y); y += 18
    else:
        line("(no range)", y, (120, 120, 120)); y += 18

    line("--- latency (ms) ---", y, (200, 200, 200)); y += 16
    for k in ("thermal", "lidar", "imu", "gps", "rf"):
        age = result.ages_ms.get(k, float("inf"))
        age_str = f"{age:6.0f}" if age != float("inf") else "  inf "
        line(f"{k:<7}: {age_str}", y); y += 13
    y += 4
    line(f"infer (total): {result.inference_ms:.1f} ms", y, (255, 255, 0)); y += 15

    return np.hstack([th_color, panel])


def run(detection_queue, shutdown_event, headless: bool = False):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    log.info("Dashboard started (headless=%s)", headless)

    if not headless:
        cv2.namedWindow("DEEPFUSE-CS HUD v2", cv2.WINDOW_AUTOSIZE)

    last_log = time.time()
    rendered = 0

    try:
        while not shutdown_event.is_set():
            try:
                result = detection_queue.get(timeout=0.1)
            except Exception:
                if not headless:
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        log.info("'q' pressed — requesting shutdown")
                        shutdown_event.set()
                        break
                continue

            if not headless:
                img = _render(result)
                cv2.imshow("DEEPFUSE-CS HUD v2", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("'q' pressed — requesting shutdown")
                    shutdown_event.set()
                    break
            rendered += 1

            now = time.time()
            if now - last_log > 2.0:
                threats = sum(1 for t in result.tracks if t.is_threat)
                log.info("rendered %d frames | tracks=%d threats=%d",
                         rendered, len(result.tracks), threats)
                last_log = now
    finally:
        if not headless:
            cv2.destroyAllWindows()
        log.info("Dashboard stopped after %d frames", rendered)
