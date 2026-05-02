"""
Multi-target tracker.

Per Operational Blueprint v2.0 section 4.I, the transformer requires a
circular buffer of 20 synchronized frames per track. This module:

  1. Associates current-frame detections with existing tracks (greedy NN).
  2. Maintains a per-track deque of the last 20 (patch, feature) pairs.
  3. Computes per-frame velocities from centroid motion.
  4. Applies IMU roll/pitch digital stabilization (pixel offsets) to the
     patch crop coordinates.
  5. Hands the pipeline `(patches, features)` tensors ready for the model
     — ONLY when a track has exactly 20 frames buffered.

Why a separate module? The transformer only classifies — it doesn't
associate detections into tracks or manage sequence history. That's this
file's job.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from fusion.detector import Candidate

log = logging.getLogger("fusion.tracker")


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

SEQ_LEN = 20                       # must match TransformerTrackClassifier.seq_len
PATCH_SIZE = 16                    # must match model img_size
GATE_PIXELS = 25.0                 # max distance for detection -> track association
MAX_MISSED_FRAMES = 8              # drop track after N frames without an update
IMU_STAB_GAIN_PX_PER_DEG = 1.5     # pixels shifted per degree of roll/pitch


# ---------------------------------------------------------------------------
# Track state
# ---------------------------------------------------------------------------

@dataclass
class Track:
    track_id: int
    # Circular buffer of 16x16 float32 patches, newest last.
    patches: deque = field(default_factory=lambda: deque(maxlen=SEQ_LEN))
    # Circular buffer of [x_norm, y_norm, vx_norm, vy_norm], newest last.
    features: deque = field(default_factory=lambda: deque(maxlen=SEQ_LEN))
    # Most recent pixel centroid (for association and dashboard overlay).
    last_cx: float = 0.0
    last_cy: float = 0.0
    last_w: int = 16
    last_h: int = 16
    frames_since_update: int = 0
    last_prob: float = 0.0         # most recent classifier probability
    confirmed: bool = False        # becomes True once the transformer ran once


# ---------------------------------------------------------------------------
# Patch extraction with IMU stabilization
# ---------------------------------------------------------------------------

def _extract_patch(frame: np.ndarray, cx: float, cy: float,
                   roll_deg: float, pitch_deg: float,
                   size: int = PATCH_SIZE) -> np.ndarray:
    """
    Crop a `size`x`size` patch centered on (cx, cy), offset by an IMU-derived
    digital-stabilization shift. Pads with zeros if the patch hits the edge.
    """
    H, W = frame.shape
    # Stabilization: roll shifts horizontally, pitch vertically. This is a
    # first-order correction — good enough for a handheld/tripod system
    # with small angular excursions. For wide-angle lenses you'd want a
    # proper homography instead.
    dx = roll_deg * IMU_STAB_GAIN_PX_PER_DEG
    dy = pitch_deg * IMU_STAB_GAIN_PX_PER_DEG

    # Top-left corner of the patch after stabilization.
    x0 = int(round(cx + dx - size / 2))
    y0 = int(round(cy + dy - size / 2))
    x1, y1 = x0 + size, y0 + size

    # Allocate zero-padded output and copy the overlapping region.
    out = np.zeros((size, size), dtype=np.float32)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x1, W), min(y1, H)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    return out


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class Tracker:
    """Greedy nearest-neighbor tracker with per-track sequence buffers."""

    def __init__(self, frame_shape: tuple[int, int]):
        self.H, self.W = frame_shape
        self.tracks: dict[int, Track] = {}
        self._next_id = 1

    # --- internal helpers ---

    def _normalize_xy(self, cx: float, cy: float) -> tuple[float, float]:
        """Pixel coords -> [-1, +1] with sensor origin at frame center."""
        nx = (cx - self.W / 2.0) / (self.W / 2.0)
        ny = (cy - self.H / 2.0) / (self.H / 2.0)
        return nx, ny

    def _associate(self, candidates: list[Candidate]) -> tuple[dict[int, Candidate], list[Candidate]]:
        """
        Greedy assignment: for each track (strongest confidence first),
        pick the closest in-gate candidate. Simple but works well when
        targets are well-separated — which is the drone case.
        """
        assigned: dict[int, Candidate] = {}
        used: set[int] = set()

        # Iterate tracks in order of most recently confirmed first — that
        # way a "locked" track keeps its detection over a new track's.
        track_order = sorted(
            self.tracks.values(),
            key=lambda t: (t.confirmed, -t.frames_since_update, t.last_prob),
            reverse=True,
        )

        for tr in track_order:
            best_idx, best_d = -1, GATE_PIXELS
            for i, c in enumerate(candidates):
                if i in used:
                    continue
                d = math.hypot(c.cx - tr.last_cx, c.cy - tr.last_cy)
                if d < best_d:
                    best_d, best_idx = d, i
            if best_idx >= 0:
                assigned[tr.track_id] = candidates[best_idx]
                used.add(best_idx)

        unassigned = [c for i, c in enumerate(candidates) if i not in used]
        return assigned, unassigned

    def _make_track(self, cand: Candidate) -> Track:
        tr = Track(track_id=self._next_id)
        self._next_id += 1
        tr.last_cx, tr.last_cy = cand.cx, cand.cy
        tr.last_w, tr.last_h = cand.w, cand.h
        return tr

    # --- public API ---

    def update(self, frame: np.ndarray, candidates: list[Candidate],
               roll_deg: float, pitch_deg: float, dt: float) -> list[int]:
        """
        Run one tracker tick.

        Args:
            frame:      H x W float32 thermal in [0, 1]
            candidates: list from detector.detect()
            roll_deg, pitch_deg: latest IMU (0, 0 if unavailable)
            dt:         seconds since the previous frame (for velocity calc)

        Returns: list of track_ids that are NOW ready for inference (buffer
        has exactly SEQ_LEN frames). Caller should call
        `build_model_inputs(track_id)` for each.
        """
        assigned, unassigned = self._associate(candidates)

        ready: list[int] = []
        seen_ids: set[int] = set()

        # --- Update assigned tracks ---
        for tid, cand in assigned.items():
            tr = self.tracks[tid]
            seen_ids.add(tid)

            # Velocity from centroid motion (px/s -> normalized units).
            nx, ny = self._normalize_xy(cand.cx, cand.cy)
            if dt > 1e-6:
                prev_nx, prev_ny = self._normalize_xy(tr.last_cx, tr.last_cy)
                vx = (nx - prev_nx) / dt
                vy = (ny - prev_ny) / dt
            else:
                vx = vy = 0.0

            patch = _extract_patch(frame, cand.cx, cand.cy, roll_deg, pitch_deg)
            tr.patches.append(patch[np.newaxis, :, :])   # (1, 16, 16)
            tr.features.append(np.array([nx, ny, vx, vy], dtype=np.float32))

            tr.last_cx, tr.last_cy = cand.cx, cand.cy
            tr.last_w, tr.last_h = cand.w, cand.h
            tr.frames_since_update = 0

            if len(tr.patches) == SEQ_LEN:
                ready.append(tid)

        # --- Create tracks for unassigned candidates ---
        for cand in unassigned:
            tr = self._make_track(cand)
            patch = _extract_patch(frame, cand.cx, cand.cy, roll_deg, pitch_deg)
            nx, ny = self._normalize_xy(cand.cx, cand.cy)
            tr.patches.append(patch[np.newaxis, :, :])
            tr.features.append(np.array([nx, ny, 0.0, 0.0], dtype=np.float32))
            self.tracks[tr.track_id] = tr
            seen_ids.add(tr.track_id)

        # --- Age/drop tracks that weren't updated ---
        to_drop = []
        for tid, tr in self.tracks.items():
            if tid in seen_ids:
                continue
            tr.frames_since_update += 1
            if tr.frames_since_update > MAX_MISSED_FRAMES:
                to_drop.append(tid)
        for tid in to_drop:
            del self.tracks[tid]
            log.debug("Dropped track %d (missed %d frames)", tid, MAX_MISSED_FRAMES + 1)

        return ready

    def build_model_inputs(self, track_id: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (patches, features) with shapes (1, 20, 1, 16, 16) and
        (1, 20, 4), ready to hand to the transformer.
        """
        tr = self.tracks[track_id]
        assert len(tr.patches) == SEQ_LEN, "build_model_inputs called too early"
        patches = np.stack(list(tr.patches), axis=0)[np.newaxis, ...]   # (1, 20, 1, 16, 16)
        features = np.stack(list(tr.features), axis=0)[np.newaxis, ...] # (1, 20, 4)
        return patches.astype(np.float32), features.astype(np.float32)

    def set_classification(self, track_id: int, prob: float) -> None:
        """Called by the fusion engine after the transformer runs."""
        tr = self.tracks.get(track_id)
        if tr is None:
            return
        tr.last_prob = prob
        tr.confirmed = True
