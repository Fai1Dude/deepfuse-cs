"""
Per-frame target detector.

The transformer expects patches CENTERED on a target, so something has to
produce (x, y) candidates before the transformer ever runs. We use a
cheap thermal-blob approach:

    1. Local adaptive thresholding to isolate hot pixels.
    2. Connected components to group them.
    3. Filter by area (reject single-pixel noise and huge bright regions).
    4. Report centroids + bounding boxes.

This is deliberately simple — a real drone stands out strongly in thermal
(hot motors/batteries on a cold sky background). If you want YOLO or a
trained detector later, replace `detect()` — the return signature is what
the tracker depends on.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Candidate:
    cx: float        # centroid x in pixels (frame coords)
    cy: float        # centroid y in pixels
    w: int           # bbox width
    h: int           # bbox height
    area: int        # blob area in pixels
    peak: float      # peak intensity [0, 1]


# Tunables — start here, adjust when you see real field data.
MIN_AREA = 4        # reject single-pixel noise
MAX_AREA = 600      # reject large bright regions (sun, hot ground patches)
THRESH_PERCENTILE = 98.0   # keep pixels above the 98th percentile


def detect(frame: np.ndarray) -> list[Candidate]:
    """
    Args:  frame  H x W float32 in [0, 1]
    Returns: list of Candidate, sorted by peak intensity (hottest first).
    """
    if frame.size == 0:
        return []

    # Adaptive threshold based on frame percentile — avoids hard-coded
    # absolute levels that break across day/night/seasons.
    thr = np.percentile(frame, THRESH_PERCENTILE)
    mask = (frame >= thr).astype(np.uint8) * 255

    # Morphological open to drop isolated pixels.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Connected components with stats.
    num, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    out: list[Candidate] = []
    for i in range(1, num):   # skip background label 0
        x, y, w, h, area = stats[i]
        if area < MIN_AREA or area > MAX_AREA:
            continue
        cx, cy = centroids[i]
        # Peak intensity inside the bbox
        patch = frame[y:y + h, x:x + w]
        peak = float(patch.max()) if patch.size else 0.0
        out.append(Candidate(cx=float(cx), cy=float(cy), w=int(w), h=int(h),
                             area=int(area), peak=peak))

    out.sort(key=lambda c: c.peak, reverse=True)
    return out
