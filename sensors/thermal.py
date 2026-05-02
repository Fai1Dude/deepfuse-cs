"""
Thermal sensor process — FLIR Lepton on /dev/video0.

INTEGRATION POINT
-----------------
This file uses OpenCV's VideoCapture as a generic placeholder. If your working
script uses libuvc or the Lepton SDK directly, replace the body of
`open_capture()` and `read_frame()` with your code. Everything else
(timestamping, shared-memory write, shutdown) stays the same.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from fusion.shared_types import SharedThermalFrame, ThermalMeta, now_ts

log = logging.getLogger("sensor.thermal")


# Lepton 3.5 native resolution. Override if you're using a different model.
THERMAL_W = 160
THERMAL_H = 120
TARGET_FPS = 30


# ---------------------------------------------------------------------------
# REPLACE THESE TWO FUNCTIONS WITH YOUR WORKING LEPTON CODE IF DIFFERENT
# ---------------------------------------------------------------------------

def open_capture(device: str = "/dev/video0"):
    """Open the Lepton via V4L2/UVC. Returns a cv2.VideoCapture."""
    # CAP_V4L2 is more reliable than the default backend on Jetson.
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open thermal device {device}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, THERMAL_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, THERMAL_H)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    return cap


def read_frame(cap) -> np.ndarray | None:
    """Read one thermal frame, min-max normalize to float32 [0, 1].

    Per v2 blueprint deployment checklist: raw 14-bit thermal should be
    mapped to [0, 1] via min-max. OpenCV returns the 14-bit data as
    uint16 on Linux (sometimes padded into a 16-bit container) or as
    8-bit BGR if the UVC driver is doing auto-gain. We handle both.
    """
    ok, frame = cap.read()
    if not ok or frame is None:
        return None

    if frame.ndim == 3:
        # UVC auto-gained BGR stream — convert to gray.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    f = frame.astype(np.float32)

    # Min-max normalize. If frame is constant (dead sensor / lens cap),
    # avoid divide-by-zero.
    lo, hi = float(f.min()), float(f.max())
    if hi - lo < 1e-6:
        return np.zeros_like(f)
    return (f - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Process entry point
# ---------------------------------------------------------------------------

def run(shm_name: str, meta_dict, shutdown_event, device: str = "/dev/video0"):
    """
    Thermal capture loop.

    Args:
        shm_name:        Name of the pre-created SharedMemory block.
        meta_dict:       Manager().dict() for sharing ThermalMeta.
        shutdown_event:  multiprocessing.Event signalling clean exit.
        device:          V4L2 device path.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    shm = SharedThermalFrame(shape=(THERMAL_H, THERMAL_W), name=shm_name, create=False)

    try:
        cap = open_capture(device)
    except Exception as e:
        log.error("Failed to open thermal capture: %s", e)
        return

    log.info("Thermal capture started on %s @ %dx%d", device, THERMAL_W, THERMAL_H)

    frame_id = 0
    target_dt = 1.0 / TARGET_FPS
    next_deadline = time.monotonic()

    try:
        while not shutdown_event.is_set():
            frame = read_frame(cap)
            if frame is None:
                # Don't tight-loop on transient failures.
                time.sleep(0.01)
                continue

            ts = now_ts()
            shm.write(frame)
            meta_dict["thermal"] = ThermalMeta(
                timestamp=ts, frame_id=frame_id, width=THERMAL_W, height=THERMAL_H
            )
            frame_id += 1

            # Pace the loop so we don't burn CPU if the device is faster than target.
            next_deadline += target_dt
            sleep_for = next_deadline - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # We're behind schedule — reset the deadline so we don't spiral.
                next_deadline = time.monotonic()
    finally:
        cap.release()
        shm.close()
        log.info("Thermal capture stopped after %d frames", frame_id)
