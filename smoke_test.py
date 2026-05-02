"""
Smoke test — runs the full v2 pipeline with mock sensors.

Mock thermal generates a persistent moving hot spot (a tight Gaussian
blob plus noise) so the detector→tracker→transformer pipeline has
something real to chew on:
  * detector picks up the hot spot each frame
  * tracker associates it with the same Track ID over frames
  * once 20 frames have accumulated, transformer runs

Run:
    python -m smoke_test
    python -m smoke_test --headless
    python -m smoke_test --duration 15
"""

from __future__ import annotations

import argparse
import logging
import math
import multiprocessing as mp
import signal
import sys
import time
from multiprocessing import shared_memory

import numpy as np

from fusion.shared_types import (
    SharedThermalFrame, ThermalMeta, LidarSample, IMUSample, GPSSample, RFSample, now_ts
)
from fusion import engine as fusion_engine
from dashboard import hud as dashboard


THERMAL_H, THERMAL_W = 120, 160


# ---------------------------------------------------------------------------
# Mock sensor processes
# ---------------------------------------------------------------------------

def mock_thermal(shm_name, latest_dict, shutdown_event):
    """Synthetic thermal with a persistent moving hot spot."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    shm = SharedThermalFrame(shape=(THERMAL_H, THERMAL_W), name=shm_name, create=False)
    yy, xx = np.meshgrid(np.arange(THERMAL_H), np.arange(THERMAL_W), indexing="ij")
    fid = 0
    t0 = time.monotonic()
    rng = np.random.default_rng(42)
    try:
        while not shutdown_event.is_set():
            t = time.monotonic() - t0
            # Drone-like moving target: slow lissajous.
            cx = THERMAL_W / 2 + 40 * math.sin(t * 0.4)
            cy = THERMAL_H / 2 + 20 * math.cos(t * 0.6)
            d2 = (xx - cx) ** 2 + (yy - cy) ** 2
            # Tight hot spot (sigma ~ 2 px) — detector's area filter will pick it up.
            frame = 0.95 * np.exp(-d2 / 8.0).astype(np.float32)
            frame += rng.normal(0, 0.015, frame.shape).astype(np.float32)
            np.clip(frame, 0, 1, out=frame)
            shm.write(frame)
            latest_dict["thermal"] = ThermalMeta(
                timestamp=now_ts(), frame_id=fid, width=THERMAL_W, height=THERMAL_H
            )
            fid += 1
            time.sleep(1.0 / 30)
    finally:
        shm.close()


def mock_lidar(latest_dict, shutdown_event):
    """TF03-180-style single-point range. 200Hz for Spec 9 sync skew."""
    rng = np.random.default_rng(0)
    t0 = time.monotonic()
    while not shutdown_event.is_set():
        t = time.monotonic() - t0
        # Smoothly varying range so HUD shows motion.
        z = 20.0 + 10.0 * math.sin(t * 0.3)
        pts = np.array([[0.0, 0.0, z]], dtype=np.float32)
        latest_dict["lidar"] = LidarSample(timestamp=now_ts(), points=pts)
        time.sleep(0.005)


def mock_imu(latest_dict, shutdown_event):
    t0 = time.monotonic()
    while not shutdown_event.is_set():
        t = time.monotonic() - t0
        latest_dict["imu"] = IMUSample(
            timestamp=now_ts(),
            roll=2.0 * math.sin(t),
            pitch=1.5 * math.cos(t * 0.7),
            yaw=(t * 5) % 360 - 180,
            gyro=(0.1, -0.05, 0.02),
            accel=(0.0, 0.0, 9.81),
        )
        time.sleep(0.01)


def mock_gps(latest_dict, shutdown_event):
    while not shutdown_event.is_set():
        latest_dict["gps"] = GPSSample(
            timestamp=now_ts(),
            latitude=26.3927, longitude=50.1810, altitude=15.2,
            fix_quality=1, num_sats=9,
        )
        time.sleep(1.0)


def mock_rf(latest_dict, shutdown_event):
    """Populate v2 RFSample fields including Doppler + velocity."""
    rng = np.random.default_rng(1)
    t0 = time.monotonic()
    while not shutdown_event.is_set():
        t = time.monotonic() - t0
        spec = rng.normal(-90, 3, 1024).astype(np.float32)
        peak_strength = 25 + 15 * math.sin(t)
        spec[512] += peak_strength
        # Simulate a target radial velocity swing ±20 m/s at 2.4 GHz.
        v_mps = 20.0 * math.sin(t * 0.5)
        carrier = 2.437e9
        doppler = v_mps * carrier / 299_792_458.0
        latest_dict["rf"] = RFSample(
            timestamp=now_ts(),
            spectrum=spec,
            peak_freq_mhz=2437.0,
            peak_power_db=float(spec[512]),
            doppler_hz=doppler,
            radial_velocity_mps=v_mps,
            corr_peak=5000.0 + 1000 * math.cos(t),
        )
        time.sleep(0.1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--duration", type=float, default=0,
                    help="Auto-stop after N seconds (0 = run forever)")
    ap.add_argument("--pytorch-weights", default=None,
                    help="Optional .pt to load; otherwise runs with random weights (fine for smoke)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    log = logging.getLogger("smoke")

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    latest_dict = manager.dict()
    shutdown_event = ctx.Event()
    detection_queue = ctx.Queue(maxsize=8)

    nbytes = THERMAL_H * THERMAL_W * np.dtype(np.float32).itemsize
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    log.info("Allocated thermal SHM: %s", shm.name)

    procs = [
        ctx.Process(target=mock_thermal, args=(shm.name, latest_dict, shutdown_event), name="mock-thermal"),
        ctx.Process(target=mock_lidar,   args=(latest_dict, shutdown_event), name="mock-lidar"),
        ctx.Process(target=mock_imu,     args=(latest_dict, shutdown_event), name="mock-imu"),
        ctx.Process(target=mock_gps,     args=(latest_dict, shutdown_event), name="mock-gps"),
        ctx.Process(target=mock_rf,      args=(latest_dict, shutdown_event), name="mock-rf"),
        ctx.Process(target=fusion_engine.run,
                    args=(shm.name, (THERMAL_H, THERMAL_W), latest_dict,
                          detection_queue, shutdown_event, None, args.pytorch_weights),
                    name="fusion"),
        ctx.Process(target=dashboard.run,
                    args=(detection_queue, shutdown_event, args.headless),
                    name="dashboard"),
    ]

    def _sig(_s, _f):
        shutdown_event.set()
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    for p in procs:
        p.start()
        log.info("Started %s pid=%d", p.name, p.pid)

    deadline = time.monotonic() + args.duration if args.duration > 0 else None
    try:
        while not shutdown_event.is_set():
            if deadline and time.monotonic() > deadline:
                log.info("Duration reached — stopping")
                shutdown_event.set()
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        shutdown_event.set()

    for p in procs:
        p.join(timeout=3)
        if p.is_alive():
            p.terminate()

    shm.close()
    shm.unlink()
    log.info("Smoke test done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
