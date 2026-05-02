"""
DEEPFUSE-CS — main supervisor.

Brings up:
    [thermal] [lidar] [imu] [gps] [rf]    -- 5 sensor producers
                       |
                  shared buffer
                       |
                   [fusion]                -- 1 consumer
                       |
                detection queue
                       |
                  [dashboard]              -- 1 visualizer

Run:
    python -m deepfuse_cs.main
    python -m deepfuse_cs.main --no-rf --no-gps          # disable specific sensors
    python -m deepfuse_cs.main --headless                # no OpenCV window
    python -m deepfuse_cs.main --thermal-dev /dev/video2 # override device path

Shutdown: Ctrl-C, or 'q' in the HUD window.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import signal
import sys
import time
from multiprocessing import shared_memory

# Sensor entry points
from sensors import thermal as sensor_thermal
from sensors import lidar   as sensor_lidar
from sensors import imu     as sensor_imu
from sensors import gps     as sensor_gps
from sensors import rf      as sensor_rf

from fusion import engine as fusion_engine
from dashboard import hud as dashboard

import numpy as np

log = logging.getLogger("supervisor")


THERMAL_SHAPE = (sensor_thermal.THERMAL_H, sensor_thermal.THERMAL_W)


def _allocate_thermal_shm() -> shared_memory.SharedMemory:
    """Create the SharedMemory block sized for one thermal frame."""
    nbytes = int(np.prod(THERMAL_SHAPE)) * np.dtype(np.float32).itemsize
    # Let the OS pick a unique name; we'll pass it to children.
    return shared_memory.SharedMemory(create=True, size=nbytes)


def parse_args():
    p = argparse.ArgumentParser(description="DEEPFUSE-CS pipeline")
    p.add_argument("--thermal-dev", default="/dev/video0")
    p.add_argument("--lidar-port",  default="/dev/ttyUSB0")
    p.add_argument("--lidar-baud",  type=int, default=115200)
    p.add_argument("--imu-port",    default="/dev/ttyUSB1")
    p.add_argument("--imu-baud",    type=int, default=115200)
    p.add_argument("--gps-port",    default="/dev/ttyTHS1")
    p.add_argument("--gps-baud",    type=int, default=9600)
    p.add_argument("--pluto-uri",   default="ip:192.168.2.1")
    p.add_argument("--no-thermal",  action="store_true")
    p.add_argument("--no-lidar",    action="store_true")
    p.add_argument("--no-imu",      action="store_true")
    p.add_argument("--no-gps",      action="store_true")
    p.add_argument("--no-rf",       action="store_true")
    p.add_argument("--headless",    action="store_true",
                   help="Run dashboard without opening an OpenCV window")
    p.add_argument("--thermal-weights", default=None,
                   help="(legacy, unused in v2)")
    p.add_argument("--rf-weights",      default=None,
                   help="(legacy, unused in v2)")
    p.add_argument("--tensorrt-engine", default=None,
                   help="Path to .engine file for TensorRT runtime. If present, used first.")
    p.add_argument("--pytorch-weights", default=None,
                   help="Path to .pt state_dict for PyTorch fallback.")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    args = parse_args()

    # 'spawn' is safer than 'fork' when CUDA/OpenCV/serial handles are involved.
    ctx = mp.get_context("spawn")

    manager = ctx.Manager()
    latest_dict = manager.dict()
    shutdown_event = ctx.Event()
    detection_queue = ctx.Queue(maxsize=8)

    thermal_shm = _allocate_thermal_shm()
    log.info("Allocated thermal SharedMemory: %s (%d bytes)",
             thermal_shm.name, thermal_shm.size)

    # ---- Process specs (name, target, args, restart_critical) ----
    # restart_critical=True means: if it dies, restart it; if restart fails
    # repeatedly, shut down the whole pipeline. False means: log and carry on
    # (a missing sensor just shows as null in the snapshot, which is fine
    # per Spec 9's non-blocking policy).
    proc_specs: list[tuple] = []

    if not args.no_thermal:
        proc_specs.append(("thermal", sensor_thermal.run,
            (thermal_shm.name, latest_dict, shutdown_event, args.thermal_dev), True))
    if not args.no_lidar:
        proc_specs.append(("lidar", sensor_lidar.run,
            (latest_dict, shutdown_event, args.lidar_port, args.lidar_baud), False))
    if not args.no_imu:
        proc_specs.append(("imu", sensor_imu.run,
            (latest_dict, shutdown_event, args.imu_port, args.imu_baud), False))
    if not args.no_gps:
        proc_specs.append(("gps", sensor_gps.run,
            (latest_dict, shutdown_event, args.gps_port, args.gps_baud), False))
    if not args.no_rf:
        proc_specs.append(("rf", sensor_rf.run,
            (latest_dict, shutdown_event, args.pluto_uri), False))

    # Fusion + dashboard are critical: without them there's no detection output.
    proc_specs.append(("fusion", fusion_engine.run,
        (thermal_shm.name, THERMAL_SHAPE, latest_dict, detection_queue,
         shutdown_event, args.tensorrt_engine, args.pytorch_weights), True))
    proc_specs.append(("dashboard", dashboard.run,
        (detection_queue, shutdown_event, args.headless), True))

    # name -> (target, args, restart_critical, restart_count, last_restart_ts)
    proc_state: dict[str, dict] = {
        name: {"target": tgt, "args": args_, "critical": crit,
               "restarts": 0, "last_restart": 0.0, "process": None}
        for (name, tgt, args_, crit) in proc_specs
    }

    def _spawn(name: str) -> mp.Process:
        st = proc_state[name]
        p = ctx.Process(target=st["target"], args=st["args"], name=name, daemon=False)
        p.start()
        st["process"] = p
        log.info("Started %s (pid=%d) [restart #%d]", name, p.pid, st["restarts"])
        return p

    # ---- Signal handler ----
    def _on_signal(signum, _frame):
        log.warning("Caught signal %d — initiating shutdown", signum)
        shutdown_event.set()
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # ---- Launch ----
    for name in proc_state:
        _spawn(name)

    # ---- Watchdog loop (Spec 7: availability) ----
    # If a critical process dies, restart it. Cap restarts to prevent flapping;
    # if a process restarts more than MAX_RESTARTS times in RESTART_WINDOW_S,
    # something is genuinely broken — give up and shut down.
    MAX_RESTARTS = 3
    RESTART_WINDOW_S = 60.0
    RESTART_BACKOFF_S = 1.0

    try:
        while not shutdown_event.is_set():
            for name, st in proc_state.items():
                p = st["process"]
                if p is None or p.is_alive():
                    continue

                exitcode = p.exitcode
                p.join(timeout=0.1)

                # Within the restart window, count restarts; outside it, reset.
                now = time.monotonic()
                if now - st["last_restart"] > RESTART_WINDOW_S:
                    st["restarts"] = 0

                if not st["critical"]:
                    log.warning("Non-critical process %s died (exit=%s); leaving down",
                                name, exitcode)
                    st["process"] = None
                    continue

                if st["restarts"] >= MAX_RESTARTS:
                    log.error("Critical process %s exceeded %d restarts in %.0fs — "
                              "shutting down for diagnosis",
                              name, MAX_RESTARTS, RESTART_WINDOW_S)
                    shutdown_event.set()
                    break

                log.warning("Critical process %s died (exit=%s) — restarting in %.1fs",
                            name, exitcode, RESTART_BACKOFF_S)
                time.sleep(RESTART_BACKOFF_S)
                st["restarts"] += 1
                st["last_restart"] = now
                _spawn(name)

            time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown_event.set()

    # ---- Graceful shutdown ----
    log.info("Joining processes...")
    for name, st in proc_state.items():
        p = st["process"]
        if p is None:
            continue
        p.join(timeout=5)
        if p.is_alive():
            log.warning("Force-terminating %s", name)
            p.terminate()
            p.join(timeout=2)

    # ---- Cleanup shared memory ----
    try:
        thermal_shm.close()
        thermal_shm.unlink()
    except Exception as e:
        log.warning("Error releasing shared memory: %s", e)

    log.info("Shutdown complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
