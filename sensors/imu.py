"""
IMU sensor process — Wimotion on /dev/ttyUSB1.

Target rate: 100 Hz (per blueprint section 2C). The Wimotion's framing
varies by configuration; replace `parse_packet()` with whatever your
working script does (binary protocol, ASCII CSV, etc.).
"""

from __future__ import annotations

import logging
import time

import serial

from fusion.shared_types import IMUSample, now_ts

log = logging.getLogger("sensor.imu")


# ---------------------------------------------------------------------------
# REPLACE WITH YOUR REAL DECODER
# ---------------------------------------------------------------------------

def parse_packet(buf: bytearray) -> tuple[IMUSample | None, int]:
    """
    Parse one IMU sample from buffer.

    Placeholder: assumes ASCII CSV lines like
        "R,P,Y,gx,gy,gz,ax,ay,az\\n"
    (degrees, deg/s, m/s^2). Adjust to your firmware's framing.
    """
    nl = buf.find(b"\n")
    if nl < 0:
        return None, 0
    line = bytes(buf[:nl]).decode("ascii", errors="ignore").strip()
    consumed = nl + 1
    parts = line.split(",")
    if len(parts) < 9:
        return None, consumed  # drop malformed line, keep advancing
    try:
        vals = [float(p) for p in parts[:9]]
    except ValueError:
        return None, consumed
    sample = IMUSample(
        timestamp=now_ts(),
        roll=vals[0], pitch=vals[1], yaw=vals[2],
        gyro=(vals[3], vals[4], vals[5]),
        accel=(vals[6], vals[7], vals[8]),
    )
    return sample, consumed


# ---------------------------------------------------------------------------
# Process entry point
# ---------------------------------------------------------------------------

def run(latest_dict, shutdown_event,
        port: str = "/dev/ttyUSB1", baud: int = 115200):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    try:
        ser = serial.Serial(port, baud, timeout=0.01)
    except Exception as e:
        log.error("Failed to open IMU port %s: %s", port, e)
        return

    log.info("IMU capture started on %s @ %d", port, baud)
    buf = bytearray()
    sample_count = 0

    try:
        while not shutdown_event.is_set():
            chunk = ser.read(256)
            if chunk:
                buf.extend(chunk)

            while True:
                sample, consumed = parse_packet(buf)
                if consumed == 0:
                    break
                del buf[:consumed]
                if sample is not None:
                    latest_dict["imu"] = sample
                    sample_count += 1

            if len(buf) > 4096:
                del buf[:-2048]

            if not chunk:
                time.sleep(0.002)  # poll fast — IMU is the highest-rate sensor
    finally:
        ser.close()
        log.info("IMU capture stopped after %d samples", sample_count)
