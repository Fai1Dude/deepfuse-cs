"""
LiDAR sensor process — Benewake TF03-180 on serial (default /dev/ttyUSB0 @ 115200).

Per project Spec 12: TF03-180 is a long-range single-point LiDAR (180 m range,
±10 cm or 1% accuracy, 1 cm resolution, 0.5° FoV, UART/CAN/IO).
This file implements the standard 9-byte UART frame format documented by
Benewake. If you're using CAN or IO instead, replace `parse_packet()`.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import serial

from fusion.shared_types import LidarSample, now_ts

log = logging.getLogger("sensor.lidar")


# ---------------------------------------------------------------------------
# Benewake TF03-180 packet decoder (per Spec 12)
# ---------------------------------------------------------------------------
# Frame format (9 bytes, little-endian):
#   [0]   0x59       header byte 1
#   [1]   0x59       header byte 2
#   [2:4] dist_lo, dist_hi    distance in cm
#   [4:6] strength_lo, strength_hi   signal strength
#   [6:8] temp_lo, temp_hi    raw temperature (chip temp in °C = raw/8 - 256)
#   [8]   checksum   = (sum of bytes [0..7]) & 0xFF
#
# The TF03-180 is a single-point ranging LiDAR (FoV 0.5°), so each frame
# yields ONE point, not a cloud. We treat it as a point on the optical axis.

PACKET_SIZE = 9
HEADER = b"\x59\x59"


def parse_packet(buf: bytearray) -> tuple[np.ndarray | None, int]:
    """
    Try to extract one TF03-180 frame from `buf`.

    Returns:
        (points, consumed_bytes)
            points: (1, 3) float32 array (x, y, z) in metres, or None.
                    For a single-point LiDAR pointed straight ahead:
                    x=0, y=0, z=distance_m.
            consumed_bytes: how many bytes to drop from the front of buf.
    """
    # Resync to header if buffer doesn't start with 0x59 0x59.
    head = buf.find(HEADER)
    if head < 0:
        # No header anywhere — drop everything but keep last byte (it might
        # be the start of a header).
        if len(buf) > 1:
            return None, len(buf) - 1
        return None, 0
    if head > 0:
        # Drop garbage before the header.
        return None, head

    if len(buf) < PACKET_SIZE:
        return None, 0  # wait for more bytes

    pkt = bytes(buf[:PACKET_SIZE])
    if (sum(pkt[:8]) & 0xFF) != pkt[8]:
        # Bad checksum — drop the header and resync from the next byte.
        return None, 1

    dist_cm = pkt[2] | (pkt[3] << 8)
    strength = pkt[4] | (pkt[5] << 8)

    # Per Benewake datasheet: strength < 100 or == 0xFFFF means unreliable.
    # Spec says detection range up to 180 m (= 18000 cm); reject implausible.
    if strength < 100 or strength == 0xFFFF or dist_cm == 0 or dist_cm > 18000:
        return None, PACKET_SIZE  # consume but don't emit

    distance_m = dist_cm / 100.0
    pts = np.array([[0.0, 0.0, distance_m]], dtype=np.float32)
    return pts, PACKET_SIZE


# ---------------------------------------------------------------------------
# Process entry point
# ---------------------------------------------------------------------------

def run(latest_dict, shutdown_event,
        port: str = "/dev/ttyUSB0", baud: int = 115200):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    try:
        ser = serial.Serial(port, baud, timeout=0.05)
    except Exception as e:
        log.error("Failed to open LiDAR port %s: %s", port, e)
        return

    log.info("LiDAR capture started on %s @ %d", port, baud)

    buf = bytearray()
    packet_count = 0

    try:
        while not shutdown_event.is_set():
            chunk = ser.read(512)
            if chunk:
                buf.extend(chunk)

            # Drain as many complete packets as currently buffered.
            while True:
                pts, consumed = parse_packet(buf)
                if pts is None:
                    break
                del buf[:consumed]
                latest_dict["lidar"] = LidarSample(timestamp=now_ts(), points=pts)
                packet_count += 1

            # Cap buffer growth in case of a bad sync — keep last 4KB.
            if len(buf) > 8192:
                del buf[:-4096]

            if not chunk:
                time.sleep(0.005)
    finally:
        ser.close()
        log.info("LiDAR capture stopped after %d packets", packet_count)
