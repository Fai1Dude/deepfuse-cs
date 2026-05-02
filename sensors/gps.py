"""
GPS sensor process — UART on /dev/ttyTHS1 (Jetson GPIO pins 8/10).

Parses NMEA $GPGGA sentences. If you also want $GPRMC for speed/heading,
extend `parse_nmea_line()`.
"""

from __future__ import annotations

import logging
import time

import serial

from fusion.shared_types import GPSSample, now_ts

log = logging.getLogger("sensor.gps")


# ---------------------------------------------------------------------------
# NMEA parsing
# ---------------------------------------------------------------------------

def _nmea_to_decimal(raw: str, hemi: str) -> float:
    """Convert NMEA ddmm.mmmm or dddmm.mmmm to signed decimal degrees."""
    if not raw or not hemi:
        return 0.0
    try:
        # Latitude has 2-digit degrees, longitude has 3.
        deg_len = 2 if hemi in ("N", "S") else 3
        deg = float(raw[:deg_len])
        minutes = float(raw[deg_len:])
        val = deg + minutes / 60.0
        if hemi in ("S", "W"):
            val = -val
        return val
    except (ValueError, IndexError):
        return 0.0


def _verify_checksum(line: str) -> bool:
    """NMEA checksum: XOR of all chars between $ and *, hex-encoded after *."""
    if "*" not in line:
        return False
    body, given = line[1:].split("*", 1)
    chk = 0
    for c in body:
        chk ^= ord(c)
    try:
        return chk == int(given.strip(), 16)
    except ValueError:
        return False


def parse_nmea_line(line: str) -> GPSSample | None:
    """Parse one $GPGGA sentence into a GPSSample, or None if invalid."""
    line = line.strip()
    if not line.startswith("$") or not _verify_checksum(line):
        return None
    body = line.split("*", 1)[0]
    fields = body.split(",")
    if len(fields) < 10 or not fields[0].endswith("GGA"):
        return None
    try:
        lat = _nmea_to_decimal(fields[2], fields[3])
        lon = _nmea_to_decimal(fields[4], fields[5])
        fix = int(fields[6]) if fields[6] else 0
        sats = int(fields[7]) if fields[7] else 0
        alt = float(fields[9]) if fields[9] else 0.0
    except (ValueError, IndexError):
        return None
    return GPSSample(
        timestamp=now_ts(),
        latitude=lat, longitude=lon, altitude=alt,
        fix_quality=fix, num_sats=sats,
    )


# ---------------------------------------------------------------------------
# Process entry point
# ---------------------------------------------------------------------------

def run(latest_dict, shutdown_event,
        port: str = "/dev/ttyTHS1", baud: int = 9600):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    try:
        ser = serial.Serial(port, baud, timeout=0.1)
    except Exception as e:
        log.error("Failed to open GPS port %s: %s", port, e)
        return

    log.info("GPS capture started on %s @ %d", port, baud)
    buf = bytearray()
    fix_count = 0

    try:
        while not shutdown_event.is_set():
            chunk = ser.read(256)
            if chunk:
                buf.extend(chunk)
            while True:
                nl = buf.find(b"\n")
                if nl < 0:
                    break
                line = bytes(buf[:nl]).decode("ascii", errors="ignore")
                del buf[:nl + 1]
                sample = parse_nmea_line(line)
                if sample is not None:
                    latest_dict["gps"] = sample
                    if sample.fix_quality > 0:
                        fix_count += 1

            if len(buf) > 4096:
                del buf[:-2048]
            if not chunk:
                time.sleep(0.05)
    finally:
        ser.close()
        log.info("GPS capture stopped, %d valid fixes", fix_count)
