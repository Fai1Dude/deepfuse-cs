"""
Convert (relative sensor X, Y in metres) + (GPS lat, lon, yaw_deg) to
absolute target (lat, lon).

Per Operational Blueprint v2.0 deployment checklist:
    "Integrate GPS Coordinate Transformation: Convert (Relative Radar X, Y)
     + (GPS Lat, Lon) → (Target Global Coordinate)."

This is a local-tangent-plane approximation (ENU / flat earth) — accurate
to <1 m within a few km of the sensor, which covers any realistic radar
range for this project. Well outside that you'd want a full geodesic
calculation via pyproj.

Conventions:
    sensor_x_m : metres EAST of the sensor     (positive = east)
    sensor_y_m : metres NORTH of the sensor    (positive = north)
    yaw_deg    : sensor heading in degrees, 0 = north, 90 = east
"""

from __future__ import annotations

import math

# WGS84 mean Earth radius.
R_EARTH_M = 6_371_000.0


def local_to_global(sensor_lat_deg: float, sensor_lon_deg: float,
                    sensor_x_m: float, sensor_y_m: float,
                    yaw_deg: float = 0.0) -> tuple[float, float]:
    """
    Rotate (x, y) by yaw to get true east/north offsets, then convert to lat/lon.
    Returns (target_lat_deg, target_lon_deg).
    """
    yaw_rad = math.radians(yaw_deg)
    # Rotate sensor-frame (x=right, y=forward) into ENU (east, north).
    east_m  = sensor_x_m * math.cos(yaw_rad) + sensor_y_m * math.sin(yaw_rad)
    north_m = -sensor_x_m * math.sin(yaw_rad) + sensor_y_m * math.cos(yaw_rad)

    # Flat-earth: small-angle conversion.
    d_lat_deg = math.degrees(north_m / R_EARTH_M)
    lat_rad = math.radians(sensor_lat_deg)
    d_lon_deg = math.degrees(east_m / (R_EARTH_M * max(math.cos(lat_rad), 1e-9)))

    return sensor_lat_deg + d_lat_deg, sensor_lon_deg + d_lon_deg
