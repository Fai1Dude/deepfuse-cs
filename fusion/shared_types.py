"""
DEEPFUSE-CS — Shared data types and the cross-process shared buffer.

The shared buffer holds the LATEST sample from each sensor, tagged with a
monotonic UTC timestamp. The fusion process reads it whenever a new thermal
frame arrives and assembles a temporally-aligned DataSnapshot.

Design notes:
  * Small structured samples (IMU / GPS / LiDAR / RF) live in a Manager dict.
    The locking overhead is negligible at these rates and the API is simple.
  * Thermal frames are large (160x120 float32 = ~75KB) and arrive at 30Hz.
    Pushing them through the Manager would copy + pickle every frame, so we
    use multiprocessing.shared_memory with a numpy view instead. Only the
    metadata (timestamp, frame counter) goes in the Manager dict.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Sensor sample dataclasses
# ---------------------------------------------------------------------------
# These are passed through the Manager so they need to be picklable. Plain
# dataclasses with primitives / numpy arrays work fine.

@dataclass
class ThermalMeta:
    """Metadata for the thermal frame stored in shared memory."""
    timestamp: float = 0.0
    frame_id: int = 0
    width: int = 0
    height: int = 0


@dataclass
class LidarSample:
    """A LiDAR point cloud snapshot, projected to (x, y, z) in metres."""
    timestamp: float = 0.0
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))


@dataclass
class IMUSample:
    """IMU orientation in degrees + angular rates in deg/s."""
    timestamp: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    gyro: tuple = (0.0, 0.0, 0.0)
    accel: tuple = (0.0, 0.0, 0.0)


@dataclass
class GPSSample:
    """GPS fix from a parsed $GPGGA NMEA sentence."""
    timestamp: float = 0.0
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    fix_quality: int = 0   # 0 = no fix, 1 = GPS, 2 = DGPS
    num_sats: int = 0


@dataclass
class RFSample:
    """RF spectrum features after FFT — kept compact for transport."""
    timestamp: float = 0.0
    # The 1D-CNN head expects a fixed-length spectrum vector.
    spectrum: np.ndarray = field(default_factory=lambda: np.zeros(1024, dtype=np.float32))
    peak_freq_mhz: float = 0.0
    peak_power_db: float = -120.0
    # --- PCL Doppler (v2 blueprint) ---
    doppler_hz: float = 0.0              # signed Doppler shift
    radial_velocity_mps: float = 0.0     # derived from doppler + carrier
    corr_peak: float = 0.0               # correlation magnitude — trust gate


# ---------------------------------------------------------------------------
# DataSnapshot — the temporally-aligned bundle the model consumes
# ---------------------------------------------------------------------------

@dataclass
class DataSnapshot:
    """A coherent multi-sensor snapshot, time-aligned to a thermal frame."""
    snapshot_id: int
    fusion_timestamp: float
    thermal_frame: np.ndarray            # H x W float32
    thermal_age_ms: float
    lidar: Optional[LidarSample]
    lidar_age_ms: float
    imu: Optional[IMUSample]
    imu_age_ms: float
    gps: Optional[GPSSample]
    gps_age_ms: float
    rf: Optional[RFSample]
    rf_age_ms: float


# ---------------------------------------------------------------------------
# Shared thermal frame — backed by multiprocessing.shared_memory
# ---------------------------------------------------------------------------

class SharedThermalFrame:
    """
    Wraps a SharedMemory block sized for one float32 thermal frame.

    Producer (sensor process):  call .write(frame, timestamp)
    Consumer (fusion process):  call .read()  -> (frame_copy, meta)

    The numpy view is rebuilt on each access because SharedMemory cannot
    be shared by reference across processes — only by name.
    """

    DEFAULT_DTYPE = np.float32

    def __init__(self, shape: tuple, name: Optional[str] = None, create: bool = False):
        self.shape = shape
        self.dtype = self.DEFAULT_DTYPE
        nbytes = int(np.prod(shape)) * np.dtype(self.dtype).itemsize
        if create:
            self.shm = shared_memory.SharedMemory(create=True, size=nbytes, name=name)
        else:
            assert name is not None, "Must provide name when attaching to existing block"
            self.shm = shared_memory.SharedMemory(name=name)
        self.name = self.shm.name

    def _view(self) -> np.ndarray:
        return np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)

    def write(self, frame: np.ndarray) -> None:
        view = self._view()
        np.copyto(view, frame.astype(self.dtype, copy=False))

    def read(self) -> np.ndarray:
        # Copy out so the caller doesn't race with the next write.
        return self._view().copy()

    def close(self) -> None:
        self.shm.close()

    def unlink(self) -> None:
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def now_ts() -> float:
    """High-resolution UTC wall time, seconds since epoch."""
    return time.time()


def age_ms(sample_ts: float, ref_ts: float) -> float:
    """Age of `sample_ts` relative to `ref_ts`, in milliseconds."""
    return (ref_ts - sample_ts) * 1000.0
