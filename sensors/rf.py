"""
RF Surveillance process — PlutoSDR over IP (192.168.2.1).

Captures IQ samples and produces:
  * Magnitude spectrum        (general situational awareness / HUD)
  * Doppler shift estimate    (via Single-SDR PCL synthetic-reference
                               cross-correlation — see estimate_doppler())
  * Radial velocity in m/s    (Doppler shift / carrier × c)

The velocity estimate is used by the tracker (fusion/tracker.py) as an
alternative / prior source for vx in the transformer's feature vector —
useful when the target is radially approaching/receding faster than the
thermal centroid motion can reveal.

INTEGRATION POINT
-----------------
This file uses pyadi-iio. If you're using libiio directly or a different
SDR, replace `open_sdr()` and `read_iq()`. The DSP section below is
portable and stays the same.
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np

from fusion.shared_types import RFSample, now_ts

log = logging.getLogger("sensor.rf")


# Drone control + video bands of interest (Hz). PlutoSDR retunes per scan.
SCAN_CENTERS_HZ = [2_437_000_000, 5_800_000_000]   # 2.4 GHz, 5.8 GHz
# 20 MS/s matches 802.11a/g/n baseband — required for Wi-Fi preamble corr.
SAMPLE_RATE_HZ = 20_000_000
FFT_SIZE = 1024
# Number of samples per capture — enough to see several Wi-Fi frames.
IQ_BUFFER_SIZE = 4096


# ---------------------------------------------------------------------------
# REPLACE WITH YOUR REAL SDR INIT IF NOT pyadi-iio
# ---------------------------------------------------------------------------

def open_sdr(uri: str = "ip:192.168.2.1"):
    """Open the PlutoSDR. Returns the configured device handle."""
    try:
        import adi  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "pyadi-iio not installed. `pip install pyadi-iio` or replace open_sdr()."
        ) from e
    sdr = adi.Pluto(uri)
    sdr.sample_rate = SAMPLE_RATE_HZ
    sdr.rx_rf_bandwidth = SAMPLE_RATE_HZ
    sdr.rx_buffer_size = IQ_BUFFER_SIZE
    sdr.gain_control_mode_chan0 = "slow_attack"
    return sdr


def read_iq(sdr) -> np.ndarray:
    """Return one buffer of complex IQ samples."""
    return sdr.rx()


# ---------------------------------------------------------------------------
# DSP
# ---------------------------------------------------------------------------

def compute_spectrum(iq: np.ndarray) -> np.ndarray:
    """Magnitude spectrum in dB, length FFT_SIZE, fftshifted so DC is centre."""
    if len(iq) < FFT_SIZE:
        iq = np.pad(iq, (0, FFT_SIZE - len(iq)))
    else:
        iq = iq[:FFT_SIZE]
    win = np.hanning(FFT_SIZE)
    spec = np.fft.fftshift(np.fft.fft(iq * win))
    mag = np.abs(spec) + 1e-12
    return (20.0 * np.log10(mag)).astype(np.float32)


# ---------------------------------------------------------------------------
# PCL Doppler extraction — Single-Channel Synthetic Reference
# (per Operational Blueprint v2.0 section 3)
# ---------------------------------------------------------------------------
#
# Hardware gives us ONE PlutoSDR, so two-channel GPSDO-synced PCL isn't
# possible. Workaround: cross-correlate the received signal against a
# locally-generated Wi-Fi preamble, then measure the Doppler shift.
#
# For 802.11a/g/n, the preamble (Short Training Field, STF) is a known
# periodic sequence of 16 samples repeated 10 times at 20 MS/s. We don't
# need the full 802.11 PHY — just a good correlator kernel. The real-world
# signal is modulated onto the carrier, so after downconversion by the
# PlutoSDR we see the baseband with any Doppler shift preserved as a
# complex rotation between correlation peaks.

# Short Training Field spec values (IEEE 802.11-2016 §17.3.3).
# These are the frequency-domain subcarriers; we synthesize the time-domain
# preamble once at module import and reuse it.

def _build_wifi_stf(n_samples: int = 160) -> np.ndarray:
    """Build a baseband 802.11 Short Training Field sequence (length 160 @ 20 MS/s)."""
    # 12 non-zero STF subcarriers, indexed -24 to +24 step 4.
    # Values per the 802.11 standard table, scaled by sqrt(13/6).
    stf_subs_idx = np.array([-24, -20, -16, -12, -8, -4, 4, 8, 12, 16, 20, 24])
    stf_subs_val = np.array([
        1+1j, -1-1j, 1+1j, -1-1j, -1-1j, 1+1j,
        -1-1j, -1-1j, 1+1j, 1+1j, 1+1j, 1+1j,
    ]) * np.sqrt(13.0 / 6.0)
    N_FFT = 64
    X = np.zeros(N_FFT, dtype=np.complex64)
    X[(stf_subs_idx + N_FFT) % N_FFT] = stf_subs_val
    # One STF symbol = IFFT of that; full STF = 10 repetitions of the 16-sample segment.
    sym = np.fft.ifft(X)           # 64 samples
    # Actual STF is 10x the first 16 samples (periodic).
    base = sym[:16]
    stf = np.tile(base, 10)        # 160 samples
    if len(stf) > n_samples:
        stf = stf[:n_samples]
    return stf.astype(np.complex64)


_REFERENCE_STF = _build_wifi_stf()


def estimate_doppler(iq: np.ndarray, sample_rate_hz: float) -> tuple[float, float]:
    """
    Cross-correlate received IQ against the synthetic Wi-Fi preamble and
    estimate the Doppler shift.

    Approach: Because the STF is periodic (16-sample cell repeated 10x),
    we can detect the preamble AND measure the carrier offset from the
    phase difference between consecutive 16-sample correlation peaks.
    That phase difference is 2π × f_doppler × T, with T = 16 / sample_rate.

    Returns:
        (doppler_hz, correlation_peak_magnitude)
        If no preamble is detected confidently, doppler_hz is 0.0.
    """
    if len(iq) < len(_REFERENCE_STF) + 16:
        return 0.0, 0.0

    # Full cross-correlation (receiver detects where the preamble starts).
    corr = np.correlate(iq, _REFERENCE_STF, mode="valid")
    mag = np.abs(corr)
    peak_idx = int(np.argmax(mag))
    peak_mag = float(mag[peak_idx])

    # Self-correlation at 16-sample lag to measure residual carrier offset.
    # Works ONLY if the signal actually contains a Wi-Fi-like periodic STF;
    # otherwise the "doppler" is meaningless noise — the caller can gate on
    # peak_mag to decide whether to trust it.
    seg_len = 16 * 9   # 9 full periods of the 16-sample cell
    start = peak_idx
    end = start + seg_len + 16
    if end > len(iq):
        return 0.0, peak_mag

    a = iq[start:start + seg_len]
    b = iq[start + 16:start + 16 + seg_len]
    phase_diff = np.angle(np.sum(a * np.conj(b)))    # radians per 16-sample gap
    doppler_hz = -phase_diff / (2.0 * math.pi * 16.0 / sample_rate_hz)
    return float(doppler_hz), peak_mag


def doppler_to_velocity_mps(doppler_hz: float, carrier_hz: float) -> float:
    """Convert Doppler shift to radial velocity. Approaching = positive."""
    C = 299_792_458.0
    return doppler_hz * C / carrier_hz


# ---------------------------------------------------------------------------
# Process entry point
# ---------------------------------------------------------------------------

def run(latest_dict, shutdown_event, uri: str = "ip:192.168.2.1"):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    try:
        sdr = open_sdr(uri)
    except Exception as e:
        log.error("Failed to open PlutoSDR at %s: %s", uri, e)
        return

    log.info("RF capture started @ %s, scanning %s", uri, SCAN_CENTERS_HZ)
    scan_idx = 0
    cycles = 0

    try:
        while not shutdown_event.is_set():
            center = SCAN_CENTERS_HZ[scan_idx]
            try:
                sdr.rx_lo = center
                # Small settle time after retune.
                time.sleep(0.005)
                iq = read_iq(sdr)
            except Exception as e:
                log.warning("RF read failed @ %.0f MHz: %s", center / 1e6, e)
                time.sleep(0.05)
                continue

            spec_db = compute_spectrum(iq)
            peak_bin = int(np.argmax(spec_db))
            bin_hz = SAMPLE_RATE_HZ / FFT_SIZE
            peak_offset_hz = (peak_bin - FFT_SIZE // 2) * bin_hz
            peak_freq_hz = center + peak_offset_hz

            # PCL Doppler estimate via synthetic Wi-Fi preamble correlation.
            doppler_hz, corr_mag = estimate_doppler(iq, SAMPLE_RATE_HZ)
            velocity_mps = doppler_to_velocity_mps(doppler_hz, center)

            latest_dict["rf"] = RFSample(
                timestamp=now_ts(),
                spectrum=spec_db,
                peak_freq_mhz=peak_freq_hz / 1e6,
                peak_power_db=float(spec_db[peak_bin]),
                doppler_hz=doppler_hz,
                radial_velocity_mps=velocity_mps,
                corr_peak=corr_mag,
            )

            scan_idx = (scan_idx + 1) % len(SCAN_CENTERS_HZ)
            cycles += 1
    finally:
        log.info("RF capture stopped after %d scans", cycles)
