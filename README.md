# DeepFuse-CS

> **Real-time multi-sensor counter-stealth pipeline** with a transformer-based track classifier, deployed on Jetson Orin Nano.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg)](https://pytorch.org/)
[![TensorRT](https://img.shields.io/badge/TensorRT-Jetson-76B900.svg)](https://developer.nvidia.com/tensorrt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

DeepFuse-CS is a senior capstone project at King Fahd University of Petroleum and Minerals (KFUPM). It fuses **thermal, LiDAR, IMU, GPS, and SDR-RF** sensors into a single live tracking and classification pipeline, all running on a Jetson Orin Nano.

<!--
  📸  Replace this comment with a real screenshot or GIF of the HUD running.
       Save it to assets/hud.png and uncomment the line below.
  ![HUD](assets/hud.png)
-->

---

## Highlights

- **Sub-10 ms inference** on Jetson Orin Nano via PyTorch → ONNX → TensorRT export.
- **5-process producer architecture** (one per sensor) writing into shared memory, decoupling capture from a fusion loop running detection → tracking → transformer classification → HUD.
- **CNN + Transformer track classifier** (d_model 128, 4 heads, 2 layers) operating on per-track sequences of `[B, 20, 1, 16, 16]` thermal patches plus `[B, 20, 4]` motion features, with calibrated probabilities (target ECE ≤ 0.08).
- **Graceful degradation:** TensorRT engine → PyTorch → NumPy dummy backend fallback chain.
- **Per-process restart watchdog** + a NumPy-only smoke test that runs without any hardware *or* PyTorch installed.

### Measured runtime performance

All targets met under load (smoke test, 60-snapshot windows):

| Spec | Target | Measured |
|---|---|---|
| End-to-end p95 latency | < 300 ms | **6.7 ms** |
| Sensor-sync skew p95 | ≤ 15 ms | **10.6 ms** |
| Longest data gap | < 5 s | **0.10 s** |
| Dashboard refresh | ≥ 5 Hz | ~30 Hz |

---

## Architecture

```
 ┌────────┐ ┌──────┐ ┌─────┐ ┌─────┐ ┌──────┐
 │thermal │ │lidar │ │ imu │ │ gps │ │  rf  │     5 producer processes
 └───┬────┘ └──┬───┘ └──┬──┘ └──┬──┘ └──┬───┘
     │ SHM     │ Manager.dict (latest sample per sensor)
     └─────────┴────────┴───────┴───────┘
                       │
                ┌──────▼──────┐
                │  detector   │  per-frame blob centroids
                └──────┬──────┘
                       │
                ┌──────▼──────┐
                │   tracker   │  greedy nearest-neighbor, per-track 20-frame buffer
                └──────┬──────┘
                       │ patches[1, 20, 1, 16, 16], features[1, 20, 4]
                ┌──────▼──────┐
                │ transformer │  TensorRT or PyTorch fallback
                └──────┬──────┘
                       │ FusionResult
                ┌──────▼──────┐
                │  dashboard  │  track IDs + global GPS coord overlay
                └─────────────┘
```

The classifier only fires when a track has 20 frames buffered. New tracks warm up for ~0.67 s at 30 Hz before their first classification.

---

## Repository layout

```
deepfuse-cs/
├── main.py                        # supervisor (per-process restart watchdog)
├── smoke_test.py                  # full-pipeline test with mock sensors
├── requirements.txt
├── sensors/
│   ├── thermal.py                 # FLIR Lepton; 14-bit → [0,1] min-max
│   ├── lidar.py                   # Benewake TF03-180 UART decoder
│   ├── imu.py                     # Wimotion IMU over serial
│   ├── gps.py                     # GPS NMEA parser
│   └── rf.py                      # PlutoSDR + PCL Doppler (synthetic Wi-Fi STF)
├── fusion/
│   ├── shared_types.py            # dataclasses + SharedThermalFrame
│   ├── detector.py                # per-frame blob detector (centroids)
│   ├── tracker.py                 # greedy NN tracker + 20-frame buffers
│   ├── gps_transform.py           # local XY + GPS fix → global lat/lon
│   └── engine.py                  # detector → tracker → transformer
├── model/
│   └── track_transformer.py       # TransformerTrackClassifier + TRT/PT backends
├── dashboard/
│   └── hud.py                     # OpenCV HUD with persistent track IDs
└── scripts/
    └── export_to_tensorrt.py      # .pt → ONNX → .engine
```

---

## Quick start

### 1. Smoke test (no hardware, no PyTorch needed)

```bash
python smoke_test.py --headless --duration 10
```

The smoke test ships a NumPy-only dummy model fallback, so it runs even without PyTorch installed. Expected output:

```
snap=60 tracks=1 infer_total=41 | e2e p95=6.9ms (<300) sync p95=10.6ms (≤15) gap_max=0.10s (<5)
```

### 2. With real hardware

```bash
pip install -r requirements.txt

# PyTorch backend
python main.py --pytorch-weights weights/transformer.pt

# TensorRT backend (sub-10 ms inference on Orin Nano)
python main.py --tensorrt-engine weights/transformer.engine
```

Press **`q`** in the HUD window or **`Ctrl-C`** to shut down.

### 3. Export model for TensorRT deployment

```bash
python scripts/export_to_tensorrt.py \
    --weights weights/transformer.pt \
    --onnx    weights/transformer.onnx \
    --engine  weights/transformer.engine
```

> **Jetson note:** prefer NVIDIA's pre-built wheels for `torch`, `opencv-python`, and `tensorrt` from JetPack rather than PyPI.

---

## Where to plug in your own code

| File | Replace | With |
|---|---|---|
| `sensors/thermal.py` | `open_capture`, `read_frame` | Lepton SDK / libuvc if different |
| `sensors/lidar.py`   | (nothing — TF03-180 decoder is real) | |
| `sensors/imu.py`     | `parse_packet` | your Wimotion frame format |
| `sensors/gps.py`     | (nothing — NMEA is standard) | |
| `sensors/rf.py`      | `open_sdr`, `read_iq` | SDR init if not pyadi-iio |
| `fusion/detector.py` | `detect` | trained YOLO / custom detector later |
| `model/track_transformer.py` | (nothing — vendored from training notebook) | just supply `--pytorch-weights` |

---

## Engineering notes

- **Zero-copy thermal frames** via `multiprocessing.shared_memory`.
- **Spawn context** (`mp.get_context("spawn")`) used everywhere to avoid CUDA/serial handle inheritance bugs that `fork` can introduce on Jetson.
- **Bounded queues** — detection queue has `maxsize=8`. If the dashboard falls behind, oldest is dropped rather than back-pressuring the fusion loop.
- **Auto backend selection:** TensorRT engine → PyTorch → NumPy dummy.
- **Spec violations are logged** every 60 snapshots — grep for `SPEC* VIOLATION` during field tests.
- **IMU-based digital stabilization** of thermal patches before they enter the model.

---

## Tech stack

**ML:** PyTorch · ONNX · TensorRT · NumPy
**Systems:** Python multiprocessing, shared memory, OpenCV
**Sensors:** FLIR Lepton (thermal), Benewake TF03-180 (LiDAR), Wimotion (IMU), generic NMEA GPS, ADALM-PlutoSDR (RF)
**Hardware:** Jetson Orin Nano

---

## License

Released under the MIT License. See [LICENSE](LICENSE).

---

## Author

**Faisal Alhamdi** — B.Sc. Software Engineering, KFUPM
[LinkedIn](https://linkedin.com/in/faisal-alhamdi) · [GitHub](https://github.com/Fai1Dude)
