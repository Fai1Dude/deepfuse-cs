"""
TransformerTrackClassifier — runtime wrapper.

Two execution paths, chosen at load time:
  1. TensorRT engine (.engine file) — production on the Orin Nano
  2. PyTorch (.pt state_dict or full model) — fallback for dev

The PyTorch architecture is VENDORED from the training notebook
(Senior_Project__DeepFuse_CS_Fin-3.ipynb) EXACTLY — same class names,
same hyperparameters, same `-logits` sign flip. Don't "clean it up":
the trained weights expect these exact operations in this exact order.

Input contract (per Operational Blueprint v2.0 section 2):
    patches:   Tensor[B, 20, 1, 16, 16]   float32, normalized to [0, 1]
    features:  Tensor[B, 20, 4]            float32, normalized coords + velocities

Output: Tensor[B]  — scalar logit per track (after the -logits flip).
        prob = sigmoid(logit).
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger("model.transformer")


# ---------------------------------------------------------------------------
# PyTorch definition — vendored verbatim from the training notebook
# ---------------------------------------------------------------------------
# We only import torch lazily so that the whole pipeline doesn't hard-fail
# in environments where torch isn't installed (e.g., if only TensorRT is
# available on the Jetson).

def _build_pytorch_model():
    import torch
    import torch.nn as nn

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=500):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe)

        def forward(self, x):
            x = x + self.pe[:, : x.size(1)]
            return x

    class TransformerTrackClassifier(nn.Module):
        def __init__(self, img_size=16, num_features=4, seq_len=20,
                     d_model=128, nhead=4, num_layers=2):
            super().__init__()
            self.seq_len = seq_len

            self.cnn = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),   # 16 -> 8
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),   # 8 -> 4
                nn.Flatten(),
            )
            self.patch_dim = 64 * 4 * 4
            self.cnn_proj = nn.Linear(self.patch_dim, d_model)
            self.feature_proj = nn.Linear(num_features, d_model)
            self.pos_enc = PositionalEncoding(d_model, max_len=seq_len)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, batch_first=True,
                dropout=0.1, dim_feedforward=256,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(d_model, 1)

        def forward(self, patches, features):
            B, T, C, H, W = patches.shape
            x = patches.reshape(B * T, C, H, W)
            cnn_out = self.cnn(x).reshape(B, T, -1)
            cnn_embed = self.cnn_proj(cnn_out)
            feat_embed = self.feature_proj(features)
            tokens = cnn_embed + feat_embed
            tokens = self.pos_enc(tokens)
            out = self.transformer(tokens)
            track_state = out[:, -1, :]
            logits = self.fc(track_state)
            # NOTE: The training notebook returns -logits. Keep this.
            return -logits

    return TransformerTrackClassifier


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

@dataclass
class ModelOutput:
    logits: np.ndarray       # (B,) raw model output (post -logits flip)
    probs: np.ndarray        # (B,) sigmoid of logits
    infer_ms: float


class _PyTorchBackend:
    def __init__(self, weights_path: Optional[str]):
        import torch
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ModelCls = _build_pytorch_model()
        self.model = ModelCls().to(self.device)

        if weights_path and os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=self.device)
            # Accept either a raw state_dict or a checkpoint dict.
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state)
            log.info("PyTorch: loaded weights from %s", weights_path)
        else:
            log.warning("PyTorch: NO weights loaded — model is RANDOM. "
                        "Pass --weights to load trained .pt")

        self.model.eval()

    def infer(self, patches: np.ndarray, features: np.ndarray) -> ModelOutput:
        import time
        torch = self.torch
        t0 = time.monotonic()
        with torch.no_grad():
            p = torch.from_numpy(patches).to(self.device, dtype=torch.float32)
            f = torch.from_numpy(features).to(self.device, dtype=torch.float32)
            logits = self.model(p, f)                    # (B, 1)
            logits = logits.squeeze(-1).cpu().numpy()    # (B,)
            probs = 1.0 / (1.0 + np.exp(-logits))
        infer_ms = (time.monotonic() - t0) * 1000.0
        return ModelOutput(logits=logits, probs=probs, infer_ms=infer_ms)


class _TensorRTBackend:
    """Uses a pre-built TensorRT engine. See `export_to_tensorrt.py` in the
    scripts section of the README for how to produce the .engine file."""

    def __init__(self, engine_path: str):
        import tensorrt as trt                 # noqa: F401  (import-guarded)
        import pycuda.autoinit                 # noqa: F401
        import pycuda.driver as cuda
        self.trt = trt
        self.cuda = cuda

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Allocate host + device buffers for patches, features, output.
        # Shapes are fixed at B=1 in the exported engine; raise if not.
        self.input_names = ["patches", "features"]
        self.output_name = "logits"
        self._alloc_bindings()
        log.info("TensorRT: loaded engine from %s", engine_path)

    def _alloc_bindings(self):
        cuda = self.cuda
        self.bindings = []
        self.host_buffers = {}
        self.device_buffers = {}
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = np.float32   # model is fp32; adjust if you export fp16.
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            host = np.empty(shape, dtype=dtype)
            device = cuda.mem_alloc(nbytes)
            self.host_buffers[name] = host
            self.device_buffers[name] = device
            self.bindings.append(int(device))

    def infer(self, patches: np.ndarray, features: np.ndarray) -> ModelOutput:
        import time
        cuda = self.cuda
        assert patches.shape[0] == 1, "Exported engine is fixed B=1; batch more at model-export time."

        t0 = time.monotonic()
        np.copyto(self.host_buffers["patches"], patches.astype(np.float32))
        np.copyto(self.host_buffers["features"], features.astype(np.float32))
        cuda.memcpy_htod(self.device_buffers["patches"], self.host_buffers["patches"])
        cuda.memcpy_htod(self.device_buffers["features"], self.host_buffers["features"])
        self.context.execute_v2(self.bindings)
        cuda.memcpy_dtoh(self.host_buffers[self.output_name],
                         self.device_buffers[self.output_name])
        logits = self.host_buffers[self.output_name].flatten().copy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        infer_ms = (time.monotonic() - t0) * 1000.0
        return ModelOutput(logits=logits, probs=probs, infer_ms=infer_ms)


class _NumpyDummyBackend:
    """Last-resort backend: pure-numpy deterministic output.

    Used only when neither TensorRT nor PyTorch is available — lets the
    fusion pipeline run end-to-end for integration testing / smoke tests
    on machines without torch installed. In production (Jetson) you will
    always have PyTorch or TensorRT, so this is never the chosen backend.
    """

    def __init__(self):
        log.warning("Neither TensorRT nor PyTorch available — using numpy dummy. "
                    "Probabilities are heuristic only.")

    def infer(self, patches: np.ndarray, features: np.ndarray):
        t0 = time.monotonic()
        # Heuristic: a real drone has high patch contrast AND non-trivial motion.
        patch_contrast = float(patches.std())
        motion = float(np.abs(features[..., 2:]).mean())   # |vx|, |vy|
        score = np.clip(patch_contrast * 2.0 + motion * 0.5, 0.0, 5.0)
        # Map [0, 5] -> logit, sigmoid -> ~[0.5, 0.99]
        logit = np.array([score - 1.0], dtype=np.float32)
        prob = 1.0 / (1.0 + np.exp(-logit))
        return ModelOutput(logits=logit, probs=prob,
                           infer_ms=(time.monotonic() - t0) * 1000.0)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def load_model(tensorrt_engine: Optional[str] = None,
               pytorch_weights: Optional[str] = None):
    """
    Load the best-available backend, in order:
        1. TensorRT engine (if path given and file exists)
        2. PyTorch (if torch is importable)
        3. Numpy dummy (smoke-test only)

    Returns an object with .infer(patches, features) -> ModelOutput.
    """
    if tensorrt_engine and os.path.exists(tensorrt_engine):
        try:
            return _TensorRTBackend(tensorrt_engine)
        except Exception as e:
            log.warning("TensorRT load failed (%s) — falling back to PyTorch", e)
    try:
        return _PyTorchBackend(pytorch_weights)
    except ImportError as e:
        log.warning("PyTorch not available (%s) — falling back to numpy dummy", e)
        return _NumpyDummyBackend()
