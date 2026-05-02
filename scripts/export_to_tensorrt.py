"""
Export TransformerTrackClassifier to TensorRT for deployment on the Orin Nano.

Pipeline: PyTorch .pt -> ONNX -> TensorRT .engine

Usage:
    # On the Jetson (where TensorRT is already installed):
    python scripts/export_to_tensorrt.py \\
        --weights /path/to/trained.pt \\
        --onnx    out/model.onnx \\
        --engine  out/model.engine

    # Or only up to ONNX (useful for checking the export on a dev machine):
    python scripts/export_to_tensorrt.py --weights trained.pt --onnx out/model.onnx --no-trt

The resulting .engine file can be passed to main.py via --tensorrt-engine.

Per Operational Blueprint v2.0 section 4.III, this is required to hit
the <10 ms inference target on the Orin Nano.
"""

from __future__ import annotations

import argparse
import os
import sys

# Make the project root importable when running this script from /scripts.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np


def export_onnx(weights_path: str, onnx_path: str,
                batch_size: int = 1, seq_len: int = 20):
    import torch
    from model.track_transformer import _build_pytorch_model

    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
    ModelCls = _build_pytorch_model()
    model = ModelCls()
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    # Dummy inputs shape EXACTLY per blueprint Input A and Input B.
    dummy_patches = torch.zeros(batch_size, seq_len, 1, 16, 16)
    dummy_features = torch.zeros(batch_size, seq_len, 4)

    torch.onnx.export(
        model, (dummy_patches, dummy_features), onnx_path,
        input_names=["patches", "features"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"[ok] wrote ONNX -> {onnx_path}")


def onnx_to_tensorrt(onnx_path: str, engine_path: str, fp16: bool = True):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    # 256 MB workspace is plenty for this small model.
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[info] FP16 enabled")

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT engine build failed")

    os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(engine)
    print(f"[ok] wrote TensorRT engine -> {engine_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to .pt state_dict")
    ap.add_argument("--onnx",    default="out/model.onnx")
    ap.add_argument("--engine",  default="out/model.engine")
    ap.add_argument("--batch",   type=int, default=1)
    ap.add_argument("--no-trt",  action="store_true",
                    help="Stop after ONNX export")
    ap.add_argument("--no-fp16", action="store_true",
                    help="Export FP32 engine (slower but byte-exact with PyTorch)")
    args = ap.parse_args()

    export_onnx(args.weights, args.onnx, batch_size=args.batch)
    if args.no_trt:
        return 0
    onnx_to_tensorrt(args.onnx, args.engine, fp16=not args.no_fp16)
    return 0


if __name__ == "__main__":
    sys.exit(main())
