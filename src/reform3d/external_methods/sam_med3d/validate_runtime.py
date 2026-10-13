"""Validate SAM-Med3D checkpoint loading and frozen feature extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from reform3d.external_methods.sam_med3d.runtime import (
    CHECKPOINT_SHA256,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CROP_SIZE,
    EXPECTED_FEATURE_DIM,
    SOURCE_COMMIT,
    SAMMed3DFeatureExtractor,
    resolve_dtype,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the SAM-Med3D runtime")
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-forward", action="store_true")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dtype = resolve_dtype(args.dtype)
    if device == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32

    extractor = SAMMed3DFeatureExtractor(
        checkpoint_path=args.checkpoint_path,
        device=device,
        dtype=dtype,
    )
    report = {
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_commit": SOURCE_COMMIT,
        "crop_size": DEFAULT_CROP_SIZE,
        "feature_type": "avg_pool",
        "feature_dim": EXPECTED_FEATURE_DIM,
        "strict_checkpoint_load": True,
        "forward_checked": not args.skip_forward,
    }

    if not args.skip_forward:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        batch = torch.rand(
            (1, 1, DEFAULT_CROP_SIZE, DEFAULT_CROP_SIZE, DEFAULT_CROP_SIZE),
            generator=generator,
        )
        first = extractor.extract(batch)["avg_pool"]
        second = extractor.extract(batch)["avg_pool"]
        if first.shape != (1, EXPECTED_FEATURE_DIM):
            raise RuntimeError(f"Unexpected feature shape: {tuple(first.shape)}")
        if not torch.isfinite(first).all():
            raise RuntimeError("SAM-Med3D produced non-finite features")
        if not torch.allclose(first, second, rtol=1e-5, atol=1e-6):
            raise RuntimeError("SAM-Med3D feature extraction is not deterministic")
        report["deterministic"] = True

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
