"""Validate MambaMIM checkpoint structure and runtime dependencies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from reform3d.external_methods.mambamim.runtime import (
    CHECKPOINT_SHA256,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CROP_SIZE,
    EXPECTED_FEATURE_DIM,
    SOURCE_COMMIT,
    MambaMIMFeatureExtractor,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the MambaMIM runtime")
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--checkpoint-only", action="store_true")
    args = parser.parse_args()

    try:
        checkpoint = torch.load(
            args.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported MambaMIM checkpoint type: {type(checkpoint)!r}")
    required_prefixes = ("embeddings.", "mae.")
    missing_prefixes = [
        prefix
        for prefix in required_prefixes
        if not any(str(key).startswith(prefix) for key in checkpoint)
    ]
    if missing_prefixes:
        raise RuntimeError(f"Missing checkpoint prefixes: {missing_prefixes}")
    if not args.checkpoint_only:
        MambaMIMFeatureExtractor(
            checkpoint_path=args.checkpoint_path,
            device="cpu",
        )

    print(
        json.dumps(
            {
                "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "source_commit": SOURCE_COMMIT,
                "crop_size": DEFAULT_CROP_SIZE,
                "feature_type": "final_stage_pool",
                "feature_dim": EXPECTED_FEATURE_DIM,
                "checkpoint_structure_valid": True,
                "strict_checkpoint_load": not args.checkpoint_only,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
