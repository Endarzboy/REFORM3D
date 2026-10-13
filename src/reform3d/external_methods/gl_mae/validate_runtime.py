"""Validate GL-MAE checkpoint loading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reform3d.external_methods.gl_mae.runtime import (
    CHECKPOINT_SHA256,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CROP_SIZE,
    EXPECTED_FEATURE_DIM,
    SOURCE_COMMIT,
    GLMAEFeatureExtractor,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the GL-MAE runtime")
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    args = parser.parse_args()

    GLMAEFeatureExtractor(
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
                "feature_type": "cls",
                "feature_dim": EXPECTED_FEATURE_DIM,
                "strict_checkpoint_load": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
