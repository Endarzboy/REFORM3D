"""Configuration contract for single-feature external encoders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch


def reference_checkpoint_sha256(
    checkpoint_path: Path,
    reference_path: Path,
    reference_sha256: str,
) -> str | None:
    """Return the recorded reference hash only for its configured path."""
    if checkpoint_path.expanduser().resolve() == reference_path.expanduser().resolve():
        return reference_sha256
    return None


@dataclass(frozen=True)
class PrimaryFeatureMethod:
    display_name: str
    model_name: str
    checkpoint_name: str
    checkpoint_path: Path
    feature_type: str
    feature_surface: str
    crop_size: int
    extractor_factory: Callable[..., Any]
    resolve_dtype: Callable[[str], torch.dtype]
    source_commit: str
    checkpoint_sha256: str
    input_preprocessing: str = "reform3d_percentile_0.05_99.95_to_minus1_1"
    default_batch_size: int = 1
