"""MambaMIM encoder runtime for REFORM-3D evaluation."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import torch

from reform3d.core.paths import resolve_external_method_checkpoint_path
from reform3d.external_methods._primary_feature import PrimaryFeatureMethod
from reform3d.external_methods._sources import source_root, add_source_path


MAMBAMIM_SOURCE_ROOT = source_root("mambamim", Path(__file__).resolve().parents[1] / "MambaMIM")
REGISTERED_CHECKPOINT_PATH = resolve_external_method_checkpoint_path(
    "mambamim",
    "mambamim_mask75.pth",
)
LOCAL_CHECKPOINT_PATH = MAMBAMIM_SOURCE_ROOT / "checkpoint" / "mambamim_mask75.pth"
DEFAULT_CHECKPOINT_PATH = REGISTERED_CHECKPOINT_PATH
DEFAULT_CHECKPOINT_NAME = "mambamim_mask75"
DEFAULT_MODEL_NAME = "MambaMIM"
DEFAULT_FEATURE_TYPE = "final_stage_pool"
SUPPORTED_FEATURE_TYPES = ("final_stage_pool",)
FEATURE_SURFACES = {
    "final_stage_pool": "official_dense_encoder_final_stage_spatial_average",
}
DEFAULT_CROP_SIZE = 96
EXPECTED_FEATURE_DIM = 384
SOURCE_COMMIT = "59e9ddafc24507cfa0a365f4eb5ff426fd4cbd24"
CHECKPOINT_SHA256 = "fd92f6cfdd2aff93f8942536f333bca7eb612b4238153c9b5accbacd9e4e1989"
INPUT_PREPROCESSING = "reform3d_percentile_0.05_99.95_to_minus1_1"
IGNORED_DENSE_INFERENCE_KEYS = ("mae.mask_token",)


def resolve_dtype(dtype_name: str) -> torch.dtype:
    normalized = str(dtype_name).strip().lower()
    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _ensure_source_available() -> None:
    add_source_path(MAMBAMIM_SOURCE_ROOT, "models/network/hymamba.py", ("models",))

def _load_encoder_class():
    _ensure_source_available()
    try:
        from models.network.hymamba import Encoder
    except ModuleNotFoundError as error:
        if error.name in {"mamba_ssm", "causal_conv1d"}:
            raise RuntimeError(
                "MambaMIM requires its dedicated environment with mamba_ssm "
                "and causal_conv1d installed"
            ) from error
        raise
    return Encoder


def build_encoder(checkpoint_path: Path | str):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"MambaMIM checkpoint not found: {checkpoint_path}")

    encoder_class = _load_encoder_class()
    model = encoder_class(
        in_channel=1,
        channels=(32, 64, 128, 192, 384),
        depths=(1, 2, 2, 2, 1),
        kernels=(3, 3, 3, 3, 3),
        exp_r=(2, 2, 4, 4, 4),
        img_size=DEFAULT_CROP_SIZE,
        depth=4,
        sparse=False,
    )
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported MambaMIM checkpoint type: {type(checkpoint)!r}")
    checkpoint = dict(checkpoint)
    for key in IGNORED_DENSE_INFERENCE_KEYS:
        checkpoint.pop(key, None)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class MambaMIMFeatureExtractor:
    """Frozen dense MambaMIM encoder exposing its final spatial stage."""

    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        self.model = build_encoder(checkpoint_path)
        self.model.to(device=device, dtype=dtype)
        self.config = SimpleNamespace(
            device=device,
            dtype=dtype,
            crop_size=DEFAULT_CROP_SIZE,
            input_size=DEFAULT_CROP_SIZE,
            checkpoint_path=str(checkpoint_path),
            source_commit=SOURCE_COMMIT,
            checkpoint_sha256=CHECKPOINT_SHA256,
        )

    @torch.inference_mode()
    def extract(self, batch: torch.Tensor, normalize: bool = True) -> Dict[str, torch.Tensor]:
        if batch.ndim != 5:
            raise ValueError(f"Expected [B, C, H, W, D] input, got {tuple(batch.shape)}")
        if batch.shape[1] != 1:
            raise ValueError(f"Expected one input channel, got {batch.shape[1]}")
        if tuple(batch.shape[2:]) != (DEFAULT_CROP_SIZE,) * 3:
            raise ValueError(
                f"MambaMIM expects {DEFAULT_CROP_SIZE}^3 inputs, got {tuple(batch.shape[2:])}"
            )
        batch = batch.to(
            device=self.config.device,
            dtype=self.config.dtype,
            non_blocking=True,
        )
        feature_maps = self.model(batch)
        if len(feature_maps) != 5:
            raise RuntimeError(f"Expected five MambaMIM feature stages, got {len(feature_maps)}")
        final_stage_pool = feature_maps[-1].flatten(2).mean(dim=-1)
        if final_stage_pool.shape[-1] != EXPECTED_FEATURE_DIM:
            raise RuntimeError(
                f"Expected {EXPECTED_FEATURE_DIM} MambaMIM features, "
                f"got {final_stage_pool.shape[-1]}"
            )
        return {"final_stage_pool": final_stage_pool}


METHOD = PrimaryFeatureMethod(
    display_name="MambaMIM",
    model_name=DEFAULT_MODEL_NAME,
    checkpoint_name=DEFAULT_CHECKPOINT_NAME,
    checkpoint_path=DEFAULT_CHECKPOINT_PATH,
    feature_type=DEFAULT_FEATURE_TYPE,
    feature_surface=FEATURE_SURFACES[DEFAULT_FEATURE_TYPE],
    crop_size=DEFAULT_CROP_SIZE,
    extractor_factory=MambaMIMFeatureExtractor,
    resolve_dtype=resolve_dtype,
    source_commit=SOURCE_COMMIT,
    checkpoint_sha256=CHECKPOINT_SHA256,
    input_preprocessing=INPUT_PREPROCESSING,
    default_batch_size=8,
)
