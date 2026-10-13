"""SAM-Med3D image-encoder runtime for REFORM-3D evaluation."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import torch
import torch.nn as nn

from reform3d.core.paths import resolve_external_method_checkpoint_path
from reform3d.external_methods._primary_feature import PrimaryFeatureMethod
from reform3d.external_methods._sources import source_root, add_source_path


SAM_MED3D_SOURCE_ROOT = source_root("sam_med3d", Path(__file__).resolve().parents[1] / "SAM-Med3D")
REGISTERED_CHECKPOINT_PATH = resolve_external_method_checkpoint_path(
    "sam_med3d",
    "sam_med3d_turbo.pth",
)
LOCAL_CHECKPOINT_PATH = SAM_MED3D_SOURCE_ROOT / "checkpoint" / "sam_med3d_turbo.pth"
DEFAULT_CHECKPOINT_PATH = REGISTERED_CHECKPOINT_PATH
DEFAULT_CHECKPOINT_NAME = "sam_med3d_turbo"
DEFAULT_MODEL_NAME = "SAM-Med3D-turbo"
DEFAULT_FEATURE_TYPE = "avg_pool"
SUPPORTED_FEATURE_TYPES = ("avg_pool",)
FEATURE_SURFACES = {
    "avg_pool": "official_image_encoder_post_neck_spatial_average",
}
DEFAULT_CROP_SIZE = 128
EXPECTED_FEATURE_DIM = 384
SOURCE_COMMIT = "f3de1fa10da98e46f49f176773d2b1e306ba131f"
CHECKPOINT_SHA256 = "899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c"
INPUT_PREPROCESSING = "reform3d_percentile_0.05_99.95_to_minus1_1"


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
    add_source_path(SAM_MED3D_SOURCE_ROOT, "segment_anything/build_sam3D.py", ("segment_anything", "utils"))

def _extract_image_encoder_state(checkpoint: object) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported SAM-Med3D checkpoint type: {type(checkpoint)!r}")

    state_dict = checkpoint
    for key in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            state_dict = value
            break

    image_encoder_state: Dict[str, torch.Tensor] = {}
    for raw_key, value in state_dict.items():
        key = str(raw_key).removeprefix("module.")
        if key.startswith("image_encoder.") and torch.is_tensor(value):
            image_encoder_state[key.removeprefix("image_encoder.")] = value
    if not image_encoder_state:
        raise RuntimeError("Checkpoint does not contain SAM-Med3D image-encoder weights")
    return image_encoder_state


def build_image_encoder(checkpoint_path: Path | str) -> nn.Module:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM-Med3D checkpoint not found: {checkpoint_path}")

    _ensure_source_available()
    from segment_anything.build_sam3D import sam_model_registry3D

    sam_model = sam_model_registry3D["vit_b_ori"](checkpoint=None)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    image_encoder_state = _extract_image_encoder_state(checkpoint)
    sam_model.image_encoder.load_state_dict(image_encoder_state, strict=True)
    image_encoder = sam_model.image_encoder
    image_encoder.eval()
    for parameter in image_encoder.parameters():
        parameter.requires_grad_(False)
    return image_encoder


class SAMMed3DFeatureExtractor:
    """Frozen SAM-Med3D image encoder exposing its pooled native output."""

    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        self.model = build_image_encoder(checkpoint_path)
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
                f"SAM-Med3D expects {DEFAULT_CROP_SIZE}^3 inputs, got {tuple(batch.shape[2:])}"
            )

        batch = batch.to(
            device=self.config.device,
            dtype=self.config.dtype,
            non_blocking=True,
        )
        image_embedding = self.model(batch)
        avg_pool = image_embedding.flatten(2).mean(dim=-1)
        if avg_pool.shape[-1] != EXPECTED_FEATURE_DIM:
            raise RuntimeError(
                f"Expected {EXPECTED_FEATURE_DIM} SAM-Med3D features, got {avg_pool.shape[-1]}"
            )
        return {"avg_pool": avg_pool}


METHOD = PrimaryFeatureMethod(
    display_name="SAM-Med3D",
    model_name=DEFAULT_MODEL_NAME,
    checkpoint_name=DEFAULT_CHECKPOINT_NAME,
    checkpoint_path=DEFAULT_CHECKPOINT_PATH,
    feature_type=DEFAULT_FEATURE_TYPE,
    feature_surface=FEATURE_SURFACES[DEFAULT_FEATURE_TYPE],
    crop_size=DEFAULT_CROP_SIZE,
    extractor_factory=SAMMed3DFeatureExtractor,
    resolve_dtype=resolve_dtype,
    source_commit=SOURCE_COMMIT,
    checkpoint_sha256=CHECKPOINT_SHA256,
    input_preprocessing=INPUT_PREPROCESSING,
    default_batch_size=2,
)
