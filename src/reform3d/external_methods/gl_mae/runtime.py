"""GL-MAE encoder runtime for REFORM-3D evaluation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import torch
import torch.nn as nn

from reform3d.core.paths import resolve_external_method_checkpoint_path
from reform3d.external_methods._primary_feature import PrimaryFeatureMethod
from reform3d.external_methods._sources import source_root, add_source_path


GL_MAE_SOURCE_ROOT = source_root("gl_mae", Path(__file__).resolve().parents[1] / "GL-MAE")
OFFICIAL_MODEL_PATH = GL_MAE_SOURCE_ROOT / "Finetune" / "models_3dvit.py"
REGISTERED_CHECKPOINT_PATH = resolve_external_method_checkpoint_path(
    "gl_mae",
    "checkpoint-1599.pth",
)
LOCAL_CHECKPOINT_PATH = GL_MAE_SOURCE_ROOT / "checkpoint" / "checkpoint-1599.pth"
DEFAULT_CHECKPOINT_PATH = REGISTERED_CHECKPOINT_PATH
DEFAULT_CHECKPOINT_NAME = "gl_mae_vit_base_patch16"
DEFAULT_MODEL_NAME = "GL-MAE ViT-B/16"
DEFAULT_FEATURE_TYPE = "cls"
SUPPORTED_FEATURE_TYPES = ("cls",)
FEATURE_SURFACES = {
    "cls": "official_encoder_final_normalized_cls_token",
}
DEFAULT_CROP_SIZE = 96
PATCH_SIZE = 16
EXPECTED_FEATURE_DIM = 768
# Pretrained positional grid (160^3 ROI / patch 16 -> 10^3) interpolated to the
# evaluation grid (96^3 ROI / patch 16 -> 6^3).
PRETRAINED_POS_GRID = (10, 10, 10)
EVALUATION_POS_GRID = (6, 6, 6)
SOURCE_COMMIT = "44cd68d6d60278a7ef25dc388f4b79704a677d4b"
CHECKPOINT_SHA256 = "88b637598b816dbc097fc92757c1d39b0b734a1c666a58ec504d34bf882c7a5e"
INPUT_PREPROCESSING = "reform3d_percentile_0.05_99.95_to_minus1_1"
# Provenance only. The released checkpoint (checkpoint['args']) recorded a CT
# window of ScaleIntensityRange(a_min=-1000, a_max=1000 -> [0, 1], clip=True) at
# a 96^3 ROI. REFORM-3D does not reproduce this window; it applies the shared
# percentile preprocessing above uniformly across CT and MR for every encoder.
ORIGINAL_CT_WINDOW = {
    "a_min": -1000.0,
    "a_max": 1000.0,
    "b_min": 0.0,
    "b_max": 1.0,
    "clip": True,
    "roi": (96, 96, 96),
}
CHECKPOINT_STUDENT_ENCODER_PREFIX = "module.backbone."


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
    if not GL_MAE_SOURCE_ROOT.exists():
        raise FileNotFoundError(f"GL-MAE source not found: {GL_MAE_SOURCE_ROOT}")
    if not OFFICIAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"GL-MAE official model not found: {OFFICIAL_MODEL_PATH}"
        )


def _load_official_module():
    """Import the official ``models_3dvit`` module directly from its file."""
    _ensure_source_available()
    spec = importlib.util.spec_from_file_location(
        "gl_mae_official_models_3dvit",
        OFFICIAL_MODEL_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load GL-MAE model from {OFFICIAL_MODEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_encoder_state(
    checkpoint: object,
    reference_state: Dict[str, torch.Tensor],
    interpolate_pos_embed_3d,
) -> Dict[str, torch.Tensor]:
    """Select and align the student encoder weights for strict loading.

    The student checkpoint stores the full pre-training backbone (encoder plus
    MAE decoder) under ``module.backbone.`` and a projection head under
    ``module.head.``. Only the encoder-only key set is retained; the decoder,
    mask token, and projection head are dropped. The pretrained positional grid
    is interpolated to the evaluation grid.
    """
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported GL-MAE checkpoint type: {type(checkpoint)!r}")
    student = checkpoint.get("student")
    if not isinstance(student, dict):
        raise RuntimeError("GL-MAE checkpoint does not contain a 'student' state")

    reference_keys = set(reference_state.keys())
    encoder_state: Dict[str, torch.Tensor] = {}
    for key, value in student.items():
        if not str(key).startswith(CHECKPOINT_STUDENT_ENCODER_PREFIX):
            continue
        stripped = str(key)[len(CHECKPOINT_STUDENT_ENCODER_PREFIX):]
        if stripped in reference_keys and torch.is_tensor(value):
            encoder_state[stripped] = value

    missing_keys = reference_keys - set(encoder_state.keys())
    if missing_keys:
        raise RuntimeError(
            f"GL-MAE checkpoint missing encoder keys: {sorted(missing_keys)}"
        )
    unexpected_keys = set(encoder_state.keys()) - reference_keys
    if unexpected_keys:
        raise RuntimeError(
            f"GL-MAE checkpoint has unexpected encoder keys: {sorted(unexpected_keys)}"
        )

    checkpoint_pos_embed = encoder_state["pos_embed"]
    reference_pos_embed = reference_state["pos_embed"]
    if tuple(checkpoint_pos_embed.shape) != tuple(reference_pos_embed.shape):
        encoder_state["pos_embed"] = interpolate_pos_embed_3d(
            checkpoint_pos_embed,
            reference_pos_embed,
            num_extra_tokens=1,
        )
    return encoder_state


class GLMAEEncoder(nn.Module):
    def __init__(self, checkpoint_path: Path | str) -> None:
        super().__init__()
        module = _load_official_module()
        self.model = module.vit_base_patch16_3d(
            img_size=(DEFAULT_CROP_SIZE, DEFAULT_CROP_SIZE, DEFAULT_CROP_SIZE),
            in_chans=1,
            encoder_only=True,
        )
        if self.model.embed_dim != EXPECTED_FEATURE_DIM:
            raise RuntimeError(
                f"Expected GL-MAE embed dim {EXPECTED_FEATURE_DIM}, "
                f"got {self.model.embed_dim}"
            )
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        encoder_state = _build_encoder_state(
            checkpoint,
            self.model.state_dict(),
            module.interpolate_pos_embed_3d,
        )
        self.model.load_state_dict(encoder_state, strict=True)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return self.model(volume)


class GLMAEFeatureExtractor:
    """Frozen GL-MAE encoder exposing its final normalized CLS representation."""

    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"GL-MAE checkpoint not found: {checkpoint_path}")
        self.model = GLMAEEncoder(checkpoint_path)
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
                f"GL-MAE expects {DEFAULT_CROP_SIZE}^3 inputs, got {tuple(batch.shape[2:])}"
            )
        batch = batch.to(
            device=self.config.device,
            dtype=self.config.dtype,
            non_blocking=True,
        )
        cls = self.model(batch)
        if cls.ndim != 2 or cls.shape[-1] != EXPECTED_FEATURE_DIM:
            raise RuntimeError(
                f"Expected [B, {EXPECTED_FEATURE_DIM}] GL-MAE features, got {tuple(cls.shape)}"
            )
        return {"cls": cls}


METHOD = PrimaryFeatureMethod(
    display_name="GL-MAE",
    model_name=DEFAULT_MODEL_NAME,
    checkpoint_name=DEFAULT_CHECKPOINT_NAME,
    checkpoint_path=DEFAULT_CHECKPOINT_PATH,
    feature_type=DEFAULT_FEATURE_TYPE,
    feature_surface=FEATURE_SURFACES[DEFAULT_FEATURE_TYPE],
    crop_size=DEFAULT_CROP_SIZE,
    extractor_factory=GLMAEFeatureExtractor,
    resolve_dtype=resolve_dtype,
    source_commit=SOURCE_COMMIT,
    checkpoint_sha256=CHECKPOINT_SHA256,
    input_preprocessing=INPUT_PREPROCESSING,
    default_batch_size=4,
)
