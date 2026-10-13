"""BrainAIC runtime and feature extractor for REFORM-3D evaluation.

The runtime loads the BrainAIC ViT backbone checkpoint and exposes a compact
``extract`` interface used by Acquisition Robustness and Cross-Modal feature
extraction.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import torch
import torch.nn as nn
from monai.networks.nets import ViT

from reform3d.core.paths import resolve_external_method_checkpoint_path


BRAINIAC_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_PATH = resolve_external_method_checkpoint_path("brainiac", "BrainIAC.ckpt")
DEFAULT_CHECKPOINT_NAME = "brainiac"
DEFAULT_MODEL_NAME = "BrainIAC"
DEFAULT_FEATURE_TYPE = "cls"
SUPPORTED_FEATURE_TYPES = ("cls", "avg_pool", "multilayer")
FEATURE_SURFACES = {
    "cls": "first_final_patch_token_legacy_cls_label",
    "avg_pool": "final_patch_token_average",
    "multilayer": "last_four_layer_first_token_concat",
}
DEFAULT_CROP_SIZE = 96
MULTILAYER_LAYER_INDICES = (-4, -3, -2, -1)


def resolve_dtype(dtype_name: str) -> torch.dtype:
    """Resolve a CLI dtype name to a torch dtype."""

    normalized = str(dtype_name).strip().lower()
    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def normalize_nonzero(batch: torch.Tensor) -> torch.Tensor:
    """Normalize nonzero voxels in each volume independently."""

    normalized = batch.clone()
    for batch_idx in range(normalized.shape[0]):
        sample = normalized[batch_idx]
        nonzero_mask = sample != 0
        if not torch.any(nonzero_mask):
            continue

        values = sample[nonzero_mask]
        mean = values.mean()
        std = values.std(unbiased=False)
        if torch.isfinite(std) and float(std) > 0.0:
            sample[nonzero_mask] = (values - mean) / std
        else:
            sample[nonzero_mask] = values - mean
    return normalized


class ViTBackboneNet(nn.Module):
    """BrainAIC ViT-B backbone exposing final and intermediate token descriptors."""

    def __init__(self, checkpoint_path: Path | str) -> None:
        super().__init__()
        self.backbone = ViT(
            in_channels=1,
            img_size=(96, 96, 96),
            patch_size=(16, 16, 16),
            hidden_size=768,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            save_attn=True,
        )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        backbone_state_dict = {
            key[len("backbone.") :]: value
            for key, value in state_dict.items()
            if key.startswith("backbone.")
        }
        self.backbone.load_state_dict(backbone_state_dict, strict=True)

    def forward_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features = self.backbone(x)
        return features[0], list(features[1])

    def extract_feature_dict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        tokens, hidden_states = self.forward_tokens(x)
        selected_layers = [self.backbone.norm(hidden_states[index])[:, 0] for index in MULTILAYER_LAYER_INDICES]
        return {
            "cls": tokens[:, 0],
            "avg_pool": tokens.mean(dim=1),
            "multilayer": torch.cat(selected_layers, dim=-1),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract_feature_dict(x)[DEFAULT_FEATURE_TYPE]


class BrainIACFeatureExtractor:
    """Feature extractor interface used by REFORM-3D external-method runs."""

    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"BrainIAC checkpoint not found: {checkpoint_path}")

        self.model = ViTBackboneNet(checkpoint_path)
        self.model.eval()
        self.model.to(device=device, dtype=dtype)
        self.config = SimpleNamespace(
            device=device,
            dtype=dtype,
            crop_size=DEFAULT_CROP_SIZE,
            input_size=DEFAULT_CROP_SIZE,
            checkpoint_path=str(checkpoint_path),
        )

    @torch.inference_mode()
    def extract(self, batch: torch.Tensor, normalize: bool = True) -> Dict[str, torch.Tensor]:
        if batch.ndim != 5:
            raise ValueError(f"Expected [B, C, H, W, D] input, got shape {tuple(batch.shape)}")
        batch = batch.to(device=self.config.device, dtype=self.config.dtype, non_blocking=True)
        if normalize:
            batch = normalize_nonzero(batch)
        return self.model.extract_feature_dict(batch)
