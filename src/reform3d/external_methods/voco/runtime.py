"""VoCo runtime for REFORM-3D evaluations."""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from reform3d.core.paths import resolve_external_method_checkpoint_path


THIS_FILE = Path(__file__).resolve()
VOCO_ROOT = THIS_FILE.parent
PROJECT_ROOT = THIS_FILE.parents[3]

DEFAULT_CHECKPOINT_PATH = resolve_external_method_checkpoint_path("voco", "VoCo_B_SSL_head.pt")
DEFAULT_CHECKPOINT_NAME = "voco_b_ssl_head"
DEFAULT_MODEL_NAME = "VoCo B SSL Head"
DEFAULT_FEATURE_TYPE = "pooled_multiscale"
SUPPORTED_FEATURE_TYPES = ("pooled_multiscale", "final_stage_pool")
FEATURE_SURFACES = {
    "pooled_multiscale": "pooled_multiscale_encoder_descriptor",
    "final_stage_pool": "final_stage_encoder_pool",
}
DEFAULT_CROP_SIZE = 96
DEFAULT_IN_CHANNELS = 1


logger = logging.getLogger(__name__)


def ensure_import_paths(extra_paths: Iterable[Path] = ()) -> None:
    """Add VoCo source directories needed by runtime checks."""

    for path in (PROJECT_ROOT, VOCO_ROOT, *extra_paths):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


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


def _load_monai_runtime() -> tuple[Any, Any, Any]:
    """Load the MONAI components used by the VoCo backbone."""

    UnetrBasicBlock = importlib.import_module("monai.networks.blocks").UnetrBasicBlock
    SwinViT = importlib.import_module("monai.networks.nets.swin_unetr").SwinTransformer
    ensure_tuple_rep = importlib.import_module("monai.utils").ensure_tuple_rep
    return UnetrBasicBlock, SwinViT, ensure_tuple_rep


def _extract_state_dict(model_payload: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if "state_dict" in model_payload:
        return dict(model_payload["state_dict"])
    if "network_weights" in model_payload:
        return dict(model_payload["network_weights"])
    if "net" in model_payload:
        return dict(model_payload["net"])
    return dict(model_payload)


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        cleaned_key = key
        if cleaned_key.startswith("module."):
            cleaned_key = cleaned_key[len("module.") :]
        if cleaned_key.startswith("backbone."):
            cleaned_key = cleaned_key[len("backbone.") :]
        if cleaned_key.startswith("_orig_mod."):
            cleaned_key = cleaned_key[len("_orig_mod.") :]
        cleaned_key = cleaned_key.replace("swin_vit", "swinViT")
        cleaned[cleaned_key] = value
    return cleaned


def infer_feature_size(state_dict: Dict[str, torch.Tensor]) -> int:
    """Infer the VoCo Swin feature size from checkpoint tensors."""

    patch_embed_key = "swinViT.patch_embed.proj.weight"
    if patch_embed_key not in state_dict:
        raise KeyError(f"Could not infer VoCo feature size from checkpoint key '{patch_embed_key}'")
    return int(state_dict[patch_embed_key].shape[0])


def infer_embedding_dim(feature_size: int) -> int:
    """Return the pooled multiscale descriptor dimension."""

    return int(feature_size * 24)


def normalize_voco_input(batch: torch.Tensor) -> torch.Tensor:
    """Map benchmark tensors to the [0, 1] input range used by VoCo."""

    normalized = batch.to(dtype=torch.float32)
    min_value = float(normalized.amin().item())
    if min_value < 0.0:
        normalized = torch.clamp(normalized, -1.0, 1.0)
        normalized = (normalized + 1.0) / 2.0
    else:
        normalized = torch.clamp(normalized, 0.0, 1.0)
    return normalized


def inspect_voco_runtime(checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH) -> Dict[str, Any]:
    """Inspect the local VoCo checkpoint and Python dependencies."""

    ensure_import_paths()
    checkpoint_path = Path(checkpoint_path)
    blockers: list[str] = []
    warnings: list[str] = []

    if not checkpoint_path.exists():
        blockers.append(f"VoCo checkpoint not found: {checkpoint_path}")

    for module_name, label in {
        "monai.networks.blocks": "MONAI network blocks",
        "monai.networks.nets.swin_unetr": "MONAI SwinUNETR",
        "monai.utils": "MONAI utilities",
    }.items():
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            blockers.append(f"Missing {label}: {exc}")
        except Exception as exc:
            blockers.append(f"Failed to import {label}: {exc}")

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if not torch.cuda.is_available():
        warnings.append("CUDA is not available in the current environment; VoCo inference will run on CPU.")

    return {
        "python_version": python_version,
        "warnings": warnings,
        "blockers": blockers,
        "checkpoint_path": str(checkpoint_path),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": torch.__version__,
    }


def format_runtime_report(report: Dict[str, Any]) -> str:
    """Format a VoCo runtime-inspection report."""

    lines = [
        f"Python: {report['python_version']}",
        f"Torch: {report['torch_version']}",
        f"CUDA available: {report['cuda_available']}",
        f"Checkpoint: {report['checkpoint_path']}",
    ]
    warnings = report.get("warnings") or []
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in warnings)
    blockers = report.get("blockers") or []
    if blockers:
        lines.append("Blockers:")
        lines.extend(f"- {blocker}" for blocker in blockers)
    else:
        lines.append("Blockers: none")
    return "\n".join(lines)


class VoCoBackbone(nn.Module):
    """SwinUNETR encoder used to reproduce VoCo pooled descriptors."""

    def __init__(
        self,
        feature_size: int,
        in_channels: int = DEFAULT_IN_CHANNELS,
        spatial_dims: int = 3,
        dropout_path_rate: float = 0.0,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        UnetrBasicBlock, SwinViT, ensure_tuple_rep = _load_monai_runtime()
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)
        norm_name = "instance"

        self.swinViT = SwinViT(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=dropout_path_rate,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            use_v2=True,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

    def pool_encs(self, encs: list[torch.Tensor]) -> list[torch.Tensor]:
        batch_size = encs[0].shape[0]
        return [F.adaptive_avg_pool3d(enc, (1, 1, 1)).view(batch_size, -1) for enc in encs]

    def forward_encs(self, encs: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(self.pool_encs(encs), dim=1)

    def extract_encoder_features(self, x_in: torch.Tensor) -> list[torch.Tensor]:
        hidden_states_out = self.swinViT(x_in)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        return [enc0, enc1, enc2, enc3, dec4]

    def extract_feature_dict(self, x_in: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled = self.pool_encs(self.extract_encoder_features(x_in))
        return {
            "pooled_multiscale": torch.cat(pooled, dim=1),
            "final_stage_pool": pooled[-1],
        }

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        return self.extract_feature_dict(x_in)[DEFAULT_FEATURE_TYPE]


def _load_voco_backbone(
    checkpoint_path: Path,
    in_channels: int = DEFAULT_IN_CHANNELS,
    spatial_dims: int = 3,
    dropout_path_rate: float = 0.0,
    use_checkpoint: bool = False,
) -> tuple[VoCoBackbone, Dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _normalize_state_dict_keys(_extract_state_dict(checkpoint))
    feature_size = infer_feature_size(state_dict)
    model = VoCoBackbone(
        feature_size=feature_size,
        in_channels=in_channels,
        spatial_dims=spatial_dims,
        dropout_path_rate=dropout_path_rate,
        use_checkpoint=use_checkpoint,
    )

    current_state_dict = model.state_dict()
    compatible_state_dict = {
        key: state_dict[key]
        if key in state_dict and tuple(state_dict[key].shape) == tuple(current_state_dict[key].shape)
        else current_state_dict[key]
        for key in current_state_dict
    }
    model.load_state_dict(compatible_state_dict, strict=True)

    missing_checkpoint_keys = [key for key in current_state_dict if key not in state_dict]
    if missing_checkpoint_keys:
        logger.warning(
            "VoCo checkpoint did not include %d backbone keys; using initialized defaults for those weights.",
            len(missing_checkpoint_keys),
        )

    metadata = {
        "feature_size": feature_size,
        "embed_dim": infer_embedding_dim(feature_size),
        "checkpoint_path": str(checkpoint_path),
    }
    return model, metadata


class VoCoFeatureExtractor:
    """VoCo feature extractor exposing pooled multiscale descriptors."""

    def __init__(
        self,
        checkpoint_path: Path,
        device: str,
        dtype: torch.dtype,
        crop_size: int = DEFAULT_CROP_SIZE,
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"VoCo checkpoint not found: {checkpoint_path}")

        model, metadata = _load_voco_backbone(checkpoint_path)
        model.eval()
        model.to(device=device, dtype=dtype)

        self.model = model
        self.config = SimpleNamespace(
            device=device,
            dtype=dtype,
            input_size=int(crop_size),
            crop_size=int(crop_size),
            checkpoint_path=str(checkpoint_path),
            feature_size=int(metadata["feature_size"]),
            embed_dim=int(metadata["embed_dim"]),
        )

    @torch.inference_mode()
    def extract(self, batch: torch.Tensor, normalize: bool = True) -> Dict[str, torch.Tensor]:
        if batch.ndim != 5:
            raise ValueError(f"Expected [B, C, D, H, W] input, got shape {tuple(batch.shape)}")

        if normalize:
            batch = normalize_voco_input(batch)
        batch = batch.to(device=self.config.device, dtype=self.config.dtype, non_blocking=True)
        return self.model.extract_feature_dict(batch)
