"""MiM encoder runtime for REFORM-3D evaluation."""

from __future__ import annotations

import io
import importlib.util
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import torch
import torch.nn as nn

from reform3d.core.paths import resolve_external_method_checkpoint_path
from reform3d.external_methods._primary_feature import PrimaryFeatureMethod
from reform3d.external_methods._sources import source_root, add_source_path


MIM_SOURCE_ROOT = source_root("mim", Path(__file__).resolve().parents[1] / "MiM")
MIM_PRETRAIN_ROOT = MIM_SOURCE_ROOT / "Pretrain"
REGISTERED_CHECKPOINT_PATH = resolve_external_method_checkpoint_path(
    "mim",
    "checkpoint-final.pth",
)
LOCAL_CHECKPOINT_PATH = MIM_SOURCE_ROOT / "checkpoint" / "ckpts.zip"
DEFAULT_CHECKPOINT_PATH = REGISTERED_CHECKPOINT_PATH
DEFAULT_CHECKPOINT_NAME = "mim_convvit_base"
DEFAULT_MODEL_NAME = "MiM ConvViT-B"
DEFAULT_FEATURE_TYPE = "avg_pool"
SUPPORTED_FEATURE_TYPES = ("avg_pool",)
FEATURE_SURFACES = {
    "avg_pool": "deterministic_unmasked_final_encoder_token_average",
}
DEFAULT_CROP_SIZE = 96
EXPECTED_FEATURE_DIM = 384
SOURCE_COMMIT = "f6c949d1e5c1df8627365a9ae9c1cdf4ee8f6019"
CHECKPOINT_SHA256 = "67fe853151bd46014982d9169870470ba18ae71063b42a804b0c90afcc77c737"
INPUT_PREPROCESSING = "reform3d_percentile_0.05_99.95_to_minus1_1"
ENCODER_PREFIXES = (
    "patch_embed1.",
    "patch_embed2.",
    "patch_embed3.",
    "patch_embed4.",
    "patch_embed5.",
    "stage1_output_decode.",
    "stage2_output_decode.",
    "stage3_output_decode.",
    "pos_embed",
    "blocks1.",
    "blocks2.",
    "blocks3.",
    "blocks4.",
    "norm.",
)


def resolve_dtype(dtype_name: str) -> torch.dtype:
    normalized = str(dtype_name).strip().lower()
    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _load_encoder_components():
    path = MIM_PRETRAIN_ROOT / "vision_transformer.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"MiM encoder source not found: {path}; configure REFORM3D_MIM_SOURCE_ROOT"
        )
    spec = importlib.util.spec_from_file_location("reform3d_mim_vision_transformer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load MiM source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Block, module.CBlock3d, module.PatchEmbed3d

def _load_checkpoint(path: Path) -> dict:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [
                name
                for name in archive.namelist()
                if name.endswith((".pth", ".pt"))
            ]
            if len(members) != 1:
                raise RuntimeError(
                    f"Expected one MiM checkpoint in {path}, found {len(members)}"
                )
            payload = io.BytesIO(archive.read(members[0]))
            try:
                checkpoint = torch.load(
                    payload,
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                payload.seek(0)
                checkpoint = torch.load(payload, map_location="cpu")
    else:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported MiM checkpoint type: {type(checkpoint)!r}")
    return checkpoint


class MiMEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        Block, CBlock3d, PatchEmbed3d = _load_encoder_components()

        dimensions = (48, 96, 192, 384)
        image_sizes = (
            (96, 96, 96),
            (48, 48, 48),
            (24, 24, 24),
            (12, 12, 12),
        )
        self.patch_embed1 = PatchEmbed3d(
            img_size=image_sizes[0],
            patch_size=2,
            in_chans=1,
            embed_dim=dimensions[0],
        )
        self.patch_embed2 = PatchEmbed3d(
            img_size=image_sizes[1],
            patch_size=2,
            in_chans=dimensions[0],
            embed_dim=dimensions[1],
        )
        self.patch_embed3 = PatchEmbed3d(
            img_size=image_sizes[2],
            patch_size=2,
            in_chans=dimensions[1],
            embed_dim=dimensions[2],
        )
        self.patch_embed4 = PatchEmbed3d(
            img_size=image_sizes[3],
            patch_size=2,
            in_chans=dimensions[2],
            embed_dim=dimensions[3],
        )
        self.patch_embed5 = nn.Linear(dimensions[3], dimensions[3])
        self.stage1_output_decode = nn.Conv3d(
            dimensions[0],
            dimensions[3],
            kernel_size=8,
            stride=8,
        )
        self.stage2_output_decode = nn.Conv3d(
            dimensions[1],
            dimensions[3],
            kernel_size=4,
            stride=4,
        )
        self.stage3_output_decode = nn.Conv3d(
            dimensions[2],
            dimensions[3],
            kernel_size=2,
            stride=2,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed4.num_patches, dimensions[3]),
            requires_grad=False,
        )
        self.blocks1 = nn.ModuleList(
            [
                CBlock3d(
                    dim=dimensions[0],
                    num_heads=12,
                    mlp_ratio=4,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(2)
            ]
        )
        self.blocks2 = nn.ModuleList(
            [
                CBlock3d(
                    dim=dimensions[1],
                    num_heads=12,
                    mlp_ratio=4,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(2)
            ]
        )
        self.blocks3 = nn.ModuleList(
            [
                CBlock3d(
                    dim=dimensions[2],
                    num_heads=12,
                    mlp_ratio=4,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(2)
            ]
        )
        block_args = SimpleNamespace(sr_ratio=1)
        self.blocks4 = nn.ModuleList(
            [
                Block(
                    dim=dimensions[3],
                    num_heads=12,
                    mlp_ratio=4,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                    args=block_args,
                )
                for _ in range(11)
            ]
        )
        self.norm = nn.LayerNorm(dimensions[3], eps=1e-6)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        stage1 = self.patch_embed1(volume)
        for block in self.blocks1:
            stage1 = block(stage1)
        stage1_tokens = self.stage1_output_decode(stage1).flatten(2).transpose(1, 2)

        stage2 = self.patch_embed2(stage1)
        for block in self.blocks2:
            stage2 = block(stage2)
        stage2_tokens = self.stage2_output_decode(stage2).flatten(2).transpose(1, 2)

        stage3 = self.patch_embed3(stage2)
        for block in self.blocks3:
            stage3 = block(stage3)
        stage3_tokens = self.stage3_output_decode(stage3).flatten(2).transpose(1, 2)

        stage4 = self.patch_embed4(stage3).flatten(2).transpose(1, 2)
        tokens = self.patch_embed5(stage4) + self.pos_embed
        for block in self.blocks4:
            tokens = block(tokens)
        return self.norm(tokens + stage1_tokens + stage2_tokens + stage3_tokens)


def build_encoder(checkpoint_path: Path | str) -> MiMEncoder:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"MiM checkpoint not found: {checkpoint_path}")
    checkpoint = _load_checkpoint(checkpoint_path)
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict):
        raise RuntimeError("MiM checkpoint does not contain a model state")
    encoder_state = {
        str(key): value
        for key, value in state_dict.items()
        if str(key).startswith(ENCODER_PREFIXES)
    }
    model = MiMEncoder()
    model.load_state_dict(encoder_state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class MiMFeatureExtractor:
    """Frozen MiM encoder exposing the unmasked final token average."""

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
                f"MiM expects {DEFAULT_CROP_SIZE}^3 inputs, got {tuple(batch.shape[2:])}"
            )
        batch = batch.to(
            device=self.config.device,
            dtype=self.config.dtype,
            non_blocking=True,
        )
        avg_pool = self.model(batch).mean(dim=1)
        if avg_pool.shape[-1] != EXPECTED_FEATURE_DIM:
            raise RuntimeError(
                f"Expected {EXPECTED_FEATURE_DIM} MiM features, got {avg_pool.shape[-1]}"
            )
        return {"avg_pool": avg_pool}


METHOD = PrimaryFeatureMethod(
    display_name="MiM",
    model_name=DEFAULT_MODEL_NAME,
    checkpoint_name=DEFAULT_CHECKPOINT_NAME,
    checkpoint_path=DEFAULT_CHECKPOINT_PATH,
    feature_type=DEFAULT_FEATURE_TYPE,
    feature_surface=FEATURE_SURFACES[DEFAULT_FEATURE_TYPE],
    crop_size=DEFAULT_CROP_SIZE,
    extractor_factory=MiMFeatureExtractor,
    resolve_dtype=resolve_dtype,
    source_commit=SOURCE_COMMIT,
    checkpoint_sha256=CHECKPOINT_SHA256,
    input_preprocessing=INPUT_PREPROCESSING,
    default_batch_size=4,
)
