"""Hi-End-MAE encoder runtime for REFORM-3D evaluation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from reform3d.core.paths import resolve_external_method_checkpoint_path
from reform3d.external_methods._primary_feature import PrimaryFeatureMethod


HI_END_MAE_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "Hi-End-MAE"
REGISTERED_CHECKPOINT_PATH = resolve_external_method_checkpoint_path(
    "hi_end_mae",
    "hi_end_mae_10k.pth",
)
LOCAL_CHECKPOINT_PATH = HI_END_MAE_SOURCE_ROOT / "checkpoint" / "hi_end_mae_10k.pth"
DEFAULT_CHECKPOINT_PATH = REGISTERED_CHECKPOINT_PATH
DEFAULT_CHECKPOINT_NAME = "hi_end_mae_vit_p12_1536"
DEFAULT_MODEL_NAME = "Hi-End-MAE ViT-p12/1536"
DEFAULT_FEATURE_TYPE = "cls"
SUPPORTED_FEATURE_TYPES = ("cls",)
FEATURE_SURFACES = {
    "cls": "official_encoder_final_normalized_cls_token",
}
DEFAULT_CROP_SIZE = 96
PATCH_SIZE = 12
EXPECTED_FEATURE_DIM = 1536
SOURCE_COMMIT = "b0df9c6e6cc86bfff5b6e83beb4ea6a2e9f39b1c"
CHECKPOINT_SHA256 = "549994dd1e0cf3bbe6cd2ecc604a0457178ea033f9f3ce14376b5dd994dc98b5"
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


class PatchEmbed3D(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        norm_layer=None,
        flatten: bool = True,
        in_chan_last: bool = True,
    ) -> None:
        super().__init__()
        self.img_size = (img_size, img_size, img_size)
        self.patch_size = (patch_size, patch_size, patch_size)
        self.in_chans = in_chans
        self.num_patches = 1
        self.flatten = flatten
        self.in_chan_last = in_chan_last
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, patch_count, flattened_size = x.shape
        expected_size = int(np.prod(self.img_size)) * self.in_chans
        if flattened_size != expected_size:
            raise ValueError(
                f"Expected flattened patch size {expected_size}, got {flattened_size}"
            )
        if self.in_chan_last:
            x = x.reshape(
                batch_size * patch_count,
                *self.img_size,
                self.in_chans,
            ).permute(0, 4, 1, 2, 3)
        else:
            x = x.reshape(
                batch_size * patch_count,
                self.in_chans,
                *self.img_size,
            )
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


class MLP(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dimension, dimension * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dimension * 4, dimension)
        self.drop = nn.Dropout(0.0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.fc1(tokens)
        tokens = self.act(tokens)
        tokens = self.drop(tokens)
        tokens = self.fc2(tokens)
        return self.drop(tokens)


class Attention(nn.Module):
    def __init__(self, dimension: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dimension, dimension * 3, bias=True)
        self.proj = nn.Linear(dimension, dimension)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, dimension = tokens.shape
        qkv = self.qkv(tokens).reshape(
            batch_size,
            token_count,
            3,
            self.num_heads,
            dimension // self.num_heads,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        tokens = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
        )
        tokens = tokens.transpose(1, 2).reshape(batch_size, token_count, dimension)
        return self.proj_drop(self.proj(tokens))


class TransformerBlock(nn.Module):
    def __init__(self, dimension: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension, eps=1e-6)
        self.attn = Attention(dimension, num_heads)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(dimension, eps=1e-6)
        self.mlp = MLP(dimension)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.drop_path(self.attn(self.norm1(tokens)))
        return tokens + self.drop_path(self.mlp(self.norm2(tokens)))


class VisionTransformerEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed3D(
            img_size=PATCH_SIZE,
            patch_size=PATCH_SIZE,
            in_chans=1,
            embed_dim=EXPECTED_FEATURE_DIM,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, EXPECTED_FEATURE_DIM))
        self.pos_drop = nn.Dropout(0.0)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(EXPECTED_FEATURE_DIM, num_heads=16)
                for _ in range(12)
            ]
        )
        self.norm = nn.LayerNorm(EXPECTED_FEATURE_DIM, eps=1e-6)
        self.head = nn.Identity()

    def forward(
        self,
        patches: torch.Tensor,
        position_embedding: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, patch_count, _ = patches.shape
        tokens = self.patch_embed(patches).reshape(
            batch_size,
            patch_count,
            EXPECTED_FEATURE_DIM,
        )
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_token, tokens), dim=1)
        cls_position = torch.zeros(
            batch_size,
            1,
            EXPECTED_FEATURE_DIM,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        tokens = self.pos_drop(
            tokens + torch.cat((cls_position, position_embedding), dim=1)
        )
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(self.norm(tokens))


def _patchify(volume: torch.Tensor) -> torch.Tensor:
    batch_size, channels, height, width, depth = volume.shape
    expected = (DEFAULT_CROP_SIZE,) * 3
    if (height, width, depth) != expected:
        raise ValueError(f"Expected {expected} input, got {(height, width, depth)}")
    grid = DEFAULT_CROP_SIZE // PATCH_SIZE
    patches = volume.reshape(
        batch_size,
        channels,
        grid,
        PATCH_SIZE,
        grid,
        PATCH_SIZE,
        grid,
        PATCH_SIZE,
    )
    return patches.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(
        batch_size,
        grid**3,
        PATCH_SIZE**3 * channels,
    )


def _extract_checkpoint_state(
    checkpoint: object,
) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported Hi-End-MAE checkpoint type: {type(checkpoint)!r}")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("Hi-End-MAE checkpoint state_dict is invalid")

    encoder_state = {
        str(key).removeprefix("module.encoder."): value
        for key, value in state_dict.items()
        if str(key).startswith("module.encoder.") and torch.is_tensor(value)
    }
    position_embedding = state_dict.get("module.encoder_pos_embed")
    if not encoder_state:
        raise RuntimeError("Checkpoint does not contain Hi-End-MAE encoder weights")
    if not torch.is_tensor(position_embedding):
        raise RuntimeError("Checkpoint does not contain Hi-End-MAE encoder positions")
    return encoder_state, position_embedding


class HiEndMAEEncoder(nn.Module):
    def __init__(self, checkpoint_path: Path | str) -> None:
        super().__init__()
        self.encoder = VisionTransformerEncoder()
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        encoder_state, position_embedding = _extract_checkpoint_state(checkpoint)
        self.encoder.load_state_dict(encoder_state, strict=True)
        self.register_buffer(
            "position_embedding",
            position_embedding.detach().clone(),
            persistent=True,
        )
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        patches = _patchify(volume)
        position_embedding = self.position_embedding.expand(
            volume.shape[0],
            -1,
            -1,
        )
        tokens = self.encoder(patches, position_embedding)
        return tokens[:, 0]


class HiEndMAEFeatureExtractor:
    """Frozen Hi-End-MAE encoder exposing its learned CLS representation."""

    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Hi-End-MAE checkpoint not found: {checkpoint_path}")
        self.model = HiEndMAEEncoder(checkpoint_path)
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
        batch = batch.to(
            device=self.config.device,
            dtype=self.config.dtype,
            non_blocking=True,
        )
        cls = self.model(batch)
        if cls.shape[-1] != EXPECTED_FEATURE_DIM:
            raise RuntimeError(
                f"Expected {EXPECTED_FEATURE_DIM} Hi-End-MAE features, got {cls.shape[-1]}"
            )
        return {"cls": cls}


METHOD = PrimaryFeatureMethod(
    display_name="Hi-End-MAE",
    model_name=DEFAULT_MODEL_NAME,
    checkpoint_name=DEFAULT_CHECKPOINT_NAME,
    checkpoint_path=DEFAULT_CHECKPOINT_PATH,
    feature_type=DEFAULT_FEATURE_TYPE,
    feature_surface=FEATURE_SURFACES[DEFAULT_FEATURE_TYPE],
    crop_size=DEFAULT_CROP_SIZE,
    extractor_factory=HiEndMAEFeatureExtractor,
    resolve_dtype=resolve_dtype,
    source_commit=SOURCE_COMMIT,
    checkpoint_sha256=CHECKPOINT_SHA256,
    input_preprocessing=INPUT_PREPROCESSING,
    default_batch_size=4,
)
