"""Extract frozen 3D ViT features for Acquisition Robustness.

This module loads a registered checkpoint, batches preprocessed manifest
volumes, and writes feature bundles for CLS, average-pooled patch, or
multi-layer representations. The evaluation pipeline reads these bundles when
computing native-spacing and controlled-perturbation metrics.

Inputs:
    checkpoint_name: checkpoint key or alias from ``reform3d.core.checkpoints``.
    manifest_path: Acquisition manifest JSON.
    feature_type: ``cls``, ``avg_pool``, or ``multilayer``.
"""

import logging
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from reform3d.encoders.med3dino.vision_transformer import vit_large_3d, vit_base_3d
from reform3d.encoders.checkpoints import load_encoder_state

try:
    from .config import (
        CHECKPOINTS,
        CheckpointConfig,
        get_acquisition_manifest_path,
        get_output_paths,
        get_checkpoint_feature_dir,
        get_dataset_name_from_manifest_path,
        get_manifest_variant_from_manifest_path,
        resolve_acquisition_manifest_path,
    )
    from reform3d.core.checkpoints import resolve_checkpoint_config
except ImportError:  # pragma: no cover - direct script execution fallback
    from config import (
        CHECKPOINTS,
        CheckpointConfig,
        get_acquisition_manifest_path,
        get_output_paths,
        get_checkpoint_feature_dir,
        get_dataset_name_from_manifest_path,
        get_manifest_variant_from_manifest_path,
        resolve_acquisition_manifest_path,
    )
    from reform3d.core.checkpoints import resolve_checkpoint_config

logger = logging.getLogger(__name__)


def _resolve_features_dir(
    output_dir: Optional[Path],
    analysis_name: str = "default",
    manifest_variant: str = "original_bins",
) -> Path:
    if output_dir is not None:
        return output_dir
    return get_output_paths(analysis_name, manifest_variant)["features"]


@dataclass
class ExtractionConfig:
    """Feature extraction configuration; CPU inference uses float32."""
    extract_cls: bool = True
    extract_avg_pool: bool = True
    extract_multilayer: bool = True
    n_last_layers: int = 4  # Number of layers from end for multi-layer
    batch_size: int = 8
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    # For position embedding interpolation
    input_size: int = 96  # Will be overridden by checkpoint config

    def __post_init__(self):
        if torch.device(self.device).type == "cpu":
            self.dtype = torch.float32


class FeatureExtractor:
    """
    Feature extractor for 3D medical imaging foundation models.

    Supports extracting:
    - CLS token: Global representation [B, embed_dim]
    - Avg-pooled patches: Spatial-averaged representation [B, embed_dim]
    - Multi-layer: Concatenated features from last N layers [B, N * embed_dim]
    """

    def __init__(
        self,
        checkpoint_name: str,
        config: Optional[ExtractionConfig] = None,
        device: Optional[str] = None,
        checkpoint_path: Optional[Path] = None,
    ):
        """
        Initialize feature extractor.

        Args:
            checkpoint_name: checkpoint key from CHECKPOINTS (for example, "med3dino_rel_96")
            config: Extraction configuration
            device: Override device from config
        """
        self.config = replace(config) if config is not None else ExtractionConfig()
        if device is not None:
            self.config = replace(self.config, device=device)

        # Load checkpoint config
        try:
            self.ckpt_config = resolve_checkpoint_config(checkpoint_name)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        if checkpoint_path is not None:
            self.ckpt_config = replace(
                self.ckpt_config, raw_checkpoint_path=Path(checkpoint_path).expanduser().resolve()
            )
        self.checkpoint_name = self.ckpt_config.checkpoint_key

        # Override input size from checkpoint
        self.config.input_size = self.ckpt_config.crop_size

        # Build and load model
        self.model = self._build_model()
        self._load_weights()

        logger.info(
            f"Initialized FeatureExtractor with {checkpoint_name} "
            f"(crop_size={self.config.input_size}, embed_dim={self.embed_dim})"
        )

    def _build_model(self) -> nn.Module:
        """Build the ViT 3D model based on checkpoint architecture."""
        # Use block_chunks=4 to match checkpoint structure from FSDP training
        # Use init_values=1e-5 to enable LayerScale (matches training config)
        arch = getattr(self.ckpt_config, 'arch', 'vit_large_3d')

        model_kwargs = dict(
            img_size=self.config.input_size,
            patch_size=16,
            in_chans=1,
            drop_path_rate=0.0,  # No dropout during eval
            block_chunks=4,  # Match FSDP training structure
            init_values=1e-5,  # Enable LayerScale to load ls1.gamma, ls2.gamma
        )

        if arch == "vit_base_3d":
            logger.info(f"Building ViT-Base 3D model (embed_dim=768)")
            model = vit_base_3d(**model_kwargs)
        elif arch == "vit_large_3d":
            logger.info(f"Building ViT-Large 3D model (embed_dim=1024)")
            model = vit_large_3d(**model_kwargs)
        else:
            raise ValueError(f"Unsupported encoder architecture: {arch}")

        model.eval()
        model.to(self.config.device)

        self.embed_dim = model.embed_dim
        self.n_blocks = model.n_blocks

        return model

    def _load_weights(self):
        """Load pretrained weights from checkpoint."""
        ckpt_path = self.ckpt_config.path

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}. "
                f"Please ensure checkpoints are downloaded."
            )

        logger.info(f"Loading weights from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        # Extract teacher weights
        if "teacher" in checkpoint:
            state_dict = checkpoint["teacher"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Remove prefixes
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

        # Handle position embedding size mismatch via interpolation
        state_dict = self._interpolate_pos_embed_if_needed(state_dict)

        load_encoder_state(
            self.model,
            state_dict,
            allowed_extra_prefixes=("dino_head.", "ibot_head."),
        )

        logger.info(f"Weights loaded successfully from {ckpt_path.name}")

    def _interpolate_pos_embed_if_needed(self, state_dict: dict) -> dict:
        """
        Interpolate position embeddings if checkpoint and model sizes mismatch.

        This handles cases where the checkpoint was trained at a different
        resolution than the current model expects.
        """
        if "pos_embed" not in state_dict:
            return state_dict

        ckpt_pos_embed = state_dict["pos_embed"]  # [1, N_ckpt, embed_dim]
        model_pos_embed = self.model.pos_embed    # [1, N_model, embed_dim]

        if ckpt_pos_embed.shape == model_pos_embed.shape:
            return state_dict  # No interpolation needed

        logger.warning(
            f"Position embedding size mismatch: checkpoint has {ckpt_pos_embed.shape[1]} tokens, "
            f"model expects {model_pos_embed.shape[1]} tokens. Interpolating..."
        )

        # Separate CLS token and spatial tokens
        ckpt_cls = ckpt_pos_embed[:, :1, :]  # [1, 1, embed_dim]
        ckpt_spatial = ckpt_pos_embed[:, 1:, :]  # [1, N_ckpt-1, embed_dim]

        # Calculate source and target grid sizes (assuming cubic)
        n_ckpt_spatial = ckpt_spatial.shape[1]
        n_model_spatial = model_pos_embed.shape[1] - 1

        # Cube root for 3D
        ckpt_grid = round(n_ckpt_spatial ** (1/3))
        model_grid = round(n_model_spatial ** (1/3))

        logger.info(f"Interpolating pos_embed from {ckpt_grid}³ to {model_grid}³ grid")

        # Reshape to 3D grid for interpolation
        embed_dim = ckpt_spatial.shape[-1]
        ckpt_spatial = ckpt_spatial.reshape(1, ckpt_grid, ckpt_grid, ckpt_grid, embed_dim)
        ckpt_spatial = ckpt_spatial.permute(0, 4, 1, 2, 3)  # [1, embed_dim, D, H, W]

        # Trilinear interpolation
        interpolated = F.interpolate(
            ckpt_spatial.float(),
            size=(model_grid, model_grid, model_grid),
            mode='trilinear',
            align_corners=False,
        )

        # Reshape back
        interpolated = interpolated.permute(0, 2, 3, 4, 1)  # [1, D, H, W, embed_dim]
        interpolated = interpolated.reshape(1, -1, embed_dim)  # [1, N_model-1, embed_dim]

        # Concatenate CLS token back
        new_pos_embed = torch.cat([ckpt_cls, interpolated], dim=1)
        state_dict["pos_embed"] = new_pos_embed.to(ckpt_pos_embed.dtype)

        logger.info(f"Position embedding interpolated: {ckpt_pos_embed.shape} -> {new_pos_embed.shape}")

        return state_dict

    @torch.no_grad()
    def extract(
        self,
        x: torch.Tensor,
        normalize: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from a batch of volumes.

        Args:
            x: Input tensor [B, 1, D, H, W] normalized to the checkpoint's eval preprocessing range
            normalize: L2-normalize features

        Returns:
            Dictionary with feature types as keys:
            - "cls": CLS token features [B, embed_dim]
            - "avg_pool": Avg-pooled patch tokens [B, embed_dim]
            - "multilayer": Concatenated multi-layer features [B, n_layers * embed_dim]
            - "multilayer_cls": Per-layer CLS tokens [B, n_layers, embed_dim]
        """
        x = x.to(self.config.device, dtype=self.config.dtype)

        # Validate input shape
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input [B, C, D, H, W], got shape {x.shape}")

        features = {}

        autocast_context = (
            torch.cuda.amp.autocast(dtype=self.config.dtype)
            if torch.device(self.config.device).type == "cuda"
            and self.config.dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        with autocast_context:
            if self.config.extract_multilayer:
                # Get intermediate layer outputs including CLS tokens
                layer_outputs = self.model.get_intermediate_layers(
                    x,
                    n=self.config.n_last_layers,
                    reshape=False,
                    return_class_token=True,
                    norm=True,
                )
                # layer_outputs: tuple of (patch_tokens, cls_token) for each layer

                # Extract per-layer features
                cls_tokens = []
                patch_tokens_list = []

                for patch_tokens, cls_token in layer_outputs:
                    cls_tokens.append(cls_token)
                    patch_tokens_list.append(patch_tokens)

                # Stack per-layer CLS tokens [B, n_layers, embed_dim]
                multilayer_cls = torch.stack(cls_tokens, dim=1)

                # Concatenate all layers [B, n_layers * embed_dim]
                multilayer_concat = torch.cat(cls_tokens, dim=-1)

                features["multilayer"] = multilayer_concat.float()
                features["multilayer_cls"] = multilayer_cls.float()

                # Use last layer for CLS and avg_pool
                if self.config.extract_cls:
                    features["cls"] = cls_tokens[-1].float()

                if self.config.extract_avg_pool:
                    # Average pool patch tokens from last layer
                    avg_pool = patch_tokens_list[-1].mean(dim=1)
                    features["avg_pool"] = avg_pool.float()
            else:
                # Standard forward pass
                output = self.model.forward_features(x)

                if self.config.extract_cls:
                    features["cls"] = output["x_norm_clstoken"].float()

                if self.config.extract_avg_pool:
                    avg_pool = output["x_norm_patchtokens"].mean(dim=1)
                    features["avg_pool"] = avg_pool.float()

        # L2 normalize if requested
        if normalize:
            for key in features:
                if features[key].dim() == 2:
                    features[key] = F.normalize(features[key], p=2, dim=-1)
                elif features[key].dim() == 3:
                    # Normalize along embed_dim for [B, n_layers, embed_dim]
                    features[key] = F.normalize(features[key], p=2, dim=-1)

        return features

    @torch.no_grad()
    def extract_from_dataloader(
        self,
        dataloader: DataLoader,
        normalize: bool = True,
        show_progress: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Extract features from all samples in a dataloader.

        Args:
            dataloader: PyTorch DataLoader
            normalize: L2-normalize features
            show_progress: Show tqdm progress bar

        Returns:
            Dictionary with feature arrays:
            - "cls": [N, embed_dim]
            - "avg_pool": [N, embed_dim]
            - "multilayer": [N, n_layers * embed_dim]
            - "multilayer_cls": [N, n_layers, embed_dim]
        """
        all_features = {
            "cls": [],
            "avg_pool": [],
            "multilayer": [],
            "multilayer_cls": [],
        }

        iterator = tqdm(dataloader, desc="Extracting features") if show_progress else dataloader

        for batch in iterator:
            # Handle different batch formats
            if isinstance(batch, dict):
                x = batch["image"]
            elif isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            features = self.extract(x, normalize=normalize)

            for key, value in features.items():
                all_features[key].append(value.cpu().numpy())

        # Concatenate all batches
        result = {}
        for key, value_list in all_features.items():
            if value_list:
                result[key] = np.concatenate(value_list, axis=0)

        return result


def extract_features_for_checkpoint(
    checkpoint_name: str,
    dataloader: DataLoader,
    output_dir: Optional[Path] = None,
    normalize: bool = True,
    save: bool = True,
    analysis_name: str = "default",
    manifest_variant: str = "original_bins",
) -> Dict[str, np.ndarray]:
    """
    Convenience function to extract and optionally save features.

    Args:
        checkpoint_name: checkpoint key or release-name alias
        dataloader: DataLoader with samples to process
        output_dir: Directory to save features
        normalize: L2-normalize features
        save: Whether to save features to disk
        analysis_name: Dataset/analysis namespace used when output_dir is omitted

    Returns:
        Dictionary with feature arrays
    """
    output_dir = _resolve_features_dir(output_dir, analysis_name, manifest_variant)
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = FeatureExtractor(checkpoint_name)
    checkpoint_key = extractor.checkpoint_name
    features = extractor.extract_from_dataloader(dataloader, normalize=normalize)

    if save:
        checkpoint_dir = get_checkpoint_feature_dir(output_dir, checkpoint_key, "all_features")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        output_path = checkpoint_dir / "features.npz"
        np.savez_compressed(output_path, **features)
        logger.info(f"Features saved to {output_path}")

    return features


def extract_features_all_checkpoints(
    dataloader: DataLoader,
    checkpoint_names: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    normalize: bool = True,
    analysis_name: str = "default",
    manifest_variant: str = "original_bins",
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract features using all (or specified) checkpoints.

    Args:
        dataloader: DataLoader with samples to process
        checkpoint_names: List of checkpoint names (default: all)
        output_dir: Directory to save features
        normalize: L2-normalize features

    Returns:
        Nested dict: {checkpoint_name: {feature_type: array}}
    """
    checkpoint_names = checkpoint_names or list(CHECKPOINTS.keys())
    output_dir = _resolve_features_dir(output_dir, analysis_name, manifest_variant)

    all_results = {}
    failures = {}

    for ckpt_name in checkpoint_names:
        logger.info(f"\n{'='*60}\nExtracting features with {ckpt_name}\n{'='*60}")

        try:
            features = extract_features_for_checkpoint(
                ckpt_name,
                dataloader,
                output_dir=output_dir,
                normalize=normalize,
                save=True,
                analysis_name=analysis_name,
                manifest_variant=manifest_variant,
            )
            all_results[ckpt_name] = features
        except Exception as e:
            logger.error(f"Failed to extract with {ckpt_name}: {e}")
            failures[ckpt_name] = str(e)

    if failures:
        raise RuntimeError(
            f"Incomplete checkpoint extraction: completed={list(all_results)}; "
            f"failed={failures}"
        )
    return all_results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    try:
        from .data_loader import create_acquisition_dataloader
    except ImportError:  # pragma: no cover - direct script execution fallback
        from data_loader import create_acquisition_dataloader

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Extract features from 3D medical volumes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python feature_extractor.py -m abdomenatlas/original_bins/manifest_sampled.json -c med3dino_rel_96 -a abdomenatlas\n"
            "  python feature_extractor.py -m totalsegmentermri/original_bins/manifest_sampled.json -c med3dino_rel_96 -a totalsegmentermri --all-checkpoints"
        ),
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default="med3dino_rel_96",
        help="Checkpoint key or release-name alias to use for feature extraction"
    )
    parser.add_argument(
        "--manifest", "-m",
        type=str,
        default="abdomenatlas/original_bins/manifest_sampled.json",
        help="Manifest path relative to the acquisition manifest directory"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=8,
        help="Batch size for extraction"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for features"
    )
    parser.add_argument(
        "-a", "--analysis-name",
        type=str,
        default=None,
        help="Dataset/analysis namespace for default output paths"
    )
    parser.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="Extract features with all checkpoints"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable L2 normalization"
    )

    args = parser.parse_args()

    # Set up output directory
    # Create dataloader
    manifest_arg = Path(args.manifest)
    if manifest_arg.is_absolute():
        manifest_path = manifest_arg
    elif len(manifest_arg.parts) == 1:
        manifest_path = get_acquisition_manifest_path("abdomenatlas", manifest_arg.stem, "original_bins")
    else:
        manifest_path = resolve_acquisition_manifest_path(manifest_arg)

    analysis_name = args.analysis_name or get_dataset_name_from_manifest_path(manifest_path)
    manifest_variant = get_manifest_variant_from_manifest_path(manifest_path)
    output_dir = _resolve_features_dir(
        Path(args.output_dir) if args.output_dir else None,
        analysis_name,
        manifest_variant,
    )

    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    logger.info(f"Loading manifest: {manifest_path}")

    normalize = not args.no_normalize

    if args.all_checkpoints:
        results = {}
        checkpoints_by_crop = {}
        for checkpoint_name in CHECKPOINTS:
            checkpoint = resolve_checkpoint_config(checkpoint_name)
            checkpoints_by_crop.setdefault(checkpoint.crop_size, []).append(checkpoint.checkpoint_key)

        for crop_size, checkpoint_names in sorted(checkpoints_by_crop.items()):
            dataloader = create_acquisition_dataloader(
                manifest_path=manifest_path,
                crop_size=crop_size,
                batch_size=args.batch_size,
                num_workers=4,
            )
            logger.info(f"DataLoader crop{crop_size} created with {len(dataloader.dataset)} samples")
            results.update(
                extract_features_all_checkpoints(
                    dataloader,
                    checkpoint_names=checkpoint_names,
                    output_dir=output_dir,
                    normalize=normalize,
                    analysis_name=analysis_name,
                    manifest_variant=manifest_variant,
                )
            )
        logger.info(f"\nExtracted features for {len(results)} checkpoints")
    else:
        checkpoint = resolve_checkpoint_config(args.checkpoint)
        dataloader = create_acquisition_dataloader(
            manifest_path=manifest_path,
            crop_size=checkpoint.crop_size,
            batch_size=args.batch_size,
            num_workers=4,
        )
        logger.info(f"DataLoader crop{checkpoint.crop_size} created with {len(dataloader.dataset)} samples")

        features = extract_features_for_checkpoint(
            checkpoint.checkpoint_key,
            dataloader,
            output_dir=output_dir,
            analysis_name=analysis_name,
            manifest_variant=manifest_variant,
            normalize=normalize,
        )

        print("\nFeature shapes:")
        for key, value in features.items():
            print(f"  {key}: {value.shape}")
