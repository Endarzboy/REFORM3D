"""Evaluate controlled spacing perturbations for Acquisition Robustness.

The analysis starts from source-bin volumes, resamples each case to configured
target spacings, extracts embeddings, and measures representation drift and
semantic-transfer stability. Unlike native-spacing analysis, controlled
perturbation compares multiple resampled versions of the same source case.

Inputs:
    manifest samples with source-bin eligibility.
    checkpoint feature extractor.
    controlled perturbation config with target spacings and crop size.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import json
import gc
import hashlib
import os
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# MONAI for resampling
from monai.transforms import (
    Compose,
    CropForegroundd,
    LoadImaged,
    EnsureChannelFirstd,
    Lambdad,
    ScaleIntensityRangePercentilesd,
)

from reform3d.encoders.med3dino.spacing_aware_transforms import CropForegroundSwapSliceDimsV2

from reform3d.core.paths import get_reform3d_cache_path_candidates

try:
    from .config import (
        CONTROLLED_PERTURBATION_CONFIG,
        MRI_CONTROLLED_PERTURBATION_CONFIG,
        ControlledPerturbationConfig,
        compute_anisotropy_ratio,
        PREPROCESSING_CONFIG,
        get_cache_root,
        get_checkpoint_feature_dir,
        get_dataset_name_from_manifest_path,
        get_manifest_variant_from_manifest_path,
        get_output_paths,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from config import (
        CONTROLLED_PERTURBATION_CONFIG,
        MRI_CONTROLLED_PERTURBATION_CONFIG,
        ControlledPerturbationConfig,
        compute_anisotropy_ratio,
        PREPROCESSING_CONFIG,
        get_cache_root,
        get_checkpoint_feature_dir,
        get_dataset_name_from_manifest_path,
        get_manifest_variant_from_manifest_path,
        get_output_paths,
    )

logger = logging.getLogger(__name__)


CONTROLLED_EMBEDDING_BUNDLE_NAME = "controlled_embedding_bundle.npz"
SPACING_VARIANT_KIND = "spacing"
EXTRA_ACQUISITION_VARIANT_KIND = "extra_acquisition"


def _resolve_controlled_dirs(
    output_dir: Optional[Path],
    figures_dir: Optional[Path],
    dataset_name: str,
    manifest_variant: str,
    checkpoint_name: str,
    feature_type: str,
) -> Tuple[Path, Path]:
    output_paths = get_output_paths(dataset_name, manifest_variant)
    return (
        output_dir or get_checkpoint_feature_dir(output_paths["results"], checkpoint_name, feature_type),
        figures_dir or get_checkpoint_feature_dir(output_paths["figures"], checkpoint_name, feature_type),
    )


# =============================================================================
# RESAMPLED VOLUME CACHING
# =============================================================================

def get_cache_key(file_path: Union[str, Path]) -> str:
    """Bind geometry to the resolved source file, size, and modification time."""
    path = Path(file_path).resolve()
    stat = path.stat()
    identity = [str(path), stat.st_size, stat.st_mtime_ns]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]


def get_cache_path(
    file_path: Union[str, Path],
    crop_size: int = 96,
    cache_dir: Optional[Path] = None,
    cache_signature: str = "default",
) -> Path:
    """Get cache file path for a source volume."""
    cache_key = get_cache_key(file_path)
    cache_base = cache_dir or get_cache_root()
    # Include crop_size in cache path to handle different crop sizes
    return cache_base / f"crop{crop_size}" / cache_signature / f"{cache_key}.pt"


def load_from_cache(
    file_path: Union[str, Path],
    crop_size: int = 96,
    cache_dir: Optional[Path] = None,
    cache_signature: str = "default",
) -> Optional[Dict[str, torch.Tensor]]:
    """Load cached resampled variants if available.

    Tries fast loading (weights_only=True) first, falls back to slow loading
    for old cache files with MONAI MetaTensors.
    """
    cache_path = get_cache_path(file_path, crop_size, cache_dir, cache_signature)
    for candidate_cache_path in get_reform3d_cache_path_candidates(cache_path):
        if not candidate_cache_path.exists():
            continue
        try:
            # Try fast loading first (plain tensors)
            data = torch.load(candidate_cache_path, map_location="cpu", weights_only=True)
            return data
        except Exception:
            pass  # Fall back to slow loading

        try:
            # Slow loading for MetaTensor objects
            data = torch.load(candidate_cache_path, map_location="cpu", weights_only=False)
            # Convert MetaTensors to plain tensors
            plain_data = {}
            for k, v in data.items():
                if hasattr(v, 'as_tensor'):
                    plain_data[k] = v.as_tensor().clone()
                else:
                    plain_data[k] = v
            return plain_data
        except Exception as e:
            logger.warning(f"Failed to load cache {candidate_cache_path}: {e}")
            return None
    return None


def save_to_cache(
    file_path: Union[str, Path],
    variants: Dict[str, torch.Tensor],
    crop_size: int = 96,
    cache_dir: Optional[Path] = None,
    cache_signature: str = "default",
) -> bool:
    """Save resampled variants to cache.

    Converts MONAI MetaTensors to plain tensors for faster loading.
    """
    cache_path = get_cache_path(file_path, crop_size, cache_dir, cache_signature)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Convert MetaTensors to plain tensors for faster loading
        plain_variants = {}
        for k, v in variants.items():
            if hasattr(v, 'as_tensor'):
                # MONAI MetaTensor -> regular tensor
                plain_variants[k] = v.as_tensor().clone()
            else:
                plain_variants[k] = v.clone() if isinstance(v, torch.Tensor) else v
        torch.save(plain_variants, cache_path)
        return True
    except Exception as e:
        logger.warning(f"Failed to save cache {cache_path}: {e}")
        return False


@dataclass
class PerturbedVolume:
    """Container for a volume with multiple spacing variants."""
    original_path: str
    original_spacing: Tuple[float, float, float]

    # Dict mapping target_spacing_name -> resampled tensor
    spacing_variants: Dict[str, torch.Tensor] = None

    # Metadata
    dataset: str = "unknown"
    has_label: bool = False


@dataclass
class ControlledPerturbationResults:
    """Results from controlled perturbation experiments."""
    checkpoint_name: str
    feature_type: str

    # Per-sample embedding distances across spacing variants
    # {sample_idx: {(spacing_a, spacing_b): cosine_distance}}
    pairwise_distances: Dict[int, Dict[Tuple[str, str], float]]

    # Aggregated metrics
    mean_distance_per_pair: Dict[Tuple[str, str], float]
    std_distance_per_pair: Dict[Tuple[str, str], float]

    # Representation Drift: distance from isotropic (1x1x1) to target spacing
    representation_drift: Dict[str, float]  # mean drift per target spacing
    representation_drift_std: Dict[str, float]  # std of drift per target spacing

    # CKA between spacing variants (computed on all samples)
    cka_matrix: Dict[str, Dict[str, float]]

    # Configuration
    target_spacings: Optional[List[Tuple[float, float, float]]]
    n_source_volumes: int

    # Matched semantic metrics (primary-layer semantics)
    matched_semantic_probing: Optional[Dict[str, Any]] = None
    matched_semantic_transfer: Optional[Dict[str, Any]] = None
    semantic_metadata: Optional[Dict[str, Any]] = None


def _pad_center_crop(tensor: torch.Tensor, spatial_size: Tuple[int, int, int]) -> torch.Tensor:
    """Pad then center-crop a [C, H, W, D] tensor to the requested spatial size."""
    current_size = tensor.shape[1:]
    pad_widths = []
    for current, target in zip(reversed(current_size), reversed(spatial_size)):
        total_pad = max(target - current, 0)
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        pad_widths.extend([pad_before, pad_after])

    if any(pad_widths):
        tensor = F.pad(tensor, pad_widths, mode="constant", value=PREPROCESSING_CONFIG.intensity_output_min)

    slices = [slice(None)]
    for current, target in zip(tensor.shape[1:], spatial_size):
        start = max((current - target) // 2, 0)
        slices.append(slice(start, start + target))
    return tensor[tuple(slices)]


def load_preprocessed_source_image(
    file_path: Union[str, Path],
    source_spacing: Optional[Tuple[float, float, float]] = None,
    config: Optional[ControlledPerturbationConfig] = None,
) -> Tuple[torch.Tensor, Tuple[float, float, float]]:
    """Load and preprocess a source image once so multiple crop sizes can reuse it."""
    config = config or CONTROLLED_PERTURBATION_CONFIG
    orientation_policy = getattr(config, "orientation_policy", "spacing_permutation")
    foreground_transform = (
        CropForegroundSwapSliceDimsV2(
            select_fn=lambda x: x > PREPROCESSING_CONFIG.foreground_threshold,
        )
        if orientation_policy == "spacing_permutation"
        else CropForegroundd(
            keys=["image"],
            select_fn=lambda x: x > PREPROCESSING_CONFIG.foreground_threshold,
        )
    )

    load_transforms = Compose([
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        Lambdad(
            keys=["image"],
            func=lambda x: torch.nan_to_num(x, torch.nanmean(x).item()),
        ),
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=PREPROCESSING_CONFIG.intensity_percentile_lower,
            upper=PREPROCESSING_CONFIG.intensity_percentile_upper,
            b_min=PREPROCESSING_CONFIG.intensity_output_min,
            b_max=PREPROCESSING_CONFIG.intensity_output_max,
            clip=True,
        ),
        foreground_transform,
    ])

    if source_spacing is None:
        header_img = nib.load(str(file_path))
        source_spacing = tuple(float(x) for x in header_img.header.get_zooms()[:3])

    source_anisotropy_ratio = compute_anisotropy_ratio(source_spacing)
    loaded_data = load_transforms({
        "image": str(file_path),
        "spacing": list(source_spacing),
        "anisotropy_ratio": source_anisotropy_ratio,
    })
    base_image = loaded_data["image"].contiguous()
    permuted_spacing = tuple(
        loaded_data.get(
            "spacing_permuted",
            source_spacing if orientation_policy == "identity" else [1.0, 1.0, 1.0],
        )
    )
    return base_image, permuted_spacing


def resample_preprocessed_volume(
    base_image: torch.Tensor,
    source_spacing: Tuple[float, float, float],
    variant_specs: List[Tuple[str, Tuple[float, float, float]]],
    crop_size: int = 96,
    config: Optional[ControlledPerturbationConfig] = None,
) -> Dict[str, torch.Tensor]:
    """Create spacing variants from a preprocessed source image for one crop size."""
    config = config or CONTROLLED_PERTURBATION_CONFIG
    results = {}
    base_fov = _pad_center_crop(base_image, (crop_size, crop_size, crop_size)).contiguous()

    for spacing_name, spacing in variant_specs:
        try:
            resampled = _resample_tensor_to_spacing(
                base_fov,
                source_spacing,
                spacing,
                interpolation_mode=config.interpolation_mode,
                anti_alias=config.anti_alias,
            )
            if getattr(config, "physical_fov_mm", None) is not None:
                results[spacing_name] = _extract_fixed_physical_fov(
                    resampled,
                    spacing,
                    config.physical_fov_mm,
                    (crop_size, crop_size, crop_size),
                    interpolation_mode=config.interpolation_mode,
                )
            else:
                results[spacing_name] = _interpolate_3d(
                    resampled,
                    (crop_size, crop_size, crop_size),
                    mode=config.interpolation_mode,
                ).contiguous()
            del resampled
        except Exception as e:
            logger.warning(f"Failed to resample preprocessed source to {spacing}: {e}")
            results[spacing_name] = None

    if getattr(config, "extra_perturbations", None):
        reference_name = config.reference_variant_name(source_spacing)
        reference = results.get(reference_name)
        if reference is None:
            reference = base_fov
        for perturbation_name in config.extra_perturbations:
            try:
                results[perturbation_name] = _apply_extra_acquisition_perturbation(
                    reference,
                    perturbation_name,
                    seed=int(getattr(config, "extra_perturbation_seed", 17)),
                ).contiguous()
            except Exception as e:
                logger.warning("Failed to apply extra acquisition perturbation %s: %s", perturbation_name, e)
                results[perturbation_name] = None

    return results


def _interpolate_3d(image: torch.Tensor, size: Tuple[int, int, int], mode: str) -> torch.Tensor:
    """Interpolate a [C, H, W, D] tensor while respecting PyTorch mode semantics."""
    kwargs: Dict[str, Any] = {"size": tuple(int(x) for x in size), "mode": mode}
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        kwargs["align_corners"] = False
    return F.interpolate(image.unsqueeze(0).float(), **kwargs).squeeze(0)


def _compute_antialias_sigmas(
    source_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Return Gaussian sigmas in source-voxel units for axes that are being downsampled."""
    sigmas: List[float] = []
    for src_sp, tgt_sp in zip(source_spacing, target_spacing):
        scale = float(tgt_sp) / max(float(src_sp), 1e-6)
        sigmas.append(max((scale - 1.0) / 2.0, 0.0))
    return tuple(sigmas)


def _apply_antialias_filter(
    image: torch.Tensor,
    source_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
) -> torch.Tensor:
    """Low-pass filter before downsampling to better simulate thicker slices."""
    sigma_xyz = _compute_antialias_sigmas(source_spacing, target_spacing)
    if not any(sigma > 1e-6 for sigma in sigma_xyz):
        return image

    filtered = gaussian_filter(
        image.detach().cpu().numpy().astype(np.float32, copy=False),
        sigma=(0.0,) + sigma_xyz,
        mode="nearest",
    )
    return torch.from_numpy(filtered).to(device=image.device, dtype=image.dtype)


def _resample_tensor_to_spacing(
    image: torch.Tensor,
    source_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
    interpolation_mode: str = "trilinear",
    anti_alias: bool = True,
) -> torch.Tensor:
    """Resample a [C, H, W, D] tensor to target spacing using trilinear interpolation."""
    if anti_alias:
        image = _apply_antialias_filter(image, source_spacing, target_spacing)

    spatial_shape = image.shape[1:]
    target_shape = []
    for size, src_sp, tgt_sp in zip(spatial_shape, source_spacing, target_spacing):
        new_size = max(1, int(round(size * float(src_sp) / float(tgt_sp))))
        target_shape.append(new_size)

    return _interpolate_3d(image, tuple(target_shape), interpolation_mode)


def _extract_fixed_physical_fov(
    image: torch.Tensor,
    target_spacing: Tuple[float, float, float],
    physical_fov_mm: Tuple[float, float, float],
    output_size: Tuple[int, int, int],
    interpolation_mode: str = "trilinear",
) -> torch.Tensor:
    """Crop the same physical field of view from each spacing variant, then resize to model input size."""
    crop_size_voxels = tuple(
        max(1, int(round(fov_mm / float(sp))))
        for fov_mm, sp in zip(physical_fov_mm, target_spacing)
    )

    cropped = _pad_center_crop(image, crop_size_voxels)
    resized = _interpolate_3d(cropped, output_size, interpolation_mode)
    return resized.contiguous()


def _deterministic_noise_like(image: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    """Return deterministic pseudo-noise without relying on global RNG state."""
    flat_index = torch.arange(image.numel(), device=image.device, dtype=torch.float32).reshape(image.shape)
    phase = float(seed) * 0.173
    pattern = torch.sin(flat_index * 12.9898 + phase) * torch.cos(flat_index * 78.233 + phase)
    pattern = pattern - pattern.mean()
    pattern = pattern / pattern.std().clamp_min(1e-6)
    return pattern.to(dtype=image.dtype) * float(sigma)


def _clip_intensity(image: torch.Tensor) -> torch.Tensor:
    return torch.clamp(
        image,
        min=float(PREPROCESSING_CONFIG.intensity_output_min),
        max=float(PREPROCESSING_CONFIG.intensity_output_max),
    )


def _apply_extra_acquisition_perturbation(
    image: torch.Tensor,
    perturbation_name: str,
    seed: int = 17,
) -> torch.Tensor:
    """Apply deterministic non-spacing acquisition perturbations to a [C,H,W,D] tensor."""
    name = str(perturbation_name).strip().lower().replace("-", "_")
    if name in {"noise", "dose_noise", "noise_sigma0.05", "dose_noise_sigma0.05"}:
        return _clip_intensity(image + _deterministic_noise_like(image, sigma=0.05, seed=seed))
    if name in {"noise_sigma0.10", "dose_noise_sigma0.10"}:
        return _clip_intensity(image + _deterministic_noise_like(image, sigma=0.10, seed=seed))
    if name in {"kernel_blur", "kernel_blur_sigma1.0", "reconstruction_blur"}:
        filtered = gaussian_filter(image.detach().cpu().numpy().astype(np.float32, copy=False), sigma=(0.0, 1.0, 1.0, 1.0), mode="nearest")
        return torch.from_numpy(filtered).to(device=image.device, dtype=image.dtype)
    if name in {"kernel_sharpen", "reconstruction_sharpen"}:
        blurred = _apply_extra_acquisition_perturbation(image, "kernel_blur_sigma1.0", seed=seed)
        return _clip_intensity(image + 0.35 * (image - blurred))
    if name in {"plane_reformat_coronal", "coronal_reformat"}:
        return image.permute(0, 1, 3, 2).contiguous()
    if name in {"plane_reformat_sagittal", "sagittal_reformat"}:
        return image.permute(0, 3, 2, 1).contiguous()
    raise ValueError(f"Unknown extra acquisition perturbation: {perturbation_name}")


def _gradient_energy(tensor: torch.Tensor) -> float:
    """Average squared finite-difference energy across all spatial axes."""
    spatial = tensor.float().squeeze(0)
    energies = []
    for axis in range(spatial.ndim):
        if spatial.shape[axis] <= 1:
            continue
        diff = torch.diff(spatial, dim=axis)
        energies.append(float(diff.square().mean().item()))
    if not energies:
        return 0.0
    return float(np.mean(energies))


def compute_spacing_variant_diagnostics(
    spacing_variants: Dict[str, torch.Tensor],
    reference_variant: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Quantify how strongly each spacing variant departs from the reference variant."""
    if not spacing_variants:
        return {}

    variant_names = list(spacing_variants.keys())
    reference_variant = reference_variant or variant_names[0]
    if reference_variant not in spacing_variants:
        raise ValueError(f"Reference variant {reference_variant} not found in spacing variants")

    reference = spacing_variants[reference_variant].float()
    reference_flat = reference.flatten()
    reference_gradient = _gradient_energy(reference)
    diagnostics: Dict[str, Dict[str, float]] = {}

    for variant_name, variant_tensor in spacing_variants.items():
        variant = variant_tensor.float()
        difference = variant - reference
        variant_flat = variant.flatten()
        if torch.std(reference_flat) < 1e-8 or torch.std(variant_flat) < 1e-8:
            correlation = 1.0 if torch.allclose(reference_flat, variant_flat) else 0.0
        else:
            correlation = float(torch.corrcoef(torch.stack([reference_flat, variant_flat]))[0, 1].item())

        diagnostics[variant_name] = {
            "mean_absolute_difference": float(difference.abs().mean().item()),
            "rmse": float(torch.sqrt(difference.square().mean()).item()),
            "max_absolute_difference": float(difference.abs().max().item()),
            "normalized_cross_correlation": correlation,
            "gradient_energy": _gradient_energy(variant),
            "gradient_energy_ratio_vs_reference": float(
                _gradient_energy(variant) / max(reference_gradient, 1e-8)
            ),
        }

    return diagnostics


def resample_volume(
    file_path: Union[str, Path],
    variant_specs: List[Tuple[str, Tuple[float, float, float]]],
    crop_size: int = 96,
    source_spacing: Optional[Tuple[float, float, float]] = None,
    config: Optional[ControlledPerturbationConfig] = None,
) -> Dict[str, torch.Tensor]:
    """
    Resample a single volume to multiple target spacings.

    Memory-efficient: loads volume once, processes each spacing sequentially
    with explicit garbage collection to handle large whole-body scans.

    Args:
        file_path: Path to NIfTI file
        variant_specs: List of (variant_name, target_spacing)
        crop_size: Final crop size

    Returns:
        Dict mapping spacing_name -> tensor [1, D, H, W]
    """
    config = config or CONTROLLED_PERTURBATION_CONFIG
    try:
        base_image, source_spacing = load_preprocessed_source_image(
            file_path,
            source_spacing=source_spacing,
            config=config,
        )
    except Exception as e:
        logger.warning(f"Failed to load {file_path}: {e}")
        return {name: None for name, _ in variant_specs}

    try:
        return resample_preprocessed_volume(
            base_image,
            source_spacing,
            variant_specs,
            crop_size=crop_size,
            config=config,
        )
    finally:
        del base_image


def prepare_perturbed_dataset(
    manifest_path: Union[str, Path],
    config: Optional[ControlledPerturbationConfig] = None,
    crop_size: int = 96,
    cache_dir: Optional[Path] = None,
) -> List[PerturbedVolume]:
    """
    Prepare a dataset of perturbed volumes from the configured native-observational source bin.

    Args:
        manifest_path: Path to manifest JSON
        config: Perturbation configuration
        crop_size: Crop size for transformed volumes
        cache_dir: Directory for resampled volume cache

    Returns:
        List of PerturbedVolume objects
    """
    config = config or CONTROLLED_PERTURBATION_CONFIG
    cache_dir = cache_dir or get_cache_root()

    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)

    volumes = manifest["volumes"]

    # Filter to source bin (isotropic volumes)
    source_volumes = [v for v in volumes if v["anisotropy_bin"] == config.source_bin]
    logger.info(f"Found {len(source_volumes)} volumes in source bin {config.source_bin}")

    # Limit if specified
    if config.n_source_volumes is not None:
        source_volumes = source_volumes[:config.n_source_volumes]

    logger.info(f"Processing {len(source_volumes)} volumes")

    # Check cache statistics by file existence (fast - no loading)
    cache_signature = config.cache_signature()
    cache_hits = 0
    for vol_info in source_volumes:
        cache_path = get_cache_path(vol_info["file_path"], crop_size, cache_dir, cache_signature)
        if cache_path.exists():
            cache_hits += 1

    # Determine progress bar description based on cache status
    if cache_hits == len(source_volumes):
        logger.info(f"All {cache_hits} volumes found in cache - loading from disk")
        progress_desc = "Loading cached volumes"
    elif cache_hits > 0:
        logger.info(f"Cache: {cache_hits}/{len(source_volumes)} found, {len(source_volumes) - cache_hits} need resampling")
        progress_desc = "Loading/Resampling volumes"
    else:
        logger.info(f"No cache found - resampling all {len(source_volumes)} volumes")
        progress_desc = "Resampling volumes"

    # Resample each volume with caching and memory cleanup
    perturbed_volumes = []
    actual_cache_hits = 0

    for i, vol_info in enumerate(tqdm(source_volumes, desc=progress_desc)):
        file_path = vol_info["file_path"]
        source_spacing = tuple(vol_info["spacing"])
        variant_specs = config.resolve_variant_specs(source_spacing)
        expected_variants = [name for name, _ in variant_specs] + list(config.extra_perturbations or [])
        if len(expected_variants) < 2 or len(expected_variants) != len(set(expected_variants)):
            raise ValueError("Controlled protocol requires at least two uniquely named variants")

        # Try loading from cache first
        variants = load_from_cache(file_path, crop_size, cache_dir, cache_signature)

        cache_miss = variants is None
        if not cache_miss:
            actual_cache_hits += 1
        else:
            # Cache miss - resample and save
            variants = resample_volume(
                file_path,
                variant_specs,
                crop_size=crop_size,
                source_spacing=source_spacing,
                config=config,
            )

        if not isinstance(variants, dict) or set(variants) != set(expected_variants):
            raise ValueError(f"Controlled geometry cache has incomplete protocol variants: {file_path}")
        variants = {name: variants[name] for name in expected_variants}
        if any(not isinstance(value, torch.Tensor) or tuple(value.shape) != (1, crop_size, crop_size, crop_size) or not torch.isfinite(value).all() for value in variants.values()):
            raise ValueError(f"Controlled geometry is incomplete, malformed, or nonfinite: {file_path}")
        if cache_miss:
            save_to_cache(file_path, variants, crop_size, cache_dir, cache_signature)

        perturbed_volumes.append(PerturbedVolume(
            original_path=file_path,
            original_spacing=source_spacing,
            spacing_variants=variants,
            dataset=vol_info.get("dataset", "unknown"),
            has_label=vol_info.get("has_label", False),
        ))

        """
        Example cache structure after running this function:
        reform3d_caches/acquisition_robustness/<dataset>/<variant>/crop96/
        ├── 9a56b1548efd1d2b.pt     # One source volume
        │   └── Dict with 3 keys:
        │       ├── "1.0x1.0x1.0": tensor [1, 96, 96, 96]  # Isotropic
        │       ├── "1.0x1.0x3.0": tensor [1, 96, 96, 96]  # Moderate anisotropy
        │       └── "1.0x1.0x5.0": tensor [1, 96, 96, 96]  # High anisotropy
        ├── ab12cd34ef56gh78.pt     # Another source volume
        │   └── Same 3 spacing variants
        └── ... (998 files total when complete)
        """

        # Aggressive memory cleanup after each volume (large PET-CT scans need this)
        del variants
        gc.collect()

    logger.info(f"Successfully prepared {len(perturbed_volumes)} perturbed volumes")
    logger.info(f"Cache statistics: {actual_cache_hits} hits, {len(source_volumes) - actual_cache_hits} misses")
    return perturbed_volumes


def extract_features_for_variants(
    perturbed_volumes: List[PerturbedVolume],
    extractor,  # FeatureExtractor instance
    feature_type: str = "cls",
    batch_size: int = 8,
) -> Dict[str, np.ndarray]:
    """
    Extract features for all spacing variants of all volumes.

    Args:
        perturbed_volumes: List of PerturbedVolume objects
        extractor: FeatureExtractor instance
        feature_type: Type of features to extract
        batch_size: Batch size for extraction

    Returns:
        Dict mapping spacing_name -> feature_matrix [N, D]
    """
    if not perturbed_volumes:
        return {}

    # Get spacing names from first volume
    spacing_names = list(perturbed_volumes[0].spacing_variants.keys())

    # Collect features per spacing variant
    features_by_spacing = {name: [] for name in spacing_names}

    for i, vol in enumerate(tqdm(perturbed_volumes, desc="Extracting features")):
        for spacing_name, tensor in vol.spacing_variants.items():
            # Extract features for this variant
            with torch.no_grad():
                # Add batch dimension
                batch = tensor.unsqueeze(0).to(extractor.config.device)
                batch = batch.to(extractor.config.dtype)

                # Extract features
                features_dict = extractor.extract(batch)
                features = features_dict[feature_type].cpu().numpy()

                features_by_spacing[spacing_name].append(features[0])

                # Free GPU memory immediately
                del batch, features_dict

        # Clear tensor variants after extraction to free memory
        vol.spacing_variants = None

        # Periodic memory cleanup
        if (i + 1) % 25 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Stack into arrays
    return {name: np.stack(feats) for name, feats in features_by_spacing.items()}


def compute_pairwise_distances(
    features_by_spacing: Dict[str, np.ndarray],
) -> Tuple[Dict[int, Dict[Tuple[str, str], float]], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    """
    Compute pairwise cosine distances between spacing variants.

    For each sample, computes the distance between its representations
    under different spacing conditions.

    Args:
        features_by_spacing: Dict mapping spacing_name -> features [N, D]

    Returns:
        Tuple of:
        - per_sample_distances: {sample_idx: {(spacing_a, spacing_b): distance}}
        - mean_distances: {(spacing_a, spacing_b): mean_distance}
        - std_distances: {(spacing_a, spacing_b): std_distance}
    """
    from scipy.spatial.distance import cosine

    spacing_names = list(features_by_spacing.keys())
    n_samples = len(features_by_spacing[spacing_names[0]])

    per_sample_distances = {}
    pair_distances = {(a, b): [] for i, a in enumerate(spacing_names)
                      for b in spacing_names[i+1:]}

    for sample_idx in range(n_samples):
        per_sample_distances[sample_idx] = {}

        for i, spacing_a in enumerate(spacing_names):
            for spacing_b in spacing_names[i+1:]:
                feat_a = features_by_spacing[spacing_a][sample_idx]
                feat_b = features_by_spacing[spacing_b][sample_idx]

                # Cosine distance (1 - cosine_similarity)
                dist = cosine(feat_a, feat_b)

                per_sample_distances[sample_idx][(spacing_a, spacing_b)] = float(dist)
                pair_distances[(spacing_a, spacing_b)].append(dist)

    # Compute means and stds
    mean_distances = {pair: float(np.mean(dists))
                      for pair, dists in pair_distances.items()}
    std_distances = {pair: float(np.std(dists))
                     for pair, dists in pair_distances.items()}

    return per_sample_distances, mean_distances, std_distances


def compute_cka_between_variants(
    features_by_spacing: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """
    Compute CKA between all pairs of spacing variants.

    Args:
        features_by_spacing: Dict mapping spacing_name -> features [N, D]

    Returns:
        CKA matrix as nested dict
    """
    try:
        from .representation_geometry_analysis import cka_linear
    except ImportError:  # pragma: no cover - direct script execution fallback
        from representation_geometry_analysis import cka_linear

    spacing_names = list(features_by_spacing.keys())
    cka_matrix = {name: {} for name in spacing_names}

    for spacing_a in spacing_names:
        for spacing_b in spacing_names:
            feat_a = features_by_spacing[spacing_a]
            feat_b = features_by_spacing[spacing_b]

            cka_val = cka_linear(feat_a, feat_b)
            cka_matrix[spacing_a][spacing_b] = float(cka_val)

    return cka_matrix


def _resolve_variant_metadata(
    config: ControlledPerturbationConfig,
    perturbed_volumes: List[PerturbedVolume],
    variant_names: List[str],
) -> Dict[str, Any]:
    """Resolve metadata for spacing and non-spacing acquisition variants."""
    if not perturbed_volumes:
        raise ValueError("Cannot resolve spacing metadata without perturbed volumes")

    variant_specs = config.resolve_variant_specs(perturbed_volumes[0].original_spacing)
    spacing_by_name = {
        name: tuple(float(value) for value in target_spacing)
        for name, target_spacing in variant_specs
    }
    extra_names = set(config.extra_perturbations or [])

    z_spacings_mm = np.full(len(variant_names), np.nan, dtype=np.float32)
    target_spacings_mm = np.full((len(variant_names), 3), np.nan, dtype=np.float32)
    variant_kinds: List[str] = []
    unknown_names: List[str] = []

    for index, variant_name in enumerate(variant_names):
        if variant_name in spacing_by_name:
            target_spacing = spacing_by_name[variant_name]
            variant_kinds.append(SPACING_VARIANT_KIND)
            target_spacings_mm[index] = np.asarray(target_spacing, dtype=np.float32)
            z_spacings_mm[index] = float(target_spacing[2])
        elif variant_name in extra_names:
            variant_kinds.append(EXTRA_ACQUISITION_VARIANT_KIND)
        else:
            unknown_names.append(variant_name)

    if unknown_names:
        raise ValueError(f"Missing perturbation metadata for variants: {unknown_names}")

    return {
        "variant_kinds": np.asarray(variant_kinds, dtype=str),
        "spacing_variant_mask": np.asarray(
            [kind == SPACING_VARIANT_KIND for kind in variant_kinds],
            dtype=bool,
        ),
        "extra_acquisition_variant_mask": np.asarray(
            [kind == EXTRA_ACQUISITION_VARIANT_KIND for kind in variant_kinds],
            dtype=bool,
        ),
        "spacing_values_z_mm": z_spacings_mm,
        "target_spacings_mm": target_spacings_mm,
        "spacing_variants": [
            name for name, kind in zip(variant_names, variant_kinds)
            if kind == SPACING_VARIANT_KIND
        ],
        "extra_acquisition_variants": [
            name for name, kind in zip(variant_names, variant_kinds)
            if kind == EXTRA_ACQUISITION_VARIANT_KIND
        ],
    }


def _variant_metadata_for_json(metadata: Dict[str, Any], variant_names: List[str]) -> Dict[str, Any]:
    kinds = [str(value) for value in np.asarray(metadata["variant_kinds"]).tolist()]
    z_values = np.asarray(metadata["spacing_values_z_mm"], dtype=np.float32)
    target_spacings = np.asarray(metadata["target_spacings_mm"], dtype=np.float32)
    return {
        "variant_kinds": dict(zip(variant_names, kinds)),
        "spacing_variants": list(metadata["spacing_variants"]),
        "extra_acquisition_variants": list(metadata["extra_acquisition_variants"]),
        "spacing_values_z_mm": {
            name: (None if not np.isfinite(float(value)) else float(value))
            for name, value in zip(variant_names, z_values)
        },
        "target_spacings_mm": {
            name: [
                None if not np.isfinite(float(axis_value)) else float(axis_value)
                for axis_value in target_spacings[index]
            ]
            for index, name in enumerate(variant_names)
        },
    }


def save_controlled_embedding_bundle(
    output_dir: Path,
    features_by_spacing: Dict[str, np.ndarray],
    perturbed_volumes: List[PerturbedVolume],
    config: ControlledPerturbationConfig,
    checkpoint_name: str,
    feature_type: str,
    dataset_name: str,
    manifest_variant: str,
    reference_variant: str,
) -> Path:
    """Persist matched controlled-perturbation embeddings for downstream controlled-geometry analysis."""
    spacing_names = list(features_by_spacing.keys())
    if reference_variant not in features_by_spacing:
        raise ValueError(f"Reference variant {reference_variant} missing from controlled-perturbation features")

    variant_metadata = _resolve_variant_metadata(config, perturbed_volumes, spacing_names)
    z_spacings_mm = variant_metadata["spacing_values_z_mm"]
    variant_kinds = variant_metadata["variant_kinds"]
    reference_index = spacing_names.index(reference_variant)
    if variant_kinds[reference_index] != SPACING_VARIANT_KIND:
        raise ValueError(f"Reference variant {reference_variant} must be a spacing variant")
    reference_z_mm = float(z_spacings_mm[reference_index])
    spacing_mask = variant_metadata["spacing_variant_mask"]
    log_spacing_severity = np.full(len(spacing_names), np.nan, dtype=np.float32)
    log_spacing_severity[spacing_mask] = np.log(
        np.clip(z_spacings_mm[spacing_mask], 1e-8, None) / max(reference_z_mm, 1e-8)
    )
    case_ids = np.asarray([str(volume.original_path) for volume in perturbed_volumes])

    bundle_arrays: Dict[str, Any] = {
        "dataset": np.asarray(dataset_name),
        "manifest_variant": np.asarray(manifest_variant),
        "checkpoint": np.asarray(checkpoint_name),
        "feature_type": np.asarray(feature_type),
        "perturbation_protocol": np.asarray(config.protocol_name),
        "case_ids": case_ids,
        "spacing_names": np.asarray(spacing_names, dtype=str),
        "variant_names": np.asarray(spacing_names, dtype=str),
        "variant_kinds": variant_kinds,
        "spacing_variant_mask": spacing_mask,
        "extra_acquisition_variant_mask": variant_metadata["extra_acquisition_variant_mask"],
        "spacing_variants": np.asarray(variant_metadata["spacing_variants"], dtype=str),
        "extra_acquisition_variants": np.asarray(variant_metadata["extra_acquisition_variants"], dtype=str),
        "reference_spacing_name": np.asarray(reference_variant),
        "reference_variant_name": np.asarray(reference_variant),
        "spacing_values_z_mm": z_spacings_mm,
        "target_spacings_mm": variant_metadata["target_spacings_mm"],
        "log_spacing_severity": log_spacing_severity.astype(np.float32),
    }
    for spacing_name, features in features_by_spacing.items():
        bundle_arrays[f"features__{spacing_name}"] = features.astype(np.float32, copy=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / CONTROLLED_EMBEDDING_BUNDLE_NAME
    with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".npz", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        np.savez_compressed(temporary_path, **bundle_arrays)
        temporary_path.replace(bundle_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    logger.info(f"Saved controlled embedding bundle to {bundle_path}")
    return bundle_path


def _load_controlled_features(
    path: Path,
    volumes: List[PerturbedVolume],
    config: ControlledPerturbationConfig,
    checkpoint_name: str,
    feature_type: str,
    dataset_name: str,
) -> Tuple[Dict[str, np.ndarray], bool]:
    """Read the analysis bundle, accepting the old extraction schema for migration."""
    expected_names = config.describe_targets() + list(config.extra_perturbations or [])
    expected_paths = [str(volume.original_path) for volume in volumes]
    try:
        with np.load(path, allow_pickle=False) as bundle:
            rich_bundle = "case_ids" in bundle
            case_key = "case_ids" if rich_bundle else "source_paths"
            if bundle[case_key].tolist() != expected_paths:
                raise ValueError("Controlled feature cache has different source cases or case order")
            names = bundle["variant_names"].tolist()
            if len(names) != len(set(names)) or set(names) != set(expected_names):
                raise ValueError("Controlled feature cache has duplicate or incomplete protocol variants")
            if rich_bundle:
                identity = {
                    "checkpoint": checkpoint_name,
                    "feature_type": feature_type,
                    "dataset": dataset_name,
                    "perturbation_protocol": config.protocol_name,
                }
                for key, expected in identity.items():
                    if bundle[key].shape != () or bundle[key].item() != expected:
                        raise ValueError(f"Controlled embedding bundle has different {key}")
                expected_metadata = _resolve_variant_metadata(config, volumes, names)
                if not np.array_equal(bundle["target_spacings_mm"], expected_metadata["target_spacings_mm"], equal_nan=True):
                    raise ValueError("Controlled embedding bundle has different target spacings")
                if bundle["reference_variant_name"].item() != expected_names[0]:
                    raise ValueError("Controlled embedding bundle has a different reference variant")
            features = {
                name: bundle[f"features__{name}" if rich_bundle else f"features_{index}"].copy()
                for index, name in enumerate(names)
            }
    except (KeyError, OSError, TypeError) as error:
        raise ValueError(f"Incomplete or unreadable controlled embedding bundle: {path}") from error
    return features, rich_bundle


def _build_matched_split_indices(
    semantic_labels: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build shared case-disjoint folds for matched semantic evaluation.

    When semantic targets are single-label or one-hot, stratified folds are used.
    For genuinely multi-label targets, the code falls back to shuffled K-folds
    because sklearn does not provide iterative multilabel stratification.
    """
    n_samples = len(semantic_labels)
    effective_splits = min(n_splits, n_samples)
    if effective_splits < 2:
        raise ValueError("Matched semantic evaluation requires at least two samples")

    stratify_labels = None
    if semantic_labels.ndim == 2 and semantic_labels.shape[1] == 1:
        stratify_labels = semantic_labels[:, 0]
    elif semantic_labels.ndim == 2 and np.all(semantic_labels.sum(axis=1) == 1):
        stratify_labels = np.argmax(semantic_labels, axis=1)

    if stratify_labels is not None:
        unique_classes, counts = np.unique(stratify_labels, return_counts=True)
        if len(unique_classes) > 1 and counts.min() >= effective_splits:
            splitter = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)
            return [(train_idx, test_idx) for train_idx, test_idx in splitter.split(np.zeros(n_samples), stratify_labels)]

    splitter = KFold(n_splits=effective_splits, shuffle=True, random_state=random_state)
    return [(train_idx, test_idx) for train_idx, test_idx in splitter.split(np.zeros(n_samples))]


def _cross_validated_multilabel_balanced_accuracy(
    features: np.ndarray,
    semantic_labels: np.ndarray,
    fold_indices: List[Tuple[np.ndarray, np.ndarray]],
    random_state: int = 42,
) -> Tuple[float, Dict[int, float]]:
    """Evaluate multi-label balanced accuracy with shared precomputed folds."""
    if semantic_labels.ndim != 2 or semantic_labels.shape[1] == 0:
        return 0.0, {}

    per_label_scores: Dict[int, List[float]] = {label_idx: [] for label_idx in range(semantic_labels.shape[1])}

    for label_idx in range(semantic_labels.shape[1]):
        label_column = semantic_labels[:, label_idx]
        for train_idx, test_idx in fold_indices:
            y_train = label_column[train_idx]
            y_test = label_column[test_idx]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            scaler = StandardScaler()
            X_train = scaler.fit_transform(features[train_idx])
            X_test = scaler.transform(features[test_idx])

            clf = LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                random_state=random_state,
                n_jobs=4,
            )
            try:
                clf.fit(X_train, y_train)
                predictions = clf.predict(X_test)
                per_label_scores[label_idx].append(float(balanced_accuracy_score(y_test, predictions)))
            except Exception as e:
                logger.warning(f"Matched semantic probe failed for label {label_idx}: {e}")

    mean_per_label = {
        label_idx: float(np.mean(scores))
        for label_idx, scores in per_label_scores.items()
        if scores
    }
    mean_score = float(np.mean(list(mean_per_label.values()))) if mean_per_label else 0.0
    return mean_score, mean_per_label


def _train_and_evaluate_multilabel_transfer_once(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    random_state: int = 42,
    min_class_count: int = 2,
) -> Tuple[float, Dict[int, float]]:
    """Train one classifier per label on one variant and evaluate on another."""
    if train_labels.ndim != 2 or test_labels.ndim != 2 or train_labels.shape[1] == 0:
        return 0.0, {}

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)

    per_label_scores: Dict[int, float] = {}
    valid_scores: List[float] = []
    for label_idx in range(train_labels.shape[1]):
        y_train = train_labels[:, label_idx]
        y_test = test_labels[:, label_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        pos_train = int(y_train.sum())
        neg_train = int(len(y_train) - pos_train)
        pos_test = int(y_test.sum())
        neg_test = int(len(y_test) - pos_test)
        if min(pos_train, neg_train, pos_test, neg_test) < min_class_count:
            continue

        clf = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state,
            n_jobs=4,
        )
        try:
            clf.fit(train_scaled, y_train)
            predictions = clf.predict(test_scaled)
            score = float(balanced_accuracy_score(y_test, predictions))
            per_label_scores[label_idx] = score
            valid_scores.append(score)
        except Exception as e:
            logger.warning(f"Matched semantic transfer failed for label {label_idx}: {e}")

    if not valid_scores:
        return 0.0, {}
    return float(np.mean(valid_scores)), per_label_scores


def _load_matched_semantic_targets(
    manifest_path: Union[str, Path],
    perturbed_volumes: List[PerturbedVolume],
    cache_dir: Optional[Path],
) -> Tuple[Optional[np.ndarray], List[str], Optional[Dict[str, Any]], Optional[np.ndarray]]:
    """Load semantic targets aligned to the prepared perturbation dataset order."""
    if not perturbed_volumes:
        return None, [], None, None

    datasets = sorted({volume.dataset for volume in perturbed_volumes})
    if len(datasets) != 1:
        logger.warning(
            "Matched semantic evaluation currently expects a single-dataset manifest; "
            f"found datasets: {datasets}. Skipping semantic metrics."
        )
        return None, [], None, None

    try:
        from .semantic_labels import compute_semantic_task_labels
    except ImportError:  # pragma: no cover - direct script execution fallback
        from semantic_labels import compute_semantic_task_labels

    target_dataset = datasets[0]
    semantic_labels_full, semantic_paths, semantic_metadata = compute_semantic_task_labels(
        manifest_path,
        mode="auto",
        filter_dataset=target_dataset,
        cache_dir=cache_dir,
    )
    if semantic_labels_full.ndim != 2 or semantic_labels_full.shape[1] == 0:
        logger.warning("No informative semantic targets available for matched perturbation semantics")
        return None, [], semantic_metadata, None

    path_to_index = {path: idx for idx, path in enumerate(semantic_paths)}
    valid_indices: List[int] = []
    aligned_labels: List[np.ndarray] = []
    for sample_index, volume in enumerate(perturbed_volumes):
        if volume.original_path in path_to_index:
            valid_indices.append(sample_index)
            aligned_labels.append(semantic_labels_full[path_to_index[volume.original_path]])

    if len(aligned_labels) < 10:
        logger.warning(
            f"Only {len(aligned_labels)} perturbation samples carry semantic labels; skipping matched semantic metrics"
        )
        return None, semantic_metadata.get("organ_names", []) if semantic_metadata else [], semantic_metadata, None

    return (
        np.stack(aligned_labels),
        semantic_metadata.get("organ_names", []) if semantic_metadata else [],
        semantic_metadata,
        np.asarray(valid_indices, dtype=np.int64),
    )


def run_matched_semantic_probing(
    features_by_spacing: Dict[str, np.ndarray],
    semantic_labels: np.ndarray,
    label_names: List[str],
    checkpoint_name: str,
    feature_type: str,
    output_dir: Path,
    n_cv_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Evaluate semantic accessibility independently for each perturbation variant."""
    fold_indices = _build_matched_split_indices(semantic_labels, n_splits=n_cv_splits, random_state=random_state)

    variant_scores: Dict[str, float] = {}
    per_label_scores: Dict[str, Dict[int, float]] = {}
    for variant_name, variant_features in features_by_spacing.items():
        mean_score, label_score_map = _cross_validated_multilabel_balanced_accuracy(
            variant_features,
            semantic_labels,
            fold_indices,
            random_state=random_state,
        )
        variant_scores[variant_name] = mean_score
        per_label_scores[variant_name] = label_score_map

    results = {
        "checkpoint": checkpoint_name,
        "feature_type": feature_type,
        "metric_name": "mean_balanced_accuracy",
        "n_labels": int(semantic_labels.shape[1]),
        "label_names": label_names,
        "variant_balanced_accuracies": variant_scores,
        "per_label_balanced_accuracies": {
            variant_name: {str(label_idx): score for label_idx, score in label_map.items()}
            for variant_name, label_map in per_label_scores.items()
        },
        "mean_variant_accuracy": float(np.mean(list(variant_scores.values()))) if variant_scores else 0.0,
    }

    results_path = output_dir / "matched_semantic_probing.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved matched semantic probing results to {results_path}")
    return results


def compute_matched_semantic_transfer(
    features_by_spacing: Dict[str, np.ndarray],
    semantic_labels: np.ndarray,
    label_names: List[str],
    checkpoint_name: str,
    feature_type: str,
    n_cv_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Compute semantic transfer between perturbation variants with shared folds."""
    fold_indices = _build_matched_split_indices(semantic_labels, n_splits=n_cv_splits, random_state=random_state)
    variant_names = list(features_by_spacing.keys())

    transfer_matrix: Dict[str, Dict[str, float]] = {name: {} for name in variant_names}
    per_label_transfer: Dict[str, Dict[str, Dict[int, float]]] = {name: {} for name in variant_names}

    for train_variant in variant_names:
        for test_variant in variant_names:
            fold_scores: List[float] = []
            label_score_lists: Dict[int, List[float]] = {label_idx: [] for label_idx in range(semantic_labels.shape[1])}

            for train_idx, test_idx in fold_indices:
                score, label_scores = _train_and_evaluate_multilabel_transfer_once(
                    features_by_spacing[train_variant][train_idx],
                    semantic_labels[train_idx],
                    features_by_spacing[test_variant][test_idx],
                    semantic_labels[test_idx],
                    random_state=random_state,
                )
                if score > 0:
                    fold_scores.append(score)
                for label_idx, label_score in label_scores.items():
                    label_score_lists[label_idx].append(label_score)

            transfer_matrix[train_variant][test_variant] = float(np.mean(fold_scores)) if fold_scores else 0.0
            per_label_transfer[train_variant][test_variant] = {
                label_idx: float(np.mean(scores))
                for label_idx, scores in label_score_lists.items()
                if scores
            }

    in_variant_accuracies = {variant: transfer_matrix[variant][variant] for variant in variant_names}
    cross_variant_scores = [
        transfer_matrix[train_variant][test_variant]
        for train_variant in variant_names
        for test_variant in variant_names
        if train_variant != test_variant
    ]
    cross_variant_accuracy = float(np.mean(cross_variant_scores)) if cross_variant_scores else 0.0
    in_variant_mean = float(np.mean(list(in_variant_accuracies.values()))) if in_variant_accuracies else 0.0

    results = {
        "checkpoint": checkpoint_name,
        "feature_type": feature_type,
        "metric_name": "mean_balanced_accuracy",
        "n_labels": int(semantic_labels.shape[1]),
        "label_names": label_names,
        "transfer_matrix": transfer_matrix,
        "per_label_transfer": {
            train_variant: {
                test_variant: {str(label_idx): score for label_idx, score in label_scores.items()}
                for test_variant, label_scores in test_map.items()
            }
            for train_variant, test_map in per_label_transfer.items()
        },
        "in_variant_accuracies": in_variant_accuracies,
        "cross_variant_accuracy": cross_variant_accuracy,
        "transfer_gap": in_variant_mean - cross_variant_accuracy,
    }
    return results


def run_matched_semantic_transfer(
    features_by_spacing: Dict[str, np.ndarray],
    semantic_labels: np.ndarray,
    label_names: List[str],
    checkpoint_name: str,
    feature_type: str,
    output_dir: Path,
    n_cv_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Evaluate semantic transfer between perturbation variants with shared folds."""
    results = compute_matched_semantic_transfer(
        features_by_spacing,
        semantic_labels,
        label_names,
        checkpoint_name,
        feature_type,
        n_cv_splits=n_cv_splits,
        random_state=random_state,
    )

    results_path = output_dir / "matched_semantic_transfer.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved matched semantic transfer results to {results_path}")
    return results


def plot_drift_curve(
    representation_drift: Dict[str, float],
    representation_drift_std: Dict[str, float],
    checkpoint_name: str,
    feature_type: str = "cls",
    output_dir: Optional[Path] = None,
    analysis_name: str = "default",
) -> Path:
    """
    Plot representation drift vs spacing anisotropy ratio.

    Creates a curve showing how much embeddings drift from isotropic
    as spacing becomes more anisotropic.

    Args:
        representation_drift: Mean drift per target spacing
        representation_drift_std: Std of drift per target spacing
        checkpoint_name: Name of checkpoint
        feature_type: Feature type used
        output_dir: Output directory for figure

    Returns:
        Path to saved figure
    """
    import matplotlib.pyplot as plt

    output_dir = output_dir or get_output_paths(analysis_name)["figures"]
    output_dir.mkdir(parents=True, exist_ok=True)

    def _parse_variant_axis(spacing_name: str) -> Tuple[float, str]:
        """Map a spacing variant name to a plottable x-axis value and axis mode.

        Supported formats:
        - fixed absolute spacing protocol: "1.0x1.0x3.0" -> anisotropy ratio
        - MR native in-plane protocol: "native_xy_z4.0" -> target z spacing
        """
        if "x" in spacing_name:
            parts = spacing_name.split("x")
            if len(parts) == 3:
                sx, sy, sz = float(parts[0]), float(parts[1]), float(parts[2])
                return max(sx, sy, sz) / min(sx, sy, sz), "anisotropy_ratio"

        if spacing_name.startswith("native_xy_z"):
            z_spacing = float(spacing_name.removeprefix("native_xy_z"))
            return z_spacing, "target_z_spacing"

        raise ValueError(f"Unsupported spacing variant name for plotting: {spacing_name}")

    x_values = []
    drifts = []
    stds = []
    axis_modes = set()

    parsed_points = []
    for spacing_name, drift in representation_drift.items():
        x_value, axis_mode = _parse_variant_axis(spacing_name)
        parsed_points.append((x_value, spacing_name, drift))
        axis_modes.add(axis_mode)

    if len(axis_modes) != 1:
        raise ValueError(f"Mixed perturbation axis modes are not supported: {sorted(axis_modes)}")

    axis_mode = next(iter(axis_modes))

    for x_value, spacing_name, drift in sorted(parsed_points, key=lambda item: item[0]):
        x_values.append(x_value)
        drifts.append(drift)
        stds.append(representation_drift_std.get(spacing_name, 0))

    # Add baseline point (reference variant -> itself = 0 drift).
    baseline_x = 1.0 if axis_mode == "anisotropy_ratio" else min(x_values)
    x_values = [baseline_x] + x_values
    drifts = [0.0] + drifts
    stds = [0.0] + stds

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(x_values, drifts, yerr=stds,
                fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8,
                color='#2E86AB', ecolor='#A23B72')

    xlabel = (
        'Anisotropy Ratio (max spacing / min spacing)'
        if axis_mode == "anisotropy_ratio"
        else 'Target Through-Plane Spacing (mm)'
    )
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Representation Drift (cosine distance)', fontsize=12)
    ax.set_title(f'Representation Drift under Spacing Perturbation\n{checkpoint_name} ({feature_type})',
                 fontsize=14)

    x_min = min(x_values)
    x_max = max(x_values)
    if axis_mode == "anisotropy_ratio":
        ax.set_xlim(0.8, x_max * 1.1)
    else:
        padding = max((x_max - x_min) * 0.1, 0.25)
        ax.set_xlim(max(0.0, x_min - padding), x_max + padding)
    ax.set_ylim(0, max(d + s for d, s in zip(drifts, stds)) * 1.15)

    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add annotation for key points
    for i, (x_value, drift, std) in enumerate(zip(x_values[1:], drifts[1:], stds[1:]), 1):
        ax.annotate(f'{drift:.3f}±{std:.3f}',
                   (x_value, drift),
                   textcoords="offset points",
                   xytext=(0, 10),
                   ha='center', fontsize=9)

    plt.tight_layout()

    # Save figure
    fig_path = output_dir / "drift_curve.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Saved drift curve to {fig_path}")
    return fig_path


def run_controlled_perturbation(
    manifest_path: Union[str, Path],
    checkpoint_name: str,
    feature_type: str = "cls",
    config: Optional[ControlledPerturbationConfig] = None,
    output_dir: Optional[Path] = None,
    figures_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    semantic_cache_dir: Optional[Path] = None,
    analysis_name: str = "default",
    random_state: int = 42,
    compute_matched_semantics: bool = True,
    compute_matched_semantic_probing: Optional[bool] = None,
    compute_matched_semantic_transfer: Optional[bool] = None,
    compute_drift_figure: bool = True,
    feature_cache_path: Optional[Path] = None,
    device: str = "cuda",
    batch_size: int = 1,
) -> ControlledPerturbationResults:
    """
    Run controlled spacing perturbation experiments.

    This experiment isolates spacing as a causal factor by:
    1. Taking isotropic volumes from Bin 0
    2. Resampling each to multiple target spacings
    3. Extracting features for each spacing variant
    4. Comparing representations across spacing conditions

    Args:
        manifest_path: Path to manifest JSON
        checkpoint_name: Name of checkpoint to evaluate
        feature_type: Type of features
        config: Perturbation configuration
        output_dir: Output directory for results
        figures_dir: Output directory for figures
        cache_dir: Directory for resampled volume cache
        semantic_cache_dir: Directory for matched semantic label cache
        feature_cache_path: Optional existing embedding bundle to reuse. The
            canonical bundle lives in output_dir/features and also serves as
            the inference cache on subsequent calls.

    Returns:
        ControlledPerturbationResults dataclass
    """
    try:
        from .feature_extractor import FeatureExtractor, ExtractionConfig
        from .config import CHECKPOINTS
    except ImportError:  # pragma: no cover - direct script execution fallback
        from feature_extractor import FeatureExtractor, ExtractionConfig
        from config import CHECKPOINTS

    if config is None:
        with open(manifest_path) as f:
            manifest = json.load(f)
        modality = manifest.get("modality")
        config = MRI_CONTROLLED_PERTURBATION_CONFIG if modality == "mr" else CONTROLLED_PERTURBATION_CONFIG
    expected_variants = config.describe_targets() + list(config.extra_perturbations or [])
    if len(expected_variants) < 2 or len(expected_variants) != len(set(expected_variants)):
        raise ValueError("Controlled protocol requires at least two uniquely named variants")
    dataset_name = analysis_name if analysis_name != "default" else get_dataset_name_from_manifest_path(Path(manifest_path))
    manifest_variant = get_manifest_variant_from_manifest_path(Path(manifest_path))
    output_dir, figures_dir = _resolve_controlled_dirs(
        output_dir,
        figures_dir,
        dataset_name,
        manifest_variant,
        checkpoint_name,
        feature_type,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "features" / CONTROLLED_EMBEDDING_BUNDLE_NAME
    cache_path = Path(feature_cache_path) if feature_cache_path is not None else bundle_path
    if bundle_path.is_symlink() or bundle_path.parent.is_symlink():
        raise ValueError(f"Controlled embeddings must be stored inside the encoder folder: {bundle_path}")
    if bundle_path.exists() and cache_path.absolute() != bundle_path.absolute():
        if not cache_path.exists() or not bundle_path.samefile(cache_path):
            raise ValueError(f"A different controlled embedding bundle already exists: {bundle_path}")

    # Use provided cache_dir or default
    effective_cache_dir = cache_dir or get_cache_root(dataset_name, manifest_variant)

    logger.info("Running controlled spacing perturbation")
    logger.info(f"Checkpoint: {checkpoint_name}")
    logger.info(f"Perturbation protocol: {config.protocol_name}")
    logger.info(f"Target variants: {config.describe_targets()}")
    logger.info(f"Source bin: {config.source_bin}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Figures dir: {figures_dir}")
    logger.info(f"Cache dir: {effective_cache_dir}")

    # Get crop size from checkpoint
    ckpt_config = CHECKPOINTS[checkpoint_name]
    crop_size = ckpt_config.crop_size

    # Prepare perturbed dataset (using the effective cache dir)
    perturbed_volumes = prepare_perturbed_dataset(
        manifest_path, config, crop_size, cache_dir=effective_cache_dir
    )

    if not perturbed_volumes:
        raise ValueError("No eligible controlled source volumes")

    expected_paths = [str(volume.original_path) for volume in perturbed_volumes]
    reused_features = cache_path.exists()
    rich_bundle = False
    if reused_features:
        features_by_spacing, rich_bundle = _load_controlled_features(
            cache_path, perturbed_volumes, config, checkpoint_name, feature_type, dataset_name,
        )
    else:
        extraction = ExtractionConfig(extract_cls=feature_type == "cls", extract_avg_pool=feature_type == "avg_pool", extract_multilayer=feature_type == "multilayer", device=device, batch_size=batch_size)
        extractor = FeatureExtractor(checkpoint_name, config=extraction)
        features_by_spacing = extract_features_for_variants(perturbed_volumes, extractor, feature_type, batch_size=batch_size)
        del extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if set(features_by_spacing) != set(expected_variants):
        raise ValueError("Controlled embeddings have incomplete protocol variants")
    features_by_spacing = {name: features_by_spacing[name] for name in expected_variants}
    if any(array.dtype.kind not in "fi" or array.ndim != 2 or array.shape[0] != len(expected_paths) or array.shape[1] < 1 or not np.isfinite(array).all() for array in features_by_spacing.values()):
        raise ValueError("Incomplete or nonfinite controlled embeddings")
    if len({array.shape[1] for array in features_by_spacing.values()}) != 1:
        raise ValueError("Controlled variants have inconsistent feature dimensions")
    variant_names = list(features_by_spacing.keys())
    variant_metadata = _resolve_variant_metadata(config, perturbed_volumes, variant_names)
    variant_metadata_json = _variant_metadata_for_json(variant_metadata, variant_names)

    # Compute pairwise distances
    logger.info("Computing pairwise distances...")
    per_sample_dists, mean_dists, std_dists = compute_pairwise_distances(features_by_spacing)

    # Compute representation drift from isotropic (1x1x1)
    original_name = next(iter(features_by_spacing.keys()))
    representation_drift = {}
    representation_drift_std = {}
    for spacing_name in features_by_spacing.keys():
        if spacing_name != original_name:
            pair = tuple(sorted([original_name, spacing_name]))
            if pair in mean_dists:
                representation_drift[spacing_name] = mean_dists[pair]
                representation_drift_std[spacing_name] = std_dists[pair]

    # Compute CKA between variants
    logger.info("Computing CKA between spacing variants...")
    cka_matrix = compute_cka_between_variants(features_by_spacing)

    if rich_bundle:
        if cache_path.absolute() != bundle_path.absolute():
            bundle_path.parent.mkdir(parents=True, exist_ok=True)
            if bundle_path.exists():
                if not bundle_path.samefile(cache_path):
                    raise ValueError(f"A different controlled embedding bundle already exists: {bundle_path}")
            else:
                os.link(cache_path, bundle_path)
    else:
        bundle_path = save_controlled_embedding_bundle(
            output_dir=bundle_path.parent,
            features_by_spacing=features_by_spacing,
            perturbed_volumes=perturbed_volumes,
            config=config,
            checkpoint_name=checkpoint_name,
            feature_type=feature_type,
            dataset_name=dataset_name,
            manifest_variant=manifest_variant,
            reference_variant=original_name,
        )

    # Compute matched semantic probing/transfer if semantic targets are available.
    matched_semantic_probing = None
    matched_semantic_transfer = None
    semantic_metadata = None
    if compute_matched_semantic_probing is None:
        compute_matched_semantic_probing = compute_matched_semantics
    if compute_matched_semantic_transfer is None:
        compute_matched_semantic_transfer = compute_matched_semantics
    compute_any_matched_semantics = bool(compute_matched_semantic_probing or compute_matched_semantic_transfer)
    # Use a shared dataset-level directory for semantic label cache so that
    # parallel checkpoint/feature-type runs reuse one extraction pass instead
    # of each re-extracting the full manifest independently.
    shared_semantic_cache_dir = semantic_cache_dir or get_output_paths(dataset_name, manifest_variant)["results"]
    if compute_any_matched_semantics:
        semantic_labels, label_names, semantic_metadata, valid_indices = _load_matched_semantic_targets(
            manifest_path,
            perturbed_volumes,
            shared_semantic_cache_dir,
        )
        if semantic_labels is not None and valid_indices is not None:
            semantic_features_by_spacing = {
                spacing_name: spacing_features[valid_indices]
                for spacing_name, spacing_features in features_by_spacing.items()
            }
            if compute_matched_semantic_probing:
                logger.info("Computing matched semantic probing under controlled perturbation...")
                matched_semantic_probing = run_matched_semantic_probing(
                    semantic_features_by_spacing,
                    semantic_labels,
                    label_names,
                    checkpoint_name,
                    feature_type,
                    output_dir,
                    random_state=random_state,
                )
            else:
                logger.info("Skipping matched controlled semantic probing")

            if compute_matched_semantic_transfer:
                logger.info("Computing matched semantic transfer across perturbation variants...")
                matched_semantic_transfer = run_matched_semantic_transfer(
                    semantic_features_by_spacing,
                    semantic_labels,
                    label_names,
                    checkpoint_name,
                    feature_type,
                    output_dir,
                    random_state=random_state,
                )
            else:
                logger.info("Skipping matched controlled semantic transfer")
    else:
        logger.info("Skipping matched controlled semantic probing/transfer")

    if compute_drift_figure:
        spacing_drift = {
            variant: representation_drift[variant]
            for variant in variant_metadata["spacing_variants"]
            if variant in representation_drift
        }
        spacing_drift_std = {
            variant: representation_drift_std[variant]
            for variant in spacing_drift
        }
        if spacing_drift:
            logger.info("Generating spacing drift curve visualization...")
            plot_drift_curve(
                representation_drift=spacing_drift,
                representation_drift_std=spacing_drift_std,
                checkpoint_name=checkpoint_name,
                feature_type=feature_type,
                output_dir=figures_dir,
            )
        else:
            logger.info("Skipping drift curve: no non-reference spacing variants available")
    else:
        logger.info("Skipping controlled drift figure")

    # Build results
    results = ControlledPerturbationResults(
        checkpoint_name=checkpoint_name,
        feature_type=feature_type,
        pairwise_distances=per_sample_dists,
        mean_distance_per_pair=mean_dists,
        std_distance_per_pair=std_dists,
        representation_drift=representation_drift,
        representation_drift_std=representation_drift_std,
        cka_matrix=cka_matrix,
        matched_semantic_probing=matched_semantic_probing,
        matched_semantic_transfer=matched_semantic_transfer,
        semantic_metadata=semantic_metadata,
        target_spacings=config.target_spacings,
        n_source_volumes=len(perturbed_volumes),
    )

    # Save results
    # Convert tuple keys to strings for JSON
    mean_dists_json = {f"{k[0]}___{k[1]}": v for k, v in mean_dists.items()}
    std_dists_json = {f"{k[0]}___{k[1]}": v for k, v in std_dists.items()}
    per_sample_json = {
        str(idx): {f"{k[0]}___{k[1]}": v for k, v in dists.items()}
        for idx, dists in per_sample_dists.items()
    }

    results_dict = {
        "checkpoint": checkpoint_name,
        "feature_type": feature_type,
        "task_name": "controlled_spacing_perturbation",
        "alternate_name": "controlled_perturbation",
        "perturbation_protocol": config.protocol_name,
        "target_variants": config.describe_targets(),
        "target_spacings": [list(s) for s in config.target_spacings] if config.target_spacings is not None else None,
        "target_z_spacings": list(config.target_z_spacings) if config.target_z_spacings is not None else None,
        "source_bin": config.source_bin,
        "reference_variant": original_name,
        "variant_metadata": variant_metadata_json,
        "controlled_embedding_bundle": bundle_path.relative_to(output_dir).as_posix(),
        "n_source_volumes": len(perturbed_volumes),
        "representation_drift": representation_drift,
        "representation_drift_std": representation_drift_std,
        "mean_distance_per_pair": mean_dists_json,
        "std_distance_per_pair": std_dists_json,
        "cka_matrix": cka_matrix,
        "matched_semantic_probing": matched_semantic_probing,
        "matched_semantic_transfer": matched_semantic_transfer,
        "semantic_metadata": semantic_metadata,
        "drift_figure": "drift_curve.png" if compute_drift_figure else None,
        "drift_figure_skipped": not compute_drift_figure,
        # "per_sample_distances": per_sample_json,  # Large, omit by default
    }

    results_path = output_dir / "perturbation_robustness.json"
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    logger.info(f"Saved controlled perturbation results to {results_path}")

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Controlled perturbation summary: {checkpoint_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Volumes processed: {len(perturbed_volumes)}")

    logger.info("\nRepresentation drift from isotropic baseline:")
    for spacing, drift in sorted(representation_drift.items()):
        std = representation_drift_std.get(spacing, 0)
        logger.info(f"  1.0x1.0x1.0 → {spacing}: {drift:.4f} ± {std:.4f}")

    logger.info(f"\nInterpretation: Lower drift = more spacing-robust representation")

    logger.info("\nPairwise distances across spacing variants:")
    for pair, dist in sorted(mean_dists.items()):
        std = std_dists.get(pair, 0)
        logger.info(f"  {pair[0]} <-> {pair[1]}: {dist:.4f} ± {std:.4f}")

    logger.info("\nCKA matrix across spacing variants:")
    for spacing_a in cka_matrix.keys():
        row = [f"{cka_matrix[spacing_a][spacing_b]:.3f}" for spacing_b in cka_matrix.keys()]
        logger.info(f"  {spacing_a}: {' '.join(row)}")

    return results


def main():
    """Main entry point for controlled perturbation experiments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Acquisition Robustness: Controlled Spacing Perturbation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python perturbation_robustness_analysis.py -m abdomenatlas/original_bins/manifest_sampled.json -a abdomenatlas -c med3dino_rel_96 -f cls\n"
            "  python perturbation_robustness_analysis.py -m totalsegmentermri/original_bins/manifest_sampled.json -a totalsegmentermri -c med3dino_rel_96 -f avg_pool -n 24\n"
            "  python perturbation_robustness_analysis.py -m totalsegmenter_ct/original_bins/manifest_sampled.json -a totalsegmenter_ct -c med3dino_rel_96 -f cls -n 24"
        ),
    )
    parser.add_argument(
        "-m", "--manifest",
        type=Path,
        required=True,
        help="Path to manifest JSON",
    )
    parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        default="med3dino_rel_96",
        help="Checkpoint to evaluate",
    )
    parser.add_argument(
        "-f", "--feature-type",
        type=str,
        choices=["cls", "avg_pool", "multilayer"],
        default="cls",
        help="Feature type",
    )
    parser.add_argument(
        "-n", "--n-volumes",
        type=int,
        default=None,
        help="Number of source volumes (default: all)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "-a", "--analysis-name",
        type=str,
        default=None,
        help="Dataset/analysis namespace for default output paths",
    )

    args = parser.parse_args()

    # Override config
    config = ControlledPerturbationConfig()
    if args.n_volumes:
        config.n_source_volumes = args.n_volumes

    analysis_name = args.analysis_name or get_dataset_name_from_manifest_path(args.manifest)
    manifest_variant = get_manifest_variant_from_manifest_path(args.manifest)
    output_paths = get_output_paths(analysis_name, manifest_variant)

    # Run experiment
    results = run_controlled_perturbation(
        manifest_path=args.manifest,
        checkpoint_name=args.checkpoint,
        feature_type=args.feature_type,
        config=config,
        output_dir=args.output_dir,
        figures_dir=None,
        cache_dir=output_paths["cache_root"],
        analysis_name=analysis_name,
    )

    if results:
        logger.info("Controlled perturbation completed successfully")
    else:
        logger.error("Controlled perturbation failed")


if __name__ == "__main__":
    main()
