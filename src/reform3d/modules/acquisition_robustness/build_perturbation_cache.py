#!/usr/bin/env python
"""Optionally prewarm controlled-spacing inputs for the direct acquisition runner.

    python -m reform3d.modules.acquisition_robustness.build_perturbation_cache \
        --manifest /path/to/manifest.json --crop-size 96 112

The direct runner's configured cache root and protocol signature determine the
output location. The YAML workflow prepares its own geometry caches on demand.
Volumes are processed individually; compatible cached volumes are reused.
"""

import argparse
import gc
import json
import sys
import logging
from pathlib import Path
from tqdm import tqdm

try:
    from .perturbation_robustness_analysis import (
        get_cache_path,
        load_preprocessed_source_image,
        resample_preprocessed_volume,
        resample_volume,
        save_to_cache,
    )
    from .config import (
        ACQUISITION_MANIFESTS,
        ControlledPerturbationConfig,
        get_binning_scheme_from_manifest,
        get_cache_root,
        get_controlled_perturbation_config_from_manifest,
        get_dataset_name_from_manifest_path,
        get_manifest_variant_from_manifest_path,
        resolve_acquisition_manifest_path,
    )
    from .protocol_artifacts import apply_controlled_perturbation_overrides
except ImportError:  # pragma: no cover - direct script execution fallback
    from perturbation_robustness_analysis import (
        get_cache_path,
        load_preprocessed_source_image,
        resample_preprocessed_volume,
        resample_volume,
        save_to_cache,
    )
    from config import (
        ACQUISITION_MANIFESTS,
        ControlledPerturbationConfig,
        get_binning_scheme_from_manifest,
        get_cache_root,
        get_controlled_perturbation_config_from_manifest,
        get_dataset_name_from_manifest_path,
        get_manifest_variant_from_manifest_path,
        resolve_acquisition_manifest_path,
    )
    from protocol_artifacts import apply_controlled_perturbation_overrides

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


CACHE_BUILD_GC_INTERVAL = 16


def infer_dataset_name(manifest_path: Path, manifest_data: dict) -> str:
    """Infer dataset name from manifest layout, falling back to manifest metadata."""
    dataset_name = get_dataset_name_from_manifest_path(manifest_path, fallback="")
    if dataset_name:
        return dataset_name
    return manifest_data.get("dataset", "default")


def infer_manifest_variant(manifest_path: Path, manifest_data: dict) -> str:
    """Infer manifest variant from manifest layout, falling back to manifest metadata."""
    manifest_variant = get_manifest_variant_from_manifest_path(manifest_path, fallback="")
    if manifest_variant:
        return manifest_variant

    binning_scheme = get_binning_scheme_from_manifest(manifest_data)
    if binning_scheme == "coarse_ratio_thickness":
        return "coarse_bins"
    return "original_bins"


def generate_cache_for_crop_size(
    manifest_path: Path,
    crop_size: int,
    cache_base_dir: Path,
    perturbation_config: ControlledPerturbationConfig,
) -> None:
    """Generate cache for a single crop size."""
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"Generating cache for crop_size={crop_size}")
    logger.info("=" * 70)

    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return

    logger.info(f"Manifest: {manifest_path}")
    logger.info(f"Cache dir: {cache_base_dir}")
    logger.info(f"Protocol: {perturbation_config.protocol_name}")
    logger.info(f"Target variants: {perturbation_config.describe_targets()}")
    logger.info(f"Source bin: {perturbation_config.source_bin}")

    # Create cache directory
    cache_dir = cache_base_dir / f"crop{crop_size}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)
    volumes = manifest["volumes"]
    cache_signature = perturbation_config.cache_signature()

    # Filter to source bin (isotropic)
    source_volumes = [v for v in volumes if v.get("anisotropy_bin") == perturbation_config.source_bin]
    logger.info(f"Found {len(source_volumes)} volumes in source bin {perturbation_config.source_bin}")

    # Find uncached volumes ONLY (fast file existence check)
    uncached_volumes = []
    for vol_info in source_volumes:
        cache_path = get_cache_path(
            vol_info["file_path"],
            crop_size,
            cache_dir=cache_base_dir,
            cache_signature=cache_signature,
        )
        if not cache_path.exists():
            uncached_volumes.append(vol_info)

    cached_count = len(source_volumes) - len(uncached_volumes)
    logger.info(f"Cache status: {cached_count}/{len(source_volumes)} already cached")
    logger.info(f"Need to resample: {len(uncached_volumes)} volumes")

    if len(uncached_volumes) == 0:
        logger.info("All volumes already cached! Nothing to do.")
        return

    # Process ONLY uncached volumes (no memory accumulation)
    success_count = 0
    fail_count = 0

    for vol_info in tqdm(uncached_volumes, desc=f"Resampling (crop{crop_size})"):
        file_path = vol_info["file_path"]

        try:
            variant_specs = perturbation_config.resolve_variant_specs(tuple(vol_info["spacing"]))

            # Resample volume
            variants = resample_volume(
                file_path,
                variant_specs,
                crop_size=crop_size,
                source_spacing=tuple(vol_info["spacing"]),
                config=perturbation_config,
            )

            # Save to cache if all variants succeeded
            if not any(v is None for v in variants.values()):
                save_to_cache(
                    file_path,
                    variants,
                    crop_size,
                    cache_dir=cache_base_dir,
                    cache_signature=cache_signature,
                )
                success_count += 1
            else:
                logger.warning(f"Skipping {Path(file_path).name} - some variants failed")
                fail_count += 1

            # CRITICAL: Free memory immediately after each volume
            del variants
            gc.collect()

        except Exception as e:
            logger.error(f"Failed to process {Path(file_path).name}: {e}")
            fail_count += 1
            gc.collect()

    logger.info("=" * 70)
    logger.info(f"Cache generation complete for crop_size={crop_size}!")
    logger.info(f"Successfully cached: {success_count}")
    logger.info(f"Failed: {fail_count}")
    logger.info(f"Total cached: {cached_count + success_count}/{len(source_volumes)}")
    logger.info(f"Cache location: {cache_dir}")

    # Estimate cache size
    if cache_dir.exists():
        total_size = sum(f.stat().st_size for f in cache_dir.rglob("*.pt"))
        logger.info(f"Cache size (crop{crop_size}): {total_size / 1e9:.2f} GB")
    logger.info("=" * 70)


def generate_cache_for_crop_sizes(
    manifest_path: Path,
    crop_sizes: list[int],
    cache_base_dir: Path,
    perturbation_config: ControlledPerturbationConfig,
) -> None:
    """Generate cache for multiple crop sizes while loading each source volume only once."""
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"Generating cache for crop sizes={sorted(set(crop_sizes))}")
    logger.info("=" * 70)

    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)
    volumes = manifest["volumes"]
    cache_signature = perturbation_config.cache_signature()
    source_volumes = [v for v in volumes if v.get("anisotropy_bin") == perturbation_config.source_bin]
    crop_sizes = sorted(set(int(crop_size) for crop_size in crop_sizes))

    cached_counts = {crop_size: 0 for crop_size in crop_sizes}
    success_counts = {crop_size: 0 for crop_size in crop_sizes}
    fail_counts = {crop_size: 0 for crop_size in crop_sizes}
    pending = []

    logger.info(f"Found {len(source_volumes)} volumes in source bin {perturbation_config.source_bin}")

    for vol_info in source_volumes:
        missing_crop_sizes = []
        for crop_size in crop_sizes:
            cache_path = get_cache_path(
                vol_info["file_path"],
                crop_size,
                cache_dir=cache_base_dir,
                cache_signature=cache_signature,
            )
            if cache_path.exists():
                cached_counts[crop_size] += 1
            else:
                missing_crop_sizes.append(crop_size)
        if missing_crop_sizes:
            pending.append((vol_info, missing_crop_sizes))

    for crop_size in crop_sizes:
        logger.info(
            "Cache status crop%d: %d/%d already cached",
            crop_size,
            cached_counts[crop_size],
            len(source_volumes),
        )

    if not pending:
        logger.info("All requested volumes are already cached for every crop size.")
        return

    logger.info(f"Need to resample: {len(pending)} source volumes")

    for index, (vol_info, missing_crop_sizes) in enumerate(tqdm(pending, desc="Resampling shared source volumes"), start=1):
        file_path = vol_info["file_path"]

        try:
            variant_specs = perturbation_config.resolve_variant_specs(tuple(vol_info["spacing"]))
            base_image, permuted_spacing = load_preprocessed_source_image(
                file_path,
                source_spacing=tuple(vol_info["spacing"]),
                config=perturbation_config,
            )
        except Exception as e:
            logger.error(f"Failed to load {Path(file_path).name}: {e}")
            for crop_size in missing_crop_sizes:
                fail_counts[crop_size] += 1
            if index % CACHE_BUILD_GC_INTERVAL == 0:
                gc.collect()
            continue

        try:
            for crop_size in missing_crop_sizes:
                try:
                    variants = resample_preprocessed_volume(
                        base_image,
                        permuted_spacing,
                        variant_specs,
                        crop_size=crop_size,
                        config=perturbation_config,
                    )
                    if not any(v is None for v in variants.values()):
                        save_to_cache(
                            file_path,
                            variants,
                            crop_size,
                            cache_dir=cache_base_dir,
                            cache_signature=cache_signature,
                        )
                        success_counts[crop_size] += 1
                    else:
                        logger.warning(f"Skipping {Path(file_path).name} crop{crop_size} - some variants failed")
                        fail_counts[crop_size] += 1
                    del variants
                except Exception as e:
                    logger.error(f"Failed to process {Path(file_path).name} for crop{crop_size}: {e}")
                    fail_counts[crop_size] += 1
        finally:
            del base_image

        if index % CACHE_BUILD_GC_INTERVAL == 0:
            gc.collect()

    gc.collect()

    logger.info("=" * 70)
    logger.info("Shared cache generation complete")
    for crop_size in crop_sizes:
        cache_dir = cache_base_dir / f"crop{crop_size}"
        logger.info(
            "crop%d: cached=%d success=%d fail=%d total=%d/%d",
            crop_size,
            cached_counts[crop_size],
            success_counts[crop_size],
            fail_counts[crop_size],
            cached_counts[crop_size] + success_counts[crop_size],
            len(source_volumes),
        )
        if cache_dir.exists():
            total_size = sum(f.stat().st_size for f in cache_dir.rglob("*.pt"))
            logger.info(f"Cache size (crop{crop_size}): {total_size / 1e9:.2f} GB")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate controlled perturbation cache for acquisition robustness evaluation"
    )
    parser.add_argument(
        "manifest_positional",
        nargs="?",
        default=None,
        help="Optional positional manifest path. Equivalent to --manifest."
    )
    parser.add_argument(
        "--crop-size", "-c",
        type=int,
        nargs="+",
        default=[96],
        help="Crop size(s) to generate cache for (default: 96). Can specify multiple: --crop-size 96 112"
    )
    parser.add_argument(
        "--manifest", "-m",
        type=str,
        default=None,
        help="Path to manifest JSON file"
    )
    parser.add_argument(
        "--analysis-name", "-a",
        type=str,
        default=None,
        help="Dataset name override for cache routing (e.g., 'abdomenatlas', 'kits23')"
    )
    parser.add_argument(
        "--orientation-policy",
        default="spacing_permutation",
        choices=("spacing_permutation", "identity"),
        help="Orientation standardization used before controlled resampling.",
    )
    parser.add_argument(
        "--physical-fov-mm",
        nargs=3,
        type=float,
        default=None,
        metavar=("H_MM", "W_MM", "D_MM"),
        help="Optional fixed physical FOV crop after resampling and before resizing.",
    )
    parser.add_argument(
        "--extra-perturbations",
        action="store_true",
        help="Include deterministic kernel, noise/dose, and plane-reformatting variants in the cache.",
    )
    args = parser.parse_args()

    manifest_arg = args.manifest or args.manifest_positional
    if manifest_arg is None:
        parser.error("the following arguments are required: --manifest/-m or a positional manifest path")

    manifest_path = Path(manifest_arg)
    if not manifest_path.is_absolute():
        candidate_paths = [
            Path.cwd() / manifest_path,
            Path(__file__).resolve().parent.parent / manifest_path,
            resolve_acquisition_manifest_path(manifest_path),
            resolve_acquisition_manifest_path(manifest_path.name),
        ]
        resolved_path = None
        for candidate in candidate_paths:
            if candidate.exists():
                resolved_path = candidate.resolve()
                break
        if resolved_path is not None:
            manifest_path = resolved_path

    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest_data = json.load(f)

    inferred_dataset_name = infer_dataset_name(manifest_path, manifest_data)
    dataset_name = args.analysis_name
    if dataset_name in {None, "default"}:
        dataset_name = inferred_dataset_name
    manifest_variant = infer_manifest_variant(manifest_path, manifest_data)
    cache_base_dir = get_cache_root(dataset_name, manifest_variant)
    cache_base_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Pre-generating controlled perturbation cache (MEMORY-EFFICIENT)")
    logger.info("=" * 70)
    logger.info(f"Dataset name: {dataset_name}")
    logger.info(f"Inferred dataset name: {inferred_dataset_name}")
    logger.info(f"Manifest variant: {manifest_variant}")
    logger.info(f"Manifest: {manifest_path}")
    logger.info(f"Cache base: {cache_base_dir}")
    logger.info(f"Crop sizes to generate: {args.crop_size}")
    perturbation_config = apply_controlled_perturbation_overrides(
        get_controlled_perturbation_config_from_manifest(manifest_data),
        orientation_policy=args.orientation_policy,
        physical_fov_mm=args.physical_fov_mm,
        extra_perturbations=args.extra_perturbations,
    )
    logger.info(f"Protocol: {perturbation_config.protocol_name}")
    logger.info(f"Target variants: {perturbation_config.describe_targets()}")
    logger.info(f"Cache signature: {perturbation_config.cache_signature()}")

    generate_cache_for_crop_sizes(
        manifest_path,
        args.crop_size,
        cache_base_dir,
        perturbation_config,
    )


if __name__ == "__main__":
    main()
