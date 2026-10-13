"""Cross-Modal Alignment helpers for external baseline methods."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from reform3d.modules.cross_modal_alignment import feature_extractor as cross_modal_features
from reform3d.modules.cross_modal_alignment.config import (
    DEFAULT_MANIFEST_VARIANT,
    get_checkpoint_feature_npz_path,
    get_cross_modal_manifest_path,
    get_dataset_name_from_manifest_path,
    get_manifest_variant_from_manifest_path,
    get_output_paths,
    normalize_cross_modal_dataset_name,
    normalize_feature_type,
)


logger = logging.getLogger(__name__)

SINGLE_ORGAN_DATASET = "chaos_ct_mr"
SINGLE_ORGAN_NAME = "liver"
SINGLE_ORGAN_PAIR_ID_FIELD = "patient_id"


def resolve_cross_modal_manifest_path(
    dataset: str,
    manifest_path: Optional[Path],
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> Path:
    """Resolve a Cross-Modal manifest from an explicit path or dataset key."""

    if manifest_path is not None:
        resolved = Path(manifest_path).resolve()
    else:
        resolved = get_cross_modal_manifest_path(
            normalize_cross_modal_dataset_name(dataset),
            "sampled",
            manifest_variant,
        )
    if not resolved.exists():
        raise FileNotFoundError(
            f"Cross-Modal manifest not found: {resolved}. "
            "Build the manifest first or launch through reform3d/scripts/run_external_methods.sh "
            "so missing manifests are prepared before evaluation."
        )
    return resolved


def default_cross_modal_embedding_path(
    manifest_path: Path,
    checkpoint_name: str,
    feature_type: str,
) -> Path:
    """Return the default feature-bundle path for one Cross-Modal manifest."""

    dataset_name = get_dataset_name_from_manifest_path(manifest_path)
    manifest_variant = get_manifest_variant_from_manifest_path(manifest_path)
    base_dir = get_output_paths(dataset_name, manifest_variant)["features"]
    return get_checkpoint_feature_npz_path(base_dir, checkpoint_name, feature_type)


def run_cross_modal_evaluation_cli(
    *,
    manifest_path: Path,
    checkpoint_name: str,
    feature_type: str,
    embeddings_npz: Optional[Path] = None,
    analysis_name: Optional[str] = None,
    manifest_variant: Optional[str] = None,
    required_modalities: tuple[str, ...] = ("ct", "mr"),
    min_samples_per_modality: int = 5,
    max_samples_per_pool: Optional[int] = None,
    max_queries_per_organ: Optional[int] = None,
    bootstrap_resamples: int = 1000,
    seed: int = 42,
) -> None:
    """Run Cross-Modal metrics against an external method feature bundle."""

    manifest_path = Path(manifest_path).resolve()
    dataset_name = get_dataset_name_from_manifest_path(manifest_path)
    analysis_name = analysis_name or dataset_name
    manifest_variant = manifest_variant or get_manifest_variant_from_manifest_path(manifest_path)
    embeddings_npz = Path(embeddings_npz).resolve() if embeddings_npz else default_cross_modal_embedding_path(
        manifest_path,
        checkpoint_name,
        feature_type,
    )

    if dataset_name == SINGLE_ORGAN_DATASET:
        argv = [
            sys.executable,
            "-m",
            "reform3d.modules.cross_modal_alignment.single_organ_alignment",
            "-m",
            str(manifest_path),
            "--analysis-name",
            analysis_name,
            "--manifest-variant",
            str(manifest_variant),
            "--embeddings-npz",
            str(embeddings_npz),
            "--checkpoint-name",
            checkpoint_name,
            "--feature-type",
            feature_type,
            "--organ",
            SINGLE_ORGAN_NAME,
            "--pair-id-field",
            SINGLE_ORGAN_PAIR_ID_FIELD,
            "--required-modalities",
            *[str(item).lower() for item in required_modalities],
        ]
    else:
        argv = [
            sys.executable,
            "-m",
            "reform3d.modules.cross_modal_alignment.evaluation_pipeline",
            "-m",
            str(manifest_path),
            "--analysis-name",
            analysis_name,
            "--manifest-variant",
            str(manifest_variant),
            "--embeddings-npz",
            str(embeddings_npz),
            "--checkpoint-name",
            checkpoint_name,
            "--feature-type",
            feature_type,
            "--required-modalities",
            *[str(item).lower() for item in required_modalities],
            "--min-samples-per-modality",
            str(min_samples_per_modality),
            "--bootstrap-resamples",
            str(bootstrap_resamples),
            "--seed",
            str(seed),
        ]
        if max_samples_per_pool is not None:
            argv.extend(["--max-samples-per-pool", str(max_samples_per_pool)])
        if max_queries_per_organ is not None:
            argv.extend(["--max-queries-per-organ", str(max_queries_per_organ)])

    subprocess.run(argv, check=True)


def extract_cross_modal_features(
    *,
    manifest_path: Path,
    checkpoint_path: Path,
    checkpoint_name: str,
    feature_type: str,
    extractor_factory: Callable[..., Any],
    crop_size: int,
    output_npz: Optional[Path] = None,
    batch_size: int = 8,
    bbox_margin: int = cross_modal_features.DEFAULT_BBOX_MARGIN,
    outside_organ_mask: str = "none",
    min_mask_voxels: int = 100,
    max_samples: Optional[int] = None,
    crop_cache_dir: Optional[Path] = None,
    crop_cache_workers: int = 4,
    overwrite_crop_cache: bool = False,
    disable_crop_cache: bool = False,
    build_crop_cache_only: bool = False,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    extractor_kwargs: Optional[Dict[str, Any]] = None,
    result_metadata: Optional[Dict[str, Any]] = None,
) -> Path | None:
    """Extract organ embeddings for a Cross-Modal manifest using an external model."""

    manifest_path = Path(manifest_path).resolve()
    feature_type = normalize_feature_type(feature_type)
    samples = cross_modal_features._load_manifest_samples(manifest_path, max_samples=max_samples)
    if not samples:
        raise ValueError(f"No samples loaded from {manifest_path}")

    outside_organ_mask = cross_modal_features._normalize_outside_organ_mask_mode(outside_organ_mask)
    context_ablation_tag = cross_modal_features._context_ablation_tag(bbox_margin, outside_organ_mask)

    resolved_crop_cache_dir: Path | None = None
    if not disable_crop_cache or build_crop_cache_only:
        resolved_crop_cache_dir = cross_modal_features._resolve_crop_cache_dir(
            manifest_path,
            int(crop_size),
            crop_cache_dir,
            context_ablation_tag=context_ablation_tag,
        )
        cross_modal_features.ensure_crop_cache(
            samples=samples,
            manifest_path=manifest_path,
            crop_size=int(crop_size),
            bbox_margin=bbox_margin,
            min_mask_voxels=min_mask_voxels,
            outside_organ_mask=outside_organ_mask,
            cache_dir=resolved_crop_cache_dir,
            workers=crop_cache_workers,
            overwrite=overwrite_crop_cache,
        )
        logger.info("Crop cache ready at %s", resolved_crop_cache_dir)

    if build_crop_cache_only:
        return None

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    resolved_dtype = dtype
    if resolved_device == "cpu" and resolved_dtype in {torch.float16, torch.bfloat16}:
        logger.warning("Downgrading dtype to float32 on CPU")
        resolved_dtype = torch.float32

    extractor_kwargs = extractor_kwargs or {}
    extractor = extractor_factory(
        checkpoint_path=Path(checkpoint_path),
        device=resolved_device,
        dtype=resolved_dtype,
        **extractor_kwargs,
    )

    output_npz = (
        Path(output_npz).resolve()
        if output_npz
        else cross_modal_features._default_output_npz(
            manifest_path,
            checkpoint_name,
            feature_type,
            context_ablation_tag=context_ablation_tag,
        )
    )
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_npz.with_name(output_npz.stem + "_summary.json")

    feature_rows: List[np.ndarray] = []
    feature_ids: List[str] = []
    failures: List[Dict[str, str]] = []
    grouped_samples = cross_modal_features._group_samples_by_case(samples)

    if resolved_crop_cache_dir and cross_modal_features._crop_cache_index_path(resolved_crop_cache_dir).exists():
        failures = cross_modal_features._extract_from_crop_cache(
            extractor=extractor,
            cache_dir=resolved_crop_cache_dir,
            batch_size=batch_size,
            feature_type=feature_type,
            feature_rows=feature_rows,
            feature_ids=feature_ids,
        )
    else:
        pending_tensors: List[torch.Tensor] = []
        pending_ids: List[str] = []
        progress = tqdm(total=len(samples), desc=f"Extracting organ {feature_type} features", unit="sample", dynamic_ncols=True)
        for (_, image_path_str), case_samples in grouped_samples.items():
            image_path = Path(image_path_str)
            try:
                case_image, perm = cross_modal_features._load_case_image(image_path, case_samples[0].spacing)
            except Exception as exc:
                for sample in case_samples:
                    failures.append({"sample_id": sample.sample_id, "reason": f"image_load_failed: {exc}"})
                    progress.update(1)
                logger.warning("Failed to load %s: %s", image_path, exc)
                continue

            for sample in case_samples:
                try:
                    mask = cross_modal_features._load_mask(sample, perm)
                    voxel_count = int(mask.sum())
                    if voxel_count < min_mask_voxels:
                        raise ValueError(f"mask_too_small:{voxel_count}")
                    if mask.shape != case_image.shape[1:]:
                        raise ValueError(f"mask_shape_mismatch:{mask.shape}!={case_image.shape[1:]}")

                    bbox = cross_modal_features._get_bounding_box(mask, margin=bbox_margin)
                    if bbox is None:
                        raise ValueError("empty_mask")

                    pending_tensors.append(
                        cross_modal_features._resize_crop(
                            case_image,
                            bbox,
                            int(crop_size),
                            mask=mask,
                            outside_organ_mask=outside_organ_mask,
                        )
                    )
                    pending_ids.append(sample.sample_id)
                    if len(pending_tensors) >= batch_size:
                        cross_modal_features._flush_batch(
                            extractor,
                            pending_tensors,
                            pending_ids,
                            feature_type,
                            feature_rows,
                            feature_ids,
                        )
                except Exception as exc:
                    failures.append({"sample_id": sample.sample_id, "reason": str(exc)})
                finally:
                    progress.update(1)

            progress.set_postfix(extracted=len(feature_ids), failed=len(failures))

        cross_modal_features._flush_batch(
            extractor,
            pending_tensors,
            pending_ids,
            feature_type,
            feature_rows,
            feature_ids,
        )
        progress.close()

    cross_modal_features.require_complete_extraction(samples, feature_ids, failures, summary_path)

    if not feature_rows:
        raise RuntimeError("No features were extracted successfully")

    feature_matrix = np.stack(feature_rows, axis=0).astype(np.float32, copy=False)
    sample_ids = np.asarray(feature_ids, dtype=object)
    np.savez_compressed(output_npz, features=feature_matrix, sample_ids=sample_ids)

    summary = {
        "status": "complete",
        "manifest_path": str(manifest_path),
        "checkpoint": checkpoint_name,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "feature_type": feature_type,
        "device": resolved_device,
        "dtype": str(resolved_dtype).replace("torch.", ""),
        "crop_size": int(crop_size),
        "bbox_margin": bbox_margin,
        "outside_organ_mask": outside_organ_mask,
        "context_ablation_tag": context_ablation_tag,
        "min_mask_voxels": min_mask_voxels,
        "n_requested_samples": len(samples),
        "n_extracted_samples": len(feature_ids),
        "n_failed_samples": len(failures),
        "crop_cache_dir": str(resolved_crop_cache_dir) if resolved_crop_cache_dir else None,
        "output_npz": str(output_npz),
        "failure_examples": failures[:25],
        "feature_shape": list(feature_matrix.shape),
    }
    if result_metadata:
        summary.update(result_metadata)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %d organ %s embeddings to %s", len(feature_ids), feature_type, output_npz)
    return output_npz
