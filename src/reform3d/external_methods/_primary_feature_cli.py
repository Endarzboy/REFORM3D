"""Shared command-line adapters for single-feature external encoders."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from reform3d.external_methods._primary_feature import PrimaryFeatureMethod
from reform3d.external_methods._acquisition import (
    resolve_acquisition_manifest_path,
    run_controlled_perturbation_analysis,
)
from reform3d.external_methods._anatomical import (
    resolve_anatomical_manifest_path,
    run_anatomical_evaluation_cli,
)
from reform3d.external_methods._cross_modal import (
    extract_cross_modal_features,
    resolve_cross_modal_manifest_path,
    run_cross_modal_evaluation_cli,
)
from reform3d.modules.cross_modal_alignment.feature_extractor import (
    DEFAULT_BBOX_MARGIN,
    OUTSIDE_ORGAN_MASK_MODES,
)

def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _checkpoint_sha256(
    method: PrimaryFeatureMethod,
    checkpoint_path: Path,
) -> str | None:
    if checkpoint_path.expanduser().resolve() == method.checkpoint_path.expanduser().resolve():
        return method.checkpoint_sha256
    return None


def run_controlled_cli(method: PrimaryFeatureMethod) -> None:
    parser = argparse.ArgumentParser(
        description=f"Run {method.display_name} controlled Acquisition Robustness evaluation"
    )
    parser.add_argument("--dataset", type=str, default="jhu_stroke")
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--manifest-variant", default="original_bins")
    parser.add_argument("--checkpoint-path", type=Path, default=method.checkpoint_path)
    parser.add_argument("--checkpoint-name", type=str, default=method.checkpoint_name)
    parser.add_argument("--feature-type", default=method.feature_type, choices=[method.feature_type])
    parser.add_argument("--analysis-name", type=str, default="default")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--n-source-volumes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=method.default_batch_size)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast-seeded-evaluation", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    _configure_logging(args.log_level)
    run_controlled_perturbation_analysis(
        manifest_path=resolve_acquisition_manifest_path(
            args.dataset,
            args.manifest_path,
            args.manifest_variant,
        ),
        checkpoint_path=args.checkpoint_path,
        checkpoint_name=args.checkpoint_name,
        feature_type=args.feature_type,
        extractor_factory=method.extractor_factory,
        model_name=method.model_name,
        crop_size=method.crop_size,
        analysis_name=args.analysis_name,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        cache_dir=args.cache_dir,
        n_source_volumes=args.n_source_volumes,
        feature_batch_size=args.batch_size,
        device=args.device,
        dtype=method.resolve_dtype(args.dtype),
        result_metadata={
            "feature_surface": method.feature_surface,
            "input_preprocessing": method.input_preprocessing,
            "source_commit": method.source_commit,
            "checkpoint_sha256": _checkpoint_sha256(
                method,
                args.checkpoint_path,
            ),
        },
        random_state=args.seed,
        compute_matched_semantics=True,
        compute_matched_semantic_probing=not args.fast_seeded_evaluation,
        compute_matched_semantic_transfer=True,
        compute_drift_figure=not args.fast_seeded_evaluation,
    )


def run_cross_modal_extract_cli(method: PrimaryFeatureMethod) -> None:
    parser = argparse.ArgumentParser(
        description=f"Extract {method.display_name} organ embeddings for Cross-Modal Alignment"
    )
    parser.add_argument("--dataset", type=str, default="amos22_ct_mr")
    parser.add_argument("-m", "--manifest", type=Path, default=None)
    parser.add_argument("--manifest-variant", default="core")
    parser.add_argument("--checkpoint-path", type=Path, default=method.checkpoint_path)
    parser.add_argument("--checkpoint-name", type=str, default=method.checkpoint_name)
    parser.add_argument("--feature-type", default=method.feature_type, choices=[method.feature_type])
    parser.add_argument("--crop-size", type=int, default=method.crop_size)
    parser.add_argument("-o", "--output-npz", default=None)
    parser.add_argument("--batch-size", type=int, default=method.default_batch_size)
    parser.add_argument("--bbox-margin", type=int, default=DEFAULT_BBOX_MARGIN)
    parser.add_argument(
        "--outside-organ-mask",
        default="none",
        choices=list(OUTSIDE_ORGAN_MASK_MODES),
    )
    parser.add_argument("--min-mask-voxels", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--crop-cache-dir", type=Path, default=None)
    parser.add_argument("--crop-cache-workers", type=int, default=4)
    parser.add_argument("--overwrite-crop-cache", action="store_true")
    parser.add_argument("--disable-crop-cache", action="store_true")
    parser.add_argument("--build-crop-cache-only", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    _configure_logging(args.log_level)
    if int(args.crop_size) != method.crop_size:
        raise ValueError(
            f"{method.display_name} expects crop size {method.crop_size}, got {args.crop_size}"
        )
    extract_cross_modal_features(
        manifest_path=resolve_cross_modal_manifest_path(
            args.dataset,
            args.manifest,
            args.manifest_variant,
        ),
        checkpoint_path=args.checkpoint_path,
        checkpoint_name=args.checkpoint_name,
        feature_type=args.feature_type,
        extractor_factory=method.extractor_factory,
        crop_size=method.crop_size,
        output_npz=Path(args.output_npz) if args.output_npz else None,
        batch_size=args.batch_size,
        bbox_margin=args.bbox_margin,
        outside_organ_mask=args.outside_organ_mask,
        min_mask_voxels=args.min_mask_voxels,
        max_samples=args.max_samples,
        crop_cache_dir=args.crop_cache_dir,
        crop_cache_workers=args.crop_cache_workers,
        overwrite_crop_cache=args.overwrite_crop_cache,
        disable_crop_cache=args.disable_crop_cache,
        build_crop_cache_only=args.build_crop_cache_only,
        device=args.device,
        dtype=method.resolve_dtype(args.dtype),
        result_metadata={
            "feature_surface": method.feature_surface,
            "input_preprocessing": method.input_preprocessing,
            "source_commit": method.source_commit,
            "checkpoint_sha256": _checkpoint_sha256(
                method,
                args.checkpoint_path,
            ),
        },
    )


def run_cross_modal_evaluate_cli(method: PrimaryFeatureMethod) -> None:
    parser = argparse.ArgumentParser(
        description=f"Run Cross-Modal Alignment for {method.display_name} embeddings"
    )
    parser.add_argument("--dataset", default="amos22_ct_mr")
    parser.add_argument("-m", "--manifest", type=Path, default=None)
    parser.add_argument("-a", "--analysis-name", default=None)
    parser.add_argument("--manifest-variant", default="core")
    parser.add_argument("--embeddings-npz", type=Path, default=None)
    parser.add_argument("--checkpoint-name", default=method.checkpoint_name)
    parser.add_argument("--feature-type", default=method.feature_type, choices=[method.feature_type])
    parser.add_argument("--required-modalities", nargs="+", default=["ct", "mr"])
    parser.add_argument("--min-samples-per-modality", type=int, default=5)
    parser.add_argument("--max-samples-per-pool", type=int, default=None)
    parser.add_argument("--max-queries-per-organ", type=int, default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_cross_modal_evaluation_cli(
        manifest_path=resolve_cross_modal_manifest_path(
            args.dataset,
            args.manifest,
            args.manifest_variant,
        ),
        checkpoint_name=args.checkpoint_name,
        feature_type=args.feature_type,
        embeddings_npz=args.embeddings_npz,
        analysis_name=args.analysis_name,
        manifest_variant=args.manifest_variant,
        required_modalities=tuple(args.required_modalities),
        min_samples_per_modality=args.min_samples_per_modality,
        max_samples_per_pool=args.max_samples_per_pool,
        max_queries_per_organ=args.max_queries_per_organ,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )


def run_anatomical_cli(method: PrimaryFeatureMethod) -> None:
    parser = argparse.ArgumentParser(
        description=f"Run Anatomical Generalization for {method.display_name} embeddings"
    )
    parser.add_argument("--dataset", default="amos22_ct_mr")
    parser.add_argument("-m", "--manifest", type=Path, default=None)
    parser.add_argument("-a", "--analysis-name", default=None)
    parser.add_argument("--manifest-variant", default="core")
    parser.add_argument("--embeddings-npz", type=Path, default=None)
    parser.add_argument("--checkpoint-name", default=method.checkpoint_name)
    parser.add_argument("--feature-type", default=method.feature_type, choices=[method.feature_type])
    parser.add_argument("--required-modalities", nargs="+", default=None)
    parser.add_argument("--min-holdout-organs", type=int, default=None)
    parser.add_argument("--min-samples-per-modality", type=int, default=None)
    parser.add_argument("--few-shot-support-per-modality", type=int, default=None)
    parser.add_argument("--few-shot-query-per-modality", type=int, default=None)
    parser.add_argument("--few-shot-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args()

    run_anatomical_evaluation_cli(
        manifest_path=resolve_anatomical_manifest_path(
            args.dataset,
            args.manifest,
            args.manifest_variant,
        ),
        checkpoint_name=args.checkpoint_name,
        feature_type=args.feature_type,
        embeddings_npz=args.embeddings_npz,
        analysis_name=args.analysis_name,
        manifest_variant=args.manifest_variant,
        required_modalities=args.required_modalities,
        min_holdout_organs=args.min_holdout_organs,
        min_samples_per_modality=args.min_samples_per_modality,
        few_shot_support_per_modality=args.few_shot_support_per_modality,
        few_shot_query_per_modality=args.few_shot_query_per_modality,
        few_shot_seeds=args.few_shot_seeds,
        seed=args.seed,
        output_path=args.output_path,
    )
