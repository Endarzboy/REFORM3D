#!/usr/bin/env python

"""Run Anatomical Generalization over finished upstream organ feature bundles.

This script is the Anatomical Generalization evaluator. It supports:

    - single-bundle mode: evaluate one explicit Cross-Modal NPZ bundle
    - batch mode: discover all available bundles for a dataset surface and write
      aggregate summaries

Required parameters:
    Single-bundle mode:
        --manifest or --analysis-name
        --checkpoint-name
        --embeddings-npz

    Batch mode:
        --manifest or --analysis-name

Examples:
    python -m reform3d.modules.anatomical_generalization.evaluation_pipeline \
        --manifest /path/to/anatomical_generalization/totalsegmenter_ct_mr_anchor/core/manifest_sampled.json \
        --analysis-name totalsegmenter_ct_mr_anchor \
        --checkpoint-name med3dino_rel_96 \
        --feature-type cls \
        --embeddings-npz /path/to/organ_cls_embeddings.npz

    python -m reform3d.modules.anatomical_generalization.evaluation_pipeline \
        --analysis-name totalsegmenter_ct_mr_anchor \
        --manifest-variant core \
        --feature-types cls avg_pool
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from .config import (
    DEFAULT_CHECKPOINTS,
    DEFAULT_CHECKPOINT_POLICY,
    DEFAULT_FEW_SHOT_QUERY_PER_MODALITY,
    DEFAULT_FEW_SHOT_SEEDS,
    DEFAULT_FEW_SHOT_SUPPORT_PER_MODALITY,
    DEFAULT_MIN_HOLDOUT_ORGANS,
    DEFAULT_MIN_SAMPLES_PER_MODALITY,
    DEFAULT_REQUIRED_MODALITIES,
    FEATURE_TYPES,
    RESULT_JSON_NAME,
    SUMMARY_CSV_NAME,
    SUMMARY_JSON_NAME,
    ensure_output_directories,
    get_checkpoint_metrics_path,
    get_dataset_name_from_manifest_path,
    get_manifest_variant_from_manifest_path,
    get_output_paths,
    get_cross_modal_feature_npz_candidates,
    get_cross_modal_extension_feature_npz_path,
    get_anatomical_manifest_path,
    get_summary_csv_path,
    get_summary_json_path,
    normalize_anatomical_dataset_name,
    normalize_feature_type,
    normalize_manifest_variant,
)
from .few_shot_transfer_analysis import (
    evaluate_low_shot_target_transfer,
    evaluate_support_size_sweep,
    support_size_sweep_csv_rows,
)
from .feature_projection import (
    ContextFeaturePair,
    FeatureProjectionSettings,
    load_controlled_spacing_bundle,
    load_feature_bundle,
    normalize_projection_sources,
    projection_source_key,
)
from .data_loader import (
    get_eligible_holdout_organs,
    load_anatomical_manifest,
    summarize_anatomical_manifest,
    validate_anatomical_manifest,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MANIFEST_SUMMARY_NAME = "manifest_summary.json"


def _build_surface_b_metric_aliases(analysis: Dict[str, Any]) -> Dict[str, Any]:
    macro = analysis.get("macro_mean", {})
    aliases = {
        "balanced_accuracy": macro.get("balanced_accuracy"),
        "nearest_centroid_baseline_balanced_accuracy": macro.get("nearest_centroid_balanced_accuracy"),
        "adaptation_gain_over_nearest_centroid": macro.get("adaptation_gain_balanced_accuracy"),
        "relative_recovery_over_nearest_centroid": macro.get("relative_recovery_balanced_accuracy"),
    }
    if macro.get("projected_balanced_accuracy") is not None:
        aliases.update(
            {
                "projected_balanced_accuracy": macro.get("projected_balanced_accuracy"),
                "projected_centroid_balanced_accuracy": macro.get("projected_centroid_balanced_accuracy"),
                "projected_adaptation_gain_over_projected_centroid": macro.get(
                    "projected_adaptation_gain_balanced_accuracy"
                ),
                "projected_relative_recovery_over_projected_centroid": macro.get(
                    "projected_relative_recovery_balanced_accuracy"
                ),
            }
        )
    return aliases


def _build_surface_b_diagnostics(analysis: Dict[str, Any]) -> Dict[str, Any]:
    macro = analysis.get("macro_mean", {})
    macro_std = analysis.get("macro_mean_std", {})
    diagnostics = {
        "target_query_accuracy": macro.get("target_query_accuracy"),
        "reference_organs_accuracy": macro.get("reference_organs_accuracy"),
        "target_vs_reference_gap": macro.get("target_vs_reference_gap"),
        "transfer_efficiency": macro.get("transfer_efficiency"),
        "nearest_centroid_baseline_balanced_accuracy": macro.get("nearest_centroid_balanced_accuracy"),
        "nearest_centroid_baseline_target_query_accuracy": macro.get("nearest_centroid_target_query_accuracy"),
        "adaptation_gain_balanced_accuracy": macro.get("adaptation_gain_balanced_accuracy"),
        "adaptation_gain_target_query_accuracy": macro.get("adaptation_gain_target_query_accuracy"),
        "relative_recovery_balanced_accuracy": macro.get("relative_recovery_balanced_accuracy"),
        "seed_std_balanced_accuracy": macro_std.get("balanced_accuracy"),
        "seed_std_target_query_accuracy": macro_std.get("target_query_accuracy"),
        "seed_std_nearest_centroid_baseline_balanced_accuracy": macro_std.get(
            "nearest_centroid_balanced_accuracy"
        ),
        "seed_std_adaptation_gain_balanced_accuracy": macro_std.get("adaptation_gain_balanced_accuracy"),
        "seed_std_relative_recovery_balanced_accuracy": macro_std.get("relative_recovery_balanced_accuracy"),
    }
    if macro.get("projected_balanced_accuracy") is not None:
        diagnostics.update(
            {
                "projected_target_query_accuracy": macro.get("projected_target_query_accuracy"),
                "projected_reference_organs_accuracy": macro.get("projected_reference_organs_accuracy"),
                "projected_target_vs_reference_gap": macro.get("projected_target_vs_reference_gap"),
                "projected_transfer_efficiency": macro.get("projected_transfer_efficiency"),
                "projected_centroid_balanced_accuracy": macro.get("projected_centroid_balanced_accuracy"),
                "projected_centroid_target_query_accuracy": macro.get("projected_centroid_target_query_accuracy"),
                "projected_adaptation_gain_balanced_accuracy": macro.get(
                    "projected_adaptation_gain_balanced_accuracy"
                ),
                "projected_adaptation_gain_target_query_accuracy": macro.get(
                    "projected_adaptation_gain_target_query_accuracy"
                ),
                "projected_relative_recovery_balanced_accuracy": macro.get(
                    "projected_relative_recovery_balanced_accuracy"
                ),
                "seed_std_projected_balanced_accuracy": macro_std.get("projected_balanced_accuracy"),
                "seed_std_projected_adaptation_gain_balanced_accuracy": macro_std.get(
                    "projected_adaptation_gain_balanced_accuracy"
                ),
            }
        )
    return diagnostics


def _build_headline_metrics(surface_b_analysis: Dict[str, Any]) -> Dict[str, Any]:
    surface_b_macro = surface_b_analysis.get("macro_mean", {})
    headline = {
        "task_1_low_shot_target_adaptation_gain": {
            "primary_metric_name": "adaptation_gain_balanced_accuracy_over_nearest_centroid",
            "primary_metric_value": surface_b_macro.get("adaptation_gain_balanced_accuracy"),
            "baseline_metric_name": "nearest_centroid_balanced_accuracy",
            "baseline_metric_value": surface_b_macro.get("nearest_centroid_balanced_accuracy"),
            "adapted_metric_name": "balanced_accuracy",
            "adapted_metric_value": surface_b_macro.get("balanced_accuracy"),
            "relative_recovery": surface_b_macro.get("relative_recovery_balanced_accuracy"),
            "n_supported_organs": surface_b_analysis.get("n_supported_organs"),
            "n_evaluated_organs": surface_b_analysis.get("n_evaluated_organs"),
            "support_per_modality": surface_b_analysis.get("support_per_modality"),
        },
        "task_2_low_shot_transfer": {
            "primary_metric_name": "balanced_accuracy",
            "primary_metric_value": surface_b_macro.get("balanced_accuracy"),
            "n_supported_organs": surface_b_analysis.get("n_supported_organs"),
            "n_evaluated_organs": surface_b_analysis.get("n_evaluated_organs"),
            "support_per_modality": surface_b_analysis.get("support_per_modality"),
        },
    }
    if surface_b_macro.get("projected_balanced_accuracy") is not None:
        headline["feature_projection_low_shot_transfer"] = {
            "primary_metric_name": "projected_balanced_accuracy",
            "primary_metric_value": surface_b_macro.get("projected_balanced_accuracy"),
            "centroid_metric_name": "projected_centroid_balanced_accuracy",
            "centroid_metric_value": surface_b_macro.get("projected_centroid_balanced_accuracy"),
            "adaptation_gain_metric_name": "projected_adaptation_gain_balanced_accuracy",
            "adaptation_gain_metric_value": surface_b_macro.get(
                "projected_adaptation_gain_balanced_accuracy"
            ),
            "n_supported_organs": surface_b_analysis.get("n_supported_organs"),
            "n_evaluated_organs": surface_b_analysis.get("n_evaluated_organs"),
            "support_per_modality": surface_b_analysis.get("support_per_modality"),
        }
    return headline


def _build_anatomical_protocol(
    summary: Dict[str, Any],
    required_modalities: Sequence[str],
    supported_organs: Sequence[str],
) -> Dict[str, Any]:
    required_modalities = [str(modality).lower() for modality in required_modalities]
    surface_scope = "within_modality" if len(required_modalities) == 1 else "cross_modality"
    exposures = set((summary.get("pretraining_exposure") or {}).keys())
    if surface_scope == "within_modality":
        surface_stratum = f"{required_modalities[0]}_within_dataset_holdout_surface"
    elif exposures == {"pretraining_unseen_source"}:
        surface_stratum = "pretraining_unseen_ct_mr_validation"
    elif "pretraining_seen_source" in exposures and "pretraining_unseen_source" in exposures:
        surface_stratum = "mixed_exposure_ct_mr_anchor"
    elif exposures == {"pretraining_seen_source"}:
        surface_stratum = "pretraining_seen_ct_mr_surface"
    else:
        surface_stratum = "mixed_or_unknown_exposure_surface"
    return {
        "target_definition": "low_shot_target_organ",
        "target_definition_note": (
            "The target organ receives the same fixed support budget as every other organ; evaluation is performed "
            "on query examples from source cases disjoint from all support cases."
        ),
        "fold_policy": "one_target_organ_per_low_shot_fold",
        "support_query_split_policy": "case_disjoint_support_query",
        "within_modality_policy": "within_dataset_case_disjoint_low_shot_target_transfer",
        "surface_scope": surface_scope,
        "cross_modality_surface_stratum": surface_stratum if surface_scope == "cross_modality" else None,
        "within_modality_surface_stratum": surface_stratum if surface_scope == "within_modality" else None,
        "checkpoint_policy": DEFAULT_CHECKPOINT_POLICY,
        "required_modalities": list(required_modalities),
        "supported_organs": list(sorted(supported_organs)),
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a JSON artifact with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Write a flat summary table for batch Anatomical results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_batch_summary(
    summary_output: Path,
    csv_output: Path,
    analysis_name: str,
    manifest_path: Path,
    manifest_variant: str,
    checkpoints: Sequence[str],
    feature_types: Sequence[str],
    rows: Sequence[Dict[str, Any]],
    missing_entries: Sequence[Dict[str, Any]],
) -> None:
    """Write the current batch summary payload and CSV snapshot."""
    summary_payload = {
        "analysis_name": analysis_name,
        "manifest_path": str(manifest_path),
        "manifest_variant": manifest_variant,
        "requested_checkpoints": list(checkpoints),
        "requested_feature_types": list(feature_types),
        "completed_rows": list(rows),
        "missing_entries": list(missing_entries),
    }
    _write_json(summary_output, summary_payload)
    _write_csv(csv_output, rows)


def _load_embedding_bundle(npz_path: Path, manifest_samples: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    from reform3d.core.features import load_organ_feature_bundle
    return load_organ_feature_bundle(npz_path, [sample["sample_id"] for sample in manifest_samples], dtype=np.float32)

def _feature_adapter_output_path(
    baseline_output_path: Path,
    source_key: str,
) -> Path:
    return baseline_output_path.parent / "feature_adapter" / "projection" / source_key / RESULT_JSON_NAME


def _resolve_spacing_bundle_path(
    dataset_name: str,
    manifest_variant: str,
    checkpoint_name: str,
    feature_type: str,
) -> Path:
    from reform3d.modules.acquisition_robustness.config import (
        get_checkpoint_feature_dir as get_acquisition_checkpoint_feature_dir,
        get_output_paths as get_acquisition_output_paths,
        normalize_manifest_variant as normalize_acquisition_manifest_variant,
    )

    acquisition_paths = get_acquisition_output_paths(
        dataset_name,
        normalize_acquisition_manifest_variant(manifest_variant),
    )
    return (
        get_acquisition_checkpoint_feature_dir(acquisition_paths["results"], checkpoint_name, feature_type)
        / "controlled_embedding_bundle.npz"
    )


def _build_feature_projection_settings(
    adapter_name: str,
    projection_sources: Sequence[str] | None,
    analysis_name: str,
    manifest_variant: str,
    checkpoint_name: str,
    feature_type: str,
    samples: Sequence[Dict[str, Any]],
    context_source_surface: str,
    context_reference_surface: str,
    spacing_datasets: Sequence[str] | None,
    spacing_manifest_variant: str,
    spacing_reference_variant: str | None,
    spacing_target_variants: Sequence[str] | None,
    components: int,
    strength: float,
) -> FeatureProjectionSettings | None:
    adapter_name = str(adapter_name).strip().lower()
    if adapter_name == "none":
        return None
    if adapter_name != "projection":
        raise ValueError(f"Unknown feature adapter: {adapter_name}")

    sources = normalize_projection_sources(projection_sources)
    if not sources:
        raise ValueError("--projection-sources is required when --feature-adapter projection is used")
    sample_ids = [str(sample["sample_id"]) for sample in samples]

    context_pair = None
    if "context" in sources:
        source_path = get_cross_modal_extension_feature_npz_path(
            analysis_name,
            manifest_variant,
            checkpoint_name,
            feature_type,
            context_source_surface,
        )
        reference_path = get_cross_modal_extension_feature_npz_path(
            analysis_name,
            manifest_variant,
            checkpoint_name,
            feature_type,
            context_reference_surface,
        )
        if not source_path.exists():
            raise FileNotFoundError(f"Missing context source feature bundle: {source_path}")
        if not reference_path.exists():
            raise FileNotFoundError(f"Missing context reference feature bundle: {reference_path}")
        context_pair = ContextFeaturePair(
            source_surface=context_source_surface,
            reference_surface=context_reference_surface,
            source_path=source_path,
            reference_path=reference_path,
            source_by_id=load_feature_bundle(source_path, sample_ids),
            reference_by_id=load_feature_bundle(reference_path, sample_ids),
        )

    spacing_bundles = []
    if "spacing" in sources:
        if not spacing_datasets:
            raise ValueError("--projection-spacing-datasets is required when spacing projection is requested")
        for spacing_dataset in spacing_datasets:
            bundle_path = _resolve_spacing_bundle_path(
                spacing_dataset,
                spacing_manifest_variant,
                checkpoint_name,
                feature_type,
            )
            if not bundle_path.exists():
                raise FileNotFoundError(f"Missing controlled spacing embedding bundle: {bundle_path}")
            spacing_bundles.append(
                load_controlled_spacing_bundle(
                    bundle_path=bundle_path,
                    dataset=spacing_dataset,
                    manifest_variant=spacing_manifest_variant,
                    reference_variant=spacing_reference_variant,
                    target_variants=spacing_target_variants,
                )
            )

    return FeatureProjectionSettings(
        sources=sources,
        components=int(components),
        strength=float(strength),
        context_pair=context_pair,
        spacing_bundles=tuple(spacing_bundles),
        metadata={
            "fit_scope": {
                "context": "support_samples" if "context" in sources else None,
                "modality": "support_samples" if "modality" in sources else None,
                "spacing": "controlled_embedding_bundles" if "spacing" in sources else None,
            },
            "analysis_name": analysis_name,
            "manifest_variant": manifest_variant,
            "checkpoint_name": checkpoint_name,
            "feature_type": feature_type,
        },
    )


def _checkpoint_sort_key(checkpoint_name: str) -> Tuple[int, str]:
    """Sort known checkpoints according to the Anatomical evaluation order."""
    known_order = {name: idx for idx, name in enumerate(DEFAULT_CHECKPOINTS)}
    return (known_order.get(checkpoint_name, len(known_order)), checkpoint_name)


def _ordered_checkpoint_names(checkpoint_names: Iterable[str]) -> List[str]:
    """Deduplicate and order checkpoint names for batch evaluation."""
    unique_names = list(dict.fromkeys(checkpoint_names))
    return sorted(unique_names, key=_checkpoint_sort_key)


def _metric_value(payload: Dict[str, Any], *keys: str) -> Any:
    """Safely read a nested metric from an Anatomical result payload."""
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _write_manifest_summary(
    output_root: Path,
    analysis_name: str,
    manifest_path: Path,
    validation: Dict[str, Any],
    summary: Dict[str, Any],
) -> None:
    """Persist manifest validation and summary metadata for one run root."""
    _write_json(
        output_root / MANIFEST_SUMMARY_NAME,
        {
            "analysis_name": analysis_name,
            "manifest_path": str(manifest_path),
            "validation": validation,
            "summary": summary,
        },
    )


def _evaluate_single_bundle(
    manifest_path: Path,
    analysis_name: str,
    manifest_variant: str,
    checkpoint_name: str,
    feature_type: str,
    embeddings_npz: Path,
    required_modalities: Sequence[str],
    min_holdout_organs: int,
    min_samples_per_modality: int,
    few_shot_support_per_modality: int,
    few_shot_query_per_modality: int | None,
    few_shot_seeds: Sequence[int],
    seed: int,
    output_path: Optional[Path],
    manifest_summary_dir: Optional[Path] = None,
    support_size_sweep: Sequence[int] | None = None,
    feature_adapter: str = "none",
    projection_sources: Sequence[str] | None = None,
    projection_context_source_surface: str = "baseline_full_bbox_context",
    projection_context_reference_surface: str = "organ_only_bbox0",
    projection_spacing_datasets: Sequence[str] | None = None,
    projection_spacing_manifest_variant: str = "original_bins",
    projection_spacing_reference_variant: str | None = None,
    projection_spacing_target_variants: Sequence[str] | None = None,
    projection_components: int = 4,
    projection_strength: float = 1.0,
) -> Dict[str, Any]:
    """Evaluate one checkpoint/feature bundle against one Anatomical manifest.

    Required parameters:
        manifest_path: Anatomical manifest JSON path.
        analysis_name: dataset namespace for outputs.
        manifest_variant: manifest variant label, usually ``core``.
        checkpoint_name: checkpoint label.
        feature_type: one of ``cls``, ``avg_pool``, or ``multilayer``.
        embeddings_npz: explicit Cross-Modal embedding bundle.
    """
    progress = tqdm(total=6, desc=f"Anatomical {checkpoint_name}/{feature_type}", unit="stage", dynamic_ncols=True, leave=False)
    ensure_output_directories(analysis_name, manifest_variant)
    output_paths = get_output_paths(analysis_name, manifest_variant)
    progress.set_postfix_str("resolved paths")
    progress.update(1)

    samples = load_anatomical_manifest(manifest_path)
    progress.set_postfix_str("loaded manifest")
    progress.update(1)
    validation = validate_anatomical_manifest(
        samples,
        required_modalities=required_modalities,
        min_holdout_organs=min_holdout_organs,
        min_samples_per_modality=min_samples_per_modality,
    )
    if not validation["is_valid"]:
        raise ValueError(f"Anatomical manifest validation failed for {manifest_path}: {validation}")
    supported_organs = get_eligible_holdout_organs(
        samples,
        required_modalities=required_modalities,
        min_samples_per_modality=min_samples_per_modality,
    )
    progress.set_postfix_str("validated manifest")
    progress.update(1)
    features_by_id = _load_embedding_bundle(embeddings_npz, samples)
    progress.set_postfix_str("loaded embeddings")
    progress.update(1)
    feature_projection = _build_feature_projection_settings(
        adapter_name=feature_adapter,
        projection_sources=projection_sources,
        analysis_name=analysis_name,
        manifest_variant=manifest_variant,
        checkpoint_name=checkpoint_name,
        feature_type=feature_type,
        samples=samples,
        context_source_surface=projection_context_source_surface,
        context_reference_surface=projection_context_reference_surface,
        spacing_datasets=projection_spacing_datasets,
        spacing_manifest_variant=projection_spacing_manifest_variant,
        spacing_reference_variant=projection_spacing_reference_variant,
        spacing_target_variants=projection_spacing_target_variants,
        components=projection_components,
        strength=projection_strength,
    )
    few_shot_analysis = evaluate_low_shot_target_transfer(
        features_by_id,
        samples,
        supported_organs=supported_organs,
        required_modalities=required_modalities,
        support_per_modality=few_shot_support_per_modality,
        query_per_modality=few_shot_query_per_modality,
        seeds=few_shot_seeds,
        feature_projection=feature_projection,
    )
    progress.set_postfix_str("surface B")
    progress.update(1)
    baseline_output_path = output_path or get_checkpoint_metrics_path(output_paths["results"], checkpoint_name, feature_type)
    resolved_output_path = (
        _feature_adapter_output_path(baseline_output_path, feature_projection.source_key())
        if feature_projection is not None and feature_projection.enabled
        else baseline_output_path
    )
    support_sweep_payload = None
    if support_size_sweep:
        support_sweep_payload = evaluate_support_size_sweep(
            features_by_id,
            samples,
            supported_organs=supported_organs,
            required_modalities=required_modalities,
            support_sizes=support_size_sweep,
            query_per_modality=few_shot_query_per_modality,
            seeds=few_shot_seeds,
            feature_projection=feature_projection,
        )
        _write_json(resolved_output_path.parent / "support_size_sweep.json", support_sweep_payload)
        _write_csv(resolved_output_path.parent / "support_size_sweep.csv", support_size_sweep_csv_rows(support_sweep_payload))
    manifest_summary = summarize_anatomical_manifest(samples)
    anatomical_protocol = _build_anatomical_protocol(manifest_summary, required_modalities, supported_organs)
    surface_b_payload = {
        "status": few_shot_analysis.get("status"),
        "surface_name": "surface_b_few_shot_transfer_evidence",
        "surface_scope": anatomical_protocol.get("surface_scope"),
        "anatomical_protocol": anatomical_protocol,
        "metric_aliases": _build_surface_b_metric_aliases(few_shot_analysis),
        "diagnostics": _build_surface_b_diagnostics(few_shot_analysis),
        "few_shot_transfer_analysis": few_shot_analysis,
    }
    headline_metrics = _build_headline_metrics(few_shot_analysis)

    payload = {
        "analysis_name": analysis_name,
        "checkpoint_name": checkpoint_name,
        "feature_type": feature_type,
        "manifest_path": str(manifest_path),
        "embeddings_npz": str(embeddings_npz),
        "required_modalities": list(required_modalities),
        "feature_adapter": feature_projection.to_payload()
        if feature_projection is not None and feature_projection.enabled
        else {"adapter": "none"},
        "anatomical_protocol": anatomical_protocol,
        "headline_metrics": headline_metrics,
        "metric_contract": {
            "headline_metrics": [
                "headline_metrics.task_1_low_shot_target_adaptation_gain.primary_metric_value",
                "headline_metrics.task_1_low_shot_target_adaptation_gain.baseline_metric_value",
                "headline_metrics.task_1_low_shot_target_adaptation_gain.adapted_metric_value",
                "headline_metrics.task_2_low_shot_transfer.primary_metric_value",
                "headline_metrics.task_2_low_shot_transfer.n_evaluated_organs",
            ],
            "diagnostic_metrics": [
                "surface_b_few_shot_transfer_evidence.diagnostics",
                "surface_b_few_shot_transfer_evidence.few_shot_transfer_analysis",
            ],
        },
        "manifest_validation": validation,
        "manifest_summary": manifest_summary,
        "support_size_sweep": support_sweep_payload,
        "surface_b_few_shot_transfer_evidence": surface_b_payload,
    }

    _write_manifest_summary(
        manifest_summary_dir or output_paths["results"],
        analysis_name,
        manifest_path,
        validation,
        payload["manifest_summary"],
    )
    _write_json(resolved_output_path, payload)
    if feature_projection is not None and feature_projection.enabled:
        _write_json(
            resolved_output_path.parent / "projection_diagnostics.json",
            {
                "analysis_name": analysis_name,
                "checkpoint_name": checkpoint_name,
                "feature_type": feature_type,
                "result_path": str(resolved_output_path),
                "feature_adapter": feature_projection.to_payload(),
                "few_shot_projection": few_shot_analysis.get("feature_projection"),
                "macro_mean": few_shot_analysis.get("macro_mean"),
                "macro_mean_std": few_shot_analysis.get("macro_mean_std"),
            },
        )
    progress.set_postfix_str("wrote metrics")
    progress.update(1)
    progress.close()
    return payload


def _discover_bundle_map(
    analysis_name: str,
    manifest_variant: str,
    checkpoints: Sequence[str],
    feature_types: Sequence[str],
) -> Dict[Tuple[str, str], Path]:
    """Discover which requested upstream feature bundles already exist on disk.

    Batch discovery reuses the organ-embedding layout owned by
    Cross-Modal Alignment.
    """
    bundle_map: Dict[Tuple[str, str], Path] = {}
    for checkpoint_name in checkpoints:
        for feature_type in feature_types:
            for candidate in get_cross_modal_feature_npz_candidates(
                analysis_name,
                manifest_variant,
                checkpoint_name,
                feature_type,
            ):
                if candidate.exists():
                    bundle_map[(checkpoint_name, feature_type)] = candidate
                    break
    return bundle_map


def _build_summary_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one Anatomical result payload into one batch-summary row."""
    headline_metrics = payload.get("headline_metrics", {})
    task_1_gain = headline_metrics.get("task_1_low_shot_target_adaptation_gain", {})
    task_2 = headline_metrics.get("task_2_low_shot_transfer", {})
    projection = headline_metrics.get("feature_projection_low_shot_transfer", {})
    adapter_payload = payload.get("feature_adapter") or {}
    row = {
        "checkpoint_name": payload.get("checkpoint_name"),
        "feature_type": payload.get("feature_type"),
        "feature_adapter": adapter_payload.get("adapter"),
        "projection_sources": "+".join(adapter_payload.get("sources", [])) if adapter_payload.get("sources") else None,
        "surface_b_status": (payload.get("surface_b_few_shot_transfer_evidence") or {}).get("status"),
        "n_supported_organs": task_1_gain.get("n_supported_organs"),
        "n_evaluated_organs": task_1_gain.get("n_evaluated_organs"),
        "task_1_adaptation_gain_balanced_accuracy": task_1_gain.get("primary_metric_value"),
        "task_1_nearest_centroid_baseline_balanced_accuracy": task_1_gain.get("baseline_metric_value"),
        "task_1_adapted_balanced_accuracy": task_1_gain.get("adapted_metric_value"),
        "task_1_relative_recovery": task_1_gain.get("relative_recovery"),
        "task_2_n_evaluated_organs": task_2.get("n_evaluated_organs"),
        "task_2_balanced_accuracy": task_2.get("primary_metric_value"),
        "task_2_support_per_modality": task_2.get("support_per_modality"),
    }
    if projection:
        row.update(
            {
                "projected_balanced_accuracy": projection.get("primary_metric_value"),
                "projected_centroid_balanced_accuracy": projection.get("centroid_metric_value"),
                "projected_adaptation_gain_balanced_accuracy": projection.get("adaptation_gain_metric_value"),
            }
        )
    return row


def main() -> None:
    """CLI entrypoint for Anatomical Generalization evaluation.

    The command resolves one Anatomical manifest, validates it, and then runs
    either single-bundle or batch evaluation depending on whether
    ``--embeddings-npz`` was provided.
    """
    parser = argparse.ArgumentParser(
        description="Anatomical Generalization over frozen organ embedding bundles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m reform3d.modules.anatomical_generalization.evaluation_pipeline -m /path/to/manifest_sampled.json --analysis-name totalsegmenter_ct_mr_anchor --checkpoint-name med3dino_rel_96 --feature-type cls --embeddings-npz /path/to/organ_cls_embeddings.npz\n"
            "  python -m reform3d.modules.anatomical_generalization.evaluation_pipeline --analysis-name totalsegmenter_ct_mr_anchor --manifest-variant core --feature-types cls avg_pool"
        ),
    )
    parser.add_argument("-m", "--manifest", default=None, help="Path to the Anatomical manifest JSON")
    parser.add_argument("-a", "--analysis-name", default=None, help="Dataset namespace for outputs")
    parser.add_argument("--manifest-variant", default="core", help="Manifest variant label")
    parser.add_argument("--embeddings-npz", default=None, help="Optional NPZ bundle with Cross-Modal features")
    parser.add_argument("--checkpoint-name", default=None, help="Checkpoint label for single-bundle mode")
    parser.add_argument("--feature-type", default="cls", help="Feature family label for single-bundle mode")
    parser.add_argument(
        "--feature-types",
        nargs="+",
        default=None,
        help="Feature families for batch mode. Defaults to all registered Cross-Modal feature families.",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        help="Optional checkpoint subset for batch mode",
    )
    parser.add_argument(
        "--required-modalities",
        nargs="+",
        default=list(DEFAULT_REQUIRED_MODALITIES),
        help="Required modalities for low-shot target-organ support",
    )
    parser.add_argument(
        "--min-holdout-organs",
        type=int,
        default=DEFAULT_MIN_HOLDOUT_ORGANS,
        help="Minimum number of cross-modal organs required by the manifest",
    )
    parser.add_argument(
        "--min-samples-per-modality",
        type=int,
        default=DEFAULT_MIN_SAMPLES_PER_MODALITY,
        help="Minimum unique case support per organ and modality for target eligibility",
    )
    parser.add_argument(
        "--few-shot-support-per-modality",
        type=int,
        default=DEFAULT_FEW_SHOT_SUPPORT_PER_MODALITY,
        help="Balanced Surface B support budget per organ and modality",
    )
    parser.add_argument(
        "--few-shot-query-per-modality",
        type=int,
        default=DEFAULT_FEW_SHOT_QUERY_PER_MODALITY,
        help="Optional cap for the balanced Surface B query budget per organ and modality",
    )
    parser.add_argument(
        "--few-shot-seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_FEW_SHOT_SEEDS),
        help="Random seeds used for Surface B balanced support/query sampling",
    )
    parser.add_argument(
        "--support-size-sweep",
        nargs="*",
        type=int,
        default=None,
        help="Optional support-per-modality sweep. Use without values for 1 2 4 8.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", default=None, help="Explicit output path for single-bundle mode")
    parser.add_argument("--manifest-summary-output-dir", default=None, help="Optional manifest summary directory")
    parser.add_argument("--summary-output", default=None, help=f"Optional batch summary JSON path ({SUMMARY_JSON_NAME})")
    parser.add_argument("--csv-output", default=None, help=f"Optional batch summary CSV path ({SUMMARY_CSV_NAME})")
    parser.add_argument("--require-complete", action="store_true", help="Fail batch mode if any requested bundle is missing")
    parser.add_argument("--feature-adapter", choices=["none", "projection"], default="none")
    parser.add_argument(
        "--projection-sources",
        nargs="+",
        default=None,
        choices=["context", "spacing", "modality"],
        help="Feature-difference sources used to fit the projection basis.",
    )
    parser.add_argument("--projection-components", type=int, default=4)
    parser.add_argument("--projection-strength", type=float, default=1.0)
    parser.add_argument("--projection-context-source-surface", default="baseline_full_bbox_context")
    parser.add_argument("--projection-context-reference-surface", default="organ_only_bbox0")
    parser.add_argument(
        "--projection-spacing-datasets",
        nargs="+",
        default=None,
        help="Acquisition Robustness datasets that provide controlled spacing embedding bundles.",
    )
    parser.add_argument("--projection-spacing-manifest-variant", default="original_bins")
    parser.add_argument("--projection-spacing-reference-variant", default=None)
    parser.add_argument("--projection-spacing-target-variants", nargs="+", default=None)
    args = parser.parse_args()
    support_size_sweep = (
        [1, 2, 4, 8]
        if args.support_size_sweep == []
        else args.support_size_sweep
    )

    manifest_variant = normalize_manifest_variant(args.manifest_variant)
    manifest_path = Path(args.manifest).resolve() if args.manifest else None
    if manifest_path is None and args.analysis_name is None:
        raise ValueError("Either --manifest or --analysis-name is required")

    analysis_name = normalize_anatomical_dataset_name(args.analysis_name or get_dataset_name_from_manifest_path(manifest_path))
    resolved_manifest_path = manifest_path or get_anatomical_manifest_path(analysis_name, "sampled", manifest_variant)
    if not resolved_manifest_path.exists():
        raise FileNotFoundError(f"Anatomical manifest not found: {resolved_manifest_path}")

    ensure_output_directories(analysis_name, manifest_variant)
    output_paths = get_output_paths(analysis_name, manifest_variant)
    required_modalities = [str(modality).strip().lower() for modality in args.required_modalities]

    if args.embeddings_npz:
        if not args.checkpoint_name:
            raise ValueError("--checkpoint-name is required in single-bundle mode")
        feature_type = normalize_feature_type(args.feature_type)
        logger.info(
            "Starting Anatomical single-bundle evaluation: analysis=%s checkpoint=%s feature_type=%s manifest=%s embeddings=%s",
            analysis_name,
            args.checkpoint_name,
            feature_type,
            resolved_manifest_path,
            Path(args.embeddings_npz).resolve(),
        )
        start_time = perf_counter()
        payload = _evaluate_single_bundle(
            manifest_path=resolved_manifest_path,
            analysis_name=analysis_name,
            manifest_variant=manifest_variant,
            checkpoint_name=args.checkpoint_name,
            feature_type=feature_type,
            embeddings_npz=Path(args.embeddings_npz).resolve(),
            required_modalities=required_modalities,
            min_holdout_organs=args.min_holdout_organs,
            min_samples_per_modality=args.min_samples_per_modality,
            few_shot_support_per_modality=args.few_shot_support_per_modality,
            few_shot_query_per_modality=args.few_shot_query_per_modality,
            few_shot_seeds=args.few_shot_seeds,
            seed=args.seed,
            output_path=Path(args.output_path).resolve() if args.output_path else None,
            manifest_summary_dir=Path(args.manifest_summary_output_dir).resolve() if args.manifest_summary_output_dir else None,
            support_size_sweep=support_size_sweep,
            feature_adapter=args.feature_adapter,
            projection_sources=args.projection_sources,
            projection_context_source_surface=args.projection_context_source_surface,
            projection_context_reference_surface=args.projection_context_reference_surface,
            projection_spacing_datasets=args.projection_spacing_datasets,
            projection_spacing_manifest_variant=args.projection_spacing_manifest_variant,
            projection_spacing_reference_variant=args.projection_spacing_reference_variant,
            projection_spacing_target_variants=args.projection_spacing_target_variants,
            projection_components=args.projection_components,
            projection_strength=args.projection_strength,
        )
        logger.info(
            "Completed Anatomical single-bundle evaluation: checkpoint=%s feature_type=%s status=%s elapsed_sec=%.1f",
            args.checkpoint_name,
            feature_type,
            payload["surface_b_few_shot_transfer_evidence"].get("status"),
            perf_counter() - start_time,
        )
        if payload["surface_b_few_shot_transfer_evidence"].get("status") != "ok":
            raise ValueError(f"Anatomical evaluation did not produce few-shot transfer scores for {args.embeddings_npz}")
        return

    checkpoints = _ordered_checkpoint_names(args.checkpoints or DEFAULT_CHECKPOINTS)
    feature_types = [normalize_feature_type(feature_type) for feature_type in (args.feature_types or FEATURE_TYPES)]
    bundle_map = _discover_bundle_map(analysis_name, manifest_variant, checkpoints, feature_types)
    rows: List[Dict[str, Any]] = []
    missing_entries: List[Dict[str, Any]] = []
    default_summary_root = output_paths["results"]
    if args.feature_adapter == "projection":
        default_summary_root = (
            output_paths["results"]
            / "feature_adapter"
            / "projection"
            / projection_source_key(args.projection_sources or [])
        )
    summary_output = Path(args.summary_output).resolve() if args.summary_output else get_summary_json_path(default_summary_root)
    csv_output = Path(args.csv_output).resolve() if args.csv_output else get_summary_csv_path(default_summary_root)
    total_requested = len(checkpoints) * len(feature_types)
    logger.info(
        "Starting Anatomical batch evaluation: analysis=%s variant=%s requested_bundles=%d available_bundles=%d checkpoints=%s feature_types=%s",
        analysis_name,
        manifest_variant,
        total_requested,
        len(bundle_map),
        checkpoints,
        feature_types,
    )
    processed_bundle_count = 0

    checkpoint_iterator = tqdm(checkpoints, desc="Anatomical checkpoints", unit="ckpt", dynamic_ncols=True)
    for checkpoint_index, checkpoint_name in enumerate(checkpoint_iterator, start=1):
        logger.info(
            "Starting checkpoint %s (%d/%d)",
            checkpoint_name,
            checkpoint_index,
            len(checkpoints),
        )
        feature_iterator = tqdm(feature_types, desc=f"Anatomical features {checkpoint_name}", unit="feature", dynamic_ncols=True, leave=False)
        for feature_index, feature_type in enumerate(feature_iterator, start=1):
            processed_bundle_count += 1
            embeddings_npz = bundle_map.get((checkpoint_name, feature_type))
            if embeddings_npz is None:
                missing_entries.append({
                    "checkpoint_name": checkpoint_name,
                    "feature_type": feature_type,
                    "reason": "missing_cross_modal_feature_bundle",
                })
                logger.warning(
                    "Skipping missing Cross-Modal bundle %d/%d: checkpoint=%s feature_type=%s",
                    processed_bundle_count,
                    total_requested,
                    checkpoint_name,
                    feature_type,
                )
                _write_batch_summary(
                    summary_output=summary_output,
                    csv_output=csv_output,
                    analysis_name=analysis_name,
                    manifest_path=resolved_manifest_path,
                    manifest_variant=manifest_variant,
                    checkpoints=checkpoints,
                    feature_types=feature_types,
                    rows=rows,
                    missing_entries=missing_entries,
                )
                continue
            logger.info(
                "Evaluating bundle %d/%d: checkpoint=%s (%d/%d) feature_type=%s (%d/%d)",
                processed_bundle_count,
                total_requested,
                checkpoint_name,
                checkpoint_index,
                len(checkpoints),
                feature_type,
                feature_index,
                len(feature_types),
            )
            bundle_start_time = perf_counter()
            payload = _evaluate_single_bundle(
                manifest_path=resolved_manifest_path,
                analysis_name=analysis_name,
                manifest_variant=manifest_variant,
                checkpoint_name=checkpoint_name,
                feature_type=feature_type,
                embeddings_npz=embeddings_npz,
                required_modalities=required_modalities,
                min_holdout_organs=args.min_holdout_organs,
                min_samples_per_modality=args.min_samples_per_modality,
                few_shot_support_per_modality=args.few_shot_support_per_modality,
                few_shot_query_per_modality=args.few_shot_query_per_modality,
                few_shot_seeds=args.few_shot_seeds,
                seed=args.seed,
                output_path=None,
                manifest_summary_dir=Path(args.manifest_summary_output_dir).resolve() if args.manifest_summary_output_dir else None,
                support_size_sweep=support_size_sweep,
                feature_adapter=args.feature_adapter,
                projection_sources=args.projection_sources,
                projection_context_source_surface=args.projection_context_source_surface,
                projection_context_reference_surface=args.projection_context_reference_surface,
                projection_spacing_datasets=args.projection_spacing_datasets,
                projection_spacing_manifest_variant=args.projection_spacing_manifest_variant,
                projection_spacing_reference_variant=args.projection_spacing_reference_variant,
                projection_spacing_target_variants=args.projection_spacing_target_variants,
                projection_components=args.projection_components,
                projection_strength=args.projection_strength,
            )
            rows.append(_build_summary_row(payload))
            _write_batch_summary(
                summary_output=summary_output,
                csv_output=csv_output,
                analysis_name=analysis_name,
                manifest_path=resolved_manifest_path,
                manifest_variant=manifest_variant,
                checkpoints=checkpoints,
                feature_types=feature_types,
                rows=rows,
                missing_entries=missing_entries,
            )
            logger.info(
                "Completed bundle %d/%d: checkpoint=%s feature_type=%s status=%s elapsed_sec=%.1f completed_rows=%d missing_entries=%d",
                processed_bundle_count,
                total_requested,
                checkpoint_name,
                feature_type,
                payload.get("surface_b_few_shot_transfer_evidence", {}).get("status"),
                perf_counter() - bundle_start_time,
                len(rows),
                len(missing_entries),
            )

        logger.info(
            "Finished checkpoint %s (%d/%d): completed_rows=%d missing_entries=%d",
            checkpoint_name,
            checkpoint_index,
            len(checkpoints),
            len(rows),
            len(missing_entries),
        )

    _write_batch_summary(
        summary_output=summary_output,
        csv_output=csv_output,
        analysis_name=analysis_name,
        manifest_path=resolved_manifest_path,
        manifest_variant=manifest_variant,
        checkpoints=checkpoints,
        feature_types=feature_types,
        rows=rows,
        missing_entries=missing_entries,
    )

    if args.require_complete and missing_entries:
        raise ValueError(f"Anatomical batch run is incomplete: {missing_entries}")

    logger.info("Wrote Anatomical batch summary to %s", summary_output)


if __name__ == "__main__":
    main()
