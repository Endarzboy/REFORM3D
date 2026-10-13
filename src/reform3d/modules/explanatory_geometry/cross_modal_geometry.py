#!/usr/bin/env python

"""Cross-Modal explanatory geometry analysis for CT-MR organ embeddings.

This module adds an explanatory layer over finished Cross-Modal feature
bundles. It does not change the Cross-Modal primary-metric contract. Instead,
it reads existing embedding NPZ files and writes a compact geometry
surface that helps explain whether a checkpoint is organ-dominant, modality-
dominant, selectively aligned, or geometrically collapsed.

Execution modes
---------------
1. Single-checkpoint mode
    Evaluate one Cross-Modal feature bundle and write a sibling
   ``cross_modal_geometry.json`` result under the configured results
   tree.

2. Batch mode
   Discover Cross-Modal embedding bundles under one dataset-level outputs tree,
   evaluate every requested checkpoint and feature family, and write dataset-
   level JSON and CSV summaries.

Paired metrics are guarded by manifest evidence type. Cohort-level anchor
surfaces never fabricate same-case correspondences.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from reform3d.core.checkpoints import FULL_CHECKPOINT_ORDER
from reform3d.modules.cross_modal_alignment.config import (
    DEFAULT_REQUIRED_MODALITIES,
    FEATURE_TYPES,
    PRIMARY_METRICS_FILE_NAME,
    ensure_output_directories,
    get_feature_root_candidates,
    get_checkpoint_metrics_path,
    get_feature_embedding_file_name,
    get_output_paths,
    get_cross_modal_manifest_path,
    normalize_cross_modal_dataset_name,
    normalize_manifest_variant,
    normalize_feature_type,
    resolve_existing_checkpoint_metrics_path,
)
from reform3d.modules.cross_modal_alignment.correspondence import (
    CorrespondenceConfig,
    build_population_cohorts,
)
from reform3d.modules.cross_modal_alignment.alignment_metrics import (
    compute_global_alignment_summary,
    linear_cka,
)
from reform3d.modules.cross_modal_alignment.data_loader import load_cross_modal_manifest


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PRIMARY_METRICS_NAME = PRIMARY_METRICS_FILE_NAME
RESULT_JSON_NAME = "cross_modal_geometry.json"
SUMMARY_JSON_NAME = "cross_modal_geometry_summary.json"
SUMMARY_CSV_NAME = "cross_modal_geometry_summary.csv"

DEFAULT_FEATURE_TYPES = list(FEATURE_TYPES)
DEFAULT_CHECKPOINTS = list(FULL_CHECKPOINT_ORDER)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _get_nested(payload: Dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _ordered_checkpoint_names(checkpoint_names: Iterable[str]) -> List[str]:
    known_order = {name: idx for idx, name in enumerate(DEFAULT_CHECKPOINTS)}
    unique_names = list(dict.fromkeys(checkpoint_names))
    return sorted(unique_names, key=lambda name: (known_order.get(name, len(known_order)), name))


def _checkpoint_sort_key(checkpoint_name: str) -> Tuple[int, str]:
    known_order = {name: idx for idx, name in enumerate(DEFAULT_CHECKPOINTS)}
    return (known_order.get(checkpoint_name, len(known_order)), checkpoint_name)


def _l2_normalize_rows(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.clip(norms, 1e-8, None)


def _l2_normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return vector.astype(np.float32, copy=False)
    return (vector / norm).astype(np.float32, copy=False)


def _load_embedding_bundle(npz_path: Path, manifest_samples: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    payload = np.load(npz_path, allow_pickle=True)
    if "features" not in payload:
        raise ValueError(f"Embedding bundle {npz_path} must contain a 'features' array")

    features = np.asarray(payload["features"], dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"Expected features to have shape [n_samples, dim], got {features.shape}")

    if "sample_ids" in payload:
        sample_ids = [str(item) for item in payload["sample_ids"].tolist()]
    else:
        if features.shape[0] != len(manifest_samples):
            raise ValueError(
                "Embedding bundle omitted sample_ids, so manifest order must match features length exactly"
            )
        sample_ids = [str(sample["sample_id"]) for sample in manifest_samples]

    if len(sample_ids) != features.shape[0]:
        raise ValueError("sample_ids length does not match features length")

    return {sample_id: features[idx] for idx, sample_id in enumerate(sample_ids)}


def _embedding_summary_path(npz_path: Path) -> Path:
    return npz_path.with_name(npz_path.stem + "_summary.json")


def _load_optional_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_manifest_metadata(manifest_path: Path) -> Dict[str, Any]:
    meta_path = manifest_path.with_name("manifest_meta.json")
    return _load_optional_json(meta_path) or {}


def _infer_single_manifest_path(
    manifest_path: Optional[Path],
    analysis_name: Optional[str],
    manifest_variant: str,
    embeddings_npz: Path,
) -> Path:
    if manifest_path is not None:
        return manifest_path.resolve()

    summary_payload = _load_optional_json(_embedding_summary_path(embeddings_npz))
    if summary_payload and summary_payload.get("manifest_path"):
        return Path(str(summary_payload["manifest_path"])).resolve()

    if analysis_name is None:
        raise ValueError("Single-checkpoint mode requires --manifest, --analysis-name, or a feature summary JSON")

    return get_cross_modal_manifest_path(
        dataset_name=analysis_name,
        manifest_kind="sampled",
        manifest_variant=manifest_variant,
    ).resolve()


def _infer_analysis_name(manifest_path: Path, explicit_name: Optional[str]) -> str:
    if explicit_name:
        return normalize_cross_modal_dataset_name(explicit_name)
    try:
        return normalize_cross_modal_dataset_name(manifest_path.resolve().parents[1].name)
    except IndexError as exc:
        raise ValueError(f"Could not infer analysis name from manifest path {manifest_path}") from exc


def _default_single_output(npz_path: Path) -> Path:
    parts = list(npz_path.parts)
    if "features" not in parts:
        return npz_path.with_name(RESULT_JSON_NAME)
    features_idx = parts.index("features")
    result_parts = parts[:]
    result_parts[features_idx] = "results"
    return Path(*result_parts[:-1]) / RESULT_JSON_NAME


def _rowwise_cosine_distance(features_a: np.ndarray, features_b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(features_a, axis=1) * np.linalg.norm(features_b, axis=1)
    cosine_similarity = np.sum(features_a * features_b, axis=1) / np.clip(denom, 1e-8, None)
    return 1.0 - np.clip(cosine_similarity, -1.0, 1.0)


def _compute_uniformity(
    features: np.ndarray,
    max_points: int,
    seed: int,
    temperature: float = 2.0,
) -> Dict[str, Any]:
    if features.shape[0] < 2:
        return {
            "status": "skipped",
            "reason": "need_at_least_two_samples",
            "value": None,
        }

    rng = np.random.default_rng(seed)
    if features.shape[0] > max_points:
        indices = rng.choice(features.shape[0], size=max_points, replace=False)
        sampled = features[indices]
    else:
        sampled = features

    similarity = np.clip(sampled @ sampled.T, -1.0, 1.0)
    squared_distance = np.clip(2.0 - 2.0 * similarity, 0.0, None)
    upper = squared_distance[np.triu_indices(sampled.shape[0], k=1)]
    if upper.size == 0:
        return {
            "status": "skipped",
            "reason": "no_unique_pairs_after_sampling",
            "value": None,
        }

    value = float(np.log(np.mean(np.exp(-temperature * upper))))
    return {
        "status": "ok",
        "value": value,
        "temperature": float(temperature),
        "n_points": int(sampled.shape[0]),
        "n_pairs": int(upper.size),
    }


def _compute_spectral_geometry(features: np.ndarray) -> Dict[str, Any]:
    if features.shape[0] == 0:
        return {
            "status": "skipped",
            "reason": "no_samples",
            "effective_rank": None,
            "stable_rank": None,
            "participation_ratio": None,
            "top1_variance_ratio": None,
        }

    centered = features - features.mean(axis=0, keepdims=True)
    if not np.any(centered):
        return {
            "status": "ok",
            "effective_rank": 0.0,
            "stable_rank": 0.0,
            "participation_ratio": 0.0,
            "top1_variance_ratio": 1.0,
        }

    singular_values = np.linalg.svd(centered, compute_uv=False, full_matrices=False)
    power = singular_values ** 2
    total_power = float(np.sum(power))
    if total_power <= 0.0:
        return {
            "status": "ok",
            "effective_rank": 0.0,
            "stable_rank": 0.0,
            "participation_ratio": 0.0,
            "top1_variance_ratio": 1.0,
        }

    probs = power / total_power
    entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, None))))
    stable_rank = total_power / max(float(np.max(power)), 1e-12)
    participation_ratio = (total_power ** 2) / max(float(np.sum(power ** 2)), 1e-12)

    return {
        "status": "ok",
        "effective_rank": float(math.exp(entropy)),
        "stable_rank": float(stable_rank),
        "participation_ratio": float(participation_ratio),
        "top1_variance_ratio": float(probs[0]),
    }


def _compute_centroid_alignment_by_organ(
    normalized_features_by_id: Dict[str, np.ndarray],
    samples: Sequence[Dict[str, Any]],
    supported_organs: Sequence[str],
    required_modalities: Sequence[str],
) -> Dict[str, Any]:
    organ_set = set(supported_organs)
    grouped: Dict[str, Dict[str, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))

    for sample in samples:
        organ = sample.get("primary_organ")
        modality = sample.get("modality")
        sample_id = sample.get("sample_id")
        if organ not in organ_set or modality not in required_modalities or sample_id not in normalized_features_by_id:
            continue
        grouped[organ][modality].append(normalized_features_by_id[sample_id])

    per_organ: Dict[str, Any] = {}
    centroid_distances: List[float] = []
    modality_a, modality_b = required_modalities[:2]
    for organ in tqdm(sorted(organ_set), desc="Cross-Modal geometry centroid organs", unit="organ", dynamic_ncols=True, leave=False):
        features_a = grouped[organ].get(modality_a, [])
        features_b = grouped[organ].get(modality_b, [])
        if not features_a or not features_b:
            per_organ[organ] = {
                "status": "skipped",
                "reason": "missing_cross_modal_support",
                modality_a: len(features_a),
                modality_b: len(features_b),
            }
            continue

        stack_a = np.stack(features_a, axis=0)
        stack_b = np.stack(features_b, axis=0)
        centroid_a = _l2_normalize_vector(np.mean(stack_a, axis=0))
        centroid_b = _l2_normalize_vector(np.mean(stack_b, axis=0))
        cosine_similarity = float(np.clip(np.dot(centroid_a, centroid_b), -1.0, 1.0))
        cosine_distance = 1.0 - cosine_similarity
        centroid_distances.append(cosine_distance)

        per_organ[organ] = {
            "status": "ok",
            modality_a: int(stack_a.shape[0]),
            modality_b: int(stack_b.shape[0]),
            "cross_modal_centroid_cosine_similarity": cosine_similarity,
            "cross_modal_centroid_cosine_distance": cosine_distance,
            "within_modality_dispersion": {
                modality_a: float(np.mean(1.0 - np.clip(stack_a @ centroid_a, -1.0, 1.0))),
                modality_b: float(np.mean(1.0 - np.clip(stack_b @ centroid_b, -1.0, 1.0))),
            },
        }

    return {
        "required_modalities": list(required_modalities),
        "macro_mean_centroid_distance": float(np.mean(centroid_distances)) if centroid_distances else None,
        "macro_median_centroid_distance": float(np.median(centroid_distances)) if centroid_distances else None,
        "macro_mean_centroid_similarity": float(1.0 - np.mean(centroid_distances)) if centroid_distances else None,
        "n_supported_organs": len(per_organ),
        "n_ok_organs": sum(1 for item in per_organ.values() if item.get("status") == "ok"),
        "per_organ": per_organ,
    }


def _compute_modality_centroid_geometry(
    normalized_features_by_id: Dict[str, np.ndarray],
    samples: Sequence[Dict[str, Any]],
    required_modalities: Sequence[str],
) -> Dict[str, Any]:
    grouped: Dict[str, List[np.ndarray]] = defaultdict(list)
    for sample in samples:
        modality = sample.get("modality")
        sample_id = sample.get("sample_id")
        if modality not in required_modalities or sample_id not in normalized_features_by_id:
            continue
        grouped[modality].append(normalized_features_by_id[sample_id])

    modality_a, modality_b = required_modalities[:2]
    features_a = grouped.get(modality_a, [])
    features_b = grouped.get(modality_b, [])
    if not features_a or not features_b:
        return {
            "status": "skipped",
            "reason": "missing_required_modalities",
            "required_modalities": list(required_modalities),
        }

    centroid_a = _l2_normalize_vector(np.mean(np.stack(features_a, axis=0), axis=0))
    centroid_b = _l2_normalize_vector(np.mean(np.stack(features_b, axis=0), axis=0))
    cosine_similarity = float(np.clip(np.dot(centroid_a, centroid_b), -1.0, 1.0))
    cosine_distance = 1.0 - cosine_similarity

    return {
        "status": "ok",
        "required_modalities": list(required_modalities),
        modality_a: int(len(features_a)),
        modality_b: int(len(features_b)),
        "cross_modal_centroid_cosine_similarity": cosine_similarity,
        "cross_modal_centroid_cosine_distance": cosine_distance,
    }


def _compute_paired_case_cka_by_organ(
    normalized_features_by_id: Dict[str, np.ndarray],
    samples: Sequence[Dict[str, Any]],
    supported_organs: Sequence[str],
    required_modalities: Sequence[str],
    evidence_type: Optional[str],
) -> Dict[str, Any]:
    if evidence_type != "paired_same_case_ct_mr":
        return {
            "status": "skipped",
            "reason": "evidence_type_not_paired_same_case",
            "evidence_type": evidence_type,
        }

    organ_set = set(supported_organs)
    modality_a, modality_b = required_modalities[:2]
    paired_rows: Dict[str, Dict[str, Dict[str, List[np.ndarray]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for sample in samples:
        organ = sample.get("primary_organ")
        modality = sample.get("modality")
        sample_id = sample.get("sample_id")
        pair_id = sample.get("patient_id") or sample.get("study_id")
        if organ not in organ_set or modality not in required_modalities or sample_id not in normalized_features_by_id:
            continue
        if pair_id is None:
            continue
        paired_rows[organ][str(pair_id)][modality].append(normalized_features_by_id[sample_id])

    per_organ: Dict[str, Any] = {}
    cka_values: List[float] = []
    pair_distances: List[float] = []
    for organ in tqdm(sorted(organ_set), desc="Cross-Modal geometry paired CKA organs", unit="organ", dynamic_ncols=True, leave=False):
        matched_a: List[np.ndarray] = []
        matched_b: List[np.ndarray] = []
        for _, modality_map in sorted(paired_rows[organ].items()):
            features_a = modality_map.get(modality_a, [])
            features_b = modality_map.get(modality_b, [])
            if not features_a or not features_b:
                continue
            matched_a.append(_l2_normalize_vector(np.mean(np.stack(features_a, axis=0), axis=0)))
            matched_b.append(_l2_normalize_vector(np.mean(np.stack(features_b, axis=0), axis=0)))

        if len(matched_a) < 2 or len(matched_b) < 2:
            per_organ[organ] = {
                "status": "skipped",
                "reason": "insufficient_paired_case_support",
                "n_paired_cases": int(min(len(matched_a), len(matched_b))),
            }
            continue

        matrix_a = np.stack(matched_a, axis=0)
        matrix_b = np.stack(matched_b, axis=0)
        cka_value = float(linear_cka(matrix_a, matrix_b))
        mean_pair_distance = float(np.mean(_rowwise_cosine_distance(matrix_a, matrix_b)))
        cka_values.append(cka_value)
        pair_distances.append(mean_pair_distance)
        per_organ[organ] = {
            "status": "ok",
            "n_paired_cases": int(matrix_a.shape[0]),
            "paired_case_linear_cka": cka_value,
            "paired_case_mean_cosine_distance": mean_pair_distance,
        }

    return {
        "status": "ok",
        "evidence_type": evidence_type,
        "required_modalities": list(required_modalities),
        "macro_mean_cka": float(np.mean(cka_values)) if cka_values else None,
        "macro_median_cka": float(np.median(cka_values)) if cka_values else None,
        "macro_mean_pair_distance": float(np.mean(pair_distances)) if pair_distances else None,
        "n_ok_organs": int(len(cka_values)),
        "per_organ": per_organ,
    }


def _load_primary_metrics_context(primary_metrics_path: Path) -> Dict[str, Any]:
    payload = _load_optional_json(primary_metrics_path)
    if payload is None:
        return {
            "status": "missing",
            "primary_metrics_path": str(primary_metrics_path),
        }

    return {
        "status": "ok",
        "primary_metrics_path": str(primary_metrics_path),
        "retrieval_map": _get_nested(payload, "balanced_cross_modal_retrieval", "bidirectional_mean", "map"),
        "retrieval_top1": _get_nested(payload, "balanced_cross_modal_retrieval", "bidirectional_mean", "top@1"),
        "retrieval_top5": _get_nested(payload, "balanced_cross_modal_retrieval", "bidirectional_mean", "top@5"),
        "mean_margin": _get_nested(payload, "anatomy_over_modality_margin", "overall", "mean_margin"),
        "fraction_positive_margin": _get_nested(
            payload, "anatomy_over_modality_margin", "overall", "fraction_positive_margin"
        ),
        "modality_ilisi": _get_nested(payload, "balanced_lisi", "overall", "modality_ilisi", "mean"),
        "organ_clisi": _get_nested(payload, "balanced_lisi", "overall", "organ_clisi", "mean"),
    }


def compute_cross_modal_explanatory_geometry(
    manifest_path: Path,
    embeddings_npz: Path,
    analysis_name: str,
    checkpoint_name: str,
    feature_type: str,
    min_samples_per_modality: int,
    uniformity_max_points: int,
    seed: int,
    primary_metrics_path: Optional[Path] = None,
) -> Dict[str, Any]:
    samples = load_cross_modal_manifest(manifest_path)
    manifest_meta = _load_manifest_metadata(manifest_path)
    evidence_type = (
        manifest_meta.get("evidence_type")
        or next((sample.get("raw_sample", {}).get("evidence_type") for sample in samples if sample.get("raw_sample")), None)
    )

    features_by_id = _load_embedding_bundle(embeddings_npz, samples)
    cohorts = build_population_cohorts(
        samples,
        CorrespondenceConfig(
            required_modalities=tuple(DEFAULT_REQUIRED_MODALITIES),
            min_samples_per_modality=min_samples_per_modality,
        ),
    )
    supported_organs = sorted(cohorts.get("supported_organs", {}).keys())
    required_modalities = tuple(cohorts.get("required_modalities") or DEFAULT_REQUIRED_MODALITIES)

    eligible_samples = [
        sample
        for sample in samples
        if sample.get("sample_id") in features_by_id
        and sample.get("primary_organ") in supported_organs
        and sample.get("modality") in required_modalities
    ]
    if not eligible_samples:
        raise ValueError(f"No eligible Cross-Modal samples with features were found for {embeddings_npz}")

    sample_ids = [str(sample["sample_id"]) for sample in eligible_samples]
    normalized_feature_matrix = _l2_normalize_rows(
        np.stack([features_by_id[sample_id] for sample_id in sample_ids], axis=0)
    )
    normalized_features_by_id = {
        sample_id: normalized_feature_matrix[idx] for idx, sample_id in enumerate(sample_ids)
    }

    organ_labels = [str(sample["primary_organ"]) for sample in eligible_samples]
    modality_labels = [str(sample["modality"]) for sample in eligible_samples]
    global_alignment = compute_global_alignment_summary(
        normalized_feature_matrix,
        organ_labels,
        modality_labels,
    )
    modality_centroid_geometry = _compute_modality_centroid_geometry(
        normalized_features_by_id,
        eligible_samples,
        required_modalities,
    )
    organ_centroid_alignment = _compute_centroid_alignment_by_organ(
        normalized_features_by_id,
        eligible_samples,
        supported_organs,
        required_modalities,
    )
    paired_case_cka = _compute_paired_case_cka_by_organ(
        normalized_features_by_id,
        eligible_samples,
        supported_organs,
        required_modalities,
        evidence_type,
    )

    by_modality_indices: Dict[str, np.ndarray] = {}
    for modality in required_modalities:
        indices = [idx for idx, label in enumerate(modality_labels) if label == modality]
        by_modality_indices[modality] = np.asarray(indices, dtype=np.int64)

    uniformity = {
        "overall": _compute_uniformity(normalized_feature_matrix, uniformity_max_points, seed),
        "by_modality": {
            modality: _compute_uniformity(
                normalized_feature_matrix[indices],
                min(uniformity_max_points, max(len(indices), 1)),
                seed,
            )
            for modality, indices in by_modality_indices.items()
        },
    }
    spectral_geometry = {
        "overall": _compute_spectral_geometry(normalized_feature_matrix),
        "by_modality": {
            modality: _compute_spectral_geometry(normalized_feature_matrix[indices])
            for modality, indices in by_modality_indices.items()
        },
    }

    resolved_primary_metrics_path = primary_metrics_path
    if resolved_primary_metrics_path is None:
        inferred_primary = _default_single_output(embeddings_npz).parents[1] / PRIMARY_METRICS_NAME
        resolved_primary_metrics_path = inferred_primary
    primary_context = _load_primary_metrics_context(resolved_primary_metrics_path)

    organ_silhouette = _get_nested(global_alignment, "organ_silhouette", "score")
    modality_silhouette = _get_nested(global_alignment, "modality_silhouette", "score")
    summary_metrics = {
        "organ_silhouette": organ_silhouette,
        "modality_silhouette": modality_silhouette,
        "organ_over_modality_ratio": global_alignment.get("organ_over_modality_ratio"),
        "dominant_axis": global_alignment.get("dominant_axis"),
        "modality_centroid_distance": modality_centroid_geometry.get("cross_modal_centroid_cosine_distance"),
        "organ_centroid_distance_macro": organ_centroid_alignment.get("macro_mean_centroid_distance"),
        "paired_case_cka_macro": paired_case_cka.get("macro_mean_cka"),
        "effective_rank_overall": _get_nested(spectral_geometry, "overall", "effective_rank"),
        "top1_variance_ratio_overall": _get_nested(spectral_geometry, "overall", "top1_variance_ratio"),
        "uniformity_overall": _get_nested(uniformity, "overall", "value"),
        "cross_modal_retrieval_map": primary_context.get("retrieval_map"),
        "cross_modal_mean_margin": primary_context.get("mean_margin"),
        "cross_modal_modality_ilisi": primary_context.get("modality_ilisi"),
        "cross_modal_organ_clisi": primary_context.get("organ_clisi"),
    }

    return {
        "analysis_version": "cross_modal_explanatory_geometry_v1",
        "analysis_name": analysis_name,
        "checkpoint_name": checkpoint_name,
        "feature_type": feature_type,
        "manifest_path": str(manifest_path),
        "embeddings_npz": str(embeddings_npz),
        "geometry_feature_space": "l2_normalized_embeddings",
        "evidence_type": evidence_type,
        "required_modalities": list(required_modalities),
        "n_samples": int(len(eligible_samples)),
        "n_supported_organs": int(len(supported_organs)),
        "supported_organs": supported_organs,
        "primary_metrics_context": primary_context,
        "global_alignment": global_alignment,
        "modality_centroid_geometry": modality_centroid_geometry,
        "organ_centroid_alignment": organ_centroid_alignment,
        "paired_case_cka": paired_case_cka,
        "uniformity": uniformity,
        "spectral_geometry": spectral_geometry,
        "summary_metrics": summary_metrics,
    }


def evaluate_single_checkpoint(
    manifest_path: Path,
    embeddings_npz: Path,
    analysis_name: str,
    manifest_variant: str,
    checkpoint_name: str,
    feature_type: str,
    output_path: Optional[Path],
    min_samples_per_modality: int,
    uniformity_max_points: int,
    seed: int,
) -> Dict[str, Any]:
    resolved_output_path = (
        output_path
        or get_output_paths(analysis_name, manifest_variant)["results"] / checkpoint_name / feature_type / RESULT_JSON_NAME
    )
    default_primary_metrics_path = resolve_existing_checkpoint_metrics_path(
        analysis_name,
        manifest_variant,
        checkpoint_name,
        feature_type,
    )
    results = compute_cross_modal_explanatory_geometry(
        manifest_path=manifest_path,
        embeddings_npz=embeddings_npz,
        analysis_name=analysis_name,
        checkpoint_name=checkpoint_name,
        feature_type=feature_type,
        min_samples_per_modality=min_samples_per_modality,
        uniformity_max_points=uniformity_max_points,
        seed=seed,
        primary_metrics_path=default_primary_metrics_path,
    )
    _write_json(resolved_output_path, results)
    logger.info("Saved Cross-Modal explanatory geometry to %s", resolved_output_path)
    return results


def _resolve_batch_roots(
    analysis_name: str,
    manifest_variant: str,
) -> Tuple[Tuple[Path, Path], Path]:
    paths = get_output_paths(analysis_name, manifest_variant)
    return get_feature_root_candidates(analysis_name, manifest_variant), paths["results"]


def _batch_output_paths(
    results_root: Path,
    checkpoint_name: str,
    feature_type: str,
    summary_output: Optional[Path],
    csv_output: Optional[Path],
) -> Tuple[Path, Path, Path]:
    per_checkpoint_output = results_root / checkpoint_name / feature_type / RESULT_JSON_NAME
    resolved_summary_output = summary_output or results_root / SUMMARY_JSON_NAME
    resolved_csv_output = csv_output or results_root / SUMMARY_CSV_NAME
    return per_checkpoint_output, resolved_summary_output, resolved_csv_output


def _collect_summary_row(
    analysis_name: str,
    checkpoint_name: str,
    feature_type: str,
    result_path: Path,
    results: Dict[str, Any],
) -> Dict[str, Any]:
    summary = results.get("summary_metrics", {})
    return {
        "analysis_name": analysis_name,
        "checkpoint_name": checkpoint_name,
        "feature_type": feature_type,
        "organ_silhouette": summary.get("organ_silhouette"),
        "modality_silhouette": summary.get("modality_silhouette"),
        "organ_over_modality_ratio": summary.get("organ_over_modality_ratio"),
        "dominant_axis": summary.get("dominant_axis"),
        "modality_centroid_distance": summary.get("modality_centroid_distance"),
        "organ_centroid_distance_macro": summary.get("organ_centroid_distance_macro"),
        "paired_case_cka_macro": summary.get("paired_case_cka_macro"),
        "effective_rank_overall": summary.get("effective_rank_overall"),
        "top1_variance_ratio_overall": summary.get("top1_variance_ratio_overall"),
        "uniformity_overall": summary.get("uniformity_overall"),
        "cross_modal_retrieval_map": summary.get("cross_modal_retrieval_map"),
        "cross_modal_mean_margin": summary.get("cross_modal_mean_margin"),
        "cross_modal_modality_ilisi": summary.get("cross_modal_modality_ilisi"),
        "cross_modal_organ_clisi": summary.get("cross_modal_organ_clisi"),
        "result_path": str(result_path),
    }


def _discover_embeddings(
    features_roots: Sequence[Path],
    checkpoints: Sequence[str],
    feature_types: Sequence[str],
) -> Tuple[Dict[Tuple[str, str], Path], List[Tuple[str, str]]]:
    bundle_map: Dict[Tuple[str, str], Path] = {}
    missing_pairs: List[Tuple[str, str]] = []
    for checkpoint_name in checkpoints:
        for feature_type in feature_types:
            normalized_feature_type = normalize_feature_type(feature_type)
            pair = (checkpoint_name, feature_type)
            npz_path = None
            for features_root in features_roots:
                candidate = (
                    features_root
                    / checkpoint_name
                    / normalized_feature_type
                    / get_feature_embedding_file_name(normalized_feature_type)
                )
                if candidate.exists():
                    npz_path = candidate
                    break
            if npz_path is None:
                missing_pairs.append(pair)
            else:
                bundle_map[pair] = npz_path
    return bundle_map, missing_pairs


def _best_row(rows: Sequence[Dict[str, Any]], key: str, reverse: bool) -> Optional[Dict[str, Any]]:
    valid_rows = [row for row in rows if row.get(key) is not None]
    if not valid_rows:
        return None
    return max(valid_rows, key=lambda row: row[key]) if reverse else min(valid_rows, key=lambda row: row[key])


def run_batch_evaluation(
    analysis_name: str,
    manifest_path: Path,
    manifest_variant: str,
    checkpoints: Sequence[str],
    feature_types: Sequence[str],
    min_samples_per_modality: int,
    uniformity_max_points: int,
    seed: int,
    summary_output: Optional[Path],
    csv_output: Optional[Path],
    require_complete: bool,
) -> Dict[str, Any]:
    feature_roots, results_root = _resolve_batch_roots(analysis_name, manifest_variant)
    ordered_checkpoints = _ordered_checkpoint_names(checkpoints)
    bundle_map, missing_pairs = _discover_embeddings(feature_roots, ordered_checkpoints, feature_types)

    rows: List[Dict[str, Any]] = []
    checkpoint_iterator = tqdm(ordered_checkpoints, desc="Cross-Modal geometry checkpoints", unit="ckpt", dynamic_ncols=True)
    for checkpoint_name in checkpoint_iterator:
        feature_iterator = tqdm(feature_types, desc=f"Cross-Modal geometry features {checkpoint_name}", unit="feature", dynamic_ncols=True, leave=False)
        for feature_type in feature_iterator:
            pair = (checkpoint_name, feature_type)
            embeddings_npz = bundle_map.get(pair)
            if embeddings_npz is None:
                continue
            result_output, _, _ = _batch_output_paths(
                results_root,
                checkpoint_name,
                feature_type,
                summary_output,
                csv_output,
            )
            results = evaluate_single_checkpoint(
                manifest_path=manifest_path,
                embeddings_npz=embeddings_npz,
                analysis_name=analysis_name,
                manifest_variant=manifest_variant,
                checkpoint_name=checkpoint_name,
                feature_type=feature_type,
                output_path=result_output,
                min_samples_per_modality=min_samples_per_modality,
                uniformity_max_points=uniformity_max_points,
                seed=seed,
            )
            rows.append(_collect_summary_row(analysis_name, checkpoint_name, feature_type, result_output, results))

    rows = sorted(rows, key=lambda row: (_checkpoint_sort_key(row["checkpoint_name"]), row["feature_type"]))
    _, resolved_summary_output, resolved_csv_output = _batch_output_paths(
        results_root,
        ordered_checkpoints[0] if ordered_checkpoints else "",
        feature_types[0] if feature_types else "cls",
        summary_output,
        csv_output,
    )
    if rows:
        _write_csv(resolved_csv_output, rows)

    summary = {
        "analysis_version": "cross_modal_explanatory_geometry_v1_batch",
        "analysis_name": analysis_name,
        "manifest_path": str(manifest_path),
        "manifest_variant": manifest_variant,
        "features_root": str(feature_roots[0]),
        "feature_root_candidates": [str(root) for root in feature_roots],
        "results_root": str(results_root),
        "requested_checkpoints": ordered_checkpoints,
        "requested_feature_types": list(feature_types),
        "rows": rows,
        "missing": [
            {"checkpoint_name": checkpoint_name, "feature_type": feature_type}
            for checkpoint_name, feature_type in missing_pairs
        ],
        "n_completed": len(rows),
        "n_missing": len(missing_pairs),
        "best_by_anatomy_dominance_ratio": _best_row(rows, "organ_over_modality_ratio", reverse=True),
        "best_by_modality_centroid_distance": _best_row(rows, "modality_centroid_distance", reverse=False),
        "best_by_organ_centroid_distance": _best_row(rows, "organ_centroid_distance_macro", reverse=False),
        "best_by_paired_case_cka": _best_row(rows, "paired_case_cka_macro", reverse=True),
        "best_by_effective_rank": _best_row(rows, "effective_rank_overall", reverse=True),
        "best_by_uniformity": _best_row(rows, "uniformity_overall", reverse=False),
    }
    _write_json(resolved_summary_output, summary)
    logger.info("Saved Cross-Modal explanatory batch summary to %s", resolved_summary_output)
    if rows:
        logger.info("Saved Cross-Modal explanatory batch CSV to %s", resolved_csv_output)

    if require_complete and missing_pairs:
        preview = ", ".join(f"{checkpoint}/{feature_type}" for checkpoint, feature_type in missing_pairs[:10])
        raise RuntimeError(
            f"Cross-Modal explanatory geometry batch is incomplete: {len(missing_pairs)} embedding bundle(s) are missing under {feature_roots}. "
            f"Examples: {preview}"
        )

    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-Modal explanatory geometry analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Single checkpoint:\n"
            "    python -m reform3d.modules.explanatory_geometry.cross_modal_geometry --embeddings-npz /path/to/organ_cls_embeddings.npz --analysis-name totalsegmenter_ct_mr_anchor\n\n"
            "  Full dataset batch:\n"
            "    python -m reform3d.modules.explanatory_geometry.cross_modal_geometry --analysis-name totalsegmenter_ct_mr_anchor --manifest-variant core\n"
        ),
    )

    parser.add_argument("--embeddings-npz", type=Path, default=None, help="Single-checkpoint NPZ bundle to analyze")
    parser.add_argument("--manifest", type=Path, default=None, help="Cross-Modal manifest path")
    parser.add_argument("--analysis-name", type=str, default=None, help="Cross-Modal analysis namespace")
    parser.add_argument("--manifest-variant", type=str, default="core", help="Cross-Modal manifest variant")
    parser.add_argument("--checkpoint-name", type=str, default=None, help="Checkpoint label for single mode")
    parser.add_argument("--feature-type", type=str, default=None, help="Feature family label for single mode")
    parser.add_argument("--output", type=Path, default=None, help="Single-checkpoint output JSON path")

    parser.add_argument("--checkpoints", nargs="*", default=None, help="Checkpoint names for batch mode")
    parser.add_argument("--feature-types", nargs="*", default=None, help="Feature types for batch mode")
    parser.add_argument("--summary-output", type=Path, default=None, help="Batch summary JSON output path")
    parser.add_argument("--csv-output", type=Path, default=None, help="Batch summary CSV output path")
    parser.add_argument("--require-complete", action="store_true", help="Fail if any requested bundle is missing")
    parser.add_argument(
        "--min-samples-per-modality",
        type=int,
        default=5,
        help="Minimum organ support per modality used to define the supported cohort set",
    )
    parser.add_argument(
        "--uniformity-max-points",
        type=int,
        default=1024,
        help="Maximum number of embeddings sampled when estimating Wang-Isola uniformity",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling-based geometry estimates")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    manifest_variant = normalize_manifest_variant(args.manifest_variant)

    if args.embeddings_npz is not None:
        embeddings_npz = args.embeddings_npz.resolve()
        manifest_path = _infer_single_manifest_path(args.manifest, args.analysis_name, manifest_variant, embeddings_npz)
        analysis_name = _infer_analysis_name(manifest_path, args.analysis_name)
        checkpoint_name = args.checkpoint_name or embeddings_npz.parent.parent.name
        feature_type = normalize_feature_type(args.feature_type or embeddings_npz.parent.name)
        ensure_output_directories(analysis_name, manifest_variant)
        evaluate_single_checkpoint(
            manifest_path=manifest_path,
            embeddings_npz=embeddings_npz,
            analysis_name=analysis_name,
            manifest_variant=manifest_variant,
            checkpoint_name=checkpoint_name,
            feature_type=feature_type,
            output_path=args.output.resolve() if args.output is not None else None,
            min_samples_per_modality=args.min_samples_per_modality,
            uniformity_max_points=args.uniformity_max_points,
            seed=args.seed,
        )
        return

    if not args.analysis_name:
        raise ValueError("Batch mode requires --analysis-name when --embeddings-npz is not provided")
    analysis_name = normalize_cross_modal_dataset_name(args.analysis_name)

    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else get_cross_modal_manifest_path(
            dataset_name=analysis_name,
            manifest_kind="sampled",
            manifest_variant=manifest_variant,
        ).resolve()
    )
    ensure_output_directories(analysis_name, manifest_variant)
    run_batch_evaluation(
        analysis_name=analysis_name,
        manifest_path=manifest_path,
        manifest_variant=manifest_variant,
        checkpoints=args.checkpoints or DEFAULT_CHECKPOINTS,
        feature_types=[normalize_feature_type(feature_type) for feature_type in (args.feature_types or DEFAULT_FEATURE_TYPES)],
        min_samples_per_modality=args.min_samples_per_modality,
        uniformity_max_points=args.uniformity_max_points,
        seed=args.seed,
        summary_output=args.summary_output.resolve() if args.summary_output is not None else None,
        csv_output=args.csv_output.resolve() if args.csv_output is not None else None,
        require_complete=args.require_complete,
    )


if __name__ == "__main__":
    main()
