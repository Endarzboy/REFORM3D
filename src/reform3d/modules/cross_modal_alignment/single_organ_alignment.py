#!/usr/bin/env python

"""Evaluate a single-organ patient-id-matched CT/MR alignment surface.

This script is intended for datasets such as CHAOS where the shared-organ set is
too narrow for the full organ-balanced Cross-Modal retrieval contract, but
patient-id overlap still permits matched-case validation on one organ.

The evaluator writes a separate artifact with:

- patient-id-matched bidirectional retrieval within one organ
- paired-case linear CKA
- paired-case mean cosine distance

Usage examples:
    python -m reform3d.modules.cross_modal_alignment.single_organ_alignment \
        -m chaos_ct_mr/core/manifest_sampled.json \
        --checkpoint-name 3dino \
        --feature-type cls \
        --organ liver \
        --embeddings-npz /path/to/organ_cls_embeddings.npz
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from .config import (
    DEFAULT_MANIFEST_VARIANT,
    DEFAULT_REQUIRED_MODALITIES,
    ensure_output_directories,
    get_checkpoint_metrics_dir,
    get_dataset_name_from_manifest_path,
    get_manifest_variant_from_manifest_path,
    get_output_paths,
    normalize_cross_modal_dataset_name,
    normalize_feature_type,
    normalize_manifest_variant,
)
from .alignment_metrics import linear_cka
from .data_loader import load_cross_modal_manifest
from .evaluation_pipeline import (
    _load_embedding_bundle,
    _resolve_embeddings_path,
    _resolve_manifest_path,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SINGLE_ORGAN_ALIGNMENT_FILE_NAME = "single_organ_alignment.json"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return vector.astype(np.float32, copy=False)
    return (vector / norm).astype(np.float32, copy=False)


def _average_precision_for_single_positive(ranking: np.ndarray, positive_index: int) -> float:
    hit_positions = np.flatnonzero(ranking == positive_index)
    if hit_positions.size == 0:
        return 0.0
    return float(1.0 / float(hit_positions[0] + 1))


def _paired_case_sort_key(pair_id: str) -> Tuple[int, str]:
    return (0, int(pair_id)) if str(pair_id).isdigit() else (1, str(pair_id))


def _build_paired_case_records(
    samples: Sequence[Dict[str, Any]],
    features_by_id: Dict[str, np.ndarray],
    organ: str,
    required_modalities: Sequence[str],
    pair_id_field: str,
) -> List[Dict[str, Any]]:
    required_modalities = tuple(str(modality).lower() for modality in required_modalities)
    organ = str(organ).strip().lower()
    rows: Dict[str, Dict[str, List[np.ndarray]]] = {}

    sample_iterator = tqdm(samples, desc=f"Single-organ {organ} samples", unit="sample", dynamic_ncols=True, leave=False)
    for sample in sample_iterator:
        sample_id = sample.get("sample_id")
        if sample_id not in features_by_id:
            continue
        if str(sample.get("primary_organ") or "").strip().lower() != organ:
            continue
        modality = str(sample.get("modality") or "").strip().lower()
        if modality not in required_modalities:
            continue
        pair_id = sample.get(pair_id_field) or sample.get("study_id")
        if pair_id is None:
            continue
        rows.setdefault(str(pair_id), {}).setdefault(modality, []).append(np.asarray(features_by_id[sample_id]))

    paired_cases: List[Dict[str, Any]] = []
    pair_iterator = tqdm(sorted(rows, key=_paired_case_sort_key), desc=f"Single-organ {organ} paired cases", unit="pair", dynamic_ncols=True, leave=False)
    for pair_id in pair_iterator:
        modality_map = rows[pair_id]
        if not all(modality_map.get(modality) for modality in required_modalities):
            continue
        aggregated_features = {
            modality: _l2_normalize(np.mean(np.stack(modality_map[modality], axis=0), axis=0))
            for modality in required_modalities
        }
        paired_cases.append(
            {
                "pair_id": pair_id,
                "organ": organ,
                "feature_count_by_modality": {
                    modality: len(modality_map[modality]) for modality in required_modalities
                },
                "features": aggregated_features,
            }
        )

    return paired_cases


def _compute_directional_patient_retrieval(
    paired_cases: Sequence[Dict[str, Any]],
    query_modality: str,
    target_modality: str,
    top_ks: Sequence[int] = (1, 5),
) -> Dict[str, Any]:
    if len(paired_cases) < 2:
        return {
            "status": "skipped",
            "reason": "need_at_least_two_paired_cases_for_single_organ_retrieval",
            "n_paired_cases": len(paired_cases),
        }

    query_features = np.stack([case["features"][query_modality] for case in paired_cases], axis=0)
    target_features = np.stack([case["features"][target_modality] for case in paired_cases], axis=0)
    similarity = np.asarray(query_features @ target_features.T, dtype=np.float64)
    top_ks = tuple(sorted({int(k) for k in top_ks if int(k) > 0}))

    top_counts = {k: 0 for k in top_ks}
    average_precision_total = 0.0
    positive_ranks: List[int] = []
    per_pair: List[Dict[str, Any]] = []
    query_iterator = tqdm(paired_cases, desc=f"Single-organ retrieval {query_modality}->{target_modality}", unit="query", dynamic_ncols=True, leave=False)
    for query_index, case in enumerate(query_iterator):
        ranking = np.argsort(-similarity[query_index])
        positive_rank = int(np.flatnonzero(ranking == query_index)[0] + 1)
        positive_ranks.append(positive_rank)
        average_precision_total += _average_precision_for_single_positive(ranking, query_index)
        for k in top_ks:
            top_counts[k] += int(positive_rank <= k)
        per_pair.append(
            {
                "pair_id": case["pair_id"],
                "positive_rank": positive_rank,
                "positive_similarity": float(similarity[query_index, query_index]),
            }
        )

    n_queries = len(paired_cases)
    return {
        "status": "ok",
        "query_modality": query_modality,
        "target_modality": target_modality,
        "n_queries": n_queries,
        "n_targets": len(paired_cases),
        "overall": {
            **{f"top@{k}": float(top_counts[k] / n_queries) for k in top_ks},
            "map": float(average_precision_total / n_queries),
            "mean_positive_rank": float(np.mean(positive_ranks)),
            "median_positive_rank": float(np.median(positive_ranks)),
        },
        "per_pair": per_pair,
    }


def _compute_bidirectional_patient_retrieval(
    paired_cases: Sequence[Dict[str, Any]],
    required_modalities: Sequence[str],
    top_ks: Sequence[int] = (1, 5),
) -> Dict[str, Any]:
    modality_a, modality_b = tuple(required_modalities[:2])
    a_to_b = _compute_directional_patient_retrieval(paired_cases, modality_a, modality_b, top_ks=top_ks)
    b_to_a = _compute_directional_patient_retrieval(paired_cases, modality_b, modality_a, top_ks=top_ks)

    if a_to_b.get("status") != "ok" or b_to_a.get("status") != "ok":
        return {
            "status": "skipped",
            "reason": "directional_single_organ_retrieval_unavailable",
            f"{modality_a}_to_{modality_b}": a_to_b,
            f"{modality_b}_to_{modality_a}": b_to_a,
        }

    shared_keys = sorted(set(a_to_b["overall"]) & set(b_to_a["overall"]))
    bidirectional_mean = {
        key: float((a_to_b["overall"][key] + b_to_a["overall"][key]) / 2.0) for key in shared_keys
    }
    return {
        "status": "ok",
        f"{modality_a}_to_{modality_b}": a_to_b,
        f"{modality_b}_to_{modality_a}": b_to_a,
        "bidirectional_mean": bidirectional_mean,
    }


def _compute_paired_case_geometry(
    paired_cases: Sequence[Dict[str, Any]],
    required_modalities: Sequence[str],
) -> Dict[str, Any]:
    if len(paired_cases) < 2:
        return {
            "status": "skipped",
            "reason": "need_at_least_two_paired_cases_for_geometry",
            "n_paired_cases": len(paired_cases),
        }

    modality_a, modality_b = tuple(required_modalities[:2])
    matrix_a = np.stack([case["features"][modality_a] for case in paired_cases], axis=0)
    matrix_b = np.stack([case["features"][modality_b] for case in paired_cases], axis=0)
    pairwise_cosine_distance = 1.0 - np.sum(matrix_a * matrix_b, axis=1)
    return {
        "status": "ok",
        "n_paired_cases": int(len(paired_cases)),
        "paired_case_linear_cka": float(linear_cka(matrix_a, matrix_b)),
        "paired_case_mean_cosine_distance": float(np.mean(pairwise_cosine_distance)),
        "paired_case_median_cosine_distance": float(np.median(pairwise_cosine_distance)),
    }


def _default_output_path(analysis_name: str, manifest_variant: str, checkpoint_name: str, feature_type: str) -> Path:
    metrics_dir = get_checkpoint_metrics_dir(
        get_output_paths(analysis_name, manifest_variant)["results"],
        checkpoint_name,
        feature_type,
    )
    return metrics_dir / SINGLE_ORGAN_ALIGNMENT_FILE_NAME


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a single-organ patient-id-matched CT-MR alignment surface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-m", "--manifest", required=True, help="Path to the Cross-Modal manifest JSON")
    parser.add_argument("-a", "--analysis-name", default=None, help="Dataset namespace for outputs")
    parser.add_argument("--manifest-variant", default=None, help="Optional manifest variant override")
    parser.add_argument("--checkpoint-name", required=True, help="Checkpoint label for output organization")
    parser.add_argument("--feature-type", default="cls", help="Feature family label")
    parser.add_argument("--organ", required=True, help="Single organ to evaluate")
    parser.add_argument("--pair-id-field", default="patient_id", help="Manifest field used to define cross-modal pairs")
    parser.add_argument(
        "--required-modalities",
        nargs="+",
        default=list(DEFAULT_REQUIRED_MODALITIES),
        help="Required modalities used to form matched pairs",
    )
    parser.add_argument("--embeddings-npz", required=True, help="Feature bundle NPZ path")
    parser.add_argument("--output-json", default=None, help="Optional explicit output path")
    args = parser.parse_args()

    requested_manifest_variant = normalize_manifest_variant(args.manifest_variant) if args.manifest_variant else None
    manifest_path = _resolve_manifest_path(
        args.manifest,
        args.analysis_name,
        requested_manifest_variant,
        args.embeddings_npz,
    )
    embeddings_path = _resolve_embeddings_path(args.embeddings_npz)
    feature_type = normalize_feature_type(args.feature_type)
    analysis_name = normalize_cross_modal_dataset_name(args.analysis_name or get_dataset_name_from_manifest_path(manifest_path))
    manifest_variant = get_manifest_variant_from_manifest_path(
        manifest_path,
        fallback=requested_manifest_variant or DEFAULT_MANIFEST_VARIANT,
    )
    ensure_output_directories(analysis_name, manifest_variant)

    samples = load_cross_modal_manifest(manifest_path)
    features_by_id = _load_embedding_bundle(embeddings_path, samples)
    required_modalities = tuple(str(modality).lower() for modality in args.required_modalities)
    paired_cases = _build_paired_case_records(
        samples=samples,
        features_by_id=features_by_id,
        organ=args.organ,
        required_modalities=required_modalities,
        pair_id_field=args.pair_id_field,
    )

    retrieval = _compute_bidirectional_patient_retrieval(
        paired_cases=paired_cases,
        required_modalities=required_modalities,
    )
    geometry = _compute_paired_case_geometry(
        paired_cases=paired_cases,
        required_modalities=required_modalities,
    )

    output_path = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else _default_output_path(analysis_name, manifest_variant, args.checkpoint_name, feature_type)
    )
    payload = {
        "analysis_name": analysis_name,
        "manifest_path": str(manifest_path),
        "manifest_variant": manifest_variant,
        "checkpoint_name": args.checkpoint_name,
        "feature_type": feature_type,
        "organ": str(args.organ).strip().lower(),
        "pair_id_field": args.pair_id_field,
        "pairing_basis": "patient_id_matched_single_organ_validation",
        "required_modalities": list(required_modalities),
        "embeddings_npz": str(embeddings_path),
        "n_paired_cases": len(paired_cases),
        "paired_case_ids": [case["pair_id"] for case in paired_cases],
        "feature_count_by_pair": {
            case["pair_id"]: case["feature_count_by_modality"] for case in paired_cases
        },
        "single_organ_bidirectional_retrieval": retrieval,
        "paired_case_geometry": geometry,
        "interpretation_scope": (
            "Single-organ patient-id-matched validation only. This result does not replace the Cross-Modal "
            "multi-organ retrieval, LISI, or anatomy-over-modality metrics."
        ),
    }
    _write_json(output_path, payload)
    logger.info("Wrote single-organ alignment result to %s", output_path)


if __name__ == "__main__":
    main()
