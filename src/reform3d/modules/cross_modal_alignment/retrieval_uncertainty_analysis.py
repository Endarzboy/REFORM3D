#!/usr/bin/env python
"""Post-hoc retrieval uncertainty for Cross-Modal Alignment.

The script reads existing Cross-Modal organ embedding bundles and manifests. It
does not rerun feature extraction. It materializes per-query AP/top-k rows,
query-bootstrap CIs, and optional paired checkpoint differences against a
reference checkpoint.

Examples:
    python -m reform3d.modules.cross_modal_alignment.retrieval_uncertainty_analysis \
        --analysis-name amos22_ct_mr \
        --manifest-variant core \
        --feature-type cls \
        --reference-checkpoint 3dino \
        --bootstrap-resamples 1000

    python -m reform3d.modules.cross_modal_alignment.retrieval_uncertainty_analysis \
        --analysis-name amos22_ct_mr \
        --checkpoints med3dino_rel_96 med3dino_sa_96 3dino
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_: Any):  # type: ignore[no-redef]
        return iterable

from .config import (
    FEATURE_TYPES,
    get_feature_root_candidates,
    get_checkpoint_feature_npz_path,
    get_output_paths,
    get_cross_modal_manifest_path,
    normalize_cross_modal_dataset_name,
    normalize_feature_type,
    normalize_manifest_variant,
    resolve_existing_checkpoint_feature_npz_path,
)
from .data_loader import load_cross_modal_manifest
from .retrieval_metrics import compute_bidirectional_cross_modal_retrieval


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_embedding_bundle(npz_path: Path, samples: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    payload = np.load(npz_path, allow_pickle=True)
    features = np.asarray(payload["features"], dtype=np.float32)
    if "sample_ids" in payload:
        sample_ids = [str(item) for item in payload["sample_ids"].tolist()]
    else:
        sample_ids = [str(sample["sample_id"]) for sample in samples]
    if features.shape[0] != len(sample_ids):
        raise ValueError(f"Feature rows and sample_ids differ for {npz_path}")
    return {sample_id: features[index] for index, sample_id in enumerate(sample_ids)}


def _supported_organs(samples: Sequence[Dict[str, Any]], required_modalities: Sequence[str]) -> list[str]:
    organs_by_modality: Dict[str, set[str]] = {str(modality): set() for modality in required_modalities}
    for sample in samples:
        modality = str(sample.get("modality"))
        organ = sample.get("primary_organ")
        if modality in organs_by_modality and organ:
            organs_by_modality[modality].add(str(organ))
    common_organs: set[str] | None = None
    for organs in organs_by_modality.values():
        common_organs = set(organs) if common_organs is None else common_organs & organs
    return sorted(common_organs or [])


def _discover_checkpoints(features_roots: Sequence[Path], feature_type: str) -> list[str]:
    checkpoints = []
    seen = set()
    for features_root in features_roots:
        for checkpoint_dir in sorted(features_root.iterdir() if features_root.exists() else []):
            if not checkpoint_dir.is_dir() or checkpoint_dir.name in seen:
                continue
            if get_checkpoint_feature_npz_path(features_root, checkpoint_dir.name, feature_type).exists():
                checkpoints.append(checkpoint_dir.name)
                seen.add(checkpoint_dir.name)
    return checkpoints


def _query_key(row: Dict[str, Any]) -> tuple[str, str, str]:
    return (str(row["sample_id"]), str(row["query_modality"]), str(row["target_modality"]))


def _flatten_query_rows(checkpoint_name: str, retrieval: Dict[str, Any]) -> list[Dict[str, Any]]:
    rows = []
    for direction_key in ("ct_to_mr", "mr_to_ct"):
        direction = retrieval.get(direction_key, {})
        for row in direction.get("query_results", []):
            rows.append({"checkpoint_name": checkpoint_name, "direction": direction_key, **row})
    return rows


def _bootstrap_paired_difference(
    checkpoint_rows: Sequence[Dict[str, Any]],
    reference_rows: Sequence[Dict[str, Any]],
    metric_key: str,
    bootstrap_resamples: int,
    seed: int,
) -> Dict[str, Any] | None:
    reference_by_key = {_query_key(row): row for row in reference_rows if row.get(metric_key) is not None}
    paired_differences = []
    for row in checkpoint_rows:
        key = _query_key(row)
        if key not in reference_by_key or row.get(metric_key) is None:
            continue
        paired_differences.append(float(row[metric_key]) - float(reference_by_key[key][metric_key]))
    if not paired_differences:
        return None

    values = np.asarray(paired_differences, dtype=np.float64)
    observed_difference = float(np.mean(values))
    result = {
        "mean_difference": observed_difference,
        "n_paired_queries": int(values.size),
        "bootstrap_unit": "paired_query",
    }
    if bootstrap_resamples > 0 and values.size > 1:
        rng = np.random.default_rng(int(seed))
        bootstrap_values = np.empty(int(bootstrap_resamples), dtype=np.float64)
        for resample_index in range(int(bootstrap_resamples)):
            indices = rng.integers(0, values.size, size=values.size)
            bootstrap_values[resample_index] = float(np.mean(values[indices]))
        sign_flip_means = np.empty(int(bootstrap_resamples), dtype=np.float64)
        for resample_index in range(int(bootstrap_resamples)):
            signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float64), size=values.size, replace=True)
            sign_flip_means[resample_index] = float(np.mean(values * signs))
        result.update(
            {
                "ci_lower": float(np.percentile(bootstrap_values, 2.5)),
                "ci_upper": float(np.percentile(bootstrap_values, 97.5)),
                "bootstrap_resamples": int(bootstrap_resamples),
                "sign_flip_resamples": int(bootstrap_resamples),
                "two_sided_sign_flip_p": float(
                    (1.0 + np.sum(np.abs(sign_flip_means) >= abs(observed_difference)))
                    / (float(bootstrap_resamples) + 1.0)
                ),
            }
        )
    return result


def _extract_ci_summary(checkpoint_name: str, retrieval: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"checkpoint_name": checkpoint_name, "status": retrieval.get("status")}
    bidirectional = retrieval.get("bidirectional_mean") or {}
    bootstrap = retrieval.get("bidirectional_bootstrap_95ci") or {}
    for metric_key in ("map", "top@1", "top@5"):
        safe_key = metric_key.replace("@", "at")
        row[f"{safe_key}_mean"] = bidirectional.get(metric_key)
        metric_ci = bootstrap.get(metric_key) or {}
        row[f"{safe_key}_ci_lower"] = metric_ci.get("ci_lower")
        row[f"{safe_key}_ci_upper"] = metric_ci.get("ci_upper")
    return row


def _stable_seed_offset(*parts: str) -> int:
    text = "|".join(str(part) for part in parts)
    return sum((index + 1) * ord(character) for index, character in enumerate(text)) % 10000


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Cross-Modal retrieval uncertainty from saved organ embedding bundles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m reform3d.modules.cross_modal_alignment.retrieval_uncertainty_analysis --analysis-name amos22_ct_mr --reference-checkpoint 3dino --bootstrap-resamples 1000\n"
            "  python -m reform3d.modules.cross_modal_alignment.retrieval_uncertainty_analysis --analysis-name amos22_ct_mr --checkpoints med3dino_rel_96 med3dino_sa_96 3dino\n"
        ),
    )
    parser.add_argument("--analysis-name", required=True)
    parser.add_argument("--manifest-variant", default="core")
    parser.add_argument("--feature-type", default="cls", choices=FEATURE_TYPES)
    parser.add_argument("--checkpoints", nargs="+", default=None)
    parser.add_argument("--reference-checkpoint", default=None)
    parser.add_argument("--required-modalities", nargs="+", default=["ct", "mr"])
    parser.add_argument("--max-queries-per-organ", type=int, default=None)
    parser.add_argument("--max-targets-per-organ", type=int, default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--query-csv", type=Path, default=None)
    args = parser.parse_args()

    analysis_name = normalize_cross_modal_dataset_name(args.analysis_name)
    manifest_variant = normalize_manifest_variant(args.manifest_variant)
    feature_type = normalize_feature_type(args.feature_type)
    output_paths = get_output_paths(analysis_name, manifest_variant)
    manifest_path = get_cross_modal_manifest_path(analysis_name, "sampled", manifest_variant)
    samples = load_cross_modal_manifest(manifest_path)
    supported_organs = _supported_organs(samples, args.required_modalities)
    feature_roots = get_feature_root_candidates(analysis_name, manifest_variant)
    checkpoints = args.checkpoints or _discover_checkpoints(feature_roots, feature_type)

    retrieval_by_checkpoint: Dict[str, Dict[str, Any]] = {}
    query_rows_by_checkpoint: Dict[str, list[Dict[str, Any]]] = {}
    skipped = []
    for checkpoint_name in tqdm(checkpoints, desc="Cross-Modal retrieval uncertainty"):
        embeddings_path = resolve_existing_checkpoint_feature_npz_path(
            analysis_name,
            manifest_variant,
            checkpoint_name,
            feature_type,
        )
        if not embeddings_path.exists():
            skipped.append({"checkpoint_name": checkpoint_name, "reason": "missing_embedding_bundle"})
            continue
        features_by_id = _load_embedding_bundle(embeddings_path, samples)
        retrieval = compute_bidirectional_cross_modal_retrieval(
            features_by_id,
            samples,
            supported_organs=supported_organs,
            max_queries_per_organ=args.max_queries_per_organ,
            max_targets_per_organ=args.max_targets_per_organ,
            seed=args.seed,
            return_query_results=True,
            bootstrap_resamples=args.bootstrap_resamples,
        )
        retrieval_by_checkpoint[checkpoint_name] = retrieval
        query_rows_by_checkpoint[checkpoint_name] = _flatten_query_rows(checkpoint_name, retrieval)

    reference_checkpoint = args.reference_checkpoint
    if reference_checkpoint is None and "3dino" in retrieval_by_checkpoint:
        reference_checkpoint = "3dino"

    paired_differences: Dict[str, Dict[str, Any]] = {}
    if reference_checkpoint and reference_checkpoint in query_rows_by_checkpoint:
        reference_rows = query_rows_by_checkpoint[reference_checkpoint]
        for checkpoint_name, checkpoint_rows in query_rows_by_checkpoint.items():
            if checkpoint_name == reference_checkpoint:
                continue
            paired_differences[checkpoint_name] = {}
            for metric_key in ("map", "top@1", "top@5"):
                difference = _bootstrap_paired_difference(
                    checkpoint_rows,
                    reference_rows,
                    metric_key=metric_key,
                    bootstrap_resamples=args.bootstrap_resamples,
                    seed=args.seed + _stable_seed_offset(checkpoint_name, metric_key),
                )
                if difference is not None:
                    paired_differences[checkpoint_name][metric_key] = difference

    output_json = args.output_json or output_paths["results"] / "statistical_rigor" / f"retrieval_uncertainty_{feature_type}.json"
    summary_csv = args.summary_csv or output_json.with_suffix(".csv")
    query_csv = args.query_csv or output_json.with_name(output_json.stem + "_query_rows.csv")
    all_query_rows = [row for rows in query_rows_by_checkpoint.values() for row in rows]
    summary_rows = [_extract_ci_summary(checkpoint, retrieval) for checkpoint, retrieval in retrieval_by_checkpoint.items()]
    payload = {
        "analysis_scope": "cross_modal_retrieval_fixed_checkpoint_query_uncertainty",
        "analysis_name": analysis_name,
        "manifest_path": str(manifest_path),
        "manifest_variant": manifest_variant,
        "feature_type": feature_type,
        "supported_organs": supported_organs,
        "bootstrap_unit": "query_for_ci; paired_query_for_checkpoint_differences",
        "training_run_variance": "not_estimated_from_single_checkpoint_artifacts",
        "reference_checkpoint": reference_checkpoint,
        "summary_rows": summary_rows,
        "paired_differences_vs_reference": paired_differences,
        "skipped": skipped,
    }
    _write_json(output_json, payload)
    _write_csv(summary_csv, summary_rows)
    _write_csv(query_csv, all_query_rows)
    print(f"Wrote {output_json}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {query_csv}")


if __name__ == "__main__":
    main()
