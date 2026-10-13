#!/usr/bin/env python
"""Post-hoc retrieval uncertainty for Cross-Modal single-organ alignment surfaces.

This script reads existing ``single_organ_alignment.json`` payloads. It
does not rerun feature extraction. It materializes per-query retrieval rows,
query-bootstrap CIs, and optional paired checkpoint differences against a
reference checkpoint using the patient-matched single-organ retrieval surface.

Examples:
    python -m reform3d.modules.cross_modal_alignment.single_organ_retrieval_uncertainty \
        --analysis-name chaos_ct_mr \
        --manifest-variant core \
        --feature-type cls \
        --reference-checkpoint 3dino \
        --bootstrap-resamples 1000

    python -m reform3d.modules.cross_modal_alignment.single_organ_retrieval_uncertainty \
        --analysis-name chaos_ct_mr \
        --checkpoints med3dino_rel_96 med3dino_sa_96 3dino
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_: Any):  # type: ignore[no-redef]
        return iterable

from .config import (
    FEATURE_TYPES,
    get_output_paths,
    get_results_root_candidates,
    normalize_cross_modal_dataset_name,
    normalize_feature_type,
    normalize_manifest_variant,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _single_organ_payload_path(results_root: Path, checkpoint_name: str, feature_type: str) -> Path:
    checkpoint_root = results_root / checkpoint_name
    if feature_type == "cls":
        return checkpoint_root / "single_organ_alignment.json"
    return checkpoint_root / feature_type / "single_organ_alignment.json"


def _resolve_single_organ_payload_path(results_roots: Sequence[Path], checkpoint_name: str, feature_type: str) -> Path:
    for results_root in results_roots:
        candidate = _single_organ_payload_path(results_root, checkpoint_name, feature_type)
        if candidate.exists():
            return candidate
    fallback_root = results_roots[0] if results_roots else Path()
    return _single_organ_payload_path(fallback_root, checkpoint_name, feature_type)


def _discover_checkpoints(results_roots: Sequence[Path], feature_type: str) -> list[str]:
    checkpoints = []
    seen = set()
    for results_root in results_roots:
        for checkpoint_dir in sorted(results_root.iterdir() if results_root.exists() else []):
            if not checkpoint_dir.is_dir() or checkpoint_dir.name in seen:
                continue
            if _single_organ_payload_path(results_root, checkpoint_dir.name, feature_type).exists():
                checkpoints.append(checkpoint_dir.name)
                seen.add(checkpoint_dir.name)
    return checkpoints


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _query_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row["pair_id"]), str(row["direction"]))


def _average_precision_from_rank(rank: int | float | None) -> float | None:
    if rank is None:
        return None
    rank_value = int(rank)
    if rank_value <= 0:
        return None
    return 1.0 / float(rank_value)


def _flatten_query_rows(checkpoint_name: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    retrieval = payload.get("single_organ_bidirectional_retrieval") or {}
    rows: list[dict[str, Any]] = []
    for direction_key in ("ct_to_mr", "mr_to_ct"):
        direction = retrieval.get(direction_key) or {}
        for per_pair in direction.get("per_pair", []):
            positive_rank = per_pair.get("positive_rank")
            if positive_rank is None:
                continue
            ap = _average_precision_from_rank(positive_rank)
            rows.append(
                {
                    "checkpoint_name": checkpoint_name,
                    "direction": direction_key,
                    "pair_id": str(per_pair.get("pair_id")),
                    "query_modality": direction.get("query_modality"),
                    "target_modality": direction.get("target_modality"),
                    "positive_rank": int(positive_rank),
                    "positive_similarity": per_pair.get("positive_similarity"),
                    "map": ap,
                    "top@1": 1.0 if int(positive_rank) <= 1 else 0.0,
                    "top@5": 1.0 if int(positive_rank) <= 5 else 0.0,
                }
            )
    return rows


def _bootstrap_metric_summary(values: np.ndarray, bootstrap_resamples: int, seed: int) -> dict[str, Any]:
    summary = {"mean": float(np.mean(values))}
    if bootstrap_resamples > 0 and values.size > 1:
        rng = np.random.default_rng(int(seed))
        bootstrap_values = np.empty(int(bootstrap_resamples), dtype=np.float64)
        for resample_index in tqdm(range(int(bootstrap_resamples)), desc="Single-organ metric bootstrap", unit="resample", dynamic_ncols=True, leave=False):
            indices = rng.integers(0, values.size, size=values.size)
            bootstrap_values[resample_index] = float(np.mean(values[indices]))
        summary.update(
            {
                "ci_lower": float(np.percentile(bootstrap_values, 2.5)),
                "ci_upper": float(np.percentile(bootstrap_values, 97.5)),
                "bootstrap_resamples": int(bootstrap_resamples),
            }
        )
    return summary


def _extract_ci_summary(checkpoint_name: str, query_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    row: dict[str, Any] = {"checkpoint_name": checkpoint_name, "status": "ok" if query_rows else "no_queries"}
    for metric_key in ("map", "top@1", "top@5"):
        safe_key = metric_key.replace("@", "at")
        values = np.asarray([float(query_row[metric_key]) for query_row in query_rows if query_row.get(metric_key) is not None], dtype=np.float64)
        if values.size == 0:
            row[f"{safe_key}_mean"] = None
            row[f"{safe_key}_ci_lower"] = None
            row[f"{safe_key}_ci_upper"] = None
            continue
        summary = _bootstrap_metric_summary(values, bootstrap_resamples=0, seed=0)
        row[f"{safe_key}_mean"] = summary["mean"]
        row[f"{safe_key}_ci_lower"] = None
        row[f"{safe_key}_ci_upper"] = None
    return row


def _stable_seed_offset(*parts: str) -> int:
    text = "|".join(str(part) for part in parts)
    return sum((index + 1) * ord(character) for index, character in enumerate(text)) % 10000


def _bootstrap_query_cis(
    summary_rows: list[dict[str, Any]],
    query_rows_by_checkpoint: dict[str, list[dict[str, Any]]],
    bootstrap_resamples: int,
    seed: int,
) -> None:
    if bootstrap_resamples <= 0:
        return
    for row in summary_rows:
        checkpoint_name = row["checkpoint_name"]
        query_rows = query_rows_by_checkpoint.get(checkpoint_name, [])
        for metric_key in ("map", "top@1", "top@5"):
            safe_key = metric_key.replace("@", "at")
            values = np.asarray(
                [float(query_row[metric_key]) for query_row in query_rows if query_row.get(metric_key) is not None],
                dtype=np.float64,
            )
            if values.size <= 1:
                continue
            summary = _bootstrap_metric_summary(
                values,
                bootstrap_resamples=bootstrap_resamples,
                seed=seed + _stable_seed_offset(checkpoint_name, metric_key),
            )
            row[f"{safe_key}_ci_lower"] = summary.get("ci_lower")
            row[f"{safe_key}_ci_upper"] = summary.get("ci_upper")


def _bootstrap_paired_difference(
    checkpoint_rows: Sequence[dict[str, Any]],
    reference_rows: Sequence[dict[str, Any]],
    metric_key: str,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any] | None:
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
        for resample_index in tqdm(range(int(bootstrap_resamples)), desc="Single-organ paired-difference bootstrap", unit="resample", dynamic_ncols=True, leave=False):
            indices = rng.integers(0, values.size, size=values.size)
            bootstrap_values[resample_index] = float(np.mean(values[indices]))
        sign_flip_means = np.empty(int(bootstrap_resamples), dtype=np.float64)
        for resample_index in tqdm(range(int(bootstrap_resamples)), desc="Single-organ sign-flip bootstrap", unit="resample", dynamic_ncols=True, leave=False):
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Cross-Modal single-organ retrieval uncertainty from saved patient-matched alignment payloads",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m reform3d.modules.cross_modal_alignment.single_organ_retrieval_uncertainty --analysis-name chaos_ct_mr --feature-type cls --reference-checkpoint 3dino --bootstrap-resamples 1000\n"
            "  python -m reform3d.modules.cross_modal_alignment.single_organ_retrieval_uncertainty --analysis-name chaos_ct_mr --checkpoints med3dino_rel_96 med3dino_sa_96 3dino\n"
        ),
    )
    parser.add_argument("--analysis-name", required=True)
    parser.add_argument("--manifest-variant", default="core")
    parser.add_argument("--feature-type", default="cls", choices=FEATURE_TYPES)
    parser.add_argument("--checkpoints", nargs="+", default=None)
    parser.add_argument("--reference-checkpoint", default=None)
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
    results_roots = get_results_root_candidates(analysis_name, manifest_variant)
    checkpoints = args.checkpoints or _discover_checkpoints(results_roots, feature_type)

    query_rows_by_checkpoint: dict[str, list[dict[str, Any]]] = {}
    payloads_by_checkpoint: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, str]] = []
    for checkpoint_name in tqdm(checkpoints, desc="Single-organ retrieval uncertainty"):
        payload_path = _resolve_single_organ_payload_path(results_roots, checkpoint_name, feature_type)
        if not payload_path.exists():
            skipped.append({"checkpoint_name": checkpoint_name, "reason": "missing_single_organ_alignment_payload"})
            continue
        payload = _load_json(payload_path)
        payloads_by_checkpoint[checkpoint_name] = payload
        query_rows_by_checkpoint[checkpoint_name] = _flatten_query_rows(checkpoint_name, payload)

    reference_checkpoint = args.reference_checkpoint
    if reference_checkpoint is None and "3dino" in query_rows_by_checkpoint:
        reference_checkpoint = "3dino"

    summary_rows = [_extract_ci_summary(checkpoint_name, query_rows_by_checkpoint[checkpoint_name]) for checkpoint_name in query_rows_by_checkpoint]
    _bootstrap_query_cis(summary_rows, query_rows_by_checkpoint, args.bootstrap_resamples, args.seed)

    paired_differences: dict[str, dict[str, Any]] = {}
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

    representative_payload = next(iter(payloads_by_checkpoint.values()), {})
    output_json = args.output_json or output_paths["results"] / "statistical_rigor" / f"single_organ_retrieval_uncertainty_{feature_type}.json"
    summary_csv = args.summary_csv or output_json.with_suffix(".csv")
    query_csv = args.query_csv or output_json.with_name(output_json.stem + "_query_rows.csv")
    all_query_rows = [row for rows in query_rows_by_checkpoint.values() for row in rows]
    payload = {
        "analysis_scope": "cross_modal_single_organ_patient_matched_retrieval_fixed_checkpoint_query_uncertainty",
        "analysis_name": analysis_name,
        "interpretation_scope": representative_payload.get("interpretation_scope"),
        "manifest_path": representative_payload.get("manifest_path"),
        "manifest_variant": manifest_variant,
        "feature_type": feature_type,
        "organ": representative_payload.get("organ"),
        "pair_id_field": representative_payload.get("pair_id_field"),
        "pairing_basis": representative_payload.get("pairing_basis"),
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
