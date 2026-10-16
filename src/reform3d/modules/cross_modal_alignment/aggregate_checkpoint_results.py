#!/usr/bin/env python

"""Aggregate Cross-Modal checkpoint results into comparison tables.

This script reads the per-checkpoint Cross-Modal primary metric files produced by
the evaluation pipeline and writes a compact comparison table in both CSV and
JSON formats under the Cross-Modal results root.

Usage examples:
    python -m reform3d.modules.cross_modal_alignment.aggregate_checkpoint_results \
        --analysis-name amos22_ct_mr \
        --manifest-variant core

    python -m reform3d.modules.cross_modal_alignment.aggregate_checkpoint_results \
        --results-dir /path/to/results
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from reform3d.core.checkpoints import resolve_checkpoint_config
from .config import (
    CHAOS_CROSS_MODAL_DATASET,
    get_checkpoint_comparison_csv_path,
    get_checkpoint_comparison_json_path,
    get_existing_results_root,
    get_output_paths,
    normalize_cross_modal_dataset_name,
    normalize_feature_type,
)


SINGLE_ORGAN_ALIGNMENT_FILE_NAME = "single_organ_alignment.json"


def _get_metric(metrics: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metrics:
            return metrics[key]
    return None


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _checkpoint_family(checkpoint_name: str) -> str:
    return resolve_checkpoint_config(checkpoint_name).model_family


def _checkpoint_crop_size(checkpoint_name: str) -> int | None:
    return resolve_checkpoint_config(checkpoint_name).crop_size


def _collect_rows(results_dir: Path, feature_type: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if feature_type == "cls":
        metrics_paths = sorted(results_dir.glob("*/primary_metrics.json"))
    else:
        metrics_paths = sorted(results_dir.glob(f"*/{feature_type}/primary_metrics.json"))

    for metrics_path in metrics_paths:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        checkpoint_name = payload.get("checkpoint_name") or metrics_path.parent.name
        margin = payload.get("anatomy_over_modality_margin", {})
        retrieval = payload.get("balanced_cross_modal_retrieval", {})
        lisi = payload.get("balanced_lisi", {})
        margin_overall = margin.get("overall") or {}
        ct_to_mr = retrieval.get("ct_to_mr") or {}
        mr_to_ct = retrieval.get("mr_to_ct") or {}
        bidirectional_mean = retrieval.get("bidirectional_mean") or {}
        ct_to_mr_overall = ct_to_mr.get("overall") or {}
        mr_to_ct_overall = mr_to_ct.get("overall") or {}
        overall_lisi = lisi.get("overall", {})
        modality_ilisi = overall_lisi.get("modality_ilisi", {})
        organ_clisi = overall_lisi.get("organ_clisi", {})
        bootstrap_ci = margin_overall.get("bootstrap_95ci") or {}

        row = {
            "checkpoint_name": checkpoint_name,
            "checkpoint_family": _checkpoint_family(checkpoint_name),
            "crop_size": _checkpoint_crop_size(checkpoint_name),
            "margin_status": margin.get("status"),
            "margin_reason": margin.get("reason"),
            "mean_margin": margin_overall.get("mean_margin"),
            "median_margin": margin_overall.get("median_margin"),
            "fraction_positive_margin": margin_overall.get("fraction_positive_margin"),
            "margin_ci_lower": bootstrap_ci.get("lower"),
            "margin_ci_upper": bootstrap_ci.get("upper"),
            "retrieval_status": retrieval.get("status"),
            "retrieval_top@1": _get_metric(bidirectional_mean, "top@1", "top_at_1", "hit_at_1"),
            "retrieval_top@5": _get_metric(bidirectional_mean, "top@5", "top_at_5", "hit_at_5"),
            "retrieval_map": bidirectional_mean.get("map"),
            "ct_to_mr_status": ct_to_mr.get("status"),
            "ct_to_mr_reason": ct_to_mr.get("reason"),
            "ct_to_mr_top@1": _get_metric(ct_to_mr_overall, "top@1", "top_at_1", "hit_at_1"),
            "ct_to_mr_top@5": _get_metric(ct_to_mr_overall, "top@5", "top_at_5", "hit_at_5"),
            "ct_to_mr_map": ct_to_mr_overall.get("map"),
            "mr_to_ct_status": mr_to_ct.get("status"),
            "mr_to_ct_reason": mr_to_ct.get("reason"),
            "mr_to_ct_top@1": _get_metric(mr_to_ct_overall, "top@1", "top_at_1", "hit_at_1"),
            "mr_to_ct_top@5": _get_metric(mr_to_ct_overall, "top@5", "top_at_5", "hit_at_5"),
            "mr_to_ct_map": mr_to_ct_overall.get("map"),
            "lisi_status": lisi.get("status"),
            "lisi_reason": lisi.get("reason"),
            "modality_ilisi": modality_ilisi.get("mean"),
            "organ_clisi": organ_clisi.get("mean"),
            "margin_queries": margin.get("n_queries"),
            "retrieval_ct_queries": ct_to_mr.get("n_queries"),
            "retrieval_mr_queries": mr_to_ct.get("n_queries"),
            "lisi_samples": lisi.get("n_samples"),
            "metrics_path": str(metrics_path),
        }
        rows.append(row)
    return rows


def _single_organ_metrics_paths(results_dir: Path, feature_type: str) -> List[Path]:
    if feature_type == "cls":
        return sorted(results_dir.glob(f"*/{SINGLE_ORGAN_ALIGNMENT_FILE_NAME}"))
    return sorted(results_dir.glob(f"*/{feature_type}/{SINGLE_ORGAN_ALIGNMENT_FILE_NAME}"))


def _collect_single_organ_rows(results_dir: Path, feature_type: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metrics_path in _single_organ_metrics_paths(results_dir, feature_type):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        checkpoint_name = payload.get("checkpoint_name") or (
            metrics_path.parent.name if feature_type == "cls" else metrics_path.parent.parent.name
        )
        retrieval = payload.get("single_organ_bidirectional_retrieval") or {}
        geometry = payload.get("paired_case_geometry") or {}
        ct_to_mr = retrieval.get("ct_to_mr") or {}
        mr_to_ct = retrieval.get("mr_to_ct") or {}
        bidirectional_mean = retrieval.get("bidirectional_mean") or {}
        ct_to_mr_overall = ct_to_mr.get("overall") or {}
        mr_to_ct_overall = mr_to_ct.get("overall") or {}

        row = {
            "checkpoint_name": checkpoint_name,
            "checkpoint_family": _checkpoint_family(checkpoint_name),
            "crop_size": _checkpoint_crop_size(checkpoint_name),
            "evaluation_scope": "single_organ_patient_matched",
            "organ": payload.get("organ"),
            "pairing_basis": payload.get("pairing_basis"),
            "n_paired_cases": payload.get("n_paired_cases"),
            "margin_status": "not_applicable",
            "margin_reason": "single_organ_surface_has_no_different_organ_negative_pool",
            "mean_margin": None,
            "median_margin": None,
            "fraction_positive_margin": None,
            "margin_ci_lower": None,
            "margin_ci_upper": None,
            "retrieval_status": retrieval.get("status"),
            "retrieval_top@1": _get_metric(bidirectional_mean, "top@1", "top_at_1", "hit_at_1"),
            "retrieval_top@5": _get_metric(bidirectional_mean, "top@5", "top_at_5", "hit_at_5"),
            "retrieval_map": bidirectional_mean.get("map"),
            "ct_to_mr_status": ct_to_mr.get("status"),
            "ct_to_mr_reason": ct_to_mr.get("reason"),
            "ct_to_mr_top@1": _get_metric(ct_to_mr_overall, "top@1", "top_at_1", "hit_at_1"),
            "ct_to_mr_top@5": _get_metric(ct_to_mr_overall, "top@5", "top_at_5", "hit_at_5"),
            "ct_to_mr_map": ct_to_mr_overall.get("map"),
            "mr_to_ct_status": mr_to_ct.get("status"),
            "mr_to_ct_reason": mr_to_ct.get("reason"),
            "mr_to_ct_top@1": _get_metric(mr_to_ct_overall, "top@1", "top_at_1", "hit_at_1"),
            "mr_to_ct_top@5": _get_metric(mr_to_ct_overall, "top@5", "top_at_5", "hit_at_5"),
            "mr_to_ct_map": mr_to_ct_overall.get("map"),
            "lisi_status": "not_applicable",
            "lisi_reason": "single_organ_surface_has_no_multi_organ_lisi_axis",
            "modality_ilisi": None,
            "organ_clisi": None,
            "margin_queries": None,
            "retrieval_ct_queries": ct_to_mr.get("n_queries"),
            "retrieval_mr_queries": mr_to_ct.get("n_queries"),
            "lisi_samples": None,
            "paired_case_linear_cka": geometry.get("paired_case_linear_cka"),
            "paired_case_mean_cosine_distance": geometry.get("paired_case_mean_cosine_distance"),
            "paired_case_median_cosine_distance": geometry.get("paired_case_median_cosine_distance"),
            "metrics_path": str(metrics_path),
        }
        rows.append(row)
    return rows


def _sort_metric_value(item: Dict[str, Any], key: str, reverse: bool) -> float:
    value = item.get(key)
    if value is None:
        return float("-inf") if reverse else float("inf")
    return float(value)


def _best_row(rows: List[Dict[str, Any]], key: str, reverse: bool = True) -> Dict[str, Any] | None:
    valid_rows = [row for row in rows if row.get(key) is not None]
    if not valid_rows:
        return None
    return max(valid_rows, key=lambda item: item[key]) if reverse else min(valid_rows, key=lambda item: item[key])


def _assign_rank(rows: List[Dict[str, Any]], metric_key: str, rank_key: str, reverse: bool) -> None:
    valid_rows = [row for row in rows if row.get(metric_key) is not None]
    for row in rows:
        row[rank_key] = None
    ordered_rows = sorted(valid_rows, key=lambda item: _sort_metric_value(item, metric_key, reverse), reverse=reverse)
    for index, row in enumerate(ordered_rows, start=1):
        row[rank_key] = index


def _rank_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = [dict(row) for row in rows]

    _assign_rank(ranked, "mean_margin", "rank_mean_margin", reverse=True)
    _assign_rank(ranked, "fraction_positive_margin", "rank_fraction_positive_margin", reverse=True)
    _assign_rank(ranked, "retrieval_top@1", "rank_retrieval_top@1", reverse=True)
    _assign_rank(ranked, "retrieval_top@5", "rank_retrieval_top@5", reverse=True)
    _assign_rank(ranked, "retrieval_map", "rank_retrieval_map", reverse=True)
    _assign_rank(ranked, "modality_ilisi", "rank_modality_ilisi", reverse=True)
    _assign_rank(ranked, "organ_clisi", "rank_organ_clisi", reverse=False)

    return sorted(
        ranked,
        key=lambda item: (
            item["rank_retrieval_map"] is None,
            item["rank_retrieval_map"] if item["rank_retrieval_map"] is not None else float("inf"),
            item["checkpoint_name"],
        ),
    )


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate finished Cross-Modal checkpoint metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m reform3d.modules.cross_modal_alignment.aggregate_checkpoint_results --analysis-name amos22_ct_mr --manifest-variant core\n"
            "  python -m reform3d.modules.cross_modal_alignment.aggregate_checkpoint_results --results-dir /path/to/results\n"
        ),
    )
    parser.add_argument("--analysis-name", default="amos22_ct_mr", help="Cross-Modal analysis namespace")
    parser.add_argument("--manifest-variant", default="core", help="Cross-Modal manifest variant")
    parser.add_argument("--results-dir", default=None, help="Optional explicit results directory override")
    parser.add_argument("--feature-type", default="cls", help="Feature family to aggregate")
    args = parser.parse_args()
    args.analysis_name = normalize_cross_modal_dataset_name(args.analysis_name)
    feature_type = normalize_feature_type(args.feature_type)

    if args.results_dir:
        source_results_dir = Path(args.results_dir).resolve()
        output_results_dir = source_results_dir
    else:
        source_results_dir = get_existing_results_root(args.analysis_name, args.manifest_variant)
        output_results_dir = get_output_paths(args.analysis_name, args.manifest_variant)["results"]

    if args.analysis_name == CHAOS_CROSS_MODAL_DATASET:
        rows = _collect_single_organ_rows(source_results_dir, feature_type)
        aggregation_source = SINGLE_ORGAN_ALIGNMENT_FILE_NAME
    else:
        rows = _collect_rows(source_results_dir, feature_type)
        aggregation_source = "primary_metrics.json"
    if not rows:
        raise FileNotFoundError(
            f"No per-checkpoint Cross-Modal metric files found under {source_results_dir} "
            f"for feature type {feature_type} using {aggregation_source}"
        )

    ranked_rows = _rank_rows(rows)
    csv_path = get_checkpoint_comparison_csv_path(output_results_dir, feature_type)
    json_path = get_checkpoint_comparison_json_path(output_results_dir, feature_type)

    _write_csv(csv_path, ranked_rows)
    _write_json(
        json_path,
        {
            "results_dir": str(source_results_dir),
            "summary_output_dir": str(output_results_dir),
            "aggregation_source": aggregation_source,
            "n_checkpoints": len(ranked_rows),
            "rows": ranked_rows,
            "best_by_mean_margin": _best_row(ranked_rows, "mean_margin"),
            "best_by_retrieval_top@1": _best_row(ranked_rows, "retrieval_top@1"),
            "best_by_retrieval_top@5": _best_row(ranked_rows, "retrieval_top@5"),
            "best_by_retrieval_map": _best_row(ranked_rows, "retrieval_map"),
            "best_by_modality_ilisi": _best_row(ranked_rows, "modality_ilisi"),
            "best_by_organ_clisi": _best_row(ranked_rows, "organ_clisi", reverse=False),
        },
    )

    print(f"Wrote CSV summary to {csv_path}")
    print(f"Wrote JSON summary to {json_path}")


if __name__ == "__main__":
    main()
