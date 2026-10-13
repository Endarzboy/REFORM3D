#!/usr/bin/env python
"""Summarize Anatomical Generalization adaptation-gain and seed-variance evidence.

This script reads existing ``anatomical_generalization.json`` files. It does
not rerun feature extraction.

Examples:
    python -m reform3d.modules.anatomical_generalization.statistical_rigor_summary \
        --analysis-name mmwhs_ct_mr \
        --manifest-variant core

    python -m reform3d.modules.anatomical_generalization.statistical_rigor_summary \
        --analysis-name totalsegmenter_ct_mr_anchor \
        --feature-types cls avg_pool multilayer
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
    RESULT_JSON_NAME,
    get_output_paths,
    normalize_anatomical_dataset_name,
    normalize_feature_type,
    normalize_manifest_variant,
)


SEED_METRIC_FIELDS = (
    "balanced_accuracy",
    "target_query_accuracy",
    "nearest_centroid_balanced_accuracy",
    "nearest_centroid_target_query_accuracy",
    "adaptation_gain_balanced_accuracy",
    "adaptation_gain_target_query_accuracy",
    "relative_recovery_balanced_accuracy",
)

BOOTSTRAP_METRIC_FIELDS = (
    "balanced_accuracy",
    "nearest_centroid_balanced_accuracy",
    "adaptation_gain_balanced_accuracy",
    "relative_recovery_balanced_accuracy",
)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "analysis_name",
        "checkpoint_name",
        "feature_type",
        "status",
        "bootstrap_unit",
        "bootstrap_resamples",
        "n_effective_seeds",
        "seed_uncertainty_source",
        "n_evaluated_organs",
        "balanced_accuracy_mean",
        "balanced_accuracy_seed_std",
        "balanced_accuracy_ci_lower",
        "balanced_accuracy_ci_upper",
        "balanced_accuracy_ci_status",
        "nearest_centroid_balanced_accuracy_mean",
        "nearest_centroid_balanced_accuracy_seed_std",
        "nearest_centroid_balanced_accuracy_ci_lower",
        "nearest_centroid_balanced_accuracy_ci_upper",
        "nearest_centroid_balanced_accuracy_ci_status",
        "adaptation_gain_balanced_accuracy_mean",
        "adaptation_gain_balanced_accuracy_seed_std",
        "adaptation_gain_balanced_accuracy_ci_lower",
        "adaptation_gain_balanced_accuracy_ci_upper",
        "adaptation_gain_balanced_accuracy_ci_status",
        "relative_recovery_balanced_accuracy_mean",
        "relative_recovery_balanced_accuracy_seed_std",
        "relative_recovery_balanced_accuracy_ci_lower",
        "relative_recovery_balanced_accuracy_ci_upper",
        "relative_recovery_balanced_accuracy_ci_status",
        "result_json",
        "skip_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _discover_results(results_root: Path, feature_types: Sequence[str] | None) -> list[Path]:
    paths = sorted(results_root.glob(f"*/*/{RESULT_JSON_NAME}"))
    if feature_types:
        feature_set = set(feature_types)
        paths = [path for path in paths if path.parent.name in feature_set]
    return paths


def _coalesce(primary: Any, fallback: Any) -> Any:
    return primary if primary is not None else fallback


def _mean(values: Iterable[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def _std(values: Iterable[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if len(numeric) < 2:
        return None
    mean_value = sum(numeric) / len(numeric)
    squared_error = sum((value - mean_value) ** 2 for value in numeric)
    return (squared_error / (len(numeric) - 1)) ** 0.5


def _stable_seed_offset(*parts: str) -> int:
    text = "|".join(str(part) for part in parts)
    return sum((index + 1) * ord(character) for index, character in enumerate(text)) % 10000


def _bootstrap_mean_ci(
    values: Iterable[float | None],
    *,
    bootstrap_resamples: int,
    seed: int,
) -> Dict[str, Any]:
    numeric = [float(value) for value in values if value is not None]
    summary: Dict[str, Any] = {
        "n_bootstrap_units": len(numeric),
        "ci_lower": None,
        "ci_upper": None,
    }
    if not numeric:
        summary["status"] = "missing_bootstrap_units"
        return summary
    if bootstrap_resamples <= 0:
        summary["status"] = "bootstrap_disabled"
        return summary
    if len(numeric) < 2:
        summary["status"] = "insufficient_bootstrap_units"
        return summary

    array = np.asarray(numeric, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    bootstrap_means = np.empty(int(bootstrap_resamples), dtype=np.float64)
    for resample_index in range(int(bootstrap_resamples)):
        indices = rng.integers(0, array.size, size=array.size)
        bootstrap_means[resample_index] = float(np.mean(array[indices]))

    summary.update(
        {
            "status": "ok",
            "ci_lower": float(np.percentile(bootstrap_means, 2.5)),
            "ci_upper": float(np.percentile(bootstrap_means, 97.5)),
        }
    )
    return summary


def _extract_surface_b_seed_metric_values(resolved_seed_summary: Dict[str, Any], metric_name: str) -> list[float]:
    values: list[float] = []
    for seed_result in resolved_seed_summary.get("macro_seed_results") or []:
        value = seed_result.get(metric_name)
        if value is not None:
            values.append(float(value))
    return values


def _reconstruct_macro_seed_results(analysis: Dict[str, Any]) -> list[Dict[str, Any]]:
    seed_groups: dict[int, list[Dict[str, Any]]] = {}
    for organ_payload in (analysis.get("per_organ") or {}).values():
        status = organ_payload.get("status")
        if status not in (None, "ok"):
            continue
        for seed_result in organ_payload.get("seed_results") or []:
            seed = seed_result.get("seed")
            if seed is None:
                continue
            seed_groups.setdefault(int(seed), []).append(seed_result)

    macro_seed_results: list[Dict[str, Any]] = []
    for seed in sorted(seed_groups):
        seed_results = seed_groups[seed]
        macro_seed_results.append(
            {
                "seed": int(seed),
                "n_evaluated_organs": len(seed_results),
                **{
                    field: _mean(result.get(field) for result in seed_results)
                    for field in SEED_METRIC_FIELDS
                },
            }
        )
    return macro_seed_results


def _resolve_seed_summary(analysis: Dict[str, Any]) -> Dict[str, Any]:
    macro_seed_results = _reconstruct_macro_seed_results(analysis)
    if macro_seed_results:
        return {
            "macro_mean": {
                field: _mean(result.get(field) for result in macro_seed_results)
                for field in SEED_METRIC_FIELDS
            },
            "macro_mean_std": {
                field: _std(result.get(field) for result in macro_seed_results)
                for field in SEED_METRIC_FIELDS
            },
            "macro_seed_results": macro_seed_results,
            "n_effective_seeds": len(macro_seed_results),
            "seed_uncertainty_source": "reconstructed_from_per_organ_seed_results",
        }

    return {
        "macro_mean": analysis.get("macro_mean") or {},
        "macro_mean_std": analysis.get("macro_mean_std") or {},
        "macro_seed_results": analysis.get("macro_seed_results") or [],
        "n_effective_seeds": len(analysis.get("macro_seed_results") or analysis.get("evaluation_seeds") or []),
        "seed_uncertainty_source": "raw_top_level_summary",
    }


def _extract_row(path: Path, *, bootstrap_resamples: int, bootstrap_seed: int) -> Dict[str, Any]:
    payload = _load_json(path)
    checkpoint_name = str(payload.get("checkpoint_name") or path.parent.parent.name)
    feature_type = str(payload.get("feature_type") or path.parent.name)
    analysis_name = str(payload.get("analysis_name") or "unknown")
    headline = payload.get("headline_metrics", {})
    gain_headline = headline.get("task_1_low_shot_target_adaptation_gain") or {}
    surface_b = payload.get("surface_b_few_shot_transfer_evidence") or {}
    analysis = surface_b.get("few_shot_transfer_analysis") or {}
    resolved_seed_summary = _resolve_seed_summary(analysis)
    macro = resolved_seed_summary["macro_mean"]
    macro_std = resolved_seed_summary["macro_mean_std"]
    surface_b_bootstrap = {
        metric_name: _bootstrap_mean_ci(
            _extract_surface_b_seed_metric_values(resolved_seed_summary, metric_name),
            bootstrap_resamples=bootstrap_resamples,
            seed=bootstrap_seed + _stable_seed_offset(checkpoint_name, feature_type, metric_name),
        )
        for metric_name in BOOTSTRAP_METRIC_FIELDS
    }
    nearest_centroid_mean = _coalesce(
        macro.get("nearest_centroid_balanced_accuracy"),
        gain_headline.get("baseline_metric_value"),
    )
    has_baseline = nearest_centroid_mean is not None
    return {
        "analysis_name": analysis_name,
        "checkpoint_name": checkpoint_name,
        "feature_type": feature_type,
        "status": "ok" if has_baseline else "missing_nearest_centroid_baseline",
        "bootstrap_unit": "few_shot_support_query_seed",
        "bootstrap_resamples": int(bootstrap_resamples),
        "skip_reason": None if has_baseline else "rerun_anatomical_evaluation_to_materialize_baseline",
        "n_effective_seeds": resolved_seed_summary["n_effective_seeds"],
        "seed_uncertainty_source": resolved_seed_summary["seed_uncertainty_source"],
        "n_evaluated_organs": analysis.get("n_evaluated_organs"),
        "balanced_accuracy_mean": _coalesce(macro.get("balanced_accuracy"), gain_headline.get("adapted_metric_value")),
        "balanced_accuracy_seed_std": macro_std.get("balanced_accuracy"),
        "balanced_accuracy_ci_lower": surface_b_bootstrap["balanced_accuracy"].get("ci_lower"),
        "balanced_accuracy_ci_upper": surface_b_bootstrap["balanced_accuracy"].get("ci_upper"),
        "balanced_accuracy_ci_status": surface_b_bootstrap["balanced_accuracy"].get("status"),
        "nearest_centroid_balanced_accuracy_mean": nearest_centroid_mean,
        "nearest_centroid_balanced_accuracy_seed_std": macro_std.get("nearest_centroid_balanced_accuracy"),
        "nearest_centroid_balanced_accuracy_ci_lower": surface_b_bootstrap["nearest_centroid_balanced_accuracy"].get("ci_lower"),
        "nearest_centroid_balanced_accuracy_ci_upper": surface_b_bootstrap["nearest_centroid_balanced_accuracy"].get("ci_upper"),
        "nearest_centroid_balanced_accuracy_ci_status": surface_b_bootstrap["nearest_centroid_balanced_accuracy"].get("status"),
        "adaptation_gain_balanced_accuracy_mean": _coalesce(
            macro.get("adaptation_gain_balanced_accuracy"),
            gain_headline.get("primary_metric_value"),
        ),
        "adaptation_gain_balanced_accuracy_seed_std": macro_std.get("adaptation_gain_balanced_accuracy"),
        "adaptation_gain_balanced_accuracy_ci_lower": surface_b_bootstrap["adaptation_gain_balanced_accuracy"].get("ci_lower"),
        "adaptation_gain_balanced_accuracy_ci_upper": surface_b_bootstrap["adaptation_gain_balanced_accuracy"].get("ci_upper"),
        "adaptation_gain_balanced_accuracy_ci_status": surface_b_bootstrap["adaptation_gain_balanced_accuracy"].get("status"),
        "relative_recovery_balanced_accuracy_mean": _coalesce(
            macro.get("relative_recovery_balanced_accuracy"),
            gain_headline.get("relative_recovery"),
        ),
        "relative_recovery_balanced_accuracy_seed_std": macro_std.get("relative_recovery_balanced_accuracy"),
        "relative_recovery_balanced_accuracy_ci_lower": surface_b_bootstrap["relative_recovery_balanced_accuracy"].get("ci_lower"),
        "relative_recovery_balanced_accuracy_ci_upper": surface_b_bootstrap["relative_recovery_balanced_accuracy"].get("ci_upper"),
        "relative_recovery_balanced_accuracy_ci_status": surface_b_bootstrap["relative_recovery_balanced_accuracy"].get("status"),
        "result_json": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Anatomical Generalization fixed-checkpoint seed uncertainty and adaptation gain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m reform3d.modules.anatomical_generalization.statistical_rigor_summary --analysis-name mmwhs_ct_mr\n"
            "  python -m reform3d.modules.anatomical_generalization.statistical_rigor_summary --analysis-name totalsegmenter_ct_mr_anchor --feature-types cls avg_pool multilayer\n"
        ),
    )
    parser.add_argument("--analysis-name", required=True)
    parser.add_argument("--manifest-variant", default="core")
    parser.add_argument("--feature-types", nargs="+", default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    manifest_variant = normalize_manifest_variant(args.manifest_variant)
    feature_types = [normalize_feature_type(feature_type) for feature_type in args.feature_types] if args.feature_types else None
    analysis_name = normalize_anatomical_dataset_name(args.analysis_name)
    output_paths = get_output_paths(analysis_name, manifest_variant)
    result_paths = _discover_results(output_paths["results"], feature_types or FEATURE_TYPES)
    rows = [
        _extract_row(path, bootstrap_resamples=args.bootstrap_resamples, bootstrap_seed=args.seed)
        for path in tqdm(result_paths, desc="Anatomical statistical rigor")
    ]
    output_json = args.output_json or output_paths["results"] / "statistical_rigor" / "adaptation_gain_seed_summary.json"
    output_csv = args.output_csv or output_json.with_suffix(".csv")
    payload = {
        "analysis_scope": "anatomical_low_shot_target_adaptation_gain_fixed_checkpoint_uncertainty",
        "analysis_name": analysis_name,
        "manifest_variant": manifest_variant,
        "uncertainty_unit": "few_shot_support_query_seed",
        "table_uncertainty_unit": "few_shot_support_query_seed",
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "training_run_variance": "not_estimated_from_single_checkpoint_artifacts",
        "n_result_files": len(result_paths),
        "n_ok": sum(1 for row in rows if row.get("status") == "ok"),
        "n_missing_baseline": sum(1 for row in rows if row.get("status") != "ok"),
        "rows": rows,
    }
    _write_json(output_json, payload)
    _write_csv(output_csv, rows)
    print(f"Wrote {output_json}")
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
