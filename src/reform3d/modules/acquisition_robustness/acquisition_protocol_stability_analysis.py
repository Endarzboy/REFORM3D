#!/usr/bin/env python
"""Analyze controlled acquisition-protocol sensitivity from saved artifacts.

The analysis consumes extension ``perturbation_robustness.json`` files and
sibling ``matched_semantic_transfer.json`` files. It does not extract features
or rerun encoders.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import kendalltau, t

from .config import ACQUISITION_OUTPUTS, DEFAULT_MANIFEST_VARIANT, normalize_manifest_variant


DEFAULT_DATASETS = (
    "abdomenatlas",
    "imagecas",
    "jhu_stroke",
    "kits23",
    "totalsegmentermri",
)
DEFAULT_SEEDS = (42, 123, 456)
DEFAULT_CHECKPOINTS = (
    "med3dino_iso_96",
    "med3dino_rel_96",
    "med3dino_sa_96",
)
DEFAULT_FEATURE_TYPES = ("cls", "avg_pool")
DEFAULT_SURFACES = (
    "baseline",
    "no_antialias",
    "nearest_interpolation",
    "fixed_physical_fov_192mm",
    "extra_perturbations",
)
DEFAULT_METRICS = ("drift", "cka", "semantic_transfer")
CONTRASTS = (
    ("REL-ISO", "med3dino_rel_96", "med3dino_iso_96"),
    ("SA-ISO", "med3dino_sa_96", "med3dino_iso_96"),
    ("SA-REL", "med3dino_sa_96", "med3dino_rel_96"),
)
METRIC_DIRECTION = {
    "drift": "lower",
    "cka": "higher",
    "semantic_transfer": "higher",
}
SURFACE_ROLE = {
    "baseline": "spacing_reference",
    "no_antialias": "spacing_sensitivity",
    "nearest_interpolation": "spacing_sensitivity",
    "fixed_physical_fov_192mm": "spacing_sensitivity",
    "extra_perturbations": "spacing_and_additional_perturbations",
}
TRANSFER_FILE_NAME = "matched_semantic_transfer.json"
METRIC_FILE_NAME = "perturbation_robustness.json"
DEFAULT_OUTPUT_RELATIVE = Path("protocol_stability") / "acquisition"

POINT_FIELDS = (
    "dataset",
    "manifest_variant",
    "seed",
    "surface",
    "surface_role",
    "checkpoint",
    "feature_type",
    "perturbation_protocol",
    "reference_variant",
    "target_variant",
    "variant_family",
    "metric",
    "metric_direction",
    "metric_value",
    "oriented_value",
    "n_source_volumes",
    "evaluation_seed",
    "transfer_source",
    "source_file",
)
SEED_SUMMARY_FIELDS = (
    "dataset",
    "manifest_variant",
    "surface",
    "surface_role",
    "checkpoint",
    "feature_type",
    "reference_variant",
    "target_variant",
    "variant_family",
    "metric",
    "metric_direction",
    "n_seeds",
    "n_expected_seeds",
    "complete_seed_count",
    "seed_values",
    "mean",
    "std",
    "ci95_lower",
    "ci95_upper",
)
RANK_SUMMARY_FIELDS = (
    "aggregation_level",
    "dataset",
    "manifest_variant",
    "surface",
    "feature_type",
    "target_variant",
    "variant_family",
    "metric",
    "n_comparisons",
    "mean_kendall_tau",
    "std_kendall_tau",
    "ci95_kendall_tau_lower",
    "ci95_kendall_tau_upper",
    "top_checkpoint_retention_rate",
    "exact_order_match_rate",
    "mean_rank_displacement",
)
PAIRWISE_FIELDS = (
    "dataset",
    "manifest_variant",
    "surface",
    "feature_type",
    "reference_variant",
    "target_variant",
    "variant_family",
    "metric",
    "metric_direction",
    "pairwise",
    "method_a",
    "method_b",
    "n_seeds",
    "n_expected_seeds",
    "complete_seed_count",
    "method_a_mean",
    "method_a_std",
    "method_b_mean",
    "method_b_std",
    "seed_improvements",
    "mean_improvement_a_over_b",
    "ci95_lower",
    "ci95_upper",
    "bonferroni_ci_lower",
    "bonferroni_ci_upper",
    "decision",
    "baseline_decision",
    "stability",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _resolve_output_dir(output_dir: Path, results_root: Path) -> Path:
    output_dir = output_dir.expanduser()
    if output_dir.is_absolute():
        return output_dir
    return results_root.expanduser().resolve() / output_dir


def _prepare_output_dir(output_dir: Path) -> None:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(
            f"Cannot create output directory {output_dir}. "
            "Use a writable absolute path or a path relative to --results-root."
        ) from exc
    if not output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_dir}")


def _coerce_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _std(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _mean_interval(
    values: Sequence[float],
    confidence_level: float,
) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    if std == 0.0:
        return mean, mean
    critical = float(t.ppf((1.0 + confidence_level) / 2.0, df=len(values) - 1))
    half_width = critical * std / math.sqrt(len(values))
    return float(mean - half_width), float(mean + half_width)


def _metric_path(
    results_root: Path,
    dataset: str,
    manifest_variant: str,
    seed: int,
    surface: str,
    checkpoint: str,
    feature_type: str,
) -> Path:
    return (
        results_root
        / dataset
        / manifest_variant
        / "seeds"
        / f"seed_{seed}"
        / "evaluation_extensions"
        / "controlled_perturbations"
        / surface
        / checkpoint
        / feature_type
        / METRIC_FILE_NAME
    )


def _ordered_unique(values: Sequence[Any]) -> list[str]:
    ordered: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text not in ordered:
            ordered.append(text)
    return ordered


def _variant_order(
    payload: Mapping[str, Any],
    transfer_payload: Mapping[str, Any] | None,
) -> list[str]:
    reference = payload.get("reference_variant")
    metadata = payload.get("variant_metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    cka_matrix = payload.get("cka_matrix")
    cka_matrix = cka_matrix if isinstance(cka_matrix, Mapping) else {}
    reference_cka = cka_matrix.get(reference)
    reference_cka = reference_cka if isinstance(reference_cka, Mapping) else {}
    transfer_matrix = transfer_payload.get("transfer_matrix") if isinstance(transfer_payload, Mapping) else {}
    transfer_matrix = transfer_matrix if isinstance(transfer_matrix, Mapping) else {}
    reference_transfer = transfer_matrix.get(reference)
    reference_transfer = reference_transfer if isinstance(reference_transfer, Mapping) else {}
    drift = payload.get("representation_drift")
    drift = drift if isinstance(drift, Mapping) else {}
    return _ordered_unique(
        [
            reference,
            *(payload.get("target_variants") or []),
            *(metadata.get("spacing_variants") or []),
            *(metadata.get("extra_acquisition_variants") or []),
            *drift.keys(),
            *reference_cka.keys(),
            *reference_transfer.keys(),
        ]
    )


def _variant_family(payload: Mapping[str, Any], target_variant: str) -> str:
    metadata = payload.get("variant_metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    variant_kinds = metadata.get("variant_kinds")
    variant_kinds = variant_kinds if isinstance(variant_kinds, Mapping) else {}
    kind = variant_kinds.get(target_variant)
    if kind == "extra_acquisition":
        return "additional_perturbation"
    if target_variant in set(metadata.get("extra_acquisition_variants") or []):
        return "additional_perturbation"
    return "spacing"


def _load_transfer(
    metric_path: Path,
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    sibling_path = metric_path.parent / TRANSFER_FILE_NAME
    if sibling_path.exists():
        return _read_json(sibling_path), "sibling"
    embedded = payload.get("matched_semantic_transfer")
    if isinstance(embedded, Mapping):
        return dict(embedded), "embedded"
    return None, None


def _lookup_metric(
    metric: str,
    payload: Mapping[str, Any],
    transfer_payload: Mapping[str, Any] | None,
    reference_variant: str,
    target_variant: str,
) -> float | None:
    if metric == "drift":
        drift = payload.get("representation_drift")
        if isinstance(drift, Mapping):
            value = _coerce_float(drift.get(target_variant))
            if value is not None:
                return value
        pairwise = payload.get("mean_distance_per_pair")
        if not isinstance(pairwise, Mapping):
            return None
        for pair_key in (
            f"{reference_variant}___{target_variant}",
            f"{target_variant}___{reference_variant}",
        ):
            value = _coerce_float(pairwise.get(pair_key))
            if value is not None:
                return value
        return None
    if metric == "cka":
        matrix = payload.get("cka_matrix")
        if not isinstance(matrix, Mapping):
            return None
        reference_row = matrix.get(reference_variant)
        if isinstance(reference_row, Mapping):
            value = _coerce_float(reference_row.get(target_variant))
            if value is not None:
                return value
        target_row = matrix.get(target_variant)
        if isinstance(target_row, Mapping):
            return _coerce_float(target_row.get(reference_variant))
        return None
    if metric == "semantic_transfer":
        if not isinstance(transfer_payload, Mapping):
            return None
        matrix = transfer_payload.get("transfer_matrix")
        if not isinstance(matrix, Mapping):
            return None
        reference_row = matrix.get(reference_variant)
        if not isinstance(reference_row, Mapping):
            return None
        return _coerce_float(reference_row.get(target_variant))
    raise ValueError(f"Unknown metric: {metric}")


def _diagnostic(
    diagnostics: list[dict[str, Any]],
    *,
    severity: str,
    code: str,
    path: Path,
    detail: str,
) -> None:
    diagnostics.append(
        {
            "severity": severity,
            "code": code,
            "path": str(path),
            "detail": detail,
        }
    )


def _extract_points(
    *,
    results_root: Path,
    datasets: Sequence[str],
    manifest_variant: str,
    seeds: Sequence[int],
    surfaces: Sequence[str],
    checkpoints: Sequence[str],
    feature_types: Sequence[str],
    metrics: Sequence[str],
    require_transfer_siblings: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    points: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for dataset in datasets:
        for seed in seeds:
            for surface in surfaces:
                for checkpoint in checkpoints:
                    for feature_type in feature_types:
                        metric_path = _metric_path(
                            results_root,
                            dataset,
                            manifest_variant,
                            seed,
                            surface,
                            checkpoint,
                            feature_type,
                        )
                        artifact = {
                            "dataset": dataset,
                            "manifest_variant": manifest_variant,
                            "seed": int(seed),
                            "surface": surface,
                            "checkpoint": checkpoint,
                            "feature_type": feature_type,
                            "metric_path": str(metric_path),
                        }
                        if not metric_path.exists():
                            artifact["status"] = "missing"
                            artifacts.append(artifact)
                            _diagnostic(
                                diagnostics,
                                severity="error",
                                code="missing_metric_artifact",
                                path=metric_path,
                                detail="Expected extension metric file is absent.",
                            )
                            continue

                        payload = _read_json(metric_path)
                        transfer_payload, transfer_source = _load_transfer(metric_path, payload)
                        artifact.update(
                            {
                                "status": "loaded",
                                "transfer_source": transfer_source,
                                "n_source_volumes": payload.get("n_source_volumes"),
                                "reference_variant": payload.get("reference_variant"),
                            }
                        )
                        artifacts.append(artifact)

                        for key, expected in (
                            ("checkpoint", checkpoint),
                            ("feature_type", feature_type),
                            ("evaluation_seed", seed),
                        ):
                            observed = payload.get(key)
                            if observed is not None and str(observed) != str(expected):
                                _diagnostic(
                                    diagnostics,
                                    severity="error",
                                    code=f"{key}_mismatch",
                                    path=metric_path,
                                    detail=f"Expected {expected}, found {observed}.",
                                )

                        if "semantic_transfer" in metrics:
                            if transfer_payload is None:
                                _diagnostic(
                                    diagnostics,
                                    severity="error",
                                    code="missing_semantic_transfer",
                                    path=metric_path.parent / TRANSFER_FILE_NAME,
                                    detail="Run the extension semantic-transfer backfill before full analysis.",
                                )
                            elif require_transfer_siblings and transfer_source != "sibling":
                                _diagnostic(
                                    diagnostics,
                                    severity="error",
                                    code="semantic_transfer_not_sibling",
                                    path=metric_path,
                                    detail="A sibling matched_semantic_transfer.json is required.",
                                )

                        reference_variant = payload.get("reference_variant")
                        if not isinstance(reference_variant, str) or not reference_variant:
                            _diagnostic(
                                diagnostics,
                                severity="error",
                                code="missing_reference_variant",
                                path=metric_path,
                                detail="The controlled reference variant is not defined.",
                            )
                            continue

                        variant_order = _variant_order(payload, transfer_payload)
                        targets = [variant for variant in variant_order if variant != reference_variant]
                        for target_variant in targets:
                            variant_family = _variant_family(payload, target_variant)
                            for metric in metrics:
                                value = _lookup_metric(
                                    metric,
                                    payload,
                                    transfer_payload,
                                    reference_variant,
                                    target_variant,
                                )
                                if value is None:
                                    _diagnostic(
                                        diagnostics,
                                        severity="error",
                                        code="missing_metric_value",
                                        path=metric_path,
                                        detail=f"{metric}:{reference_variant}->{target_variant}",
                                    )
                                    continue
                                orientation = -1.0 if METRIC_DIRECTION[metric] == "lower" else 1.0
                                points.append(
                                    {
                                        "dataset": dataset,
                                        "manifest_variant": manifest_variant,
                                        "seed": int(seed),
                                        "surface": surface,
                                        "surface_role": SURFACE_ROLE.get(surface, "protocol_variant"),
                                        "checkpoint": checkpoint,
                                        "feature_type": feature_type,
                                        "perturbation_protocol": payload.get("perturbation_protocol"),
                                        "reference_variant": reference_variant,
                                        "target_variant": target_variant,
                                        "variant_family": variant_family,
                                        "metric": metric,
                                        "metric_direction": METRIC_DIRECTION[metric],
                                        "metric_value": value,
                                        "oriented_value": orientation * value,
                                        "n_source_volumes": payload.get("n_source_volumes"),
                                        "evaluation_seed": payload.get("evaluation_seed"),
                                        "transfer_source": transfer_source if metric == "semantic_transfer" else None,
                                        "source_file": str(metric_path),
                                    }
                                )
    return points, artifacts, diagnostics


def _validate_protocol_schemas(
    artifacts: Sequence[Mapping[str, Any]],
    points: Sequence[Mapping[str, Any]],
    surfaces: Sequence[str],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    artifact_index = {
        (
            row["dataset"],
            row["manifest_variant"],
            row["seed"],
            row["surface"],
            row["checkpoint"],
            row["feature_type"],
        ): row
        for row in artifacts
        if row.get("status") == "loaded"
    }
    spacing_targets: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for row in points:
        if row["variant_family"] != "spacing":
            continue
        spacing_targets[
            (
                row["dataset"],
                row["manifest_variant"],
                row["seed"],
                row["surface"],
                row["checkpoint"],
                row["feature_type"],
            )
        ].add(str(row["target_variant"]))

    contexts = {
        (row["dataset"], row["manifest_variant"], row["seed"], row["checkpoint"], row["feature_type"])
        for row in artifacts
        if row.get("status") == "loaded"
    }
    for dataset, manifest_variant, seed, checkpoint, feature_type in sorted(contexts):
        baseline_key = (dataset, manifest_variant, seed, "baseline", checkpoint, feature_type)
        baseline = artifact_index.get(baseline_key)
        if baseline is None:
            continue
        for surface in surfaces:
            surface_key = (dataset, manifest_variant, seed, surface, checkpoint, feature_type)
            current = artifact_index.get(surface_key)
            if current is None:
                continue
            for field in ("n_source_volumes", "reference_variant"):
                if current.get(field) != baseline.get(field):
                    _diagnostic(
                        diagnostics,
                        severity="error",
                        code=f"protocol_{field}_mismatch",
                        path=Path(str(current["metric_path"])),
                        detail=f"Baseline={baseline.get(field)}, {surface}={current.get(field)}.",
                    )
            if spacing_targets.get(surface_key, set()) != spacing_targets.get(baseline_key, set()):
                _diagnostic(
                    diagnostics,
                    severity="error",
                    code="protocol_spacing_targets_mismatch",
                    path=Path(str(current["metric_path"])),
                    detail=(
                        f"Baseline={sorted(spacing_targets.get(baseline_key, set()))}, "
                        f"{surface}={sorted(spacing_targets.get(surface_key, set()))}."
                    ),
                )
    return diagnostics


def _seed_summaries(
    points: Sequence[Mapping[str, Any]],
    expected_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    fields = (
        "dataset",
        "manifest_variant",
        "surface",
        "surface_role",
        "checkpoint",
        "feature_type",
        "reference_variant",
        "target_variant",
        "variant_family",
        "metric",
        "metric_direction",
    )
    for row in points:
        grouped[tuple(row[field] for field in fields)].append(row)

    summaries: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        ordered = sorted(rows, key=lambda row: int(row["seed"]))
        values = [float(row["metric_value"]) for row in ordered]
        ci_lower, ci_upper = _mean_interval(values, 0.95)
        present_seeds = [int(row["seed"]) for row in ordered]
        summaries.append(
            {
                **dict(zip(fields, key)),
                "n_seeds": len(present_seeds),
                "n_expected_seeds": len(expected_seeds),
                "complete_seed_count": present_seeds == sorted(int(seed) for seed in expected_seeds),
                "seed_values": {
                    str(row["seed"]): float(row["metric_value"])
                    for row in ordered
                },
                "mean": _mean(values),
                "std": _std(values),
                "ci95_lower": ci_lower,
                "ci95_upper": ci_upper,
            }
        )
    return summaries


def _score_order(scores: Mapping[str, float], checkpoint_order: Sequence[str]) -> list[str]:
    order_index = {checkpoint: index for index, checkpoint in enumerate(checkpoint_order)}
    return sorted(scores, key=lambda checkpoint: (-scores[checkpoint], order_index.get(checkpoint, len(order_index))))


def _top_set(scores: Mapping[str, float]) -> set[str]:
    maximum = max(scores.values())
    return {
        checkpoint
        for checkpoint, score in scores.items()
        if math.isclose(score, maximum, rel_tol=1e-12, abs_tol=1e-12)
    }


def _rank_rows(
    points: Sequence[Mapping[str, Any]],
    checkpoints: Sequence[str],
    surfaces: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    references: dict[tuple[Any, ...], str] = {}
    for row in points:
        key = (
            row["dataset"],
            row["manifest_variant"],
            row["seed"],
            row["surface"],
            row["feature_type"],
            row["target_variant"],
            row["variant_family"],
            row["metric"],
        )
        indexed[key][str(row["checkpoint"])] = float(row["oriented_value"])
        references[key] = str(row["reference_variant"])

    rows: list[dict[str, Any]] = []
    for key, surface_scores in sorted(indexed.items(), key=lambda item: tuple(str(value) for value in item[0])):
        dataset, manifest_variant, seed, surface, feature_type, target_variant, variant_family, metric = key
        if variant_family != "spacing" or surface not in surfaces:
            continue
        baseline_key = (
            dataset,
            manifest_variant,
            seed,
            "baseline",
            feature_type,
            target_variant,
            variant_family,
            metric,
        )
        baseline_scores = indexed.get(baseline_key)
        if baseline_scores is None:
            continue
        if set(surface_scores) != set(checkpoints) or set(baseline_scores) != set(checkpoints):
            continue

        baseline_values = [baseline_scores[checkpoint] for checkpoint in checkpoints]
        surface_values = [surface_scores[checkpoint] for checkpoint in checkpoints]
        tau_result = kendalltau(baseline_values, surface_values, variant="b")
        tau = _coerce_float(tau_result.statistic)
        p_value = _coerce_float(tau_result.pvalue)
        baseline_order = _score_order(baseline_scores, checkpoints)
        surface_order = _score_order(surface_scores, checkpoints)
        baseline_positions = {checkpoint: index for index, checkpoint in enumerate(baseline_order)}
        surface_positions = {checkpoint: index for index, checkpoint in enumerate(surface_order)}
        mean_rank_displacement = float(
            np.mean(
                [
                    abs(baseline_positions[checkpoint] - surface_positions[checkpoint])
                    for checkpoint in checkpoints
                ]
            )
        )
        baseline_top = _top_set(baseline_scores)
        surface_top = _top_set(surface_scores)
        rows.append(
            {
                "dataset": dataset,
                "manifest_variant": manifest_variant,
                "seed": int(seed),
                "surface": surface,
                "feature_type": feature_type,
                "reference_variant": references[key],
                "target_variant": target_variant,
                "variant_family": variant_family,
                "metric": metric,
                "kendall_tau": tau,
                "kendall_p_value": p_value,
                "baseline_order": baseline_order,
                "surface_order": surface_order,
                "baseline_top": sorted(baseline_top),
                "surface_top": sorted(surface_top),
                "top_checkpoint_retained": bool(baseline_top & surface_top),
                "exact_order_match": baseline_order == surface_order,
                "mean_rank_displacement": mean_rank_displacement,
            }
        )

    summaries = _summarize_rank_rows(rows)
    return rows, summaries


def _summarize_rank_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    target_fields = (
        "dataset",
        "manifest_variant",
        "surface",
        "feature_type",
        "target_variant",
        "variant_family",
        "metric",
    )
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in target_fields)].append(row)

    summaries: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        summaries.append(_rank_summary_row("target", target_fields, key, group))

    surface_fields = (
        "dataset",
        "manifest_variant",
        "surface",
        "feature_type",
        "variant_family",
        "metric",
    )
    surface_grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        surface_grouped[tuple(row[field] for field in surface_fields)].append(row)
    for key, group in sorted(surface_grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        summary = _rank_summary_row("surface", surface_fields, key, group)
        summary["target_variant"] = "__all_spacing_targets__"
        summaries.append(summary)
    return summaries


def _rank_summary_row(
    aggregation_level: str,
    fields: Sequence[str],
    key: Sequence[Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    tau_values = [
        float(row["kendall_tau"])
        for row in rows
        if row.get("kendall_tau") is not None
    ]
    ci_lower, ci_upper = _mean_interval(tau_values, 0.95)
    return {
        "aggregation_level": aggregation_level,
        **dict(zip(fields, key)),
        "n_comparisons": len(rows),
        "mean_kendall_tau": _mean(tau_values),
        "std_kendall_tau": _std(tau_values),
        "ci95_kendall_tau_lower": ci_lower,
        "ci95_kendall_tau_upper": ci_upper,
        "top_checkpoint_retention_rate": float(np.mean([bool(row["top_checkpoint_retained"]) for row in rows])),
        "exact_order_match_rate": float(np.mean([bool(row["exact_order_match"]) for row in rows])),
        "mean_rank_displacement": float(np.mean([float(row["mean_rank_displacement"]) for row in rows])),
    }


def _decision(lower: float | None, upper: float | None) -> str:
    if lower is None or upper is None:
        return "insufficient_seeds"
    if lower > 0.0:
        return "method_a_better"
    if upper < 0.0:
        return "method_b_better"
    return "not_resolved"


def _pairwise_rows(
    points: Sequence[Mapping[str, Any]],
    expected_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    point_index = {
        (
            row["dataset"],
            row["manifest_variant"],
            row["seed"],
            row["surface"],
            row["feature_type"],
            row["target_variant"],
            row["variant_family"],
            row["metric"],
            row["checkpoint"],
        ): row
        for row in points
    }
    contexts = sorted(
        {
            (
                row["dataset"],
                row["manifest_variant"],
                row["surface"],
                row["feature_type"],
                row["reference_variant"],
                row["target_variant"],
                row["variant_family"],
                row["metric"],
                row["metric_direction"],
            )
            for row in points
        },
        key=lambda item: tuple(str(value) for value in item),
    )
    adjusted_confidence = 1.0 - (0.05 / len(CONTRASTS))
    rows: list[dict[str, Any]] = []
    for context in contexts:
        (
            dataset,
            manifest_variant,
            surface,
            feature_type,
            reference_variant,
            target_variant,
            variant_family,
            metric,
            metric_direction,
        ) = context
        for pairwise, method_a, method_b in CONTRASTS:
            method_a_values: list[float] = []
            method_b_values: list[float] = []
            improvements: list[float] = []
            present_seeds: list[int] = []
            for seed in expected_seeds:
                prefix = (
                    dataset,
                    manifest_variant,
                    seed,
                    surface,
                    feature_type,
                    target_variant,
                    variant_family,
                    metric,
                )
                row_a = point_index.get((*prefix, method_a))
                row_b = point_index.get((*prefix, method_b))
                if row_a is None or row_b is None:
                    continue
                method_a_values.append(float(row_a["metric_value"]))
                method_b_values.append(float(row_b["metric_value"]))
                improvements.append(float(row_a["oriented_value"]) - float(row_b["oriented_value"]))
                present_seeds.append(int(seed))

            ci_lower, ci_upper = _mean_interval(improvements, 0.95)
            adjusted_lower, adjusted_upper = _mean_interval(improvements, adjusted_confidence)
            rows.append(
                {
                    "dataset": dataset,
                    "manifest_variant": manifest_variant,
                    "surface": surface,
                    "feature_type": feature_type,
                    "reference_variant": reference_variant,
                    "target_variant": target_variant,
                    "variant_family": variant_family,
                    "metric": metric,
                    "metric_direction": metric_direction,
                    "pairwise": pairwise,
                    "method_a": method_a,
                    "method_b": method_b,
                    "n_seeds": len(present_seeds),
                    "n_expected_seeds": len(expected_seeds),
                    "complete_seed_count": present_seeds == [int(seed) for seed in expected_seeds],
                    "method_a_mean": _mean(method_a_values),
                    "method_a_std": _std(method_a_values),
                    "method_b_mean": _mean(method_b_values),
                    "method_b_std": _std(method_b_values),
                    "seed_improvements": {
                        str(seed): value
                        for seed, value in zip(present_seeds, improvements)
                    },
                    "mean_improvement_a_over_b": _mean(improvements),
                    "ci95_lower": ci_lower,
                    "ci95_upper": ci_upper,
                    "bonferroni_ci_lower": adjusted_lower,
                    "bonferroni_ci_upper": adjusted_upper,
                    "decision": _decision(adjusted_lower, adjusted_upper),
                    "baseline_decision": None,
                    "stability": None,
                }
            )

    baseline_decisions = {
        (
            row["dataset"],
            row["manifest_variant"],
            row["feature_type"],
            row["target_variant"],
            row["variant_family"],
            row["metric"],
            row["pairwise"],
        ): row["decision"]
        for row in rows
        if row["surface"] == "baseline"
    }
    for row in rows:
        if row["surface"] == "baseline":
            row["baseline_decision"] = row["decision"]
            row["stability"] = "reference"
            continue
        baseline_key = (
            row["dataset"],
            row["manifest_variant"],
            row["feature_type"],
            row["target_variant"],
            row["variant_family"],
            row["metric"],
            row["pairwise"],
        )
        baseline_decision = baseline_decisions.get(baseline_key)
        row["baseline_decision"] = baseline_decision
        if baseline_decision is None:
            row["stability"] = "additional_perturbation"
        elif baseline_decision == "not_resolved" and row["decision"] == "not_resolved":
            row["stability"] = "both_unresolved"
        elif baseline_decision == "not_resolved":
            row["stability"] = "baseline_unresolved"
        elif row["decision"] == "not_resolved":
            row["stability"] = "surface_unresolved"
        elif row["decision"] == baseline_decision:
            row["stability"] = "preserved"
        elif "insufficient_seeds" in {baseline_decision, row["decision"]}:
            row["stability"] = "insufficient_seeds"
        else:
            row["stability"] = "reversed"
    return rows


def _status_counts(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(field))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Analyze stability of controlled acquisition metrics across saved protocol variants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-root", type=Path, default=ACQUISITION_OUTPUTS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_RELATIVE,
        help=(
            "Output directory. Relative paths are resolved under --results-root "
            "and validated before input artifacts are loaded."
        ),
    )
    parser.add_argument("--analysis-names", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--manifest-variant", default=DEFAULT_MANIFEST_VARIANT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--surfaces", nargs="+", default=list(DEFAULT_SURFACES))
    parser.add_argument("--checkpoints", nargs="+", default=list(DEFAULT_CHECKPOINTS))
    parser.add_argument("--feature-types", nargs="+", default=list(DEFAULT_FEATURE_TYPES))
    parser.add_argument("--metrics", nargs="+", choices=DEFAULT_METRICS, default=list(DEFAULT_METRICS))
    parser.add_argument(
        "--require-transfer-siblings",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)

    datasets = tuple(dict.fromkeys(str(value) for value in args.analysis_names))
    manifest_variant = normalize_manifest_variant(args.manifest_variant)
    seeds = tuple(dict.fromkeys(int(value) for value in args.seeds))
    surfaces = tuple(dict.fromkeys(str(value) for value in args.surfaces))
    checkpoints = tuple(dict.fromkeys(str(value) for value in args.checkpoints))
    feature_types = tuple(dict.fromkeys(str(value) for value in args.feature_types))
    metrics = tuple(dict.fromkeys(str(value) for value in args.metrics))
    output_dir = _resolve_output_dir(args.output_dir, args.results_root)
    _prepare_output_dir(output_dir)
    if "baseline" not in surfaces:
        raise ValueError("--surfaces must include baseline")
    if not set(metrics).issubset(METRIC_DIRECTION):
        raise ValueError(f"Unknown metrics: {sorted(set(metrics) - set(METRIC_DIRECTION))}")
    required_contrast_checkpoints = {
        checkpoint
        for _, method_a, method_b in CONTRASTS
        for checkpoint in (method_a, method_b)
    }
    if not required_contrast_checkpoints.issubset(checkpoints):
        missing = sorted(required_contrast_checkpoints - set(checkpoints))
        raise ValueError(f"Controlled pairwise analysis requires checkpoints: {missing}")

    points, artifacts, diagnostics = _extract_points(
        results_root=args.results_root,
        datasets=datasets,
        manifest_variant=manifest_variant,
        seeds=seeds,
        surfaces=surfaces,
        checkpoints=checkpoints,
        feature_types=feature_types,
        metrics=metrics,
        require_transfer_siblings=bool(args.require_transfer_siblings),
    )
    diagnostics.extend(_validate_protocol_schemas(artifacts, points, surfaces))
    seed_summaries = _seed_summaries(points, seeds)
    rank_rows, rank_summaries = _rank_rows(points, checkpoints, surfaces)
    pairwise_rows = _pairwise_rows(points, seeds)

    error_count = sum(row["severity"] == "error" for row in diagnostics)
    paths = {
        "points_csv": output_dir / "acquisition_protocol_stability_points.csv",
        "seed_summary_csv": output_dir / "acquisition_protocol_seed_summary.csv",
        "rank_summary_csv": output_dir / "acquisition_protocol_rank_stability.csv",
        "pairwise_summary_csv": output_dir / "acquisition_protocol_pairwise_stability.csv",
        "json": output_dir / "acquisition_protocol_stability.json",
    }
    payload = {
        "task": "acquisition_protocol_stability",
        "scientific_question": (
            "Whether frozen-representation metric ordering and controlled ISO/REL/SA "
            "contrasts persist under alternative controlled-resampling implementations."
        ),
        "claim_boundary": (
            "These deterministic protocol variants test resampling and preprocessing "
            "sensitivity; they do not simulate scanner physics or establish acquisition invariance."
        ),
        "results_root": str(args.results_root),
        "selection": {
            "datasets": list(datasets),
            "manifest_variant": manifest_variant,
            "seeds": list(seeds),
            "surfaces": list(surfaces),
            "checkpoints": list(checkpoints),
            "feature_types": list(feature_types),
            "metrics": list(metrics),
            "crop_group": 96,
        },
        "metric_orientation": METRIC_DIRECTION,
        "pairwise_effect_definition": (
            "Positive values favor method_a after orienting drift lower and CKA/semantic transfer higher."
        ),
        "pairwise_interval": (
            "Two-sided t interval across evaluation seeds; Bonferroni interval uses "
            "1 - 0.05/3 confidence for the three predefined ISO/REL/SA contrasts."
        ),
        "n_expected_artifacts": (
            len(datasets) * len(seeds) * len(surfaces) * len(checkpoints) * len(feature_types)
        ),
        "n_loaded_artifacts": sum(row.get("status") == "loaded" for row in artifacts),
        "n_points": len(points),
        "n_errors": error_count,
        "artifact_status_counts": _status_counts(artifacts, "status"),
        "diagnostic_code_counts": _status_counts(diagnostics, "code"),
        "pairwise_stability_counts": _status_counts(pairwise_rows, "stability"),
        "artifacts": artifacts,
        "diagnostics": diagnostics,
        "points": points,
        "seed_summaries": seed_summaries,
        "rank_stability_by_seed": rank_rows,
        "rank_summaries": rank_summaries,
        "pairwise_summaries": pairwise_rows,
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    _write_csv(paths["points_csv"], points, POINT_FIELDS)
    _write_csv(paths["seed_summary_csv"], seed_summaries, SEED_SUMMARY_FIELDS)
    _write_csv(paths["rank_summary_csv"], rank_summaries, RANK_SUMMARY_FIELDS)
    _write_csv(paths["pairwise_summary_csv"], pairwise_rows, PAIRWISE_FIELDS)
    _write_json(paths["json"], payload)

    print(
        json.dumps(
            {
                "n_expected_artifacts": payload["n_expected_artifacts"],
                "n_loaded_artifacts": payload["n_loaded_artifacts"],
                "n_points": payload["n_points"],
                "n_errors": payload["n_errors"],
                "pairwise_stability_counts": payload["pairwise_stability_counts"],
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )
    if error_count and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
