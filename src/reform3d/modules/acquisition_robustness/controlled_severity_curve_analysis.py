#!/usr/bin/env python
"""Build severity-curve artifacts from controlled perturbation results.

This is a post-hoc reshaping step. It consumes existing
``perturbation_robustness.json`` files and does not recompute features,
drift, CKA, or matched semantic transfer.

The curve unit is the reference controlled variant evaluated against each
saved target variant, including the reference variant as the baseline point:

- drift: reference-to-target representation drift
- CKA: reference-to-target linear CKA
- semantic transfer: reference-to-target matched semantic-transfer accuracy

Examples:
    python -m reform3d.modules.acquisition_robustness.controlled_severity_curve_analysis \
        --analysis-names abdomenatlas imagecas kits23 totalsegmentermri jhu_stroke \
        --manifest-variants original_bins \
        --seeds 42 123 456
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from reform3d.core.checkpoints import CHECKPOINT_ORDER

try:
    from .config import ACQUISITION_OUTPUTS, DEFAULT_MANIFEST_VARIANT, normalize_manifest_variant
except ImportError:  # pragma: no cover - supports direct script execution.
    from config import ACQUISITION_OUTPUTS, DEFAULT_MANIFEST_VARIANT, normalize_manifest_variant


DEFAULT_FEATURE_TYPES = ("cls", "avg_pool", "multilayer")
DEFAULT_SEEDS = (42, 123, 456)
DEFAULT_OUTPUT_DIR = ACQUISITION_OUTPUTS / "statistical_rigor"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_DIR / "controlled_severity_curves.json"
DEFAULT_OUTPUT_POINTS_CSV = DEFAULT_OUTPUT_DIR / "controlled_severity_curve_points.csv"
DEFAULT_OUTPUT_SUMMARY_CSV = DEFAULT_OUTPUT_DIR / "controlled_severity_curve_summary.csv"
DEFAULT_OUTPUT_DERIVED_CSV = DEFAULT_OUTPUT_DIR / "controlled_severity_curve_derived_stats.csv"

CI95_T_CRITICAL = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


POINT_CSV_FIELDS = [
    "analysis_name",
    "manifest_variant",
    "seed",
    "checkpoint_name",
    "feature_type",
    "perturbation_protocol",
    "source_variant",
    "target_variant",
    "severity_index",
    "relative_severity_index",
    "severity_axis",
    "severity_value",
    "reference_severity_value",
    "severity_delta",
    "target_spacing_x_mm",
    "target_spacing_y_mm",
    "target_spacing_z_mm",
    "metric_family",
    "metric_name",
    "metric_value",
    "within_seed_std",
    "n_source_volumes",
    "n_semantic_labels",
    "source_file",
]

SUMMARY_CSV_FIELDS = [
    "analysis_name",
    "manifest_variant",
    "checkpoint_name",
    "feature_type",
    "perturbation_protocol",
    "source_variant",
    "target_variant",
    "severity_index",
    "relative_severity_index",
    "severity_axis",
    "severity_value",
    "reference_severity_value",
    "severity_delta",
    "target_spacing_x_mm",
    "target_spacing_y_mm",
    "target_spacing_z_mm",
    "metric_family",
    "metric_name",
    "n",
    "n_expected_seeds",
    "complete_seed_count",
    "seed_values",
    "mean",
    "std",
    "ci95_lower",
    "ci95_upper",
    "min",
    "median",
    "max",
]

DERIVED_STATS_CSV_FIELDS = [
    "analysis_name",
    "manifest_variant",
    "checkpoint_name",
    "feature_type",
    "perturbation_protocol",
    "source_variant",
    "severity_axis",
    "target_variant",
    "severity_value",
    "reference_severity_value",
    "metric_family",
    "metric_name",
    "paired_metric_family",
    "paired_metric_name",
    "statistic_name",
    "aggregation_unit",
    "n",
    "n_expected_seeds",
    "complete_seed_count",
    "seed_values",
    "mean",
    "std",
    "ci95_lower",
    "ci95_upper",
    "min",
    "median",
    "max",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)
    return value


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _std(values: Sequence[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    mean = _mean(values)
    assert mean is not None
    return float(math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)))


def _ci95(values: Sequence[float]) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    mean = _mean(values)
    std = _std(values)
    if mean is None or std is None:
        return None, None
    t_critical = CI95_T_CRITICAL.get(len(values) - 1, 1.96)
    half_width = t_critical * std / math.sqrt(len(values))
    return float(mean - half_width), float(mean + half_width)


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return float((ordered[midpoint - 1] + ordered[midpoint]) / 2.0)


def _linear_slope(x_values: Sequence[float], y_values: Sequence[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    x_mean = _mean(x_values)
    y_mean = _mean(y_values)
    if x_mean is None or y_mean is None:
        return None
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    denominator = sum((x - x_mean) ** 2 for x in x_values)
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _pearson(x_values: Sequence[float], y_values: Sequence[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 3:
        return None
    x_mean = _mean(x_values)
    y_mean = _mean(y_values)
    if x_mean is None or y_mean is None:
        return None
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    x_var = sum((x - x_mean) ** 2 for x in x_values)
    y_var = sum((y - y_mean) ** 2 for y in y_values)
    denominator = math.sqrt(x_var * y_var)
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _seed_name(seed: int | str) -> str:
    text = str(seed)
    return text if text.startswith("seed_") else f"seed_{text}"


def _parse_seed(seed_name: str) -> int | None:
    match = re.fullmatch(r"seed_(\d+)", str(seed_name))
    if match is None:
        return None
    return int(match.group(1))


def _metric_context(metric_path: Path, base_root: Path) -> dict[str, Any] | None:
    try:
        parts = metric_path.relative_to(base_root).parts
    except ValueError:
        return None
    if len(parts) >= 8 and parts[2] == "seeds" and parts[4] == "results":
        seed = _parse_seed(parts[3])
        if seed is None:
            return None
        return {
            "analysis_name": parts[0],
            "manifest_variant": parts[1],
            "seed": seed,
            "checkpoint_name": parts[5],
            "feature_type": parts[6],
        }
    if len(parts) >= 6 and parts[2] == "results":
        return {
            "analysis_name": parts[0],
            "manifest_variant": parts[1],
            "seed": None,
            "checkpoint_name": parts[3],
            "feature_type": parts[4],
        }
    return None


def _discover_metric_paths(
    *,
    base_root: Path,
    analysis_names: Sequence[str] | None,
    manifest_variants: Sequence[str] | None,
    seeds: Sequence[int] | None,
    checkpoints: Sequence[str] | None,
    feature_types: Sequence[str] | None,
    include_unseeded: bool,
) -> list[Path]:
    metric_paths = sorted(base_root.glob("*/*/seeds/seed_*/results/*/*/perturbation_robustness.json"))
    if include_unseeded:
        metric_paths.extend(sorted(base_root.glob("*/*/results/*/*/perturbation_robustness.json")))

    analysis_set = set(analysis_names) if analysis_names else None
    variant_set = {normalize_manifest_variant(variant) for variant in manifest_variants} if manifest_variants else None
    seed_set = set(int(seed) for seed in seeds) if seeds else None
    checkpoint_set = set(checkpoints) if checkpoints else None
    feature_set = set(feature_types) if feature_types else None

    selected: list[Path] = []
    for metric_path in metric_paths:
        context = _metric_context(metric_path, base_root)
        if context is None:
            continue
        if analysis_set is not None and context["analysis_name"] not in analysis_set:
            continue
        if variant_set is not None and context["manifest_variant"] not in variant_set:
            continue
        if seed_set is not None and context["seed"] not in seed_set:
            continue
        if checkpoint_set is not None and context["checkpoint_name"] not in checkpoint_set:
            continue
        if feature_set is not None and context["feature_type"] not in feature_set:
            continue
        selected.append(metric_path)
    return sorted(selected)


def _variant_order(payload: Mapping[str, Any]) -> list[str]:
    variants = [str(variant) for variant in payload.get("target_variants") or []]
    reference = payload.get("reference_variant")
    if reference is not None and str(reference) not in variants:
        variants.insert(0, str(reference))
    if variants:
        return variants

    seen: list[str] = []
    for candidate in [reference]:
        if candidate is not None and str(candidate) not in seen:
            seen.append(str(candidate))
    for container in (
        payload.get("representation_drift") or {},
        payload.get("cka_matrix") or {},
        ((payload.get("matched_semantic_transfer") or {}).get("transfer_matrix") or {}),
    ):
        if not isinstance(container, Mapping):
            continue
        for key in container:
            if str(key) not in seen:
                seen.append(str(key))
    return seen


def _spacing_by_variant(payload: Mapping[str, Any], variants: Sequence[str]) -> dict[str, tuple[float, float, float]]:
    spacings = payload.get("target_spacings")
    if not isinstance(spacings, Sequence) or isinstance(spacings, (str, bytes)):
        return {}
    mapping: dict[str, tuple[float, float, float]] = {}
    for variant, spacing in zip(variants, spacings):
        if not isinstance(spacing, Sequence) or isinstance(spacing, (str, bytes)) or len(spacing) != 3:
            continue
        numeric = tuple(_coerce_float(value) for value in spacing)
        if any(value is None for value in numeric):
            continue
        sx, sy, sz = numeric
        assert sx is not None and sy is not None and sz is not None
        mapping[str(variant)] = (sx, sy, sz)
    return mapping


def _z_by_variant(payload: Mapping[str, Any], variants: Sequence[str]) -> dict[str, float]:
    z_values = payload.get("target_z_spacings")
    if isinstance(z_values, Sequence) and not isinstance(z_values, (str, bytes)):
        mapping = {}
        for variant, z_value in zip(variants, z_values):
            numeric = _coerce_float(z_value)
            if numeric is not None:
                mapping[str(variant)] = numeric
        if mapping:
            return mapping
    mapping = {}
    for variant in variants:
        match = re.fullmatch(r"native_xy_z([0-9.]+)", str(variant))
        if match is not None:
            mapping[str(variant)] = float(match.group(1))
    return mapping


def _parse_fixed_spacing_variant(variant: str) -> tuple[float, float, float] | None:
    match = re.fullmatch(r"([0-9.]+)x([0-9.]+)x([0-9.]+)", str(variant))
    if match is None:
        return None
    return tuple(float(match.group(index)) for index in (1, 2, 3))


def _anisotropy_ratio(spacing: Sequence[float]) -> float | None:
    if not spacing:
        return None
    minimum = min(float(value) for value in spacing)
    maximum = max(float(value) for value in spacing)
    if minimum <= 0:
        return None
    return float(maximum / minimum)


def _severity_value(
    *,
    payload: Mapping[str, Any],
    variant: str,
    variants: Sequence[str],
    spacing_map: Mapping[str, tuple[float, float, float]],
    z_map: Mapping[str, float],
) -> tuple[str, float | None, tuple[float, float, float] | None]:
    protocol = str(payload.get("perturbation_protocol") or "")
    if protocol == "native_inplane_z" or str(variant).startswith("native_xy_z"):
        return "target_z_spacing_mm", z_map.get(str(variant)), None

    spacing = spacing_map.get(str(variant)) or _parse_fixed_spacing_variant(str(variant))
    if spacing is not None:
        return "anisotropy_ratio", _anisotropy_ratio(spacing), spacing

    if z_map:
        return "target_z_spacing_mm", z_map.get(str(variant)), None
    return "variant_order", float(variants.index(variant)) if variant in variants else None, None


def _nested_numeric(mapping: Any, row_key: str, col_key: str) -> float | None:
    if not isinstance(mapping, Mapping):
        return None
    row = mapping.get(row_key)
    if isinstance(row, Mapping):
        numeric = _coerce_float(row.get(col_key))
        if numeric is not None:
            return numeric
    row = mapping.get(col_key)
    if isinstance(row, Mapping):
        return _coerce_float(row.get(row_key))
    return None


def _directed_numeric(mapping: Any, row_key: str, col_key: str) -> float | None:
    if not isinstance(mapping, Mapping):
        return None
    row = mapping.get(row_key)
    if not isinstance(row, Mapping):
        return None
    return _coerce_float(row.get(col_key))


def _pair_numeric(mapping: Any, source_variant: str, target_variant: str) -> float | None:
    if not isinstance(mapping, Mapping):
        return None
    for key in (
        f"{source_variant}___{target_variant}",
        f"{source_variant}__{target_variant}",
        f"{target_variant}___{source_variant}",
        f"{target_variant}__{source_variant}",
    ):
        numeric = _coerce_float(mapping.get(key))
        if numeric is not None:
            return numeric
    return None


def _metric_rows_for_target(
    *,
    context: Mapping[str, Any],
    payload: Mapping[str, Any],
    source_file: Path,
    source_variant: str,
    target_variant: str,
    severity_metadata: Mapping[str, Any],
    include_baseline: bool,
) -> list[dict[str, Any]]:
    if target_variant == source_variant and not include_baseline:
        return []

    rows: list[dict[str, Any]] = []
    shared = {
        "analysis_name": context["analysis_name"],
        "manifest_variant": context["manifest_variant"],
        "seed": context["seed"],
        "checkpoint_name": context["checkpoint_name"],
        "feature_type": context["feature_type"],
        "perturbation_protocol": payload.get("perturbation_protocol"),
        "source_variant": source_variant,
        "target_variant": target_variant,
        "n_source_volumes": payload.get("n_source_volumes"),
        "n_semantic_labels": (payload.get("matched_semantic_transfer") or {}).get("n_labels"),
        "source_file": str(source_file),
        **severity_metadata,
    }

    if target_variant == source_variant:
        drift_value = 0.0
        drift_std = 0.0
    else:
        drift_value = _coerce_float((payload.get("representation_drift") or {}).get(target_variant))
        drift_std = _coerce_float((payload.get("representation_drift_std") or {}).get(target_variant))
        if drift_value is None:
            drift_value = _pair_numeric(payload.get("mean_distance_per_pair"), source_variant, target_variant)
            drift_std = _pair_numeric(payload.get("std_distance_per_pair"), source_variant, target_variant)
    if drift_value is not None:
        rows.append(
            {
                **shared,
                "metric_family": "controlled_drift",
                "metric_name": "reference_to_target_drift",
                "metric_value": drift_value,
                "within_seed_std": drift_std,
            }
        )

    cka_value = _nested_numeric(payload.get("cka_matrix"), source_variant, target_variant)
    if cka_value is not None:
        rows.append(
            {
                **shared,
                "metric_family": "controlled_cka",
                "metric_name": "reference_to_target_cka",
                "metric_value": cka_value,
                "within_seed_std": None,
            }
        )

    semantic_transfer = payload.get("matched_semantic_transfer") or {}
    transfer_matrix = (
        semantic_transfer.get("transfer_matrix")
        or semantic_transfer.get("directed_transfer")
        or semantic_transfer.get("transfer_results")
        or semantic_transfer.get("matrix")
    )
    semantic_value = _directed_numeric(transfer_matrix, source_variant, target_variant)
    if semantic_value is not None:
        rows.append(
            {
                **shared,
                "metric_family": "controlled_semantic_transfer",
                "metric_name": "reference_to_target_semantic_transfer",
                "metric_value": semantic_value,
                "within_seed_std": None,
            }
        )
    return rows


def _point_rows_for_metric_path(
    metric_path: Path,
    *,
    base_root: Path,
    include_baseline: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    context = _metric_context(metric_path, base_root)
    if context is None:
        return [], [{"source_file": str(metric_path), "status": "skipped", "reason": "unrecognized_metric_path"}]
    payload = _load_json(metric_path)
    variants = _variant_order(payload)
    source_variant = str(payload.get("reference_variant") or variants[0])
    if source_variant not in variants:
        variants.insert(0, source_variant)
    source_index = variants.index(source_variant)
    spacing_map = _spacing_by_variant(payload, variants)
    z_map = _z_by_variant(payload, variants)
    source_axis, source_value, _ = _severity_value(
        payload=payload,
        variant=source_variant,
        variants=variants,
        spacing_map=spacing_map,
        z_map=z_map,
    )

    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for target_index, target_variant in enumerate(variants):
        severity_axis, severity_value, spacing = _severity_value(
            payload=payload,
            variant=target_variant,
            variants=variants,
            spacing_map=spacing_map,
            z_map=z_map,
        )
        if severity_axis != source_axis and source_value is not None:
            diagnostics.append(
                {
                    "source_file": str(metric_path),
                    "status": "warning",
                    "reason": "mixed_severity_axes",
                    "source_axis": source_axis,
                    "target_axis": severity_axis,
                }
            )
        severity_metadata = {
            "severity_index": target_index,
            "relative_severity_index": target_index - source_index,
            "severity_axis": severity_axis,
            "severity_value": severity_value,
            "reference_severity_value": source_value,
            "severity_delta": (
                float(severity_value - source_value)
                if severity_value is not None and source_value is not None
                else None
            ),
            "target_spacing_x_mm": spacing[0] if spacing is not None else None,
            "target_spacing_y_mm": spacing[1] if spacing is not None else None,
            "target_spacing_z_mm": spacing[2] if spacing is not None else None,
        }
        target_rows = _metric_rows_for_target(
            context=context,
            payload=payload,
            source_file=metric_path,
            source_variant=source_variant,
            target_variant=target_variant,
            severity_metadata=severity_metadata,
            include_baseline=include_baseline,
        )
        if target_rows:
            rows.extend(target_rows)
        elif target_variant != source_variant or include_baseline:
            diagnostics.append(
                {
                    "source_file": str(metric_path),
                    "status": "warning",
                    "reason": "no_supported_metrics_for_variant",
                    "target_variant": target_variant,
                }
            )
    return rows, diagnostics


def _summary_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in SUMMARY_CSV_FIELDS[:18])


def _summarize_points(
    point_rows: Sequence[Mapping[str, Any]],
    *,
    expected_seeds: Sequence[int] | None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in point_rows:
        grouped[_summary_key(row)].append(row)

    expected_seed_count = len(set(expected_seeds)) if expected_seeds else None
    summaries: list[dict[str, Any]] = []
    for _, rows in grouped.items():
        first = rows[0]
        values = [float(row["metric_value"]) for row in rows]
        seeds = sorted({int(row["seed"]) for row in rows if row.get("seed") is not None})
        ci_lower, ci_upper = _ci95(values)
        summary = {
            **{field: first.get(field) for field in SUMMARY_CSV_FIELDS[:18]},
            "n": len(values),
            "n_expected_seeds": expected_seed_count,
            "complete_seed_count": (
                len(seeds) == expected_seed_count
                if expected_seed_count is not None
                else None
            ),
            "seed_values": seeds,
            "mean": _mean(values),
            "std": _std(values),
            "ci95_lower": ci_lower,
            "ci95_upper": ci_upper,
            "min": min(values),
            "median": _median(values),
            "max": max(values),
        }
        summaries.append(summary)
    return sorted(
        summaries,
        key=lambda row: (
            str(row.get("analysis_name")),
            str(row.get("feature_type")),
            str(row.get("metric_family")),
            str(row.get("checkpoint_name")),
            float(row.get("severity_value") if row.get("severity_value") is not None else row.get("severity_index") or 0),
            str(row.get("target_variant")),
        ),
    )


def _derived_seed_group_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("analysis_name"),
        row.get("manifest_variant"),
        row.get("checkpoint_name"),
        row.get("feature_type"),
        row.get("perturbation_protocol"),
        row.get("source_variant"),
        row.get("severity_axis"),
        row.get("metric_family"),
        row.get("metric_name"),
        row.get("seed"),
    )


def _derived_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in DERIVED_STATS_CSV_FIELDS[:16])


def _point_severity_sort_key(row: Mapping[str, Any]) -> tuple[float, int]:
    severity_value = _coerce_float(row.get("severity_value"))
    severity_index = _coerce_float(row.get("severity_index"))
    if severity_value is None:
        severity_value = severity_index if severity_index is not None else 0.0
    if severity_index is None:
        severity_index = 0.0
    return float(severity_value), int(severity_index)


def _aggregate_derived_seed_stats(
    seed_stat_rows: Sequence[Mapping[str, Any]],
    *,
    expected_seeds: Sequence[int] | None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in seed_stat_rows:
        grouped[_derived_identity(row)].append(row)

    expected_seed_count = len(set(expected_seeds)) if expected_seeds else None
    summaries: list[dict[str, Any]] = []
    for _, rows in grouped.items():
        first = rows[0]
        values = [float(row["value"]) for row in rows]
        seeds = sorted({int(row["seed"]) for row in rows if row.get("seed") is not None})
        ci_lower, ci_upper = _ci95(values)
        summaries.append(
            {
                **{field: first.get(field) for field in DERIVED_STATS_CSV_FIELDS[:16]},
                "n": len(values),
                "n_expected_seeds": expected_seed_count,
                "complete_seed_count": (
                    len(seeds) == expected_seed_count
                    if expected_seed_count is not None
                    else None
                ),
                "seed_values": seeds,
                "mean": _mean(values),
                "std": _std(values),
                "ci95_lower": ci_lower,
                "ci95_upper": ci_upper,
                "min": min(values),
                "median": _median(values),
                "max": max(values),
            }
        )
    return summaries


def _curve_stat_seed_rows(point_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in point_rows:
        if row.get("metric_family") not in {
            "controlled_drift",
            "controlled_cka",
            "controlled_semantic_transfer",
        }:
            continue
        if row.get("seed") is None:
            continue
        if _coerce_float(row.get("metric_value")) is None:
            continue
        grouped[_derived_seed_group_key(row)].append(row)

    seed_stats: list[dict[str, Any]] = []
    for _, rows in grouped.items():
        ordered = sorted(rows, key=_point_severity_sort_key)
        x_values = [
            _coerce_float(row.get("severity_value"))
            if _coerce_float(row.get("severity_value")) is not None
            else _coerce_float(row.get("severity_index"))
            for row in ordered
        ]
        y_values = [_coerce_float(row.get("metric_value")) for row in ordered]
        valid_pairs = [
            (float(x_value), float(y_value), row)
            for x_value, y_value, row in zip(x_values, y_values, ordered)
            if x_value is not None and y_value is not None
        ]
        if len(valid_pairs) < 2:
            continue

        x_numeric = [item[0] for item in valid_pairs]
        y_numeric = [item[1] for item in valid_pairs]
        template = valid_pairs[0][2]
        high_x, high_y, high_row = valid_pairs[-1]
        baseline_candidates = [
            (x_value, y_value, row)
            for x_value, y_value, row in valid_pairs
            if _coerce_float(row.get("relative_severity_index")) == 0.0
        ]
        baseline_x, baseline_y, baseline_row = baseline_candidates[0] if baseline_candidates else valid_pairs[0]

        common = {
            "analysis_name": template.get("analysis_name"),
            "manifest_variant": template.get("manifest_variant"),
            "checkpoint_name": template.get("checkpoint_name"),
            "feature_type": template.get("feature_type"),
            "perturbation_protocol": template.get("perturbation_protocol"),
            "source_variant": template.get("source_variant"),
            "severity_axis": template.get("severity_axis"),
            "metric_family": template.get("metric_family"),
            "metric_name": template.get("metric_name"),
            "paired_metric_family": None,
            "paired_metric_name": None,
            "aggregation_unit": "seed_curve",
            "seed": template.get("seed"),
        }

        slope = _linear_slope(x_numeric, y_numeric)
        if slope is not None:
            seed_stats.append(
                {
                    **common,
                    "target_variant": None,
                    "severity_value": None,
                    "reference_severity_value": baseline_x,
                    "statistic_name": "severity_slope",
                    "value": slope,
                }
            )

        seed_stats.append(
            {
                **common,
                "target_variant": high_row.get("target_variant"),
                "severity_value": high_x,
                "reference_severity_value": baseline_x,
                "statistic_name": "high_severity_delta_from_baseline",
                "value": float(high_y - baseline_y),
            }
        )
        if baseline_y != 0:
            seed_stats.append(
                {
                    **common,
                    "target_variant": high_row.get("target_variant"),
                    "severity_value": high_x,
                    "reference_severity_value": baseline_x,
                    "statistic_name": "high_severity_retention_ratio",
                    "value": float(high_y / baseline_y),
                }
            )

    return seed_stats


def _coupling_group_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("analysis_name"),
        row.get("manifest_variant"),
        row.get("checkpoint_name"),
        row.get("feature_type"),
        row.get("perturbation_protocol"),
        row.get("source_variant"),
        row.get("severity_axis"),
    )


def _geometry_semantic_coupling_rows(
    point_rows: Sequence[Mapping[str, Any]],
    *,
    expected_seeds: Sequence[int] | None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in point_rows:
        if row.get("metric_family") in {
            "controlled_drift",
            "controlled_cka",
            "controlled_semantic_transfer",
        }:
            grouped[_coupling_group_key(row)].append(row)

    expected_seed_count = len(set(expected_seeds)) if expected_seeds else None
    coupling_rows: list[dict[str, Any]] = []
    for _, rows in grouped.items():
        by_metric: dict[str, dict[tuple[Any, Any], float]] = defaultdict(dict)
        templates: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            metric_family = str(row.get("metric_family"))
            value = _coerce_float(row.get("metric_value"))
            if value is None:
                continue
            key = (row.get("seed"), row.get("target_variant"))
            by_metric[metric_family][key] = value
            templates.setdefault(metric_family, row)

        semantic = by_metric.get("controlled_semantic_transfer") or {}
        if not semantic:
            continue
        for geometry_family, metric_name in (
            ("controlled_drift", "drift_vs_semantic_transfer_pearson"),
            ("controlled_cka", "cka_vs_semantic_transfer_pearson"),
        ):
            geometry = by_metric.get(geometry_family) or {}
            common_keys = sorted(set(geometry).intersection(semantic), key=lambda item: (str(item[0]), str(item[1])))
            if len(common_keys) < 3:
                continue
            x_values = [geometry[key] for key in common_keys]
            y_values = [semantic[key] for key in common_keys]
            correlation = _pearson(x_values, y_values)
            if correlation is None:
                continue
            template = templates.get(geometry_family) or templates["controlled_semantic_transfer"]
            seeds = sorted({int(key[0]) for key in common_keys if key[0] is not None})
            coupling_rows.append(
                {
                    "analysis_name": template.get("analysis_name"),
                    "manifest_variant": template.get("manifest_variant"),
                    "checkpoint_name": template.get("checkpoint_name"),
                    "feature_type": template.get("feature_type"),
                    "perturbation_protocol": template.get("perturbation_protocol"),
                    "source_variant": template.get("source_variant"),
                    "severity_axis": template.get("severity_axis"),
                    "target_variant": None,
                    "severity_value": None,
                    "reference_severity_value": template.get("reference_severity_value"),
                    "metric_family": geometry_family,
                    "metric_name": str(templates.get(geometry_family, {}).get("metric_name") or ""),
                    "paired_metric_family": "controlled_semantic_transfer",
                    "paired_metric_name": str(templates["controlled_semantic_transfer"].get("metric_name") or ""),
                    "statistic_name": "geometry_semantic_coupling_pearson",
                    "aggregation_unit": "seed_target_points",
                    "n": len(common_keys),
                    "n_expected_seeds": expected_seed_count,
                    "complete_seed_count": (
                        len(seeds) == expected_seed_count
                        if expected_seed_count is not None
                        else None
                    ),
                    "seed_values": seeds,
                    "mean": correlation,
                    "std": None,
                    "ci95_lower": None,
                    "ci95_upper": None,
                    "min": None,
                    "median": None,
                    "max": None,
                }
            )
    return coupling_rows


def _derive_curve_statistics(
    point_rows: Sequence[Mapping[str, Any]],
    *,
    expected_seeds: Sequence[int] | None,
) -> list[dict[str, Any]]:
    seed_stats = _curve_stat_seed_rows(point_rows)
    derived_rows = _aggregate_derived_seed_stats(seed_stats, expected_seeds=expected_seeds)
    derived_rows.extend(_geometry_semantic_coupling_rows(point_rows, expected_seeds=expected_seeds))
    return sorted(
        derived_rows,
        key=lambda row: (
            str(row.get("analysis_name")),
            str(row.get("feature_type")),
            str(row.get("statistic_name")),
            str(row.get("metric_family")),
            str(row.get("checkpoint_name")),
            float(row.get("severity_value") if row.get("severity_value") is not None else 0.0),
        ),
    )


def _make_plots(summary_rows: Sequence[Mapping[str, Any]], plot_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency path.
        raise RuntimeError("matplotlib is required when --plot-dir is set") from exc

    plot_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped[
            (
                str(row.get("analysis_name")),
                str(row.get("manifest_variant")),
                str(row.get("feature_type")),
                str(row.get("metric_family")),
            )
        ].append(row)

    written: list[str] = []
    for (analysis_name, manifest_variant, feature_type, metric_family), rows in grouped.items():
        by_checkpoint: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            by_checkpoint[str(row.get("checkpoint_name"))].append(row)
        _, axis = plt.subplots(figsize=(7.5, 4.5))
        for checkpoint, checkpoint_rows in sorted(by_checkpoint.items()):
            ordered = sorted(
                checkpoint_rows,
                key=lambda row: float(row.get("severity_value") if row.get("severity_value") is not None else row.get("severity_index") or 0),
            )
            x_values = [row.get("severity_value") for row in ordered]
            y_values = [row.get("mean") for row in ordered]
            axis.plot(x_values, y_values, marker="o", linewidth=1.5, label=checkpoint)
        axis.set_title(f"{analysis_name} {feature_type} {metric_family}")
        axis.set_xlabel(str(rows[0].get("severity_axis") or "severity"))
        axis.set_ylabel(metric_family)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
        filename = f"{analysis_name}_{manifest_variant}_{feature_type}_{metric_family}.png".replace("/", "_")
        output_path = plot_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=180)
        plt.close()
        written.append(str(output_path))
    return written


def _all_or_values(values: Sequence[str] | None) -> Sequence[str] | None:
    if values is None:
        return None
    if len(values) == 1 and values[0].lower() == "all":
        return None
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build controlled perturbation severity curves from existing perturbation_robustness.json files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--analysis-names", nargs="+", default=None)
    parser.add_argument("--manifest-variants", nargs="+", default=[DEFAULT_MANIFEST_VARIANT])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=list(CHECKPOINT_ORDER),
        help="Checkpoint directory names. Use 'all' to include every discovered checkpoint directory.",
    )
    parser.add_argument(
        "--feature-types",
        nargs="+",
        default=list(DEFAULT_FEATURE_TYPES),
        help="Feature directory names. Use 'all' to include every discovered feature directory.",
    )
    parser.add_argument("--base-root", type=Path, default=ACQUISITION_OUTPUTS)
    parser.add_argument("--include-unseeded", action="store_true")
    parser.add_argument("--no-baseline", action="store_true", help="Exclude the reference variant anchor point from each curve.")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help="Optional existing controlled_severity_curves.json. When set, reuse its point/summary rows instead of rediscovering perturbation JSON files.",
    )
    parser.add_argument("--no-derived-stats", action="store_true", help="Skip curve-derived statistic rows.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-points-csv", type=Path, default=DEFAULT_OUTPUT_POINTS_CSV)
    parser.add_argument("--output-summary-csv", type=Path, default=DEFAULT_OUTPUT_SUMMARY_CSV)
    parser.add_argument("--output-derived-csv", type=Path, default=DEFAULT_OUTPUT_DERIVED_CSV)
    parser.add_argument("--plot-dir", type=Path, default=None, help="Optional directory for PNG severity-curve figures.")
    args = parser.parse_args()

    existing_payload: dict[str, Any] = {}
    if args.input_json is not None:
        existing_payload = _load_json(args.input_json)
        point_rows = list(existing_payload.get("point_rows") or [])
        summary_rows = list(existing_payload.get("summary_rows") or [])
        diagnostics = list(existing_payload.get("diagnostics") or [])
        n_metric_files = int(existing_payload.get("n_metric_files") or 0)
        if not summary_rows and point_rows:
            summary_rows = _summarize_points(point_rows, expected_seeds=args.seeds)
    else:
        checkpoints = _all_or_values(args.checkpoints)
        feature_types = _all_or_values(args.feature_types)
        metric_paths = _discover_metric_paths(
            base_root=args.base_root,
            analysis_names=args.analysis_names,
            manifest_variants=args.manifest_variants,
            seeds=args.seeds,
            checkpoints=checkpoints,
            feature_types=feature_types,
            include_unseeded=args.include_unseeded,
        )

        point_rows = []
        diagnostics = []
        for metric_path in metric_paths:
            rows, path_diagnostics = _point_rows_for_metric_path(
                metric_path,
                base_root=args.base_root,
                include_baseline=not args.no_baseline,
            )
            point_rows.extend(rows)
            diagnostics.extend(path_diagnostics)

        summary_rows = _summarize_points(point_rows, expected_seeds=args.seeds)
        n_metric_files = len(metric_paths)

    derived_rows = [] if args.no_derived_stats else _derive_curve_statistics(point_rows, expected_seeds=args.seeds)
    figure_paths: list[str] = []
    if args.plot_dir is not None:
        figure_paths = _make_plots(summary_rows, args.plot_dir)

    payload = {
        **existing_payload,
        "analysis_scope": "acquisition_controlled_perturbation_reference_to_target_severity_curves",
        "source_artifacts": "perturbation_robustness.json",
        "recompute_policy": "post_hoc_only_no_embedding_or_metric_recomputation",
        "curve_unit": "reference_controlled_variant_to_saved_target_variant",
        "baseline_included": not args.no_baseline,
        "input_json": str(args.input_json) if args.input_json is not None else None,
        "base_root": str(args.base_root),
        "analysis_names": list(args.analysis_names) if args.analysis_names else None,
        "manifest_variants": [normalize_manifest_variant(variant) for variant in args.manifest_variants],
        "seeds": list(args.seeds) if args.seeds else None,
        "checkpoints": (
            list(_all_or_values(args.checkpoints))
            if _all_or_values(args.checkpoints) is not None
            else "all"
        ),
        "feature_types": (
            list(_all_or_values(args.feature_types))
            if _all_or_values(args.feature_types) is not None
            else "all"
        ),
        "n_metric_files": n_metric_files,
        "n_point_rows": len(point_rows),
        "n_summary_rows": len(summary_rows),
        "n_derived_rows": len(derived_rows),
        "n_diagnostics": len(diagnostics),
        "output_points_csv": str(args.output_points_csv),
        "output_summary_csv": str(args.output_summary_csv),
        "output_derived_csv": str(args.output_derived_csv),
        "figure_paths": figure_paths,
        "summary_rows": summary_rows,
        "derived_rows": derived_rows,
        "point_rows": point_rows,
        "diagnostics": diagnostics,
    }
    _write_json(args.output_json, payload)
    _write_csv(args.output_points_csv, point_rows, POINT_CSV_FIELDS)
    _write_csv(args.output_summary_csv, summary_rows, SUMMARY_CSV_FIELDS)
    _write_csv(args.output_derived_csv, derived_rows, DERIVED_STATS_CSV_FIELDS)

    print(
        json.dumps(
            {
                "metric_files": n_metric_files,
                "point_rows": len(point_rows),
                "summary_rows": len(summary_rows),
                "derived_rows": len(derived_rows),
                "diagnostics": len(diagnostics),
                "output_json": str(args.output_json),
                "output_points_csv": str(args.output_points_csv),
                "output_summary_csv": str(args.output_summary_csv),
                "output_derived_csv": str(args.output_derived_csv),
                "figures": len(figure_paths),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
