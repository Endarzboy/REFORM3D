"""Contract-driven repeated-seed summaries for REFORM-3D modules."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from reform3d.core.seed_context import apply_seed_to_path, normalize_seed


MODULES = ("acquisition_robustness", "cross_modal_alignment", "anatomical_generalization")
EXTENSION_DIR_NAME = "evaluation_extensions"
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


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _lookup(payload: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for key in dotted_path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _first_present(payload: Mapping[str, Any], *keys: str) -> Any:
    """Return the first present key value, preserving valid falsy values."""
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _first_not_none(*values: Any) -> Any:
    """Return the first non-None value, preserving valid falsy values."""
    for value in values:
        if value is not None:
            return value
    return None


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
    """Return a two-sided t interval for the across-seed mean."""
    if len(values) < 2:
        return None, None
    mean = _mean(values)
    std = _std(values)
    if mean is None or std is None:
        return None, None
    t_critical = CI95_T_CRITICAL.get(len(values) - 1, 1.96)
    half_width = t_critical * std / math.sqrt(len(values))
    return float(mean - half_width), float(mean + half_width)


def _metric_key(record: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(record["checkpoint"]),
        str(record["feature_type"]),
        str(record["status"]),
        str(record["metric_family"]),
        str(record["metric_name"]),
    )


def _add_record(
    records: list[dict[str, Any]],
    *,
    seed: int,
    checkpoint: str,
    feature_type: str,
    metric_family: str,
    metric_name: str,
    metric_value: Any,
    status: str,
    source_file: Path,
) -> None:
    numeric = _coerce_float(metric_value)
    if numeric is None:
        return
    records.append(
        {
            "seed": int(seed),
            "checkpoint": checkpoint,
            "feature_type": feature_type,
            "metric_family": metric_family,
            "metric_name": metric_name,
            "metric_value": numeric,
            "status": status,
            "source_file": str(source_file),
        }
    )


def _off_diagonal_values(matrix: Mapping[str, Any]) -> list[tuple[str, float]]:
    values: list[tuple[str, float]] = []
    for row_key, row in matrix.items():
        if not isinstance(row, Mapping):
            continue
        for col_key, value in row.items():
            if str(row_key) == str(col_key):
                continue
            numeric = _coerce_float(value)
            if numeric is not None:
                values.append((f"{row_key}__{col_key}", numeric))
    return values


def _dict_numeric_items(payload: Any) -> Iterable[tuple[str, float]]:
    if not isinstance(payload, Mapping):
        return []
    items: list[tuple[str, float]] = []
    for key, value in payload.items():
        numeric = _coerce_float(value)
        if numeric is not None:
            items.append((str(key), numeric))
    return items


def _extract_transfer_matrix_items(payload: Mapping[str, Any]) -> list[tuple[str, float]]:
    matrix = (
        payload.get("transfer_matrix")
        or payload.get("directed_transfer")
        or payload.get("transfer_results")
        or payload.get("matrix")
    )
    items: list[tuple[str, float]] = []
    if not isinstance(matrix, Mapping):
        return items
    for source_key, row in matrix.items():
        if not isinstance(row, Mapping):
            continue
        for target_key, value in row.items():
            numeric = _coerce_float(value)
            if numeric is not None:
                items.append((f"{source_key}__{target_key}", numeric))
    return items


def _feature_result_dir(seed_root: Path, checkpoint: str, feature_type: str) -> Path:
    return seed_root / "results" / checkpoint / feature_type


def _add_controlled_perturbation_records(
    records: list[dict[str, Any]],
    *,
    seed: int,
    checkpoint: str,
    feature_type: str,
    surface: str,
    perturbation_path: Path,
    perturbation: Mapping[str, Any],
    status: str,
) -> None:
    drift_items = list(_dict_numeric_items(perturbation.get("representation_drift")))
    for target_name, value in drift_items:
        _add_record(
            records,
            seed=seed,
            checkpoint=checkpoint,
            feature_type=feature_type,
            metric_family=f"{surface}.controlled_drift",
            metric_name=target_name,
            metric_value=value,
            status=status,
            source_file=perturbation_path,
        )
    if drift_items:
        _add_record(
            records,
            seed=seed,
            checkpoint=checkpoint,
            feature_type=feature_type,
            metric_family=f"{surface}.controlled_drift",
            metric_name="compact_non_reference_mean_drift",
            metric_value=_mean([value for _, value in drift_items]),
            status=status,
            source_file=perturbation_path,
        )

    cka_values = _off_diagonal_values(perturbation.get("cka_matrix") or {})
    for pair_name, value in cka_values:
        _add_record(
            records,
            seed=seed,
            checkpoint=checkpoint,
            feature_type=feature_type,
            metric_family=f"{surface}.controlled_cka",
            metric_name=pair_name,
            metric_value=value,
            status=status,
            source_file=perturbation_path,
        )
    if cka_values:
        _add_record(
            records,
            seed=seed,
            checkpoint=checkpoint,
            feature_type=feature_type,
            metric_family=f"{surface}.controlled_cka",
            metric_name="compact_off_diagonal_cka",
            metric_value=_mean([value for _, value in cka_values]),
            status=status,
            source_file=perturbation_path,
        )

    semantic_transfer = perturbation.get("matched_semantic_transfer") or {}
    if isinstance(semantic_transfer, Mapping):
        _add_record(
            records,
            seed=seed,
            checkpoint=checkpoint,
            feature_type=feature_type,
            metric_family=f"{surface}.controlled_semantic_transfer",
            metric_name="cross_variant_accuracy",
            metric_value=semantic_transfer.get("cross_variant_accuracy"),
            status=status,
            source_file=perturbation_path,
        )
        for pair_name, value in _extract_transfer_matrix_items(semantic_transfer):
            _add_record(
                records,
                seed=seed,
                checkpoint=checkpoint,
                feature_type=feature_type,
                metric_family=f"{surface}.controlled_semantic_transfer",
                metric_name=pair_name,
                metric_value=value,
                status=status,
                source_file=perturbation_path,
            )


def _extract_acquisition_records(
    *,
    seed: int,
    seed_root: Path,
    checkpoints: Sequence[str],
    feature_types: Sequence[str],
    missing_files: list[str],
    include_native: bool = True,
    include_controlled: bool = True,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        for feature_type in feature_types:
            result_dir = _feature_result_dir(seed_root, checkpoint, feature_type)
            if include_controlled:
                perturbation_path = result_dir / "perturbation_robustness.json"
                perturbation = _read_json(perturbation_path)
                if perturbation is None:
                    missing_files.append(str(perturbation_path))
                else:
                    drift_items = list(_dict_numeric_items(perturbation.get("representation_drift")))
                    for target_name, value in drift_items:
                        _add_record(
                            records,
                            seed=seed,
                            checkpoint=checkpoint,
                            feature_type=feature_type,
                            metric_family="controlled_drift",
                            metric_name=target_name,
                            metric_value=value,
                            status="seeded_report",
                            source_file=perturbation_path,
                        )
                    if drift_items:
                        _add_record(
                            records,
                            seed=seed,
                            checkpoint=checkpoint,
                            feature_type=feature_type,
                            metric_family="controlled_drift",
                            metric_name="compact_non_reference_mean_drift",
                            metric_value=_mean([value for _, value in drift_items]),
                            status="seeded_report",
                            source_file=perturbation_path,
                        )

                    cka_values = _off_diagonal_values(perturbation.get("cka_matrix") or {})
                    for pair_name, value in cka_values:
                        _add_record(
                            records,
                            seed=seed,
                            checkpoint=checkpoint,
                            feature_type=feature_type,
                            metric_family="controlled_cka",
                            metric_name=pair_name,
                            metric_value=value,
                            status="seeded_report",
                            source_file=perturbation_path,
                        )
                    if cka_values:
                        _add_record(
                            records,
                            seed=seed,
                            checkpoint=checkpoint,
                            feature_type=feature_type,
                            metric_family="controlled_cka",
                            metric_name="compact_off_diagonal_cka",
                            metric_value=_mean([value for _, value in cka_values]),
                            status="seeded_report",
                            source_file=perturbation_path,
                        )

                    semantic_transfer = perturbation.get("matched_semantic_transfer") or {}
                    if isinstance(semantic_transfer, Mapping):
                        _add_record(
                            records,
                            seed=seed,
                            checkpoint=checkpoint,
                            feature_type=feature_type,
                            metric_family="controlled_semantic_transfer",
                            metric_name="cross_variant_accuracy",
                            metric_value=semantic_transfer.get("cross_variant_accuracy"),
                            status="seeded_report",
                            source_file=perturbation_path,
                        )
                        for pair_name, value in _extract_transfer_matrix_items(semantic_transfer):
                            _add_record(
                                records,
                                seed=seed,
                                checkpoint=checkpoint,
                                feature_type=feature_type,
                                metric_family="controlled_semantic_transfer",
                                metric_name=pair_name,
                                metric_value=value,
                                status="seeded_report",
                                source_file=perturbation_path,
                            )

            if not include_native:
                continue

            geometry_path = result_dir / "representation_geometry.json"
            geometry = _read_json(geometry_path)
            if geometry is None:
                missing_files.append(str(geometry_path))
            else:
                _add_record(
                    records,
                    seed=seed,
                    checkpoint=checkpoint,
                    feature_type=feature_type,
                    metric_family="native_cka",
                    metric_name="compact_mean_cross_cka",
                    metric_value=_lookup(geometry, "cka_stats.mean_cross_cka") or geometry.get("mean_cross_cka"),
                    status="seeded_report",
                    source_file=geometry_path,
                )

            semantic_path = result_dir / "semantic_transfer.json"
            semantic = _read_json(semantic_path)
            if semantic is None:
                missing_files.append(str(semantic_path))
            else:
                _add_record(
                    records,
                    seed=seed,
                    checkpoint=checkpoint,
                    feature_type=feature_type,
                    metric_family="native_semantic_transfer",
                    metric_name="cross_bin_accuracy",
                    metric_value=semantic.get("cross_bin_accuracy"),
                    status="seeded_report",
                    source_file=semantic_path,
                )
    return records


def _cross_modal_metrics_path(seed_root: Path, checkpoint: str, feature_type: str) -> Path:
    if feature_type == "cls":
        return seed_root / "results" / checkpoint / "primary_metrics.json"
    return seed_root / "results" / checkpoint / feature_type / "primary_metrics.json"


def _cross_modal_single_organ_path(seed_root: Path, checkpoint: str, feature_type: str) -> Path:
    if feature_type == "cls":
        return seed_root / "results" / checkpoint / "single_organ_alignment.json"
    return seed_root / "results" / checkpoint / feature_type / "single_organ_alignment.json"


def _extract_cross_modal_single_organ_records(
    *,
    seed: int,
    seed_root: Path,
    checkpoints: Sequence[str],
    feature_types: Sequence[str],
    missing_files: list[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        for feature_type in feature_types:
            metrics_path = _cross_modal_single_organ_path(seed_root, checkpoint, feature_type)
            payload = _read_json(metrics_path)
            if payload is None:
                missing_files.append(str(metrics_path))
                continue
            retrieval = payload.get("single_organ_bidirectional_retrieval") or {}
            bidirectional = retrieval.get("bidirectional_mean") or {}
            for key, metric_name in (("map", "retrieval_map"), ("top@1", "retrieval_top@1"), ("top@5", "retrieval_top@5")):
                _add_record(
                    records,
                    seed=seed,
                    checkpoint=checkpoint,
                    feature_type=feature_type,
                    metric_family="single_organ_bidirectional_retrieval",
                    metric_name=metric_name,
                    metric_value=_first_present(bidirectional, key, key.replace("@", "_at_")),
                    status="seeded_report",
                    source_file=metrics_path,
                )
            for direction in ("ct_to_mr", "mr_to_ct"):
                overall = (retrieval.get(direction) or {}).get("overall") or {}
                for key, metric_name in (("map", "map"), ("top@1", "top@1"), ("top@5", "top@5")):
                    _add_record(
                        records,
                        seed=seed,
                        checkpoint=checkpoint,
                        feature_type=feature_type,
                        metric_family=f"single_organ_{direction}_retrieval",
                        metric_name=metric_name,
                        metric_value=_first_present(overall, key, key.replace("@", "_at_")),
                        status="seeded_required_diagnostic",
                        source_file=metrics_path,
                    )
            geometry = payload.get("paired_case_geometry") or {}
            for metric_name in (
                "paired_case_linear_cka",
                "paired_case_mean_cosine_distance",
                "paired_case_median_cosine_distance",
            ):
                _add_record(
                    records,
                    seed=seed,
                    checkpoint=checkpoint,
                    feature_type=feature_type,
                    metric_family="single_organ_geometry",
                    metric_name=metric_name,
                    metric_value=geometry.get(metric_name),
                    status="seeded_required_diagnostic",
                    source_file=metrics_path,
                )
            _add_record(
                records,
                seed=seed,
                checkpoint=checkpoint,
                feature_type=feature_type,
                metric_family="support",
                metric_name="n_paired_cases",
                metric_value=payload.get("n_paired_cases"),
                status="seeded_required_diagnostic",
                source_file=metrics_path,
            )
    return records


def _extract_cross_modal_records(
    *,
    seed: int,
    seed_root: Path,
    dataset: str,
    checkpoints: Sequence[str],
    feature_types: Sequence[str],
    missing_files: list[str],
) -> list[dict[str, Any]]:
    if str(dataset).strip().lower() == "chaos_ct_mr":
        return _extract_cross_modal_single_organ_records(
            seed=seed,
            seed_root=seed_root,
            checkpoints=checkpoints,
            feature_types=feature_types,
            missing_files=missing_files,
        )

    records: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        for feature_type in feature_types:
            metrics_path = _cross_modal_metrics_path(seed_root, checkpoint, feature_type)
            payload = _read_json(metrics_path)
            if payload is None:
                missing_files.append(str(metrics_path))
                continue
            retrieval = payload.get("balanced_cross_modal_retrieval") or {}
            bidirectional = retrieval.get("bidirectional_mean") or {}
            for key, metric_name in (("map", "retrieval_map"), ("top@1", "retrieval_top@1"), ("top@5", "retrieval_top@5")):
                _add_record(
                    records,
                    seed=seed,
                    checkpoint=checkpoint,
                    feature_type=feature_type,
                    metric_family="bidirectional_retrieval",
                    metric_name=metric_name,
                    metric_value=_first_present(bidirectional, key, key.replace("@", "_at_")),
                    status="seeded_report",
                    source_file=metrics_path,
                )
            for direction in ("ct_to_mr", "mr_to_ct"):
                overall = (retrieval.get(direction) or {}).get("overall") or {}
                for key, metric_name in (("map", "map"), ("top@1", "top@1"), ("top@5", "top@5")):
                    _add_record(
                        records,
                        seed=seed,
                        checkpoint=checkpoint,
                        feature_type=feature_type,
                        metric_family=f"{direction}_retrieval",
                        metric_name=metric_name,
                        metric_value=_first_present(overall, key, key.replace("@", "_at_")),
                        status="seeded_required_diagnostic",
                        source_file=metrics_path,
                    )
            for metric_name in ("n_supported_organs", "n_dropped_organs", "n_queries"):
                _add_record(
                    records,
                    seed=seed,
                    checkpoint=checkpoint,
                    feature_type=feature_type,
                    metric_family="support",
                    metric_name=metric_name,
                    metric_value=_first_not_none(payload.get(metric_name), retrieval.get(metric_name)),
                    status="seeded_required_diagnostic",
                    source_file=metrics_path,
                )
    return records


def _extract_anatomical_records(
    *,
    seed: int,
    seed_root: Path,
    checkpoints: Sequence[str],
    feature_types: Sequence[str],
    missing_files: list[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        for feature_type in feature_types:
            metrics_path = seed_root / "results" / checkpoint / feature_type / "anatomical_generalization.json"
            metrics_paths = [metrics_path]
            metrics_paths.extend(
                sorted(
                    (seed_root / "results" / checkpoint / feature_type / "feature_adapter" / "projection").glob(
                        "*/anatomical_generalization.json"
                    )
                )
            )
            existing_payloads = [(path, _read_json(path)) for path in metrics_paths if path.exists()]
            existing_payloads = [(path, payload) for path, payload in existing_payloads if payload is not None]
            if not existing_payloads:
                missing_files.append(str(metrics_path))
                continue
            for path, payload in existing_payloads:
                adapter = payload.get("feature_adapter") or {}
                source_key = adapter.get("source_key")
                family_prefix = f"projection.{source_key}." if adapter.get("adapter") == "projection" else ""
                gain = _lookup(payload, "headline_metrics.task_1_low_shot_target_adaptation_gain") or {}
                task_2 = _lookup(payload, "headline_metrics.task_2_low_shot_transfer") or {}
                projection = _lookup(payload, "headline_metrics.feature_projection_low_shot_transfer") or {}
                metric_specs = [
                    (
                        f"{family_prefix}anatomical_adaptation",
                        "adaptation_gain_balanced_accuracy_over_nearest_centroid",
                        gain.get("primary_metric_value"),
                        "seeded_report",
                    ),
                    (
                        f"{family_prefix}anatomical_adaptation",
                        "nearest_centroid_balanced_accuracy",
                        gain.get("baseline_metric_value"),
                        "seeded_report",
                    ),
                    (
                        f"{family_prefix}anatomical_adaptation",
                        "adapted_balanced_accuracy",
                        _first_not_none(gain.get("adapted_metric_value"), task_2.get("primary_metric_value")),
                        "seeded_report",
                    ),
                    (
                        f"{family_prefix}support",
                        "n_supported_organs",
                        gain.get("n_supported_organs"),
                        "seeded_required_diagnostic",
                    ),
                    (
                        f"{family_prefix}support",
                        "n_evaluated_organs",
                        _first_not_none(gain.get("n_evaluated_organs"), task_2.get("n_evaluated_organs")),
                        "seeded_required_diagnostic",
                    ),
                ]
                if projection:
                    metric_specs.extend(
                        [
                            (
                                f"{family_prefix}feature_projection",
                                "projected_balanced_accuracy",
                                projection.get("primary_metric_value"),
                                "seeded_report",
                            ),
                            (
                                f"{family_prefix}feature_projection",
                                "projected_centroid_balanced_accuracy",
                                projection.get("centroid_metric_value"),
                                "seeded_report",
                            ),
                            (
                                f"{family_prefix}feature_projection",
                                "projected_adaptation_gain_balanced_accuracy",
                                projection.get("adaptation_gain_metric_value"),
                                "seeded_report",
                            ),
                        ]
                    )
                for metric_family, metric_name, value, status in metric_specs:
                    _add_record(
                        records,
                        seed=seed,
                        checkpoint=checkpoint,
                        feature_type=feature_type,
                        metric_family=metric_family,
                        metric_name=metric_name,
                        metric_value=value,
                        status=status,
                        source_file=path,
                )
    return records


def _extract_acquisition_extension_records(
    *,
    seed: int,
    extension_root: Path,
    checkpoints: Sequence[str],
    feature_types: Sequence[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    perturbation_root = extension_root / "controlled_perturbations"
    if not perturbation_root.exists():
        return records
    checkpoint_set = {str(checkpoint) for checkpoint in checkpoints}
    feature_set = {str(feature_type) for feature_type in feature_types}
    for perturbation_path in sorted(perturbation_root.glob("*/*/*/perturbation_robustness.json")):
        try:
            surface, checkpoint, feature_type, _ = perturbation_path.relative_to(perturbation_root).parts
        except ValueError:
            continue
        if checkpoint not in checkpoint_set or feature_type not in feature_set:
            continue
        payload = _read_json(perturbation_path)
        if payload is None:
            continue
        _add_controlled_perturbation_records(
            records,
            seed=seed,
            checkpoint=checkpoint,
            feature_type=feature_type,
            surface=surface,
            perturbation_path=perturbation_path,
            perturbation=payload,
            status="extension_seeded_report",
        )
    return records


def _extract_cross_modal_extension_records(
    *,
    seed: int,
    extension_root: Path,
    checkpoints: Sequence[str],
    feature_types: Sequence[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    results_root = extension_root / "results"
    if not results_root.exists():
        return records
    checkpoint_set = {str(checkpoint) for checkpoint in checkpoints}
    feature_set = {str(feature_type) for feature_type in feature_types}
    for metrics_path in sorted(results_root.glob("*/*/*/primary_metrics.json")):
        try:
            surface, checkpoint, feature_type, _ = metrics_path.relative_to(results_root).parts
        except ValueError:
            continue
        if checkpoint not in checkpoint_set or feature_type not in feature_set:
            continue
        payload = _read_json(metrics_path)
        if payload is None:
            continue
        retrieval = payload.get("balanced_cross_modal_retrieval") or {}
        bidirectional = retrieval.get("bidirectional_mean") or {}
        for key, metric_name in (("map", "retrieval_map"), ("top@1", "retrieval_top@1"), ("top@5", "retrieval_top@5")):
            _add_record(
                records,
                seed=seed,
                checkpoint=checkpoint,
                feature_type=feature_type,
                metric_family=f"{surface}.bidirectional_retrieval",
                metric_name=metric_name,
                metric_value=_first_present(bidirectional, key, key.replace("@", "_at_")),
                status="extension_seeded_report",
                source_file=metrics_path,
            )
        for direction in ("ct_to_mr", "mr_to_ct"):
            overall = (retrieval.get(direction) or {}).get("overall") or {}
            for key, metric_name in (("map", "map"), ("top@1", "top@1"), ("top@5", "top@5")):
                _add_record(
                    records,
                    seed=seed,
                    checkpoint=checkpoint,
                    feature_type=feature_type,
                    metric_family=f"{surface}.{direction}_retrieval",
                    metric_name=metric_name,
                    metric_value=_first_present(overall, key, key.replace("@", "_at_")),
                    status="extension_seeded_report",
                    source_file=metrics_path,
                )
        for metric_name in ("n_supported_organs", "n_dropped_organs", "n_queries"):
            _add_record(
                records,
                seed=seed,
                checkpoint=checkpoint,
                feature_type=feature_type,
                metric_family=f"{surface}.support",
                metric_name=metric_name,
                metric_value=_first_not_none(payload.get(metric_name), retrieval.get(metric_name)),
                status="extension_seeded_report",
                source_file=metrics_path,
            )
    return records


def _extract_anatomical_extension_records(
    *,
    seed: int,
    extension_root: Path,
    checkpoints: Sequence[str],
    feature_types: Sequence[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    sweep_root = extension_root / "support_size_sweep"
    if not sweep_root.exists():
        return records
    checkpoint_set = {str(checkpoint) for checkpoint in checkpoints}
    feature_set = {str(feature_type) for feature_type in feature_types}
    sweep_paths = sorted(sweep_root.glob("*/*/support_size_sweep.json"))
    sweep_paths.extend(sorted(sweep_root.glob("*/*/feature_adapter/projection/*/support_size_sweep.json")))
    for sweep_path in sweep_paths:
        try:
            relative_parts = sweep_path.relative_to(sweep_root).parts
        except ValueError:
            continue
        if len(relative_parts) == 3:
            checkpoint, feature_type, _ = relative_parts
            adapter_source = None
        elif len(relative_parts) == 6 and relative_parts[2:4] == ("feature_adapter", "projection"):
            checkpoint, feature_type, _, _, adapter_source, _ = relative_parts
        else:
            continue
        if checkpoint not in checkpoint_set or feature_type not in feature_set:
            continue
        payload = _read_json(sweep_path)
        if payload is None:
            continue
        for row in payload.get("rows", []):
            if not isinstance(row, Mapping):
                continue
            requested_support = row.get("requested_support_per_modality")
            support_label = f"support_{requested_support}"
            family_prefix = (
                f"projection.{adapter_source}.{support_label}"
                if adapter_source is not None
                else support_label
            )
            status = f"extension_{row.get('status') or 'unknown'}"
            for metric_name in (
                "effective_support_per_modality",
                "n_supported_organs",
                "n_evaluated_organs",
                "balanced_accuracy",
                "balanced_accuracy_seed_std",
                "nearest_centroid_balanced_accuracy",
                "adaptation_gain_balanced_accuracy",
                "adaptation_gain_balanced_accuracy_seed_std",
                "projected_balanced_accuracy",
                "projected_balanced_accuracy_seed_std",
                "projected_centroid_balanced_accuracy",
                "projected_adaptation_gain_balanced_accuracy",
                "projected_adaptation_gain_balanced_accuracy_seed_std",
                "query_per_modality",
            ):
                _add_record(
                    records,
                    seed=seed,
                    checkpoint=checkpoint,
                    feature_type=feature_type,
                    metric_family=f"{family_prefix}.support_size_sweep",
                    metric_name=metric_name,
                    metric_value=row.get(metric_name),
                    status=status,
                    source_file=sweep_path,
                )
    return records


def _extension_records_for_seed(
    *,
    module_name: str,
    seed: int,
    extension_root: Path,
    checkpoints: Sequence[str],
    feature_types: Sequence[str],
) -> list[dict[str, Any]]:
    if module_name == "acquisition_robustness":
        return _extract_acquisition_extension_records(
            seed=seed,
            extension_root=extension_root,
            checkpoints=checkpoints,
            feature_types=feature_types,
        )
    if module_name == "cross_modal_alignment":
        return _extract_cross_modal_extension_records(
            seed=seed,
            extension_root=extension_root,
            checkpoints=checkpoints,
            feature_types=feature_types,
        )
    if module_name == "anatomical_generalization":
        return _extract_anatomical_extension_records(
            seed=seed,
            extension_root=extension_root,
            checkpoints=checkpoints,
            feature_types=feature_types,
        )
    raise ValueError(f"Unsupported repeated-seed module: {module_name}")


def _records_for_seed(
    *,
    module_name: str,
    dataset: str,
    seed: int,
    seed_root: Path,
    checkpoints: Sequence[str],
    feature_types: Sequence[str],
    missing_files: list[str],
    acquisition_tracks: Sequence[str] = ("native", "controlled"),
) -> list[dict[str, Any]]:
    if module_name == "acquisition_robustness":
        requested_tracks = {str(track).strip().lower() for track in acquisition_tracks}
        return _extract_acquisition_records(
            seed=seed,
            seed_root=seed_root,
            checkpoints=checkpoints,
            feature_types=feature_types,
            missing_files=missing_files,
            include_native="native" in requested_tracks,
            include_controlled="controlled" in requested_tracks,
        )
    if module_name == "cross_modal_alignment":
        return _extract_cross_modal_records(
            seed=seed,
            seed_root=seed_root,
            dataset=dataset,
            checkpoints=checkpoints,
            feature_types=feature_types,
            missing_files=missing_files,
        )
    if module_name == "anatomical_generalization":
        return _extract_anatomical_records(
            seed=seed,
            seed_root=seed_root,
            checkpoints=checkpoints,
            feature_types=feature_types,
            missing_files=missing_files,
        )
    raise ValueError(f"Unsupported repeated-seed module: {module_name}")


def _summarize_records(
    *,
    records: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    allow_incomplete: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_metric_key(record)].append(record)

    summaries: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    expected = {int(seed) for seed in seeds}
    for key, group in sorted(grouped.items()):
        checkpoint, feature_type, status, metric_family, metric_name = key
        values_by_seed = {int(record["seed"]): float(record["metric_value"]) for record in group}
        present = set(values_by_seed)
        missing = sorted(expected - present)
        values = [values_by_seed[seed] for seed in sorted(present)]
        ci95_low, ci95_high = _ci95(values)
        source_files = {
            str(record["seed"]): record["source_file"]
            for record in sorted(group, key=lambda item: int(item["seed"]))
        }
        row = {
            "checkpoint": checkpoint,
            "feature_type": feature_type,
            "status": status,
            "metric_family": metric_family,
            "metric_name": metric_name,
            "n": len(values),
            "mean": _mean(values),
            "std": _std(values),
            "ci95_low": ci95_low,
            "ci95_high": ci95_high,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "missing_seeds": missing,
            "values_by_seed": {str(seed): values_by_seed[seed] for seed in sorted(present)},
            "source_files_by_seed": source_files,
        }
        if missing:
            incomplete.append(row)
        summaries.append(row)

    if incomplete and not allow_incomplete:
        missing_labels = ", ".join(
            f"{row['checkpoint']}/{row['feature_type']}/{row['metric_family']}.{row['metric_name']}: {row['missing_seeds']}"
            for row in incomplete[:10]
        )
        raise RuntimeError(f"Repeated-seed summary is incomplete: {missing_labels}")

    return summaries, incomplete


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "checkpoint",
        "feature_type",
        "status",
        "metric_family",
        "metric_name",
        "n",
        "mean",
        "std",
        "ci95_low",
        "ci95_high",
        "min",
        "max",
        "missing_seeds",
        "values_by_seed",
        "source_files_by_seed",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["missing_seeds"] = json.dumps(row.get("missing_seeds", []), sort_keys=True)
            serialized["values_by_seed"] = json.dumps(row.get("values_by_seed", {}), sort_keys=True)
            serialized["source_files_by_seed"] = json.dumps(row.get("source_files_by_seed", {}), sort_keys=True)
            writer.writerow(serialized)


def collect_repeated_seed_summary(
    *,
    module_name: str,
    dataset: str,
    manifest_variant: str,
    base_root: Path,
    seeds: Sequence[int | str],
    checkpoints: Sequence[str],
    feature_types: Sequence[str],
    allow_incomplete: bool = False,
    acquisition_tracks: Sequence[str] = ("native", "controlled"),
) -> dict[str, Any]:
    """Collect repeated-seed metrics without writing to the source result tree."""
    if module_name not in MODULES:
        raise ValueError(f"Unsupported repeated-seed module: {module_name}")
    normalized_seeds = [normalize_seed(seed) for seed in seeds]
    records: list[dict[str, Any]] = []
    missing_files: list[str] = []
    for seed in normalized_seeds:
        seed_root = apply_seed_to_path(base_root, seed)
        records.extend(
            _records_for_seed(
                module_name=module_name,
                dataset=dataset,
                seed=seed,
                seed_root=seed_root,
                checkpoints=checkpoints,
                feature_types=feature_types,
                missing_files=missing_files,
                acquisition_tracks=acquisition_tracks,
            )
        )

    summaries, incomplete = _summarize_records(
        records=records,
        seeds=normalized_seeds,
        allow_incomplete=allow_incomplete,
    )
    if missing_files and not allow_incomplete:
        raise RuntimeError(f"Missing repeated-seed result files: {missing_files[:10]}")

    payload = {
        "module": module_name,
        "dataset": dataset,
        "manifest_variant": manifest_variant,
        "seeds": normalized_seeds,
        "checkpoints": list(checkpoints),
        "feature_types": list(feature_types),
        "acquisition_tracks": list(acquisition_tracks) if module_name == "acquisition_robustness" else None,
        "metric_contract": "docs/reproduction.md",
        "summary_unit": "across_repeated_full_evaluation_seeds",
        "allow_incomplete": bool(allow_incomplete),
        "n_records": len(records),
        "n_summary_rows": len(summaries),
        "n_incomplete_rows": len(incomplete),
        "missing_files": missing_files,
        "rows": summaries,
    }
    return payload


def write_repeated_seed_summary(
    *,
    module_name: str,
    dataset: str,
    manifest_variant: str,
    base_root: Path,
    seeds: Sequence[int | str],
    checkpoints: Sequence[str],
    feature_types: Sequence[str],
    allow_incomplete: bool = False,
    acquisition_tracks: Sequence[str] = ("native", "controlled"),
    summary_subdir: str = "seed_summary",
) -> dict[str, Any]:
    """Write repeated-seed JSON/CSV summaries for one module/dataset/variant."""
    payload = collect_repeated_seed_summary(
        module_name=module_name,
        dataset=dataset,
        manifest_variant=manifest_variant,
        base_root=base_root,
        seeds=seeds,
        checkpoints=checkpoints,
        feature_types=feature_types,
        allow_incomplete=allow_incomplete,
        acquisition_tracks=acquisition_tracks,
    )
    normalized_seeds = payload["seeds"]
    missing_files = payload["missing_files"]
    summaries = payload["rows"]
    seed_roots = {
        str(seed): str(apply_seed_to_path(base_root, seed))
        for seed in normalized_seeds
    }
    summary_dir = base_root / summary_subdir
    manifest = {
        "module": module_name,
        "dataset": dataset,
        "manifest_variant": manifest_variant,
        "seeds": normalized_seeds,
        "seed_roots": seed_roots,
        "summary_json": str(summary_dir / "repeated_seed_summary.json"),
        "summary_csv": str(summary_dir / "repeated_seed_summary.csv"),
        "n_missing_files": len(missing_files),
        "missing_files": missing_files,
    }
    _write_json(summary_dir / "repeated_seed_summary.json", payload)
    _write_json(summary_dir / "repeated_seed_manifest.json", manifest)
    _write_csv(summary_dir / "repeated_seed_summary.csv", summaries)
    return payload


def write_extension_seed_summary(
    *,
    module_name: str,
    dataset: str,
    manifest_variant: str,
    base_root: Path,
    seeds: Sequence[int | str],
    checkpoints: Sequence[str],
    feature_types: Sequence[str],
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Write across-seed summaries for completed extension metrics."""
    if module_name not in MODULES:
        raise ValueError(f"Unsupported repeated-seed module: {module_name}")
    normalized_seeds = [normalize_seed(seed) for seed in seeds]
    records: list[dict[str, Any]] = []
    extension_roots: dict[str, str] = {}
    for seed in normalized_seeds:
        extension_root = apply_seed_to_path(base_root, seed) / EXTENSION_DIR_NAME
        extension_roots[str(seed)] = str(extension_root)
        records.extend(
            _extension_records_for_seed(
                module_name=module_name,
                seed=seed,
                extension_root=extension_root,
                checkpoints=checkpoints,
                feature_types=feature_types,
            )
        )

    summaries, incomplete = _summarize_records(
        records=records,
        seeds=normalized_seeds,
        allow_incomplete=allow_incomplete,
    )
    summary_dir = base_root / EXTENSION_DIR_NAME / "seed_summary"
    payload = {
        "module": module_name,
        "dataset": dataset,
        "manifest_variant": manifest_variant,
        "seeds": normalized_seeds,
        "checkpoints": list(checkpoints),
        "feature_types": list(feature_types),
        "summary_unit": "across_repeated_extension_evaluation_seeds",
        "confidence_interval": "two_sided_t_interval_for_mean_95pct",
        "allow_incomplete": bool(allow_incomplete),
        "n_records": len(records),
        "n_summary_rows": len(summaries),
        "n_incomplete_rows": len(incomplete),
        "extension_roots": extension_roots,
        "rows": summaries,
    }
    manifest = {
        "module": module_name,
        "dataset": dataset,
        "manifest_variant": manifest_variant,
        "seeds": normalized_seeds,
        "extension_roots": extension_roots,
        "summary_json": str(summary_dir / "extension_seed_summary.json"),
        "summary_csv": str(summary_dir / "extension_seed_summary.csv"),
    }
    _write_json(summary_dir / "extension_seed_summary.json", payload)
    _write_json(summary_dir / "extension_seed_manifest.json", manifest)
    _write_csv(summary_dir / "extension_seed_summary.csv", summaries)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write contract-driven repeated-seed summaries for REFORM-3D modules",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--module", required=True, choices=MODULES)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest-variant", required=True)
    parser.add_argument("--base-root", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--feature-types", nargs="+", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--acquisition-tracks", nargs="+", choices=("native", "controlled"), default=["native", "controlled"])
    parser.add_argument("--summary-subdir", default="seed_summary")
    parser.add_argument("--extension", action="store_true", help="Summarize metrics under evaluation_extensions/ instead of native results/.")
    args = parser.parse_args()

    if args.extension:
        payload = write_extension_seed_summary(
            module_name=args.module,
            dataset=args.dataset,
            manifest_variant=args.manifest_variant,
            base_root=args.base_root,
            seeds=args.seeds,
            checkpoints=args.checkpoints,
            feature_types=args.feature_types,
            allow_incomplete=args.allow_incomplete,
        )
        print(json.dumps({"summary_rows": payload["n_summary_rows"]}, indent=2))
    else:
        payload = write_repeated_seed_summary(
            module_name=args.module,
            dataset=args.dataset,
            manifest_variant=args.manifest_variant,
            base_root=args.base_root,
            seeds=args.seeds,
            checkpoints=args.checkpoints,
            feature_types=args.feature_types,
            allow_incomplete=args.allow_incomplete,
            acquisition_tracks=args.acquisition_tracks,
            summary_subdir=args.summary_subdir,
        )
        print(json.dumps({"summary_rows": payload["n_summary_rows"], "missing_files": payload["missing_files"]}, indent=2))


if __name__ == "__main__":
    main()
