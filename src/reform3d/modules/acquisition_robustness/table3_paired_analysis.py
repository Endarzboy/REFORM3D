#!/usr/bin/env python
"""Paired uncertainty analysis for the controlled acquisition metrics in Table 3."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from reform3d.core.markdown_tables import render_dataset_metric_table

from .controlled_uncertainty_analysis import (
    _compact_cka,
    _compact_cka_from_payload,
    _load_bundle,
    _rowwise_cosine_distance,
)


DEFAULT_RESULTS_ROOT = Path.cwd() / "workspace" / "runs"
DEFAULT_OUTPUT_DIR = Path.cwd() / "tables"

DATASETS = (
    "abdomenatlas",
    "kits23",
    "imagecas",
    "totalsegmentermri",
    "jhu_stroke",
)
DATASET_LABELS = {
    "abdomenatlas": "AbdomenAtlas",
    "kits23": "KiTS23",
    "imagecas": "ImageCAS",
    "totalsegmentermri": "TotalSegmentator-MR",
    "jhu_stroke": "JHU",
}
SEEDS = (42, 123, 456)
FEATURE_TYPES = ("cls", "avg_pool")
FEATURE_LABELS = {
    "cls": "CLS",
    "avg_pool": "Average pooled",
}
CHECKPOINTS = (
    "med3dino_iso_96",
    "med3dino_rel_96",
    "med3dino_sa_96",
)
CHECKPOINT_LABELS = {
    "med3dino_iso_96": "Med3DINO-ISO",
    "med3dino_rel_96": "Med3DINO-REL",
    "med3dino_sa_96": "Med3DINO-SA",
}
CONTRASTS = (
    ("rel_vs_iso", "med3dino_rel_96", "med3dino_iso_96"),
    ("sa_vs_iso", "med3dino_sa_96", "med3dino_iso_96"),
    ("sa_vs_rel", "med3dino_sa_96", "med3dino_rel_96"),
)
METRICS = {
    "drift": {
        "label": "Drift",
        "direction": "lower",
        "paired_unit": "matched_case",
    },
    "cka": {
        "label": "CKA",
        "direction": "higher",
        "paired_unit": "matched_case_joint_cka_recomputation",
    },
    "transfer": {
        "label": "Semantic transfer",
        "direction": "higher",
        "paired_unit": "matched_directed_spacing_transfer",
    },
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "dataset_label",
        "feature_type",
        "metric",
        "metric_direction",
        "paired_unit",
        "contrast",
        "method_a",
        "method_b",
        "n_seeds",
        "n_matched_cases_min",
        "n_matched_cases_max",
        "n_transfer_directions",
        "method_a_mean",
        "method_a_seed_std",
        "method_b_mean",
        "method_b_seed_std",
        "mean_difference_a_minus_b",
        "mean_improvement_a_over_b",
        "ci95_difference_low",
        "ci95_difference_high",
        "ci95_improvement_low",
        "ci95_improvement_high",
        "adjusted_confidence_level",
        "interval_method",
        "adjusted_difference_low",
        "adjusted_difference_high",
        "adjusted_improvement_low",
        "adjusted_improvement_high",
        "multiplicity_method",
        "decision",
        "bootstrap_resamples",
        "bootstrap_seed",
        "cka_method",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fieldnames} for row in rows)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    return sum((index + 1) * ord(character) for index, character in enumerate(text)) % 2_000_000_000


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def _bundle_path(
    results_root: Path,
    dataset: str,
    seed: int,
    checkpoint: str,
    feature_type: str,
) -> Path:
    return (
        results_root
        / "acquisition_robustness"
        / dataset
        / "original_bins"
        / "seeds"
        / f"seed_{seed}"
        / "results"
        / checkpoint
        / feature_type
        / "controlled_embedding_bundle.npz"
    )


def _metric_path(
    results_root: Path,
    dataset: str,
    seed: int,
    checkpoint: str,
    feature_type: str,
) -> Path:
    return _bundle_path(results_root, dataset, seed, checkpoint, feature_type).with_name(
        "perturbation_robustness.json"
    )


def _align_features(
    bundles: Mapping[str, tuple[dict[str, np.ndarray], list[str], str, list[str], dict[str, str]]],
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    list[str],
    str,
    list[str],
]:
    first_checkpoint = CHECKPOINTS[0]
    _, first_variants, first_reference, first_case_ids, first_kinds = bundles[first_checkpoint]
    variant_order = [
        variant
        for variant in first_variants
        if first_kinds.get(variant, "spacing") == "spacing"
    ]
    if first_reference not in variant_order:
        raise RuntimeError(f"Reference variant {first_reference} is not a spacing variant")

    common_case_ids = set(first_case_ids)
    for checkpoint in CHECKPOINTS[1:]:
        _, variants, reference, case_ids, kinds = bundles[checkpoint]
        spacing_variants = [
            variant
            for variant in variants
            if kinds.get(variant, "spacing") == "spacing"
        ]
        if spacing_variants != variant_order:
            raise RuntimeError(
                f"Variant mismatch for {checkpoint}: {spacing_variants} != {variant_order}"
            )
        if reference != first_reference:
            raise RuntimeError(f"Reference mismatch for {checkpoint}: {reference} != {first_reference}")
        common_case_ids &= set(case_ids)

    ordered_case_ids = [case_id for case_id in first_case_ids if case_id in common_case_ids]
    if len(ordered_case_ids) < 2:
        raise RuntimeError("Fewer than two matched cases across ISO, REL, and SA")

    aligned: dict[str, dict[str, np.ndarray]] = {}
    for checkpoint, (features_by_variant, _, _, case_ids, _) in bundles.items():
        index_by_case = {case_id: index for index, case_id in enumerate(case_ids)}
        indices = np.asarray([index_by_case[case_id] for case_id in ordered_case_ids], dtype=np.int64)
        aligned[checkpoint] = {
            variant: np.asarray(features_by_variant[variant][indices], dtype=np.float32)
            for variant in variant_order
        }
    return aligned, variant_order, first_reference, ordered_case_ids


def _compact_drift_per_case(
    features_by_variant: Mapping[str, np.ndarray],
    variant_order: Sequence[str],
    reference_variant: str,
) -> np.ndarray:
    reference = features_by_variant[reference_variant]
    distances = [
        _rowwise_cosine_distance(reference, features_by_variant[variant])
        for variant in variant_order
        if variant != reference_variant
    ]
    if not distances:
        raise RuntimeError("No non-reference spacing variants")
    return np.mean(np.stack(distances, axis=0), axis=0)


def _directed_transfer_values(
    payload: Mapping[str, Any],
    variant_order: Sequence[str],
) -> tuple[list[tuple[str, str]], np.ndarray]:
    semantic_transfer = payload.get("matched_semantic_transfer") or {}
    matrix = semantic_transfer.get("transfer_matrix") or {}
    keys = [
        (source_variant, target_variant)
        for source_variant in variant_order
        for target_variant in variant_order
        if source_variant != target_variant
    ]
    values = np.asarray(
        [float(matrix[source][target]) for source, target in keys],
        dtype=np.float64,
    )
    reported = float(semantic_transfer["cross_variant_accuracy"])
    if not np.isclose(float(values.mean()), reported, rtol=1e-8, atol=1e-10):
        raise RuntimeError(
            f"Transfer matrix mean {values.mean()} differs from reported value {reported}"
        )
    return keys, values


def _compact_cka_from_case_counts(
    features_by_variant: Mapping[str, np.ndarray],
    variant_order: Sequence[str],
    case_counts: np.ndarray,
) -> np.ndarray:
    counts = np.asarray(case_counts, dtype=np.float64).T
    sample_sizes = counts.sum(axis=0)
    if np.any(sample_sizes <= 1):
        raise RuntimeError("CKA resamples require at least two cases")

    grams: dict[str, np.ndarray] = {}
    weighted_grams: dict[str, np.ndarray] = {}
    gram_totals: dict[str, np.ndarray] = {}
    centered_self_inner: dict[str, np.ndarray] = {}
    for variant in variant_order:
        features = np.asarray(features_by_variant[variant], dtype=np.float64)
        gram = features @ features.T
        weighted_gram = gram @ counts
        gram_total = np.sum(counts * weighted_gram, axis=0)
        first_term = np.sum(counts * ((gram * gram) @ counts), axis=0)
        second_term = np.sum(counts * weighted_gram * weighted_gram, axis=0)
        centered_inner = (
            first_term
            - (2.0 * second_term / sample_sizes)
            + ((gram_total * gram_total) / (sample_sizes * sample_sizes))
        )
        grams[variant] = gram
        weighted_grams[variant] = weighted_gram
        gram_totals[variant] = gram_total
        centered_self_inner[variant] = np.maximum(centered_inner, 0.0)

    pair_values = []
    for source_index, source_variant in enumerate(variant_order):
        for target_variant in variant_order[source_index + 1 :]:
            source_gram = grams[source_variant]
            target_gram = grams[target_variant]
            first_term = np.sum(
                counts * ((source_gram * target_gram) @ counts),
                axis=0,
            )
            second_term = np.sum(
                counts
                * weighted_grams[source_variant]
                * weighted_grams[target_variant],
                axis=0,
            )
            centered_inner = (
                first_term
                - (2.0 * second_term / sample_sizes)
                + (
                    gram_totals[source_variant]
                    * gram_totals[target_variant]
                    / (sample_sizes * sample_sizes)
                )
            )
            denominator = np.sqrt(
                centered_self_inner[source_variant]
                * centered_self_inner[target_variant]
            )
            pair_values.append(
                np.divide(
                    centered_inner,
                    denominator,
                    out=np.zeros_like(centered_inner),
                    where=denominator > 1e-12,
                )
            )
    if not pair_values:
        raise RuntimeError("No spacing-variant pairs available for CKA")
    return np.mean(np.stack(pair_values, axis=0), axis=0)


def _analyze_seed_task(
    task: tuple[Path, str, int, str, int, int, str],
) -> dict[str, Any]:
    (
        results_root,
        dataset,
        seed,
        feature_type,
        bootstrap_resamples,
        bootstrap_seed,
        cka_method,
    ) = task
    bundles = {
        checkpoint: _load_bundle(
            _bundle_path(results_root, dataset, seed, checkpoint, feature_type)
        )
        for checkpoint in CHECKPOINTS
    }
    aligned, variant_order, reference_variant, case_ids = _align_features(bundles)
    metric_payloads = {
        checkpoint: _load_json(
            _metric_path(results_root, dataset, seed, checkpoint, feature_type)
        )
        for checkpoint in CHECKPOINTS
    }

    drift_values = {
        checkpoint: _compact_drift_per_case(
            aligned[checkpoint],
            variant_order,
            reference_variant,
        )
        for checkpoint in CHECKPOINTS
    }
    cka_points: dict[str, float] = {}
    for checkpoint in CHECKPOINTS:
        recomputed_cka = float(
            _compact_cka(
                aligned[checkpoint],
                variant_order,
                method=cka_method,
            )
        )
        reported_cka = _compact_cka_from_payload(
            metric_payloads[checkpoint],
            variant_order,
        )
        if reported_cka is None or not np.isclose(
            recomputed_cka,
            reported_cka,
            rtol=1e-7,
            atol=1e-9,
        ):
            raise RuntimeError(
                f"CKA bundle mismatch for {dataset}/seed_{seed}/"
                f"{checkpoint}/{feature_type}: {recomputed_cka} != {reported_cka}"
            )
        cka_points[checkpoint] = float(reported_cka)
    transfer_keys: list[tuple[str, str]] | None = None
    transfer_values: dict[str, np.ndarray] = {}
    for checkpoint in CHECKPOINTS:
        checkpoint_keys, checkpoint_values = _directed_transfer_values(
            metric_payloads[checkpoint],
            variant_order,
        )
        if transfer_keys is None:
            transfer_keys = checkpoint_keys
        elif checkpoint_keys != transfer_keys:
            raise RuntimeError(f"Transfer direction mismatch for {checkpoint}")
        transfer_values[checkpoint] = checkpoint_values
    assert transfer_keys is not None

    points = {
        checkpoint: {
            "drift": float(drift_values[checkpoint].mean()),
            "cka": cka_points[checkpoint],
            "transfer": float(transfer_values[checkpoint].mean()),
        }
        for checkpoint in CHECKPOINTS
    }
    bootstrap = {
        metric: {
            contrast_name: np.empty(bootstrap_resamples, dtype=np.float64)
            for contrast_name, _, _ in CONTRASTS
        }
        for metric in METRICS
    }
    rng = np.random.default_rng(
        int(bootstrap_seed) + _stable_seed(dataset, seed, feature_type)
    )
    n_cases = len(case_ids)
    n_transfer_directions = len(transfer_keys)
    case_counts = rng.multinomial(
        n_cases,
        np.full(n_cases, 1.0 / n_cases, dtype=np.float64),
        size=bootstrap_resamples,
    )
    transfer_counts = rng.multinomial(
        n_transfer_directions,
        np.full(
            n_transfer_directions,
            1.0 / n_transfer_directions,
            dtype=np.float64,
        ),
        size=bootstrap_resamples,
    )
    bootstrap_points = {
        checkpoint: {
            "drift": (
                case_counts @ np.asarray(drift_values[checkpoint], dtype=np.float64)
            )
            / n_cases,
            "cka": _compact_cka_from_case_counts(
                aligned[checkpoint],
                variant_order,
                case_counts,
            ),
            "transfer": (
                transfer_counts
                @ np.asarray(transfer_values[checkpoint], dtype=np.float64)
            )
            / n_transfer_directions,
        }
        for checkpoint in CHECKPOINTS
    }
    for contrast_name, method_a, method_b in CONTRASTS:
        for metric in METRICS:
            bootstrap[metric][contrast_name] = (
                bootstrap_points[method_a][metric]
                - bootstrap_points[method_b][metric]
            )

    return {
        "dataset": dataset,
        "seed": seed,
        "feature_type": feature_type,
        "n_matched_cases": n_cases,
        "n_transfer_directions": n_transfer_directions,
        "variant_order": variant_order,
        "reference_variant": reference_variant,
        "points": points,
        "bootstrap": bootstrap,
        "bundle_paths": {
            checkpoint: str(
                _bundle_path(results_root, dataset, seed, checkpoint, feature_type)
            )
            for checkpoint in CHECKPOINTS
        },
        "metric_paths": {
            checkpoint: str(
                _metric_path(results_root, dataset, seed, checkpoint, feature_type)
            )
            for checkpoint in CHECKPOINTS
        },
    }


def _run_seed_tasks(
    tasks: Sequence[tuple[Path, str, int, str, int, int, str]],
    workers: int,
) -> list[dict[str, Any]]:
    if workers <= 1:
        return [_analyze_seed_task(task) for task in tasks]
    results: list[dict[str, Any] | None] = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_analyze_seed_task, task): index
            for index, task in enumerate(tasks)
        }
        completed = 0
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
            completed += 1
            task = tasks[index]
            print(
                f"[{completed}/{len(tasks)}] {task[1]}/seed_{task[2]}/{task[3]}",
                flush=True,
            )
    return [result for result in results if result is not None]


def _hierarchical_distribution(
    seed_results: Sequence[Mapping[str, Any]],
    metric: str,
    contrast_name: str,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(
        int(bootstrap_seed)
        + _stable_seed(
            seed_results[0]["dataset"],
            seed_results[0]["feature_type"],
            metric,
            contrast_name,
            "aggregate",
        )
    )
    output = np.empty(bootstrap_resamples, dtype=np.float64)
    n_seeds = len(seed_results)
    for resample_index in range(bootstrap_resamples):
        selected_seed_indices = rng.integers(0, n_seeds, size=n_seeds)
        effects = []
        for seed_index in selected_seed_indices:
            seed_distribution = seed_results[int(seed_index)]["bootstrap"][metric][
                contrast_name
            ]
            effect_index = int(rng.integers(0, len(seed_distribution)))
            effects.append(float(seed_distribution[effect_index]))
        output[resample_index] = float(np.mean(effects))
    return output


def _centered_symmetric_interval(
    values: np.ndarray,
    point_estimate: float,
    confidence_level: float,
) -> tuple[float, float]:
    centered_absolute_deviations = np.abs(values - float(np.mean(values)))
    radius = float(
        np.percentile(
            centered_absolute_deviations,
            100.0 * float(confidence_level),
        )
    )
    return (
        float(point_estimate - radius),
        float(point_estimate + radius),
    )


def _aggregate_rows(
    seed_results: Sequence[Mapping[str, Any]],
    bootstrap_resamples: int,
    bootstrap_seed: int,
    cka_method: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for result in seed_results:
        key = (str(result["dataset"]), str(result["feature_type"]))
        grouped.setdefault(key, []).append(result)

    adjusted_confidence_level = 1.0 - (0.05 / len(CONTRASTS))
    rows: list[dict[str, Any]] = []
    for (dataset, feature_type), group in sorted(grouped.items()):
        ordered_group = sorted(group, key=lambda item: int(item["seed"]))
        if [int(item["seed"]) for item in ordered_group] != list(SEEDS):
            raise RuntimeError(f"Incomplete seed group for {dataset}/{feature_type}")
        for metric, metric_spec in METRICS.items():
            higher_is_better = metric_spec["direction"] == "higher"
            for contrast_name, method_a, method_b in CONTRASTS:
                method_a_points = [
                    float(item["points"][method_a][metric])
                    for item in ordered_group
                ]
                method_b_points = [
                    float(item["points"][method_b][metric])
                    for item in ordered_group
                ]
                difference_by_seed = [
                    method_a_value - method_b_value
                    for method_a_value, method_b_value in zip(
                        method_a_points,
                        method_b_points,
                    )
                ]
                mean_difference = float(np.mean(difference_by_seed))
                improvement_sign = 1.0 if higher_is_better else -1.0
                mean_improvement = improvement_sign * mean_difference
                difference_distribution = _hierarchical_distribution(
                    ordered_group,
                    metric,
                    contrast_name,
                    bootstrap_resamples,
                    bootstrap_seed,
                )
                improvement_distribution = improvement_sign * difference_distribution
                ci95_difference = _centered_symmetric_interval(
                    difference_distribution,
                    mean_difference,
                    0.95,
                )
                ci95_improvement = _centered_symmetric_interval(
                    improvement_distribution,
                    mean_improvement,
                    0.95,
                )
                adjusted_difference = _centered_symmetric_interval(
                    difference_distribution,
                    mean_difference,
                    adjusted_confidence_level,
                )
                adjusted_improvement = _centered_symmetric_interval(
                    improvement_distribution,
                    mean_improvement,
                    adjusted_confidence_level,
                )
                if adjusted_improvement[0] > 0.0:
                    decision = "method_a_better"
                elif adjusted_improvement[1] < 0.0:
                    decision = "method_b_better"
                else:
                    decision = "not_resolved"
                rows.append(
                    {
                        "dataset": dataset,
                        "dataset_label": DATASET_LABELS[dataset],
                        "feature_type": feature_type,
                        "metric": metric,
                        "metric_direction": metric_spec["direction"],
                        "paired_unit": metric_spec["paired_unit"],
                        "contrast": contrast_name,
                        "method_a": CHECKPOINT_LABELS[method_a],
                        "method_b": CHECKPOINT_LABELS[method_b],
                        "n_seeds": len(ordered_group),
                        "n_matched_cases_min": min(
                            int(item["n_matched_cases"]) for item in ordered_group
                        ),
                        "n_matched_cases_max": max(
                            int(item["n_matched_cases"]) for item in ordered_group
                        ),
                        "n_transfer_directions": min(
                            int(item["n_transfer_directions"]) for item in ordered_group
                        ),
                        "method_a_mean": float(np.mean(method_a_points)),
                        "method_a_seed_std": _sample_std(method_a_points),
                        "method_b_mean": float(np.mean(method_b_points)),
                        "method_b_seed_std": _sample_std(method_b_points),
                        "mean_difference_a_minus_b": mean_difference,
                        "mean_improvement_a_over_b": mean_improvement,
                        "ci95_difference_low": ci95_difference[0],
                        "ci95_difference_high": ci95_difference[1],
                        "ci95_improvement_low": ci95_improvement[0],
                        "ci95_improvement_high": ci95_improvement[1],
                        "adjusted_confidence_level": adjusted_confidence_level,
                        "interval_method": "hierarchical_centered_symmetric_bootstrap",
                        "adjusted_difference_low": adjusted_difference[0],
                        "adjusted_difference_high": adjusted_difference[1],
                        "adjusted_improvement_low": adjusted_improvement[0],
                        "adjusted_improvement_high": adjusted_improvement[1],
                        "multiplicity_method": (
                            "Bonferroni-adjusted centered symmetric bootstrap interval "
                            "across three "
                            "ISO/REL/SA contrasts within dataset-feature-metric"
                        ),
                        "decision": decision,
                        "bootstrap_resamples": bootstrap_resamples,
                        "bootstrap_seed": bootstrap_seed,
                        "cka_method": cka_method,
                    }
                )
    return rows


def _format_effect(row: Mapping[str, Any], adjusted: bool = False) -> str:
    if adjusted:
        low = float(row["adjusted_improvement_low"])
        high = float(row["adjusted_improvement_high"])
    else:
        low = float(row["ci95_improvement_low"])
        high = float(row["ci95_improvement_high"])
    return (
        f"{float(row['mean_improvement_a_over_b']):+.4f} "
        f"[{low:+.4f}, {high:+.4f}]"
    )


def _format_paired_cell(row: Mapping[str, Any]) -> str:
    effect = _format_effect(row, adjusted=True)
    if row["decision"] == "not_resolved":
        return effect
    return f"<strong>{effect}</strong>"


def _write_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    present_features = [
        feature_type
        for feature_type in FEATURE_TYPES
        if any(row["feature_type"] == feature_type for row in rows)
    ]
    present_datasets = [
        dataset
        for dataset in DATASETS
        if any(row["dataset"] == dataset for row in rows)
    ]
    dataset_columns = [
        (dataset, DATASET_LABELS[dataset])
        for dataset in present_datasets
    ]
    metric_columns = [
        (metric, str(metric_spec["label"]))
        for metric, metric_spec in METRICS.items()
    ]
    contrast_rows = [
        (
            contrast_name,
            f"{CHECKPOINT_LABELS[method_a]} − {CHECKPOINT_LABELS[method_b]}",
        )
        for contrast_name, method_a, method_b in CONTRASTS
    ]
    lines = [
        "# Table 3 Paired Analysis",
        "",
        "Effects are direction-normalized so that positive values favor method A. "
        "Each Markdown cell reports the effect and its Bonferroni-adjusted "
        "centered symmetric bootstrap interval. Bold cells have adjusted intervals "
        "that exclude zero. Unadjusted 95% intervals remain available in the CSV "
        "and JSON outputs. Intervals are generated "
        "from hierarchical resampling across evaluation seeds and the metric-specific "
        "paired unit.",
        "",
        "Drift and CKA jointly resample matched cases. Semantic transfer resamples "
        "the six directed cross-spacing transfer scores retained for each evaluation "
        "seed; it does not reconstruct a case-level paired transfer test.",
        "",
    ]
    for feature_type in present_features:
        lines.extend([f"## {FEATURE_LABELS[feature_type]} features", ""])
        values: dict[tuple[str, str, str], str] = {}
        for contrast_name, _ in contrast_rows:
            for dataset, _ in dataset_columns:
                for metric, _ in metric_columns:
                    row = next(
                        item
                        for item in rows
                        if item["feature_type"] == feature_type
                        and item["dataset"] == dataset
                        and item["metric"] == metric
                        and item["contrast"] == contrast_name
                    )
                    values[(contrast_name, dataset, metric)] = _format_paired_cell(row)
        lines.extend(
            render_dataset_metric_table(
                row_header="Contrast",
                rows=contrast_rows,
                datasets=dataset_columns,
                metrics=metric_columns,
                values=values,
            )
        )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_analysis(
    results_root: Path,
    output_dir: Path,
    datasets: Sequence[str],
    feature_types: Sequence[str],
    bootstrap_resamples: int,
    bootstrap_seed: int,
    cka_method: str,
    workers: int,
) -> dict[str, Any]:
    tasks = [
        (
            results_root,
            dataset,
            seed,
            feature_type,
            bootstrap_resamples,
            bootstrap_seed,
            cka_method,
        )
        for dataset in datasets
        for seed in SEEDS
        for feature_type in feature_types
    ]
    seed_results = _run_seed_tasks(tasks, workers)
    rows = _aggregate_rows(
        seed_results,
        bootstrap_resamples,
        bootstrap_seed,
        cka_method,
    )
    expected_rows = len(datasets) * len(feature_types) * len(METRICS) * len(CONTRASTS)
    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, found {len(rows)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "table3_paired_analysis.csv", rows)
    _write_markdown(output_dir / "table3_paired_analysis.md", rows)
    payload = {
        "analysis_scope": "table3_controlled_acquisition_paired_comparison",
        "results_root": str(results_root),
        "datasets": list(datasets),
        "seeds": list(SEEDS),
        "feature_types": list(feature_types),
        "checkpoints": list(CHECKPOINTS),
        "contrasts": [
            {
                "name": name,
                "method_a": method_a,
                "method_b": method_b,
            }
            for name, method_a, method_b in CONTRASTS
        ],
        "metrics": METRICS,
        "bootstrap": {
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "aggregation": "hierarchical_over_evaluation_seed_and_metric_specific_paired_unit",
            "multiplicity": (
                "Bonferroni-adjusted centered symmetric bootstrap interval across "
                "three ISO/REL/SA contrasts "
                "within each dataset-feature-metric family"
            ),
        },
        "cka_method": cka_method,
        "training_run_variance": "not_estimated",
        "semantic_transfer_limit": (
            "paired over directed spacing-transfer summaries because case-level "
            "predictions were not retained"
        ),
        "rows": rows,
        "seed_provenance": [
            {
                "dataset": result["dataset"],
                "seed": result["seed"],
                "feature_type": result["feature_type"],
                "n_matched_cases": result["n_matched_cases"],
                "n_transfer_directions": result["n_transfer_directions"],
                "variant_order": result["variant_order"],
                "reference_variant": result["reference_variant"],
                "bundle_paths": result["bundle_paths"],
                "metric_paths": result["metric_paths"],
            }
            for result in seed_results
        ],
    }
    _write_json(output_dir / "table3_paired_analysis.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute paired ISO/REL/SA comparisons for Table 3."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument(
        "--feature-types",
        nargs="+",
        choices=FEATURE_TYPES,
        default=list(FEATURE_TYPES),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--cka-method", choices=("auto", "gram", "feature"), default="auto")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_analysis(
        results_root=args.results_root,
        output_dir=args.output_dir,
        datasets=args.datasets,
        feature_types=args.feature_types,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        cka_method=args.cka_method,
        workers=args.workers,
    )
    print(f"rows: {len(payload['rows'])}")
    print(f"wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
