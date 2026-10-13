"""Aggregate Region Definition stability results with paired bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import kendalltau

from .config import CROSS_MODAL_OUTPUTS
from .region_definition_stability import (
    BASELINE_SURFACE,
    DEFAULT_CHECKPOINTS,
    DEFAULT_FEATURE_TYPES,
    DEFAULT_SEEDS,
    REGION_SURFACES,
    SUPPORTED_DATASETS,
    _evaluation_dir,
    validate_region_task,
)


BOOTSTRAP_SEED = 2026
DEFAULT_BOOTSTRAP_RESAMPLES = 50_000
RETRIEVAL_METRICS = ("map", "top@1", "top@5")
ANATOMICAL_METRICS = ("probe_ba", "centroid_ba", "adaptation_gain")
CHECKPOINT_PAIRS = (
    ("med3dino_rel_96", "med3dino_iso_96"),
    ("med3dino_sa_96", "med3dino_iso_96"),
    ("med3dino_sa_96", "med3dino_rel_96"),
)
SURFACE_COMPARISONS = (
    ("crop_extent", "tight_bbox0_context", "baseline_full_bbox_context"),
    ("outside_organ_context", "organ_only_bbox5", "baseline_full_bbox_context"),
    ("combined_crop_and_masking", "organ_only_bbox0", "baseline_full_bbox_context"),
    ("mask_morphology", "eroded_1mm_bbox5", "organ_only_bbox5"),
)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    temporary_path = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _retrieval_units(
    dataset: str,
    surface: str,
    checkpoint: str,
    feature_type: str,
) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    units: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for seed in DEFAULT_SEEDS:
        payload = _load_json(
            _evaluation_dir(dataset, surface, checkpoint, feature_type)
            / f"retrieval_seed_{seed}.json"
        )
        if payload.get("status") != "ok":
            raise ValueError(
                f"Incomplete retrieval artifact for "
                f"{dataset}/{surface}/{checkpoint}/{feature_type}/seed_{seed}"
            )
        units[int(seed)] = {}
        for direction in ("ct_to_mr", "mr_to_ct"):
            rows = payload[direction]["query_results"]
            units[int(seed)][direction] = {
                metric: np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
                for metric in RETRIEVAL_METRICS
            }
            units[int(seed)][direction]["_sample_ids"] = np.asarray(
                [str(row["sample_id"]) for row in rows],
                dtype=str,
            )
    return units


def _anatomical_units(
    dataset: str,
    surface: str,
    checkpoint: str,
    feature_type: str,
) -> dict[str, np.ndarray]:
    payload = _load_json(
        _evaluation_dir(dataset, surface, checkpoint, feature_type)
        / "anatomical_transfer.json"
    )
    if payload.get("status") != "ok":
        raise ValueError(
            f"Incomplete anatomical artifact for "
            f"{dataset}/{surface}/{checkpoint}/{feature_type}"
        )
    organs = sorted(payload["per_organ"])
    seed_to_index = {int(seed): index for index, seed in enumerate(DEFAULT_SEEDS)}
    matrices = {
        metric: np.full((len(DEFAULT_SEEDS), len(organs)), np.nan, dtype=np.float64)
        for metric in ANATOMICAL_METRICS
    }
    for organ_index, organ in enumerate(organs):
        for row in payload["per_organ"][organ]["seed_results"]:
            seed_index = seed_to_index[int(row["seed"])]
            matrices["probe_ba"][seed_index, organ_index] = float(
                row["target_query_accuracy"]
            )
            matrices["centroid_ba"][seed_index, organ_index] = float(
                row["nearest_centroid_target_query_accuracy"]
            )
            matrices["adaptation_gain"][seed_index, organ_index] = float(
                row["adaptation_gain_target_query_accuracy"]
            )
    for metric, matrix in matrices.items():
        if not np.isfinite(matrix).all():
            raise ValueError(
                f"Incomplete anatomical matrix for "
                f"{dataset}/{surface}/{checkpoint}/{feature_type}/{metric}"
            )
    return matrices


def _paired_retrieval_difference(
    target: dict[int, dict[str, dict[str, np.ndarray]]],
    reference: dict[int, dict[str, dict[str, np.ndarray]]],
) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    differences: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for seed in DEFAULT_SEEDS:
        differences[int(seed)] = {}
        for direction in ("ct_to_mr", "mr_to_ct"):
            if not np.array_equal(
                target[int(seed)][direction]["_sample_ids"],
                reference[int(seed)][direction]["_sample_ids"],
            ):
                raise ValueError(
                    f"Paired retrieval query IDs differ for seed {seed}/{direction}"
                )
            differences[int(seed)][direction] = {
                metric: (
                    target[int(seed)][direction][metric]
                    - reference[int(seed)][direction][metric]
                )
                for metric in RETRIEVAL_METRICS
            }
    return differences


def _bootstrap_retrieval_effects(
    effect_units: Sequence[dict[int, dict[str, dict[str, np.ndarray]]]],
    *,
    resamples: int,
    rng: np.random.Generator,
    chunk_size: int = 256,
) -> np.ndarray:
    comparison_count = len(effect_units)
    metric_count = len(RETRIEVAL_METRICS)
    seed_count = len(DEFAULT_SEEDS)
    within_seed = np.empty(
        (
            resamples,
            seed_count,
            seed_count,
            comparison_count,
            metric_count,
        ),
        dtype=np.float64,
    )
    for draw_index in range(seed_count):
        for seed_index, seed in enumerate(DEFAULT_SEEDS):
            direction_means = []
            for direction in ("ct_to_mr", "mr_to_ct"):
                values = np.stack(
                    [
                        np.stack(
                            [
                                effect_units[comparison][int(seed)][direction][
                                    metric
                                ]
                                for metric in RETRIEVAL_METRICS
                            ],
                            axis=0,
                        )
                        for comparison in range(comparison_count)
                    ],
                    axis=0,
                )
                unit_count = values.shape[-1]
                sampled_means = np.empty(
                    (resamples, comparison_count, metric_count),
                    dtype=np.float64,
                )
                for start in range(0, resamples, chunk_size):
                    stop = min(start + chunk_size, resamples)
                    indices = rng.integers(
                        0,
                        unit_count,
                        size=(stop - start, unit_count),
                    )
                    sampled = np.take(values, indices, axis=-1)
                    sampled_means[start:stop] = sampled.mean(axis=-1).transpose(
                        2,
                        0,
                        1,
                    )
                direction_means.append(sampled_means)
            within_seed[:, draw_index, seed_index] = (
                direction_means[0] + direction_means[1]
            ) / 2.0

    seed_indices = rng.integers(
        0,
        seed_count,
        size=(resamples, seed_count),
    )
    bootstrap = np.empty(
        (resamples, comparison_count, metric_count),
        dtype=np.float64,
    )
    row_indices = np.arange(resamples)
    for draw_index in range(seed_count):
        selected = within_seed[
            row_indices,
            draw_index,
            seed_indices[:, draw_index],
        ]
        if draw_index == 0:
            bootstrap[:] = selected
        else:
            bootstrap += selected
    return bootstrap / float(seed_count)


def _bootstrap_crossed_effects(
    effect_matrices: Sequence[dict[str, np.ndarray]],
    *,
    resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    comparison_count = len(effect_matrices)
    stacked = np.stack(
        [
            np.stack(
                [effect_matrices[index][metric] for metric in ANATOMICAL_METRICS],
                axis=0,
            )
            for index in range(comparison_count)
        ],
        axis=0,
    )
    _, metric_count, seed_count, organ_count = stacked.shape
    seed_indices = rng.integers(0, seed_count, size=(resamples, seed_count))
    organ_indices = rng.integers(0, organ_count, size=(resamples, organ_count))
    seed_counts = np.stack(
        [(seed_indices == index).sum(axis=1) for index in range(seed_count)],
        axis=1,
    ).astype(np.float64)
    organ_counts = np.stack(
        [(organ_indices == index).sum(axis=1) for index in range(organ_count)],
        axis=1,
    ).astype(np.float64)
    return (
        np.einsum(
            "rs,cmso,ro->rcm",
            seed_counts,
            stacked,
            organ_counts,
            optimize=True,
        )
        / float(seed_count * organ_count)
    )


def _interval(
    values: np.ndarray,
    *,
    family_size: int,
) -> dict[str, float | bool]:
    adjusted_alpha = 0.05 / float(family_size)
    lower_percentile = 100.0 * adjusted_alpha / 2.0
    upper_percentile = 100.0 * (1.0 - adjusted_alpha / 2.0)
    ci_lower = float(np.percentile(values, 2.5))
    ci_upper = float(np.percentile(values, 97.5))
    adjusted_lower = float(np.percentile(values, lower_percentile))
    adjusted_upper = float(np.percentile(values, upper_percentile))
    return {
        "ci95_lower": ci_lower,
        "ci95_upper": ci_upper,
        "bonferroni_ci95_lower": adjusted_lower,
        "bonferroni_ci95_upper": adjusted_upper,
        "bonferroni_excludes_zero": bool(
            adjusted_lower > 0.0 or adjusted_upper < 0.0
        ),
    }


def _retrieval_seed_values(
    units: dict[int, dict[str, dict[str, np.ndarray]]],
    metric: str,
) -> list[float]:
    return [
        float(
            (
                units[int(seed)]["ct_to_mr"][metric].mean()
                + units[int(seed)]["mr_to_ct"][metric].mean()
            )
            / 2.0
        )
        for seed in DEFAULT_SEEDS
    ]


def _anatomical_seed_values(
    matrices: dict[str, np.ndarray],
    metric: str,
) -> list[float]:
    return [float(value) for value in matrices[metric].mean(axis=1)]


def _descriptive_rows(
    datasets: Sequence[str],
    checkpoints: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for surface in REGION_SURFACES:
            for checkpoint in checkpoints:
                for feature_type in DEFAULT_FEATURE_TYPES:
                    retrieval = _retrieval_units(
                        dataset,
                        surface.name,
                        checkpoint,
                        feature_type,
                    )
                    anatomical = _anatomical_units(
                        dataset,
                        surface.name,
                        checkpoint,
                        feature_type,
                    )
                    for metric in RETRIEVAL_METRICS:
                        values = _retrieval_seed_values(retrieval, metric)
                        rows.append(
                            {
                                "dataset": dataset,
                                "surface": surface.name,
                                "checkpoint": checkpoint,
                                "feature_type": feature_type,
                                "evaluation": "retrieval",
                                "metric": metric,
                                "mean": float(np.mean(values)),
                                "std": float(np.std(values, ddof=1)),
                                "n_seeds": len(values),
                            }
                        )
                    for metric in ANATOMICAL_METRICS:
                        values = _anatomical_seed_values(anatomical, metric)
                        rows.append(
                            {
                                "dataset": dataset,
                                "surface": surface.name,
                                "checkpoint": checkpoint,
                                "feature_type": feature_type,
                                "evaluation": "anatomical",
                                "metric": metric,
                                "mean": float(np.mean(values)),
                                "std": float(np.std(values, ddof=1)),
                                "n_seeds": len(values),
                            }
                        )
    return rows


def _surface_effect_rows(
    datasets: Sequence[str],
    checkpoints: Sequence[str],
    *,
    resamples: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for dataset in datasets:
        for checkpoint in checkpoints:
            for feature_type in DEFAULT_FEATURE_TYPES:
                retrieval_effects = []
                anatomical_effects = []
                estimates: list[dict[str, float]] = []
                for _, target_surface, reference_surface in SURFACE_COMPARISONS:
                    target_retrieval = _retrieval_units(
                        dataset,
                        target_surface,
                        checkpoint,
                        feature_type,
                    )
                    reference_retrieval = _retrieval_units(
                        dataset,
                        reference_surface,
                        checkpoint,
                        feature_type,
                    )
                    retrieval_effects.append(
                        _paired_retrieval_difference(
                            target_retrieval,
                            reference_retrieval,
                        )
                    )
                    target_anatomical = _anatomical_units(
                        dataset,
                        target_surface,
                        checkpoint,
                        feature_type,
                    )
                    reference_anatomical = _anatomical_units(
                        dataset,
                        reference_surface,
                        checkpoint,
                        feature_type,
                    )
                    anatomical_effects.append(
                        {
                            metric: (
                                target_anatomical[metric]
                                - reference_anatomical[metric]
                            )
                            for metric in ANATOMICAL_METRICS
                        }
                    )
                    estimates.append(
                        {
                            **{
                                metric: float(np.mean([
                                    (
                                        retrieval_effects[-1][seed]["ct_to_mr"][
                                            metric
                                        ].mean()
                                        + retrieval_effects[-1][seed]["mr_to_ct"][
                                            metric
                                        ].mean()
                                    )
                                    / 2.0
                                    for seed in DEFAULT_SEEDS
                                ]))
                                for metric in RETRIEVAL_METRICS
                            },
                            **{
                                metric: float(
                                    anatomical_effects[-1][metric].mean()
                                )
                                for metric in ANATOMICAL_METRICS
                            },
                        }
                    )

                retrieval_bootstrap = _bootstrap_retrieval_effects(
                    retrieval_effects,
                    resamples=resamples,
                    rng=rng,
                )
                anatomical_bootstrap = _bootstrap_crossed_effects(
                    anatomical_effects,
                    resamples=resamples,
                    rng=rng,
                )
                for comparison_index, (
                    comparison,
                    target_surface,
                    reference_surface,
                ) in enumerate(SURFACE_COMPARISONS):
                    for metric_index, metric in enumerate(RETRIEVAL_METRICS):
                        rows.append(
                            {
                                "dataset": dataset,
                                "checkpoint": checkpoint,
                                "feature_type": feature_type,
                                "evaluation": "retrieval",
                                "metric": metric,
                                "comparison": comparison,
                                "target_surface": target_surface,
                                "reference_surface": reference_surface,
                                "paired_difference": estimates[comparison_index][metric],
                                **_interval(
                                    retrieval_bootstrap[
                                        :,
                                        comparison_index,
                                        metric_index,
                                    ],
                                    family_size=len(SURFACE_COMPARISONS),
                                ),
                            }
                        )
                    for metric_index, metric in enumerate(ANATOMICAL_METRICS):
                        rows.append(
                            {
                                "dataset": dataset,
                                "checkpoint": checkpoint,
                                "feature_type": feature_type,
                                "evaluation": "anatomical",
                                "metric": metric,
                                "comparison": comparison,
                                "target_surface": target_surface,
                                "reference_surface": reference_surface,
                                "paired_difference": estimates[comparison_index][metric],
                                **_interval(
                                    anatomical_bootstrap[
                                        :,
                                        comparison_index,
                                        metric_index,
                                    ],
                                    family_size=len(SURFACE_COMPARISONS),
                                ),
                            }
                        )
    return rows


def _checkpoint_effect_rows(
    datasets: Sequence[str],
    *,
    resamples: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    for dataset in datasets:
        for surface in REGION_SURFACES:
            for feature_type in DEFAULT_FEATURE_TYPES:
                retrieval_effects = []
                anatomical_effects = []
                estimates: list[dict[str, float]] = []
                for target_checkpoint, reference_checkpoint in CHECKPOINT_PAIRS:
                    target_retrieval = _retrieval_units(
                        dataset,
                        surface.name,
                        target_checkpoint,
                        feature_type,
                    )
                    reference_retrieval = _retrieval_units(
                        dataset,
                        surface.name,
                        reference_checkpoint,
                        feature_type,
                    )
                    retrieval_effects.append(
                        _paired_retrieval_difference(
                            target_retrieval,
                            reference_retrieval,
                        )
                    )
                    target_anatomical = _anatomical_units(
                        dataset,
                        surface.name,
                        target_checkpoint,
                        feature_type,
                    )
                    reference_anatomical = _anatomical_units(
                        dataset,
                        surface.name,
                        reference_checkpoint,
                        feature_type,
                    )
                    anatomical_effects.append(
                        {
                            metric: (
                                target_anatomical[metric]
                                - reference_anatomical[metric]
                            )
                            for metric in ANATOMICAL_METRICS
                        }
                    )
                    estimates.append(
                        {
                            **{
                                metric: float(np.mean([
                                    (
                                        retrieval_effects[-1][seed]["ct_to_mr"][
                                            metric
                                        ].mean()
                                        + retrieval_effects[-1][seed]["mr_to_ct"][
                                            metric
                                        ].mean()
                                    )
                                    / 2.0
                                    for seed in DEFAULT_SEEDS
                                ]))
                                for metric in RETRIEVAL_METRICS
                            },
                            **{
                                metric: float(
                                    anatomical_effects[-1][metric].mean()
                                )
                                for metric in ANATOMICAL_METRICS
                            },
                        }
                    )
                retrieval_bootstrap = _bootstrap_retrieval_effects(
                    retrieval_effects,
                    resamples=resamples,
                    rng=rng,
                )
                anatomical_bootstrap = _bootstrap_crossed_effects(
                    anatomical_effects,
                    resamples=resamples,
                    rng=rng,
                )
                for pair_index, (
                    target_checkpoint,
                    reference_checkpoint,
                ) in enumerate(CHECKPOINT_PAIRS):
                    pair_name = f"{target_checkpoint}__minus__{reference_checkpoint}"
                    for metric_index, metric in enumerate(RETRIEVAL_METRICS):
                        rows.append(
                            {
                                "dataset": dataset,
                                "surface": surface.name,
                                "feature_type": feature_type,
                                "evaluation": "retrieval",
                                "metric": metric,
                                "pair": pair_name,
                                "target_checkpoint": target_checkpoint,
                                "reference_checkpoint": reference_checkpoint,
                                "paired_difference": estimates[pair_index][metric],
                                **_interval(
                                    retrieval_bootstrap[:, pair_index, metric_index],
                                    family_size=len(CHECKPOINT_PAIRS),
                                ),
                            }
                        )
                    for metric_index, metric in enumerate(ANATOMICAL_METRICS):
                        rows.append(
                            {
                                "dataset": dataset,
                                "surface": surface.name,
                                "feature_type": feature_type,
                                "evaluation": "anatomical",
                                "metric": metric,
                                "pair": pair_name,
                                "target_checkpoint": target_checkpoint,
                                "reference_checkpoint": reference_checkpoint,
                                "paired_difference": estimates[pair_index][metric],
                                **_interval(
                                    anatomical_bootstrap[:, pair_index, metric_index],
                                    family_size=len(CHECKPOINT_PAIRS),
                                ),
                            }
                        )
    return rows


def _ranking_rows(
    descriptive_rows: Sequence[Mapping[str, Any]],
    datasets: Sequence[str],
    checkpoints: Sequence[str],
) -> list[dict[str, Any]]:
    lookup = {
        (
            str(row["dataset"]),
            str(row["surface"]),
            str(row["feature_type"]),
            str(row["evaluation"]),
            str(row["metric"]),
            str(row["checkpoint"]),
        ): float(row["mean"])
        for row in descriptive_rows
    }
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for feature_type in DEFAULT_FEATURE_TYPES:
            for evaluation, metrics in (
                ("retrieval", RETRIEVAL_METRICS),
                ("anatomical", ANATOMICAL_METRICS),
            ):
                for metric in metrics:
                    baseline_scores = {
                        checkpoint: lookup[
                            (
                                dataset,
                                BASELINE_SURFACE.name,
                                feature_type,
                                evaluation,
                                metric,
                                checkpoint,
                            )
                        ]
                        for checkpoint in checkpoints
                    }
                    baseline_order = sorted(
                        checkpoints,
                        key=lambda checkpoint: baseline_scores[checkpoint],
                        reverse=True,
                    )
                    baseline_rank = {
                        checkpoint: rank
                        for rank, checkpoint in enumerate(baseline_order, start=1)
                    }
                    for surface in REGION_SURFACES:
                        surface_scores = {
                            checkpoint: lookup[
                                (
                                    dataset,
                                    surface.name,
                                    feature_type,
                                    evaluation,
                                    metric,
                                    checkpoint,
                                )
                            ]
                            for checkpoint in checkpoints
                        }
                        surface_order = sorted(
                            checkpoints,
                            key=lambda checkpoint: surface_scores[checkpoint],
                            reverse=True,
                        )
                        surface_rank = {
                            checkpoint: rank
                            for rank, checkpoint in enumerate(surface_order, start=1)
                        }
                        baseline_vector = [
                            baseline_rank[checkpoint] for checkpoint in checkpoints
                        ]
                        surface_vector = [
                            surface_rank[checkpoint] for checkpoint in checkpoints
                        ]
                        tau = kendalltau(
                            baseline_vector,
                            surface_vector,
                        ).statistic
                        rows.append(
                            {
                                "dataset": dataset,
                                "surface": surface.name,
                                "feature_type": feature_type,
                                "evaluation": evaluation,
                                "metric": metric,
                                "kendall_tau_with_baseline": float(tau),
                                "baseline_top_checkpoint": baseline_order[0],
                                "surface_top_checkpoint": surface_order[0],
                                "top_checkpoint_retained": (
                                    baseline_order[0] == surface_order[0]
                                ),
                                "mean_rank_displacement": float(
                                    np.mean(
                                        [
                                            abs(
                                                surface_rank[checkpoint]
                                                - baseline_rank[checkpoint]
                                            )
                                            for checkpoint in checkpoints
                                        ]
                                    )
                                ),
                            }
                        )
    return rows


def _direction_stability_rows(
    checkpoint_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {
        (
            str(row["dataset"]),
            str(row["surface"]),
            str(row["feature_type"]),
            str(row["evaluation"]),
            str(row["metric"]),
            str(row["pair"]),
        ): float(row["paired_difference"])
        for row in checkpoint_rows
    }
    rows: list[dict[str, Any]] = []
    pair_names = [
        f"{target}__minus__{reference}"
        for target, reference in CHECKPOINT_PAIRS
    ]
    for key, baseline_difference in lookup.items():
        dataset, surface, feature_type, evaluation, metric, pair = key
        if surface != BASELINE_SURFACE.name:
            continue
        baseline_direction = int(np.sign(baseline_difference))
        for candidate_surface in REGION_SURFACES:
            candidate_difference = lookup[
                (
                    dataset,
                    candidate_surface.name,
                    feature_type,
                    evaluation,
                    metric,
                    pair,
                )
            ]
            rows.append(
                {
                    "dataset": dataset,
                    "surface": candidate_surface.name,
                    "feature_type": feature_type,
                    "evaluation": evaluation,
                    "metric": metric,
                    "pair": pair,
                    "baseline_paired_difference": baseline_difference,
                    "surface_paired_difference": candidate_difference,
                    "baseline_direction": baseline_direction,
                    "surface_direction": int(np.sign(candidate_difference)),
                    "direction_preserved": (
                        baseline_direction == int(np.sign(candidate_difference))
                    ),
                }
            )
    dataset_count = len({str(row["dataset"]) for row in checkpoint_rows})
    expected = (
        dataset_count
        * len(REGION_SURFACES)
        * len(DEFAULT_FEATURE_TYPES)
        * (len(RETRIEVAL_METRICS) + len(ANATOMICAL_METRICS))
        * len(pair_names)
    )
    if len(rows) != expected:
        raise RuntimeError(
            f"Expected {expected} direction-stability rows, found {len(rows)}"
        )
    return rows


def analyze_region_definition(
    datasets: Sequence[str],
    *,
    checkpoints: Sequence[str] = DEFAULT_CHECKPOINTS,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    for dataset in datasets:
        validate_region_task(
            dataset,
            checkpoints=checkpoints,
            require_evaluations=True,
        )
    descriptive = _descriptive_rows(datasets, checkpoints)
    surface_effects = _surface_effect_rows(
        datasets,
        checkpoints,
        resamples=int(bootstrap_resamples),
    )
    checkpoint_effects = _checkpoint_effect_rows(
        datasets,
        resamples=int(bootstrap_resamples),
    )
    rankings = _ranking_rows(descriptive, datasets, checkpoints)
    directions = _direction_stability_rows(checkpoint_effects)

    output_root = (
        CROSS_MODAL_OUTPUTS
        / "protocol_stability"
        / "region_definition"
        / "analysis"
    )
    output_paths = {
        "descriptive": output_root / "surface_summary.csv",
        "surface_effects": output_root / "paired_surface_effects.csv",
        "checkpoint_effects": output_root / "paired_checkpoint_effects.csv",
        "rankings": output_root / "ranking_stability.csv",
        "directions": output_root / "checkpoint_direction_stability.csv",
        "summary": output_root / "analysis_summary.json",
    }
    _write_csv(output_paths["descriptive"], descriptive)
    _write_csv(output_paths["surface_effects"], surface_effects)
    _write_csv(output_paths["checkpoint_effects"], checkpoint_effects)
    _write_csv(output_paths["rankings"], rankings)
    _write_csv(output_paths["directions"], directions)
    summary = {
        "schema_version": 1,
        "datasets": list(datasets),
        "checkpoints": list(checkpoints),
        "feature_types": list(DEFAULT_FEATURE_TYPES),
        "seeds": list(DEFAULT_SEEDS),
        "bootstrap_resamples": int(bootstrap_resamples),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "surface_family_size": len(SURFACE_COMPARISONS),
        "checkpoint_family_size": len(CHECKPOINT_PAIRS),
        "n_descriptive_rows": len(descriptive),
        "n_surface_effect_rows": len(surface_effects),
        "n_checkpoint_effect_rows": len(checkpoint_effects),
        "n_ranking_rows": len(rankings),
        "n_direction_rows": len(directions),
        "outputs": {key: str(value) for key, value in output_paths.items()},
    }
    _atomic_write_json(output_paths["summary"], summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(SUPPORTED_DATASETS),
        choices=SUPPORTED_DATASETS,
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=list(DEFAULT_CHECKPOINTS),
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
    )
    args = parser.parse_args()
    print(
        json.dumps(
            analyze_region_definition(
                args.datasets,
                checkpoints=args.checkpoints,
                bootstrap_resamples=args.bootstrap_resamples,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
