"""Aggregate organ-granularity stability results from baseline feature bundles."""

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
from .organ_granularity_stability import (
    DEFAULT_TAXONOMIES,
    SUPPORTED_DATASETS,
    _evaluation_dir,
    validate_granularity_task,
)
from .region_definition_stability import (
    DEFAULT_CHECKPOINTS,
    DEFAULT_FEATURE_TYPES,
    DEFAULT_SEEDS,
)
from .region_definition_stability_analysis import (
    ANATOMICAL_METRICS,
    BOOTSTRAP_SEED,
    CHECKPOINT_PAIRS,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    RETRIEVAL_METRICS,
    _bootstrap_crossed_effects,
    _bootstrap_retrieval_effects,
    _interval,
    _paired_retrieval_difference,
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
    taxonomy: str,
    checkpoint: str,
    feature_type: str,
) -> tuple[dict[int, dict[str, dict[str, np.ndarray]]], dict[int, dict[str, Any]]]:
    units: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    payloads: dict[int, dict[str, Any]] = {}
    for seed in DEFAULT_SEEDS:
        payload = _load_json(
            _evaluation_dir(dataset, taxonomy, checkpoint, feature_type)
            / f"retrieval_seed_{seed}.json"
        )
        if payload.get("status") != "ok":
            raise ValueError(
                f"Incomplete retrieval artifact for "
                f"{dataset}/{taxonomy}/{checkpoint}/{feature_type}/seed_{seed}"
            )
        units[int(seed)] = {}
        for direction in ("ct_to_mr", "mr_to_ct"):
            rows = payload[direction]["query_results"]
            units[int(seed)][direction] = {
                metric: np.asarray(
                    [float(row[metric]) for row in rows],
                    dtype=np.float64,
                )
                for metric in RETRIEVAL_METRICS
            }
            units[int(seed)][direction]["_sample_ids"] = np.asarray(
                [str(row["sample_id"]) for row in rows],
                dtype=str,
            )
        payloads[int(seed)] = payload
    return units, payloads


def _anatomical_units(
    dataset: str,
    taxonomy: str,
    checkpoint: str,
    feature_type: str,
) -> tuple[dict[str, np.ndarray], int]:
    payload = _load_json(
        _evaluation_dir(dataset, taxonomy, checkpoint, feature_type)
        / "anatomical_transfer.json"
    )
    if payload.get("status") != "ok":
        raise ValueError(
            f"Incomplete anatomical artifact for "
            f"{dataset}/{taxonomy}/{checkpoint}/{feature_type}"
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
    if any(not np.isfinite(values).all() for values in matrices.values()):
        raise ValueError(
            f"Incomplete anatomical matrix for "
            f"{dataset}/{taxonomy}/{checkpoint}/{feature_type}"
        )
    return matrices, len(organs)


def finite_pool_random_map_chance(
    n_organs: int,
    targets_per_organ: int,
) -> float:
    """Expected average precision of a uniformly random finite ranking."""
    total_targets = int(n_organs) * int(targets_per_organ)
    positives = int(targets_per_organ)
    if n_organs < 2 or positives <= 0 or total_targets <= positives:
        raise ValueError("Need a nontrivial balanced retrieval pool")
    reciprocal_sum = 0.0
    for rank in range(1, total_targets + 1):
        expected_hits = 1.0 + (rank - 1) * (positives - 1) / (total_targets - 1)
        reciprocal_sum += expected_hits / float(rank)
    return reciprocal_sum / float(total_targets)


def _chance_correct(score: float, chance: float) -> float:
    denominator = 1.0 - float(chance)
    if denominator <= 1e-12:
        raise ValueError(f"Invalid chance level: {chance}")
    return float((float(score) - float(chance)) / denominator)


def _retrieval_seed_values(
    payloads: Mapping[int, Mapping[str, Any]],
    metric: str,
) -> tuple[list[float], list[float]]:
    raw_values: list[float] = []
    corrected_values: list[float] = []
    for seed in DEFAULT_SEEDS:
        payload = payloads[int(seed)]
        direction_raw: list[float] = []
        direction_corrected: list[float] = []
        for direction in ("ct_to_mr", "mr_to_ct"):
            result = payload[direction]
            raw = float(result["overall"][metric])
            if metric == "map":
                chance = finite_pool_random_map_chance(
                    int(result["n_organs"]),
                    int(result["targets_per_organ"]),
                )
            else:
                chance = float(result["organ_balanced_chance"][metric])
            direction_raw.append(raw)
            direction_corrected.append(_chance_correct(raw, chance))
        raw_values.append(float(np.mean(direction_raw)))
        corrected_values.append(float(np.mean(direction_corrected)))
    return raw_values, corrected_values


def _anatomical_seed_values(
    matrices: Mapping[str, np.ndarray],
    metric: str,
    n_classes: int,
) -> tuple[list[float], list[float] | None]:
    raw_values = [float(value) for value in matrices[metric].mean(axis=1)]
    if metric == "adaptation_gain":
        return raw_values, None
    chance = 1.0 / float(n_classes)
    return raw_values, [_chance_correct(value, chance) for value in raw_values]


def _descriptive_rows(
    datasets: Sequence[str],
    checkpoints: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for taxonomy in DEFAULT_TAXONOMIES:
            for checkpoint in checkpoints:
                for feature_type in DEFAULT_FEATURE_TYPES:
                    retrieval, retrieval_payloads = _retrieval_units(
                        dataset,
                        taxonomy,
                        checkpoint,
                        feature_type,
                    )
                    del retrieval
                    anatomical, n_classes = _anatomical_units(
                        dataset,
                        taxonomy,
                        checkpoint,
                        feature_type,
                    )
                    for metric in RETRIEVAL_METRICS:
                        raw, corrected = _retrieval_seed_values(
                            retrieval_payloads,
                            metric,
                        )
                        rows.append(
                            {
                                "dataset": dataset,
                                "taxonomy": taxonomy,
                                "checkpoint": checkpoint,
                                "feature_type": feature_type,
                                "evaluation": "retrieval",
                                "metric": metric,
                                "n_classes": n_classes,
                                "raw_mean": float(np.mean(raw)),
                                "raw_std": float(np.std(raw, ddof=1)),
                                "chance_corrected_mean": float(np.mean(corrected)),
                                "chance_corrected_std": float(np.std(corrected, ddof=1)),
                                "n_seeds": len(raw),
                            }
                        )
                    for metric in ANATOMICAL_METRICS:
                        raw, corrected = _anatomical_seed_values(
                            anatomical,
                            metric,
                            n_classes,
                        )
                        row = {
                            "dataset": dataset,
                            "taxonomy": taxonomy,
                            "checkpoint": checkpoint,
                            "feature_type": feature_type,
                            "evaluation": "anatomical",
                            "metric": metric,
                            "n_classes": n_classes,
                            "raw_mean": float(np.mean(raw)),
                            "raw_std": float(np.std(raw, ddof=1)),
                            "chance_corrected_mean": None,
                            "chance_corrected_std": None,
                            "n_seeds": len(raw),
                        }
                        if corrected is not None:
                            row["chance_corrected_mean"] = float(np.mean(corrected))
                            row["chance_corrected_std"] = float(np.std(corrected, ddof=1))
                        rows.append(row)
    return rows


def _checkpoint_effect_rows(
    datasets: Sequence[str],
    *,
    resamples: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED + 17)
    for dataset in datasets:
        for taxonomy in DEFAULT_TAXONOMIES:
            for feature_type in DEFAULT_FEATURE_TYPES:
                retrieval_effects = []
                anatomical_effects = []
                estimates: list[dict[str, float]] = []
                for target_checkpoint, reference_checkpoint in CHECKPOINT_PAIRS:
                    target_retrieval, _ = _retrieval_units(
                        dataset,
                        taxonomy,
                        target_checkpoint,
                        feature_type,
                    )
                    reference_retrieval, _ = _retrieval_units(
                        dataset,
                        taxonomy,
                        reference_checkpoint,
                        feature_type,
                    )
                    retrieval_effects.append(
                        _paired_retrieval_difference(
                            target_retrieval,
                            reference_retrieval,
                        )
                    )
                    target_anatomical, _ = _anatomical_units(
                        dataset,
                        taxonomy,
                        target_checkpoint,
                        feature_type,
                    )
                    reference_anatomical, _ = _anatomical_units(
                        dataset,
                        taxonomy,
                        reference_checkpoint,
                        feature_type,
                    )
                    anatomical_effects.append(
                        {
                            metric: target_anatomical[metric]
                            - reference_anatomical[metric]
                            for metric in ANATOMICAL_METRICS
                        }
                    )
                    estimates.append(
                        {
                            **{
                                metric: float(
                                    np.mean(
                                        [
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
                                        ]
                                    )
                                )
                                for metric in RETRIEVAL_METRICS
                            },
                            **{
                                metric: float(anatomical_effects[-1][metric].mean())
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
                    pair = f"{target_checkpoint}__minus__{reference_checkpoint}"
                    for metric_index, metric in enumerate(RETRIEVAL_METRICS):
                        rows.append(
                            {
                                "dataset": dataset,
                                "taxonomy": taxonomy,
                                "feature_type": feature_type,
                                "evaluation": "retrieval",
                                "metric": metric,
                                "pair": pair,
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
                                "taxonomy": taxonomy,
                                "feature_type": feature_type,
                                "evaluation": "anatomical",
                                "metric": metric,
                                "pair": pair,
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
            str(row["taxonomy"]),
            str(row["feature_type"]),
            str(row["evaluation"]),
            str(row["metric"]),
            str(row["checkpoint"]),
        ): float(row["raw_mean"])
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
                    fine_scores = {
                        checkpoint: lookup[
                            (
                                dataset,
                                "fine_organ",
                                feature_type,
                                evaluation,
                                metric,
                                checkpoint,
                            )
                        ]
                        for checkpoint in checkpoints
                    }
                    family_scores = {
                        checkpoint: lookup[
                            (
                                dataset,
                                "organ_family",
                                feature_type,
                                evaluation,
                                metric,
                                checkpoint,
                            )
                        ]
                        for checkpoint in checkpoints
                    }
                    fine_order = sorted(
                        checkpoints,
                        key=lambda checkpoint: fine_scores[checkpoint],
                        reverse=True,
                    )
                    family_order = sorted(
                        checkpoints,
                        key=lambda checkpoint: family_scores[checkpoint],
                        reverse=True,
                    )
                    fine_rank = {
                        checkpoint: rank
                        for rank, checkpoint in enumerate(fine_order, start=1)
                    }
                    family_rank = {
                        checkpoint: rank
                        for rank, checkpoint in enumerate(family_order, start=1)
                    }
                    tau = kendalltau(
                        [fine_rank[checkpoint] for checkpoint in checkpoints],
                        [family_rank[checkpoint] for checkpoint in checkpoints],
                    ).statistic
                    rows.append(
                        {
                            "dataset": dataset,
                            "feature_type": feature_type,
                            "evaluation": evaluation,
                            "metric": metric,
                            "kendall_tau": float(tau),
                            "fine_top_checkpoint": fine_order[0],
                            "family_top_checkpoint": family_order[0],
                            "top_checkpoint_retained": fine_order[0] == family_order[0],
                            "mean_rank_displacement": float(
                                np.mean(
                                    [
                                        abs(
                                            fine_rank[checkpoint]
                                            - family_rank[checkpoint]
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
            str(row["taxonomy"]),
            str(row["feature_type"]),
            str(row["evaluation"]),
            str(row["metric"]),
            str(row["pair"]),
        ): float(row["paired_difference"])
        for row in checkpoint_rows
    }
    rows: list[dict[str, Any]] = []
    for key, fine_difference in lookup.items():
        dataset, taxonomy, feature_type, evaluation, metric, pair = key
        if taxonomy != "fine_organ":
            continue
        family_difference = lookup[
            (
                dataset,
                "organ_family",
                feature_type,
                evaluation,
                metric,
                pair,
            )
        ]
        rows.append(
            {
                "dataset": dataset,
                "feature_type": feature_type,
                "evaluation": evaluation,
                "metric": metric,
                "pair": pair,
                "fine_paired_difference": fine_difference,
                "family_paired_difference": family_difference,
                "fine_direction": int(np.sign(fine_difference)),
                "family_direction": int(np.sign(family_difference)),
                "direction_preserved": (
                    int(np.sign(fine_difference))
                    == int(np.sign(family_difference))
                ),
            }
        )
    return rows


def analyze_organ_granularity(
    datasets: Sequence[str],
    *,
    checkpoints: Sequence[str] = DEFAULT_CHECKPOINTS,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    for dataset in datasets:
        validate_granularity_task(
            dataset,
            checkpoints=checkpoints,
            require_evaluations=True,
        )
    descriptive = _descriptive_rows(datasets, checkpoints)
    checkpoint_effects = _checkpoint_effect_rows(
        datasets,
        resamples=int(bootstrap_resamples),
    )
    rankings = _ranking_rows(descriptive, datasets, checkpoints)
    directions = _direction_stability_rows(checkpoint_effects)
    output_root = (
        CROSS_MODAL_OUTPUTS
        / "protocol_stability"
        / "organ_granularity"
        / "analysis"
    )
    output_paths = {
        "descriptive": output_root / "taxonomy_summary.csv",
        "checkpoint_effects": output_root / "paired_checkpoint_effects.csv",
        "rankings": output_root / "ranking_stability.csv",
        "directions": output_root / "checkpoint_direction_stability.csv",
        "summary": output_root / "analysis_summary.json",
    }
    _write_csv(output_paths["descriptive"], descriptive)
    _write_csv(output_paths["checkpoint_effects"], checkpoint_effects)
    _write_csv(output_paths["rankings"], rankings)
    _write_csv(output_paths["directions"], directions)
    summary = {
        "schema_version": 1,
        "datasets": list(datasets),
        "taxonomies": list(DEFAULT_TAXONOMIES),
        "checkpoints": list(checkpoints),
        "feature_types": list(DEFAULT_FEATURE_TYPES),
        "seeds": list(DEFAULT_SEEDS),
        "bootstrap_resamples": int(bootstrap_resamples),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "checkpoint_family_size": len(CHECKPOINT_PAIRS),
        "n_descriptive_rows": len(descriptive),
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
            analyze_organ_granularity(
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
