#!/usr/bin/env python
"""Paired uncertainty analysis for the cross-modal retrieval metrics in Table 5."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from reform3d.core.markdown_tables import render_dataset_metric_table

from .config import (
    get_checkpoint_feature_npz_path,
    get_checkpoint_metrics_dir,
    get_checkpoint_metrics_path,
    get_cross_modal_manifest_path,
)
from .data_loader import load_cross_modal_manifest
from .retrieval_metrics import compute_bidirectional_cross_modal_retrieval


DEFAULT_RESULTS_ROOT = Path.cwd() / "workspace" / "runs"
DEFAULT_OUTPUT_DIR = Path.cwd() / "tables"

DATASETS = (
    "totalsegmenter_ct_mr_anchor",
    "mmwhs_ct_mr",
    "amos22_ct_mr",
    "chaos_ct_mr",
)
DATASET_LABELS = {
    "totalsegmenter_ct_mr_anchor": "TotalSegmentator CT-MR",
    "mmwhs_ct_mr": "MMWHS",
    "amos22_ct_mr": "AMOS22",
    "chaos_ct_mr": "CHAOS",
}
MULTI_ORGAN_DATASETS = frozenset(DATASETS[:3])
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
    "map": "mAP",
    "top@1": "Top-1",
    "top@5": "Top-5",
}
DIRECTIONS = ("ct_to_mr", "mr_to_ct")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "dataset_label",
        "feature_type",
        "retrieval_protocol",
        "metric",
        "contrast",
        "method_a",
        "method_b",
        "n_seeds",
        "n_paired_ct_to_mr_queries_min",
        "n_paired_ct_to_mr_queries_max",
        "n_paired_mr_to_ct_queries_min",
        "n_paired_mr_to_ct_queries_max",
        "method_a_mean",
        "method_a_seed_std",
        "method_b_mean",
        "method_b_seed_std",
        "mean_difference_a_minus_b",
        "ci95_difference_low",
        "ci95_difference_high",
        "adjusted_confidence_level",
        "interval_method",
        "adjusted_difference_low",
        "adjusted_difference_high",
        "multiplicity_method",
        "decision",
        "bootstrap_resamples",
        "bootstrap_seed",
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


def _seed_root(results_root: Path, dataset: str, seed: int) -> Path:
    return (
        results_root
        / "cross_modal_alignment"
        / dataset
        / "core"
        / "seeds"
        / f"seed_{seed}"
    )


def _embedding_path(
    results_root: Path,
    dataset: str,
    seed: int,
    checkpoint: str,
    feature_type: str,
) -> Path:
    return get_checkpoint_feature_npz_path(
        _seed_root(results_root, dataset, seed) / "features",
        checkpoint,
        feature_type,
    )


def _metric_path(
    results_root: Path,
    dataset: str,
    seed: int,
    checkpoint: str,
    feature_type: str,
) -> Path:
    return get_checkpoint_metrics_path(
        _seed_root(results_root, dataset, seed) / "results",
        checkpoint,
        feature_type,
    )


def _single_organ_path(
    results_root: Path,
    dataset: str,
    seed: int,
    checkpoint: str,
    feature_type: str,
) -> Path:
    return (
        get_checkpoint_metrics_dir(
            _seed_root(results_root, dataset, seed) / "results",
            checkpoint,
            feature_type,
        )
        / "single_organ_alignment.json"
    )


def _load_embedding_bundle(path: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    with np.load(path, allow_pickle=True) as payload:
        features = np.asarray(payload["features"], dtype=np.float32)
        sample_ids = [str(value) for value in np.asarray(payload["sample_ids"]).tolist()]
    if features.shape[0] != len(sample_ids):
        raise RuntimeError(f"Feature and sample counts differ for {path}")
    if len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError(f"Duplicate sample IDs in {path}")
    return (
        {
            sample_id: features[index]
            for index, sample_id in enumerate(sample_ids)
        },
        sample_ids,
    )


def _supported_organs(
    samples: Sequence[Mapping[str, Any]],
    required_modalities: Sequence[str] = ("ct", "mr"),
) -> list[str]:
    organs_by_modality = {
        modality: {
            str(sample["primary_organ"])
            for sample in samples
            if str(sample.get("modality")) == modality and sample.get("primary_organ")
        }
        for modality in required_modalities
    }
    common = set.intersection(*(set(organs) for organs in organs_by_modality.values()))
    return sorted(common)


def _normalized_multi_organ_queries(
    retrieval: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    return {
        direction: [
            {
                "sample_id": str(row["sample_id"]),
                "map": float(row["map"]),
                "top@1": float(row["top@1"]),
                "top@5": float(row["top@5"]),
            }
            for row in retrieval[direction]["query_results"]
        ]
        for direction in DIRECTIONS
    }


def _normalized_single_organ_queries(
    retrieval: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for direction in DIRECTIONS:
        rows = []
        for row in retrieval[direction]["per_pair"]:
            rank = int(row["positive_rank"])
            rows.append(
                {
                    "sample_id": str(row["pair_id"]),
                    "map": 1.0 / rank,
                    "top@1": 1.0 if rank <= 1 else 0.0,
                    "top@5": 1.0 if rank <= 5 else 0.0,
                }
            )
        output[direction] = rows
    return output


def _validate_query_points(
    points: Mapping[str, Mapping[str, float]],
    queries: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    dataset: str,
    seed: int,
    feature_type: str,
) -> None:
    for checkpoint in CHECKPOINTS:
        for metric in METRICS:
            directional_means = [
                float(np.mean([float(row[metric]) for row in queries[checkpoint][direction]]))
                for direction in DIRECTIONS
            ]
            query_mean = float(np.mean(directional_means))
            if not np.isclose(
                query_mean,
                float(points[checkpoint][metric]),
                rtol=1e-9,
                atol=1e-10,
            ):
                raise RuntimeError(
                    f"Query reconstruction mismatch for {dataset}/seed_{seed}/"
                    f"{checkpoint}/{feature_type}/{metric}: "
                    f"{query_mean} != {points[checkpoint][metric]}"
                )


def _paired_query_bootstrap(
    queries: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    bootstrap_resamples: int,
    bootstrap_seed: int,
    dataset: str,
    seed: int,
    feature_type: str,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, int]]:
    effect_columns = [
        (metric, contrast_name, method_a, method_b)
        for metric in METRICS
        for contrast_name, method_a, method_b in CONTRASTS
    ]
    distributions = np.zeros(
        (bootstrap_resamples, len(effect_columns)),
        dtype=np.float64,
    )
    query_counts: dict[str, int] = {}
    rng = np.random.default_rng(
        int(bootstrap_seed) + _stable_seed(dataset, seed, feature_type)
    )

    for direction in DIRECTIONS:
        rows_by_checkpoint = {
            checkpoint: {
                str(row["sample_id"]): row
                for row in queries[checkpoint][direction]
            }
            for checkpoint in CHECKPOINTS
        }
        ordered_query_ids = list(rows_by_checkpoint[CHECKPOINTS[0]])
        expected_query_ids = set(ordered_query_ids)
        for checkpoint in CHECKPOINTS[1:]:
            if set(rows_by_checkpoint[checkpoint]) != expected_query_ids:
                raise RuntimeError(
                    f"Paired query mismatch for {dataset}/seed_{seed}/"
                    f"{feature_type}/{direction}/{checkpoint}"
                )
        query_counts[direction] = len(ordered_query_ids)
        effect_matrix = np.empty(
            (len(ordered_query_ids), len(effect_columns)),
            dtype=np.float64,
        )
        for column_index, (metric, _, method_a, method_b) in enumerate(effect_columns):
            effect_matrix[:, column_index] = np.asarray(
                [
                    float(rows_by_checkpoint[method_a][query_id][metric])
                    - float(rows_by_checkpoint[method_b][query_id][metric])
                    for query_id in ordered_query_ids
                ],
                dtype=np.float64,
            )

        batch_size = 64
        for start in range(0, bootstrap_resamples, batch_size):
            stop = min(start + batch_size, bootstrap_resamples)
            indices = rng.integers(
                0,
                len(ordered_query_ids),
                size=(stop - start, len(ordered_query_ids)),
            )
            distributions[start:stop] += 0.5 * np.mean(
                effect_matrix[indices],
                axis=1,
                dtype=np.float64,
            )

    output = {
        metric: {
            contrast_name: distributions[:, column_index]
            for column_index, (
                column_metric,
                contrast_name,
                _,
                _,
            ) in enumerate(effect_columns)
            if column_metric == metric
        }
        for metric in METRICS
    }
    return output, query_counts


def _multi_organ_seed_task(
    results_root: Path,
    dataset: str,
    seed: int,
    feature_type: str,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    manifest_path = get_cross_modal_manifest_path(dataset, "sampled", "core")
    samples = load_cross_modal_manifest(manifest_path)
    supported_organs = _supported_organs(samples)

    embedding_paths = {
        checkpoint: _embedding_path(
            results_root,
            dataset,
            seed,
            checkpoint,
            feature_type,
        )
        for checkpoint in CHECKPOINTS
    }
    bundles = {
        checkpoint: _load_embedding_bundle(path)
        for checkpoint, path in embedding_paths.items()
    }
    reference_ids = bundles[CHECKPOINTS[0]][1]
    for checkpoint in CHECKPOINTS[1:]:
        if bundles[checkpoint][1] != reference_ids:
            raise RuntimeError(
                f"Embedding sample order mismatch for {dataset}/seed_{seed}/"
                f"{feature_type}/{checkpoint}"
            )

    metric_paths = {
        checkpoint: _metric_path(
            results_root,
            dataset,
            seed,
            checkpoint,
            feature_type,
        )
        for checkpoint in CHECKPOINTS
    }
    stored_retrieval = {
        checkpoint: _load_json(path)["balanced_cross_modal_retrieval"]
        for checkpoint, path in metric_paths.items()
    }
    recomputed_retrieval = {
        checkpoint: compute_bidirectional_cross_modal_retrieval(
            bundles[checkpoint][0],
            samples,
            supported_organs=supported_organs,
            seed=seed,
            return_query_results=True,
            bootstrap_resamples=0,
            show_progress=False,
        )
        for checkpoint in CHECKPOINTS
    }
    points: dict[str, dict[str, float]] = {}
    queries: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for checkpoint in CHECKPOINTS:
        stored_points = stored_retrieval[checkpoint]["bidirectional_mean"]
        recomputed_points = recomputed_retrieval[checkpoint]["bidirectional_mean"]
        for metric in METRICS:
            if not np.isclose(
                float(stored_points[metric]),
                float(recomputed_points[metric]),
                rtol=1e-9,
                atol=1e-10,
            ):
                raise RuntimeError(
                    f"Stored retrieval mismatch for {dataset}/seed_{seed}/"
                    f"{checkpoint}/{feature_type}/{metric}"
                )
        points[checkpoint] = {
            metric: float(stored_points[metric])
            for metric in METRICS
        }
        queries[checkpoint] = _normalized_multi_organ_queries(
            recomputed_retrieval[checkpoint]
        )

    _validate_query_points(points, queries, dataset, seed, feature_type)
    bootstrap, query_counts = _paired_query_bootstrap(
        queries,
        bootstrap_resamples,
        bootstrap_seed,
        dataset,
        seed,
        feature_type,
    )
    return {
        "dataset": dataset,
        "seed": seed,
        "feature_type": feature_type,
        "retrieval_protocol": "balanced_multi_organ_bidirectional_retrieval",
        "n_supported_organs": len(supported_organs),
        "query_counts": query_counts,
        "points": points,
        "bootstrap": bootstrap,
        "manifest_path": str(manifest_path),
        "embedding_paths": {
            checkpoint: str(path)
            for checkpoint, path in embedding_paths.items()
        },
        "metric_paths": {
            checkpoint: str(path)
            for checkpoint, path in metric_paths.items()
        },
    }


def _single_organ_seed_task(
    results_root: Path,
    dataset: str,
    seed: int,
    feature_type: str,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    metric_paths = {
        checkpoint: _single_organ_path(
            results_root,
            dataset,
            seed,
            checkpoint,
            feature_type,
        )
        for checkpoint in CHECKPOINTS
    }
    retrieval = {
        checkpoint: _load_json(path)["single_organ_bidirectional_retrieval"]
        for checkpoint, path in metric_paths.items()
    }
    points = {
        checkpoint: {
            metric: float(retrieval[checkpoint]["bidirectional_mean"][metric])
            for metric in METRICS
        }
        for checkpoint in CHECKPOINTS
    }
    queries = {
        checkpoint: _normalized_single_organ_queries(retrieval[checkpoint])
        for checkpoint in CHECKPOINTS
    }
    _validate_query_points(points, queries, dataset, seed, feature_type)
    bootstrap, query_counts = _paired_query_bootstrap(
        queries,
        bootstrap_resamples,
        bootstrap_seed,
        dataset,
        seed,
        feature_type,
    )
    representative_payload = _load_json(metric_paths[CHECKPOINTS[0]])
    return {
        "dataset": dataset,
        "seed": seed,
        "feature_type": feature_type,
        "retrieval_protocol": "single_organ_patient_matched_bidirectional_retrieval",
        "n_supported_organs": 1,
        "query_counts": query_counts,
        "points": points,
        "bootstrap": bootstrap,
        "manifest_path": str(representative_payload.get("manifest_path")),
        "embedding_paths": {},
        "metric_paths": {
            checkpoint: str(path)
            for checkpoint, path in metric_paths.items()
        },
    }


def _analyze_seed_task(
    task: tuple[Path, str, int, str, int, int],
) -> dict[str, Any]:
    (
        results_root,
        dataset,
        seed,
        feature_type,
        bootstrap_resamples,
        bootstrap_seed,
    ) = task
    if dataset in MULTI_ORGAN_DATASETS:
        return _multi_organ_seed_task(
            results_root,
            dataset,
            seed,
            feature_type,
            bootstrap_resamples,
            bootstrap_seed,
        )
    return _single_organ_seed_task(
        results_root,
        dataset,
        seed,
        feature_type,
        bootstrap_resamples,
        bootstrap_seed,
    )


def _run_seed_tasks(
    tasks: Sequence[tuple[Path, str, int, str, int, int]],
    workers: int,
) -> list[dict[str, Any]]:
    if workers <= 1:
        results = []
        for index, task in enumerate(tasks, start=1):
            results.append(_analyze_seed_task(task))
            print(
                f"[{index}/{len(tasks)}] {task[1]}/seed_{task[2]}/{task[3]}",
                flush=True,
            )
        return results

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
            distribution = seed_results[int(seed_index)]["bootstrap"][metric][
                contrast_name
            ]
            effect_index = int(rng.integers(0, len(distribution)))
            effects.append(float(distribution[effect_index]))
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
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for result in seed_results:
        key = (str(result["dataset"]), str(result["feature_type"]))
        grouped.setdefault(key, []).append(result)

    adjusted_confidence_level = 1.0 - (0.05 / len(CONTRASTS))
    rows = []
    for (dataset, feature_type), group in sorted(grouped.items()):
        ordered_group = sorted(group, key=lambda item: int(item["seed"]))
        if [int(item["seed"]) for item in ordered_group] != list(SEEDS):
            raise RuntimeError(f"Incomplete seed group for {dataset}/{feature_type}")
        protocols = {str(item["retrieval_protocol"]) for item in ordered_group}
        if len(protocols) != 1:
            raise RuntimeError(f"Retrieval protocol mismatch for {dataset}/{feature_type}")
        retrieval_protocol = protocols.pop()
        for metric in METRICS:
            for contrast_name, method_a, method_b in CONTRASTS:
                method_a_points = [
                    float(item["points"][method_a][metric])
                    for item in ordered_group
                ]
                method_b_points = [
                    float(item["points"][method_b][metric])
                    for item in ordered_group
                ]
                differences = [
                    method_a_value - method_b_value
                    for method_a_value, method_b_value in zip(
                        method_a_points,
                        method_b_points,
                    )
                ]
                mean_difference = float(np.mean(differences))
                distribution = _hierarchical_distribution(
                    ordered_group,
                    metric,
                    contrast_name,
                    bootstrap_resamples,
                    bootstrap_seed,
                )
                ci95 = _centered_symmetric_interval(
                    distribution,
                    mean_difference,
                    0.95,
                )
                adjusted = _centered_symmetric_interval(
                    distribution,
                    mean_difference,
                    adjusted_confidence_level,
                )
                if adjusted[0] > 0.0:
                    decision = "method_a_better"
                elif adjusted[1] < 0.0:
                    decision = "method_b_better"
                else:
                    decision = "not_resolved"
                rows.append(
                    {
                        "dataset": dataset,
                        "dataset_label": DATASET_LABELS[dataset],
                        "feature_type": feature_type,
                        "retrieval_protocol": retrieval_protocol,
                        "metric": metric,
                        "contrast": contrast_name,
                        "method_a": CHECKPOINT_LABELS[method_a],
                        "method_b": CHECKPOINT_LABELS[method_b],
                        "n_seeds": len(ordered_group),
                        "n_paired_ct_to_mr_queries_min": min(
                            int(item["query_counts"]["ct_to_mr"])
                            for item in ordered_group
                        ),
                        "n_paired_ct_to_mr_queries_max": max(
                            int(item["query_counts"]["ct_to_mr"])
                            for item in ordered_group
                        ),
                        "n_paired_mr_to_ct_queries_min": min(
                            int(item["query_counts"]["mr_to_ct"])
                            for item in ordered_group
                        ),
                        "n_paired_mr_to_ct_queries_max": max(
                            int(item["query_counts"]["mr_to_ct"])
                            for item in ordered_group
                        ),
                        "method_a_mean": float(np.mean(method_a_points)),
                        "method_a_seed_std": _sample_std(method_a_points),
                        "method_b_mean": float(np.mean(method_b_points)),
                        "method_b_seed_std": _sample_std(method_b_points),
                        "mean_difference_a_minus_b": mean_difference,
                        "ci95_difference_low": ci95[0],
                        "ci95_difference_high": ci95[1],
                        "adjusted_confidence_level": adjusted_confidence_level,
                        "interval_method": (
                            "hierarchical_direction_balanced_centered_"
                            "symmetric_query_bootstrap"
                        ),
                        "adjusted_difference_low": adjusted[0],
                        "adjusted_difference_high": adjusted[1],
                        "multiplicity_method": (
                            "Bonferroni-adjusted interval across three ISO/REL/SA "
                            "contrasts within dataset-feature-metric"
                        ),
                        "decision": decision,
                        "bootstrap_resamples": bootstrap_resamples,
                        "bootstrap_seed": bootstrap_seed,
                    }
                )
    return rows


def _format_signed(value: float) -> str:
    formatted = f"{float(value):+.4f}"
    if float(value) != 0.0 and formatted in {"+0.0000", "-0.0000"}:
        return f"{float(value):+.2e}"
    return formatted


def _format_effect(row: Mapping[str, Any]) -> str:
    return (
        f"{_format_signed(float(row['mean_difference_a_minus_b']))} "
        f"[{_format_signed(float(row['adjusted_difference_low']))}, "
        f"{_format_signed(float(row['adjusted_difference_high']))}]"
    )


def _format_paired_cell(row: Mapping[str, Any]) -> str:
    effect = _format_effect(row)
    if row["decision"] == "not_resolved":
        return effect
    return f"<strong>{effect}</strong>"


def _write_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    bootstrap_resamples = {
        int(row["bootstrap_resamples"])
        for row in rows
    }
    if len(bootstrap_resamples) != 1:
        raise RuntimeError("Inconsistent bootstrap resample counts")
    resample_count = bootstrap_resamples.pop()
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
        (metric, label)
        for metric, label in METRICS.items()
    ]
    contrast_rows = [
        (
            contrast_name,
            f"{CHECKPOINT_LABELS[method_a]} − {CHECKPOINT_LABELS[method_b]}",
        )
        for contrast_name, method_a, method_b in CONTRASTS
    ]
    lines = [
        "# Table 5 Paired Analysis",
        "",
        "Each cell reports the direction-balanced difference between method A and "
        "method B with its Bonferroni-adjusted centered symmetric bootstrap "
        f"interval from {resample_count:,} hierarchical resamples. Positive values "
        "favor method A. Bold cells have adjusted intervals that exclude zero. "
        "Unadjusted 95% intervals remain available in the CSV and JSON outputs.",
        "",
        "Multi-organ datasets use matched retrieval queries under the same balanced "
        "candidate pools. CHAOS uses matched patient queries on the liver-only "
        "surface. CT-to-MR and MR-to-CT effects are computed separately and then "
        "averaged with equal directional weight. These intervals quantify fixed-"
        "checkpoint evaluation variability, not independent pretraining-run "
        "variability.",
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
    )
    expected_rows = len(datasets) * len(feature_types) * len(METRICS) * len(CONTRASTS)
    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, found {len(rows)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "table5_paired_analysis.csv", rows)
    _write_markdown(output_dir / "table5_paired_analysis.md", rows)
    payload = {
        "analysis_scope": "table5_cross_modal_retrieval_paired_comparison",
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
            "aggregation": (
                "hierarchical_over_evaluation_seed_and_direction_balanced_"
                "paired_retrieval_queries"
            ),
            "multiplicity": (
                "Bonferroni-adjusted interval across three ISO/REL/SA contrasts "
                "within each dataset-feature-metric family"
            ),
        },
        "training_run_variance": "not_estimated",
        "rows": rows,
        "seed_provenance": [
            {
                "dataset": result["dataset"],
                "seed": result["seed"],
                "feature_type": result["feature_type"],
                "retrieval_protocol": result["retrieval_protocol"],
                "n_supported_organs": result["n_supported_organs"],
                "query_counts": result["query_counts"],
                "manifest_path": result["manifest_path"],
                "embedding_paths": result["embedding_paths"],
                "metric_paths": result["metric_paths"],
            }
            for result in seed_results
        ],
    }
    _write_json(output_dir / "table5_paired_analysis.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute paired ISO/REL/SA retrieval comparisons for Table 5."
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
    parser.add_argument("--bootstrap-resamples", type=int, default=50000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
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
        workers=args.workers,
    )
    print(f"rows: {len(payload['rows'])}")
    print(f"wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
