#!/usr/bin/env python
"""Paired uncertainty analysis for Anatomical Generalization metrics in Table 6."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from reform3d.core.markdown_tables import render_dataset_metric_table

from .config import get_checkpoint_metrics_path


DEFAULT_RESULTS_ROOT = Path.cwd() / "workspace" / "runs"
DEFAULT_OUTPUT_DIR = Path.cwd() / "tables"

DATASETS = (
    "totalsegmenter_ct_mr_anchor",
    "mmwhs_ct_mr",
    "amos22_ct_mr",
)
DATASET_LABELS = {
    "totalsegmenter_ct_mr_anchor": "TotalSegmentator CT-MR",
    "mmwhs_ct_mr": "MMWHS",
    "amos22_ct_mr": "AMOS22",
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
    "centroid_ba": {
        "label": "BA centroid",
        "macro_key": "nearest_centroid_balanced_accuracy",
    },
    "probe_ba": {
        "label": "BA probe",
        "macro_key": "balanced_accuracy",
    },
    "adaptation_gain": {
        "label": "Adaptation gain",
        "macro_key": "adaptation_gain_balanced_accuracy",
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
        "contrast",
        "method_a",
        "method_b",
        "paired_unit",
        "n_seeds",
        "n_paired_organs",
        "target_query_samples_per_organ_min",
        "target_query_samples_per_organ_max",
        "support_samples_min",
        "support_samples_max",
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
    return sum(
        (index + 1) * ord(character)
        for index, character in enumerate(text)
    ) % 2_000_000_000


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def _artifact_path(
    results_root: Path,
    dataset: str,
    seed: int,
    checkpoint: str,
    feature_type: str,
) -> Path:
    results_dir = (
        results_root
        / "anatomical_generalization"
        / dataset
        / "core"
        / "seeds"
        / f"seed_{seed}"
        / "results"
    )
    return get_checkpoint_metrics_path(results_dir, checkpoint, feature_type)


def _extract_checkpoint_result(
    path: Path,
    *,
    dataset: str,
    seed: int,
    checkpoint: str,
    feature_type: str,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = _load_json(path)
    if payload.get("analysis_name") != dataset:
        raise RuntimeError(f"Dataset mismatch in {path}")
    if payload.get("checkpoint_name") != checkpoint:
        raise RuntimeError(f"Checkpoint mismatch in {path}")
    if payload.get("feature_type") != feature_type:
        raise RuntimeError(f"Feature-type mismatch in {path}")

    surface = payload.get("surface_b_few_shot_transfer_evidence", {})
    analysis = surface.get("few_shot_transfer_analysis", {})
    if surface.get("status") != "ok" or analysis.get("status") != "ok":
        raise RuntimeError(f"Incomplete Anatomical Generalization result: {path}")
    if analysis.get("evaluation_seeds") != [int(seed)]:
        raise RuntimeError(f"Evaluation-seed mismatch in {path}")
    if analysis.get("support_query_split_policy") != "case_disjoint_support_query":
        raise RuntimeError(f"Unexpected split policy in {path}")

    per_organ = analysis.get("per_organ", {})
    organs = tuple(sorted(per_organ))
    if len(organs) != int(analysis.get("n_evaluated_organs", -1)):
        raise RuntimeError(f"Organ-count mismatch in {path}")

    probe_values = []
    centroid_values = []
    signatures = []
    for organ in organs:
        organ_payload = per_organ[organ]
        if organ_payload.get("status") != "ok":
            raise RuntimeError(f"Incomplete organ result for {organ}: {path}")
        seed_results = organ_payload.get("seed_results", [])
        if len(seed_results) != 1 or int(seed_results[0].get("seed", -1)) != int(seed):
            raise RuntimeError(f"Organ seed mismatch for {organ}: {path}")
        result = seed_results[0]
        probe_value = float(result["target_query_accuracy"])
        centroid_value = float(result["nearest_centroid_target_query_accuracy"])
        if not np.isclose(
            probe_value,
            float(result["per_organ_query_accuracy"][organ]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(f"Probe recall mismatch for {organ}: {path}")
        if not np.isclose(
            centroid_value,
            float(result["nearest_centroid_per_organ_query_accuracy"][organ]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(f"Centroid recall mismatch for {organ}: {path}")
        probe_values.append(probe_value)
        centroid_values.append(centroid_value)
        signatures.append(
            (
                organ,
                int(result["target_query_samples"]),
                int(result["n_support_samples"]),
                int(result["n_query_samples"]),
                int(result["support_per_modality"]),
                int(result["query_per_modality"]),
            )
        )

    probe_array = np.asarray(probe_values, dtype=np.float64)
    centroid_array = np.asarray(centroid_values, dtype=np.float64)
    organ_values = {
        "centroid_ba": centroid_array,
        "probe_ba": probe_array,
        "adaptation_gain": probe_array - centroid_array,
    }
    macro = analysis.get("macro_mean", {})
    points = {
        metric: float(macro[spec["macro_key"]])
        for metric, spec in METRICS.items()
    }
    for metric, values in organ_values.items():
        if not np.isclose(
            float(np.mean(values)),
            points[metric],
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(f"Macro reconstruction mismatch for {metric}: {path}")

    return {
        "path": str(path),
        "manifest_path": str(payload["manifest_path"]),
        "organs": organs,
        "signatures": tuple(signatures),
        "points": points,
        "organ_values": organ_values,
        "required_modalities": tuple(analysis["required_modalities"]),
        "support_per_modality": int(analysis["support_per_modality"]),
        "transfer_protocol": str(analysis["transfer_protocol"]),
    }


def _load_seed_group(
    results_root: Path,
    dataset: str,
    seed: int,
    feature_type: str,
) -> dict[str, Any]:
    checkpoint_results = {
        checkpoint: _extract_checkpoint_result(
            _artifact_path(results_root, dataset, seed, checkpoint, feature_type),
            dataset=dataset,
            seed=seed,
            checkpoint=checkpoint,
            feature_type=feature_type,
        )
        for checkpoint in CHECKPOINTS
    }
    reference = checkpoint_results[CHECKPOINTS[0]]
    for checkpoint in CHECKPOINTS[1:]:
        candidate = checkpoint_results[checkpoint]
        for field in (
            "manifest_path",
            "organs",
            "signatures",
            "required_modalities",
            "support_per_modality",
            "transfer_protocol",
        ):
            if candidate[field] != reference[field]:
                raise RuntimeError(
                    f"Matched-fold mismatch for {dataset}/seed_{seed}/"
                    f"{feature_type}/{checkpoint}: {field}"
                )

    return {
        "dataset": dataset,
        "seed": int(seed),
        "feature_type": feature_type,
        "organs": reference["organs"],
        "signatures": reference["signatures"],
        "manifest_path": reference["manifest_path"],
        "required_modalities": reference["required_modalities"],
        "support_per_modality": reference["support_per_modality"],
        "transfer_protocol": reference["transfer_protocol"],
        "points": {
            checkpoint: checkpoint_results[checkpoint]["points"]
            for checkpoint in CHECKPOINTS
        },
        "organ_values": {
            checkpoint: checkpoint_results[checkpoint]["organ_values"]
            for checkpoint in CHECKPOINTS
        },
        "artifact_paths": {
            checkpoint: checkpoint_results[checkpoint]["path"]
            for checkpoint in CHECKPOINTS
        },
    }


def _crossed_seed_organ_bootstrap(
    effect_matrix: np.ndarray,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> np.ndarray:
    if effect_matrix.ndim != 2:
        raise ValueError("Expected a seed-by-organ effect matrix")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    rng = np.random.default_rng(int(bootstrap_seed))
    n_seeds, n_organs = effect_matrix.shape
    output = np.empty(bootstrap_resamples, dtype=np.float64)
    batch_size = 4096
    for start in range(0, bootstrap_resamples, batch_size):
        stop = min(start + batch_size, bootstrap_resamples)
        batch = stop - start
        seed_indices = rng.integers(0, n_seeds, size=(batch, n_seeds))
        organ_indices = rng.integers(0, n_organs, size=(batch, n_organs))
        sampled = effect_matrix[
            seed_indices[:, :, None],
            organ_indices[:, None, :],
        ]
        output[start:stop] = np.mean(sampled, axis=(1, 2), dtype=np.float64)
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
        organs = tuple(ordered_group[0]["organs"])
        if any(tuple(item["organs"]) != organs for item in ordered_group[1:]):
            raise RuntimeError(f"Organ mismatch across seeds for {dataset}/{feature_type}")

        target_query_counts = [
            int(signature[1])
            for item in ordered_group
            for signature in item["signatures"]
        ]
        support_counts = [
            int(signature[2])
            for item in ordered_group
            for signature in item["signatures"]
        ]
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
                effect_matrix = np.stack(
                    [
                        np.asarray(item["organ_values"][method_a][metric], dtype=np.float64)
                        - np.asarray(item["organ_values"][method_b][metric], dtype=np.float64)
                        for item in ordered_group
                    ],
                    axis=0,
                )
                mean_difference = float(np.mean(effect_matrix))
                point_difference = float(
                    np.mean(method_a_points) - np.mean(method_b_points)
                )
                if not np.isclose(
                    mean_difference,
                    point_difference,
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise RuntimeError(
                        f"Paired effect mismatch for {dataset}/{feature_type}/"
                        f"{metric}/{contrast_name}"
                    )
                distribution = _crossed_seed_organ_bootstrap(
                    effect_matrix,
                    bootstrap_resamples,
                    int(bootstrap_seed)
                    + _stable_seed(dataset, feature_type, metric, contrast_name),
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
                        "metric": metric,
                        "contrast": contrast_name,
                        "method_a": CHECKPOINT_LABELS[method_a],
                        "method_b": CHECKPOINT_LABELS[method_b],
                        "paired_unit": "evaluation_seed_by_held_out_organ_recall",
                        "n_seeds": len(ordered_group),
                        "n_paired_organs": len(organs),
                        "target_query_samples_per_organ_min": min(target_query_counts),
                        "target_query_samples_per_organ_max": max(target_query_counts),
                        "support_samples_min": min(support_counts),
                        "support_samples_max": max(support_counts),
                        "method_a_mean": float(np.mean(method_a_points)),
                        "method_a_seed_std": _sample_std(method_a_points),
                        "method_b_mean": float(np.mean(method_b_points)),
                        "method_b_seed_std": _sample_std(method_b_points),
                        "mean_difference_a_minus_b": mean_difference,
                        "ci95_difference_low": ci95[0],
                        "ci95_difference_high": ci95[1],
                        "adjusted_confidence_level": adjusted_confidence_level,
                        "interval_method": (
                            "crossed_seed_organ_centered_symmetric_bootstrap"
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


def _format_paired_cell(row: Mapping[str, Any]) -> str:
    effect = (
        f"{_format_signed(float(row['mean_difference_a_minus_b']))} "
        f"[{_format_signed(float(row['adjusted_difference_low']))}, "
        f"{_format_signed(float(row['adjusted_difference_high']))}]"
    )
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
        (metric, str(spec["label"]))
        for metric, spec in METRICS.items()
    ]
    contrast_rows = [
        (
            contrast_name,
            f"{CHECKPOINT_LABELS[method_a]} − {CHECKPOINT_LABELS[method_b]}",
        )
        for contrast_name, method_a, method_b in CONTRASTS
    ]
    lines = [
        "# Table 6 Paired Analysis",
        "",
        "Each cell reports method A minus method B with its Bonferroni-adjusted "
        f"centered symmetric interval from {resample_count:,} crossed bootstrap "
        "resamples. Positive values favor method A. Bold cells have adjusted "
        "intervals that exclude zero. Unadjusted 95% intervals remain available "
        "in the CSV and JSON outputs.",
        "",
        "The paired values are held-out-organ query recalls under matched, case-"
        "disjoint support/query splits. Their macro-average exactly reconstructs "
        "balanced accuracy. Evaluation seeds and organ classes are resampled as "
        "crossed axes. Adaptation gain is probe BA minus centroid BA and should "
        "be interpreted as adaptation headroom rather than final recoverability. "
        "These intervals quantify fixed-checkpoint evaluation variability, not "
        "independent pretraining-run variability or transfer to organs absent "
        "from pretraining.",
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
) -> dict[str, Any]:
    seed_results = [
        _load_seed_group(
            results_root,
            dataset,
            seed,
            feature_type,
        )
        for dataset in datasets
        for feature_type in feature_types
        for seed in SEEDS
    ]
    rows = _aggregate_rows(
        seed_results,
        bootstrap_resamples,
        bootstrap_seed,
    )
    expected_rows = (
        len(datasets)
        * len(feature_types)
        * len(METRICS)
        * len(CONTRASTS)
    )
    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, found {len(rows)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "table6_paired_analysis.csv", rows)
    _write_markdown(output_dir / "table6_paired_analysis.md", rows)
    payload = {
        "analysis_scope": "table6_anatomical_generalization_paired_comparison",
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
                "crossed_evaluation_seed_and_held_out_organ_recall"
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
                "organs": list(result["organs"]),
                "manifest_path": result["manifest_path"],
                "required_modalities": list(result["required_modalities"]),
                "support_per_modality": result["support_per_modality"],
                "transfer_protocol": result["transfer_protocol"],
                "artifact_paths": result["artifact_paths"],
            }
            for result in seed_results
        ],
    }
    _write_json(output_dir / "table6_paired_analysis.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute paired ISO/REL/SA comparisons for Table 6."
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
    )
    print(f"rows: {len(payload['rows'])}")
    print(f"wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
