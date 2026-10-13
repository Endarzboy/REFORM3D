#!/usr/bin/env python
"""Generate compact repeated-seed tables for the main REFORM-3D metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .markdown_tables import render_dataset_metric_table
from .repeated_seed_summary import collect_repeated_seed_summary


DEFAULT_RESULTS_ROOT = Path.cwd() / "workspace" / "runs"
DEFAULT_OUTPUT_DIR = Path.cwd() / "tables"

SEEDS = (42, 123, 456)
CHECKPOINTS = (
    "med3dino_iso_96",
    "med3dino_rel_96",
    "med3dino_sa_96",
)
FEATURE_TYPES = ("cls", "avg_pool")

CHECKPOINT_LABELS = {
    "med3dino_iso_96": "Med3DINO-ISO",
    "med3dino_rel_96": "Med3DINO-REL",
    "med3dino_sa_96": "Med3DINO-SA",
}
FEATURE_LABELS = {
    "cls": "CLS",
    "avg_pool": "Average pooled",
}
DATASET_LABELS = {
    "abdomenatlas": "AbdomenAtlas",
    "kits23": "KiTS23",
    "imagecas": "ImageCAS",
    "totalsegmentermri": "TotalSegmentator-MR",
    "jhu_stroke": "JHU",
    "totalsegmenter_ct_mr_anchor": "TotalSegmentator CT-MR",
    "mmwhs_ct_mr": "MMWHS",
    "amos22_ct_mr": "AMOS22",
    "chaos_ct_mr": "CHAOS",
}

TABLE_SPECS: dict[str, dict[str, Any]] = {
    "table3": {
        "title": "Table 3: Controlled Acquisition Robustness",
        "module": "acquisition_robustness",
        "manifest_variant": "original_bins",
        "datasets": (
            "abdomenatlas",
            "kits23",
            "imagecas",
            "totalsegmentermri",
            "jhu_stroke",
        ),
        "metrics": (
            {
                "key": "drift",
                "label": "Drift ↓",
                "family": "controlled_drift",
                "name": "compact_non_reference_mean_drift",
            },
            {
                "key": "cka",
                "label": "CKA ↑",
                "family": "controlled_cka",
                "name": "compact_off_diagonal_cka",
            },
            {
                "key": "transfer",
                "label": "Transfer ↑",
                "family": "controlled_semantic_transfer",
                "name": "cross_variant_accuracy",
            },
        ),
        "expected_rows": 30,
    },
    "table5": {
        "title": "Table 5: Cross-Modal Anatomical Alignment",
        "module": "cross_modal_alignment",
        "manifest_variant": "core",
        "datasets": (
            "totalsegmenter_ct_mr_anchor",
            "mmwhs_ct_mr",
            "amos22_ct_mr",
            "chaos_ct_mr",
        ),
        "metrics": (
            {
                "key": "map",
                "label": "mAP ↑",
                "family": "bidirectional_retrieval",
                "name": "retrieval_map",
            },
            {
                "key": "top1",
                "label": "Top-1 ↑",
                "family": "bidirectional_retrieval",
                "name": "retrieval_top@1",
            },
            {
                "key": "top5",
                "label": "Top-5 ↑",
                "family": "bidirectional_retrieval",
                "name": "retrieval_top@5",
            },
        ),
        "expected_rows": 24,
    },
    "table6": {
        "title": "Table 6: Anatomical Generalization",
        "module": "anatomical_generalization",
        "manifest_variant": "core",
        "datasets": (
            "totalsegmenter_ct_mr_anchor",
            "mmwhs_ct_mr",
            "amos22_ct_mr",
        ),
        "metrics": (
            {
                "key": "centroid_ba",
                "label": "BA centroid ↑",
                "family": "anatomical_adaptation",
                "name": "nearest_centroid_balanced_accuracy",
            },
            {
                "key": "probe_ba",
                "label": "BA probe ↑",
                "family": "anatomical_adaptation",
                "name": "adapted_balanced_accuracy",
            },
            {
                "key": "adaptation_gain",
                "label": "Adaptation gain ↑",
                "family": "anatomical_adaptation",
                "name": "adaptation_gain_balanced_accuracy_over_nearest_centroid",
            },
        ),
        "expected_rows": 18,
    },
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], metric_keys: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "table",
        "dataset",
        "dataset_label",
        "crop_size",
        "feature_type",
        "checkpoint",
        "method",
        "n_seeds",
    ]
    for metric_key in metric_keys:
        fieldnames.extend(
            [
                f"{metric_key}_mean",
                f"{metric_key}_std",
                f"{metric_key}_ci95_low",
                f"{metric_key}_ci95_high",
                *[f"{metric_key}_seed_{seed}" for seed in SEEDS],
            ]
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fieldnames} for row in rows)


def _base_root(results_root: Path, module: str, dataset: str, manifest_variant: str) -> Path:
    return results_root / module / dataset / manifest_variant


def _summary_index(summary_rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str, str], Mapping[str, Any]]:
    index: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for row in summary_rows:
        if row.get("status") != "seeded_report":
            continue
        key = (
            str(row["checkpoint"]),
            str(row["feature_type"]),
            str(row["metric_family"]),
            str(row["metric_name"]),
        )
        if key in index:
            raise RuntimeError(f"Duplicate seeded metric row: {key}")
        index[key] = row
    return index


def _compact_rows(results_root: Path, table_name: str, spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    module = str(spec["module"])
    manifest_variant = str(spec["manifest_variant"])
    metrics = list(spec["metrics"])
    output: list[dict[str, Any]] = []
    for dataset in spec["datasets"]:
        summary = collect_repeated_seed_summary(
            module_name=module,
            dataset=dataset,
            manifest_variant=manifest_variant,
            base_root=_base_root(results_root, module, dataset, manifest_variant),
            seeds=SEEDS,
            checkpoints=CHECKPOINTS,
            feature_types=FEATURE_TYPES,
            acquisition_tracks=("controlled",),
        )
        index = _summary_index(summary["rows"])
        for feature_type in FEATURE_TYPES:
            for checkpoint in CHECKPOINTS:
                compact: dict[str, Any] = {
                    "table": table_name,
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS[dataset],
                    "crop_size": 96,
                    "feature_type": feature_type,
                    "checkpoint": checkpoint,
                    "method": CHECKPOINT_LABELS[checkpoint],
                    "n_seeds": len(SEEDS),
                    "provenance": {},
                }
                for metric in metrics:
                    metric_key = str(metric["key"])
                    metric_family = str(metric["family"])
                    if table_name == "table5" and dataset == "chaos_ct_mr":
                        metric_family = "single_organ_bidirectional_retrieval"
                    summary_key = (
                        checkpoint,
                        feature_type,
                        metric_family,
                        str(metric["name"]),
                    )
                    if summary_key not in index:
                        raise RuntimeError(
                            f"Missing {table_name} metric for {dataset}/{checkpoint}/"
                            f"{feature_type}: {metric['family']}.{metric['name']}"
                        )
                    metric_row = index[summary_key]
                    expected_seed_keys = {str(seed) for seed in SEEDS}
                    if (
                        int(metric_row["n"]) != len(SEEDS)
                        or metric_row.get("missing_seeds")
                        or set(metric_row.get("values_by_seed", {})) != expected_seed_keys
                        or set(metric_row.get("source_files_by_seed", {})) != expected_seed_keys
                    ):
                        raise RuntimeError(
                            f"Incomplete {table_name} seeds for {dataset}/{checkpoint}/"
                            f"{feature_type}/{metric_key}"
                        )
                    values = [metric_row.get(key) for key in ("mean", "std", "ci95_low", "ci95_high")]
                    values.extend(metric_row["values_by_seed"].values())
                    try:
                        finite = all(math.isfinite(float(value)) for value in values)
                    except (TypeError, ValueError):
                        finite = False
                    if not finite or float(metric_row["std"]) < 0:
                        raise RuntimeError(
                            f"Invalid {table_name} metric values for {dataset}/{checkpoint}/{feature_type}/{metric_key}"
                        )
                    compact[f"{metric_key}_mean"] = metric_row["mean"]
                    compact[f"{metric_key}_std"] = metric_row["std"]
                    compact[f"{metric_key}_ci95_low"] = metric_row["ci95_low"]
                    compact[f"{metric_key}_ci95_high"] = metric_row["ci95_high"]
                    values_by_seed = metric_row["values_by_seed"]
                    for seed in SEEDS:
                        compact[f"{metric_key}_seed_{seed}"] = values_by_seed[str(seed)]
                    compact["provenance"][metric_key] = metric_row["source_files_by_seed"]
                output.append(compact)
    expected_rows = int(spec["expected_rows"])
    if len(output) != expected_rows:
        raise RuntimeError(f"{table_name} expected {expected_rows} rows, found {len(output)}")
    return output


def _format_mean_std(mean: Any, std: Any) -> str:
    return f"{float(mean):.4f} ± {float(std):.4f}"


def _markdown_table(spec: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [f"## {spec['title']}", ""]
    metrics = list(spec["metrics"])
    dataset_columns = [
        (str(dataset), DATASET_LABELS[str(dataset)])
        for dataset in spec["datasets"]
    ]
    metric_columns = [
        (str(metric["key"]), str(metric["label"]))
        for metric in metrics
    ]
    method_rows = [
        (checkpoint, CHECKPOINT_LABELS[checkpoint])
        for checkpoint in CHECKPOINTS
    ]
    for feature_type in FEATURE_TYPES:
        values: dict[tuple[str, str, str], str] = {}
        for dataset, _ in dataset_columns:
            for checkpoint, _ in method_rows:
                row = next(
                    item
                    for item in rows
                    if item["dataset"] == dataset
                    and item["feature_type"] == feature_type
                    and item["checkpoint"] == checkpoint
                )
                for metric_key, _ in metric_columns:
                    values[(checkpoint, dataset, metric_key)] = _format_mean_std(
                        row[f"{metric_key}_mean"],
                        row[f"{metric_key}_std"],
                    )
        lines.extend([f"### {FEATURE_LABELS[feature_type]} features", ""])
        lines.extend(
            render_dataset_metric_table(
                row_header="Method",
                rows=method_rows,
                datasets=dataset_columns,
                metrics=metric_columns,
                values=values,
            )
        )
        lines.append("")
    return "\n".join(lines)


def generate_tables(results_root: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables: dict[str, Any] = {}
    markdown_sections = [
        "# REFORM-3D Repeated-Seed Summary Tables",
        "",
        "Values are mean ± sample standard deviation across evaluation seeds "
        "42, 123, and 456 for fixed pretrained checkpoints. The CSV and JSON "
        "outputs also retain two-sided 95% t-intervals for the across-seed mean. "
        "These intervals do not test paired checkpoint differences.",
        "",
        "ISO, REL, and SA use the 96³ checkpoint group. CLS is the headline "
        "feature type and average pooling is secondary.",
        "",
    ]
    for table_name, spec in TABLE_SPECS.items():
        rows = _compact_rows(results_root, table_name, spec)
        metric_keys = [str(metric["key"]) for metric in spec["metrics"]]
        _write_csv(output_dir / f"{table_name}_seed_summary.csv", rows, metric_keys)
        tables[table_name] = {
            "title": spec["title"],
            "module": spec["module"],
            "manifest_variant": spec["manifest_variant"],
            "datasets": list(spec["datasets"]),
            "metrics": list(spec["metrics"]),
            "rows": rows,
        }
        markdown_sections.append(_markdown_table(spec, rows))
    payload = {
        "analysis_scope": "main_reform3d_repeated_evaluation_seed_summary",
        "results_root": str(results_root),
        "seeds": list(SEEDS),
        "checkpoints": list(CHECKPOINTS),
        "crop_size": 96,
        "feature_types": list(FEATURE_TYPES),
        "uncertainty": {
            "mean": "arithmetic mean across evaluation seeds",
            "std": "sample standard deviation across evaluation seeds",
            "ci95": "two-sided t interval for the across-seed mean",
            "paired_significance": "not_computed_in_this_step",
            "training_run_variance": "not_estimated",
        },
        "tables": tables,
    }
    _write_json(output_dir / "seed_summary_tables.json", payload)
    (output_dir / "seed_summary_tables.md").write_text(
        "\n".join(markdown_sections).rstrip() + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate compact repeated-seed tables for main REFORM-3D metrics."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = generate_tables(args.results_root, args.output_dir)
    for table_name, table in payload["tables"].items():
        print(f"{table_name}: {len(table['rows'])} rows")
    print(f"wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
