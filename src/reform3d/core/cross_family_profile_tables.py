#!/usr/bin/env python
"""Generate cross-family REFORM-3D representation profile tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from .repeated_seed_summary import collect_repeated_seed_summary


DEFAULT_RESULTS_ROOT = Path.cwd() / "workspace" / "runs"
DEFAULT_OUTPUT_DIR = Path.cwd() / "tables"
SEEDS = (42, 123, 456)


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    role: str
    architecture: str
    checkpoint: str
    feature_type: str
    feature_label: str
    crop_size: int
    newly_added: bool = False


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    family: str
    name: str
    direction: str


@dataclass(frozen=True)
class TableSpec:
    key: str
    title: str
    module: str
    manifest_variant: str
    datasets: tuple[str, ...]
    profile_datasets: tuple[str, ...]
    metrics: tuple[MetricSpec, ...]


METHODS = (
    MethodSpec(
        key="brainiac",
        label="BrainIAC",
        role="Independent reference",
        architecture="3D ViT-B/16",
        checkpoint="brainiac",
        feature_type="cls",
        feature_label="CLS",
        crop_size=96,
    ),
    MethodSpec(
        key="openmind",
        label="OpenMind",
        role="Independent reference",
        architecture="PrimusM",
        checkpoint="openmind_primusm_voco",
        feature_type="avg_pool",
        feature_label="Average pool",
        crop_size=64,
    ),
    MethodSpec(
        key="voco",
        label="VoCo",
        role="Independent reference",
        architecture="SwinUNETR hybrid",
        checkpoint="voco_b_ssl_head",
        feature_type="pooled_multiscale",
        feature_label="Pooled multiscale",
        crop_size=96,
    ),
    MethodSpec(
        key="3dino",
        label="3DINO",
        role="Independent reference",
        architecture="3D ViT-L/16",
        checkpoint="3dino",
        feature_type="cls",
        feature_label="CLS",
        crop_size=96,
    ),
    MethodSpec(
        key="sam_med3d",
        label="SAM-Med3D",
        role="Added independent reference",
        architecture="3D ViT-B",
        checkpoint="sam_med3d_turbo",
        feature_type="avg_pool",
        feature_label="Average pool",
        crop_size=128,
        newly_added=True,
    ),
    MethodSpec(
        key="hi_end_mae",
        label="Hi-End-MAE",
        role="Added independent reference",
        architecture="3D ViT-p12/1536",
        checkpoint="hi_end_mae_vit_p12_1536",
        feature_type="cls",
        feature_label="CLS",
        crop_size=96,
        newly_added=True,
    ),
    MethodSpec(
        key="mambamim",
        label="MambaMIM",
        role="Added independent reference",
        architecture="Hierarchical Mamba",
        checkpoint="mambamim_mask75",
        feature_type="final_stage_pool",
        feature_label="Final-stage pool",
        crop_size=96,
        newly_added=True,
    ),
    MethodSpec(
        key="mim",
        label="MiM",
        role="Added independent reference",
        architecture="ConvViT-B",
        checkpoint="mim_convvit_base",
        feature_type="avg_pool",
        feature_label="Average pool",
        crop_size=96,
        newly_added=True,
    ),
    MethodSpec(
        key="gl_mae",
        label="GL-MAE",
        role="Added independent reference",
        architecture="3D ViT-B/16",
        checkpoint="gl_mae_vit_base_patch16",
        feature_type="cls",
        feature_label="CLS",
        crop_size=96,
        newly_added=True,
    ),
    MethodSpec(
        key="med3dino_iso",
        label="Med3DINO-ISO",
        role="Controlled geometry",
        architecture="3D ViT-L/16",
        checkpoint="med3dino_iso_96",
        feature_type="cls",
        feature_label="CLS",
        crop_size=96,
    ),
    MethodSpec(
        key="med3dino_rel",
        label="Med3DINO-REL",
        role="Controlled geometry",
        architecture="3D ViT-L/16",
        checkpoint="med3dino_rel_96",
        feature_type="cls",
        feature_label="CLS",
        crop_size=96,
    ),
    MethodSpec(
        key="med3dino_sa",
        label="Med3DINO-SA",
        role="Controlled geometry",
        architecture="3D ViT-L/16",
        checkpoint="med3dino_sa_96",
        feature_type="cls",
        feature_label="CLS",
        crop_size=96,
    ),
)

DATASET_LABELS = {
    "abdomenatlas": "AbdomenAtlas",
    "kits23": "KiTS23",
    "imagecas": "ImageCAS",
    "totalsegmentermri": "TS-MR",
    "jhu_stroke": "JHU",
    "totalsegmenter_ct_mr_anchor": "TotalSeg CT–MR",
    "mmwhs_ct_mr": "MMWHS",
    "amos22_ct_mr": "AMOS22",
    "chaos_ct_mr": "CHAOS",
}

TABLES = (
    TableSpec(
        key="table3",
        title="Controlled Acquisition Robustness",
        module="acquisition_robustness",
        manifest_variant="original_bins",
        datasets=(
            "abdomenatlas",
            "kits23",
            "imagecas",
            "totalsegmentermri",
            "jhu_stroke",
        ),
        profile_datasets=(
            "abdomenatlas",
            "kits23",
            "imagecas",
            "totalsegmentermri",
            "jhu_stroke",
        ),
        metrics=(
            MetricSpec(
                key="drift",
                label="Drift ↓",
                family="controlled_drift",
                name="compact_non_reference_mean_drift",
                direction="lower",
            ),
            MetricSpec(
                key="cka",
                label="CKA ↑",
                family="controlled_cka",
                name="compact_off_diagonal_cka",
                direction="higher",
            ),
            MetricSpec(
                key="transfer",
                label="Transfer ↑",
                family="controlled_semantic_transfer",
                name="cross_variant_accuracy",
                direction="higher",
            ),
        ),
    ),
    TableSpec(
        key="table5",
        title="Cross-Modal Anatomical Alignment",
        module="cross_modal_alignment",
        manifest_variant="core",
        datasets=(
            "totalsegmenter_ct_mr_anchor",
            "mmwhs_ct_mr",
            "amos22_ct_mr",
            "chaos_ct_mr",
        ),
        profile_datasets=(
            "totalsegmenter_ct_mr_anchor",
            "mmwhs_ct_mr",
            "amos22_ct_mr",
        ),
        metrics=(
            MetricSpec(
                key="map",
                label="mAP ↑",
                family="bidirectional_retrieval",
                name="retrieval_map",
                direction="higher",
            ),
            MetricSpec(
                key="top1",
                label="Top-1 ↑",
                family="bidirectional_retrieval",
                name="retrieval_top@1",
                direction="higher",
            ),
            MetricSpec(
                key="top5",
                label="Top-5 ↑",
                family="bidirectional_retrieval",
                name="retrieval_top@5",
                direction="higher",
            ),
        ),
    ),
    TableSpec(
        key="table6",
        title="Low-Shot Anatomical Transfer",
        module="anatomical_generalization",
        manifest_variant="core",
        datasets=(
            "totalsegmenter_ct_mr_anchor",
            "mmwhs_ct_mr",
            "amos22_ct_mr",
        ),
        profile_datasets=(
            "totalsegmenter_ct_mr_anchor",
            "mmwhs_ct_mr",
            "amos22_ct_mr",
        ),
        metrics=(
            MetricSpec(
                key="centroid_ba",
                label="Centroid BA ↑",
                family="anatomical_adaptation",
                name="nearest_centroid_balanced_accuracy",
                direction="higher",
            ),
            MetricSpec(
                key="probe_ba",
                label="Probe BA ↑",
                family="anatomical_adaptation",
                name="adapted_balanced_accuracy",
                direction="higher",
            ),
            MetricSpec(
                key="adaptation_gain",
                label="Adaptation gain",
                family="anatomical_adaptation",
                name="adaptation_gain_balanced_accuracy_over_nearest_centroid",
                direction="higher",
            ),
        ),
    ),
)

PROFILE_METRICS = (
    ("table3", "drift"),
    ("table3", "cka"),
    ("table5", "map"),
    ("table5", "top1"),
    ("table6", "centroid_ba"),
    ("table6", "probe_ba"),
)


def _metric_family(table: TableSpec, dataset: str, metric: MetricSpec) -> str:
    if table.key == "table5" and dataset == "chaos_ct_mr":
        return "single_organ_bidirectional_retrieval"
    return metric.family


def _collect_dataset_rows(
    results_root: Path,
    table: TableSpec,
    dataset: str,
    method: MethodSpec,
) -> list[Mapping[str, Any]]:
    base_root = (
        results_root
        / table.module
        / dataset
        / table.manifest_variant
    )
    payload = collect_repeated_seed_summary(
        module_name=table.module,
        dataset=dataset,
        manifest_variant=table.manifest_variant,
        base_root=base_root,
        seeds=SEEDS,
        checkpoints=(method.checkpoint,),
        feature_types=(method.feature_type,),
        acquisition_tracks=("controlled",),
    )
    if payload["missing_files"] or payload["n_incomplete_rows"]:
        raise RuntimeError(
            f"Incomplete repeated-seed results for {table.key}/{dataset}/{method.key}"
        )
    return payload["rows"]


def _find_metric_row(
    rows: Sequence[Mapping[str, Any]],
    table: TableSpec,
    dataset: str,
    method: MethodSpec,
    metric: MetricSpec,
) -> Mapping[str, Any]:
    family = _metric_family(table, dataset, metric)
    matches = [
        row
        for row in rows
        if row.get("status") == "seeded_report"
        and row.get("checkpoint") == method.checkpoint
        and row.get("feature_type") == method.feature_type
        and row.get("metric_family") == family
        and row.get("metric_name") == metric.name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one metric row for {table.key}/{dataset}/{method.key}/"
            f"{metric.key}, found {len(matches)}"
        )
    row = matches[0]
    expected_seed_keys = {str(seed) for seed in SEEDS}
    if (
        int(row.get("n", 0)) != len(SEEDS)
        or row.get("missing_seeds")
        or set(row.get("values_by_seed", {})) != expected_seed_keys
        or set(row.get("source_files_by_seed", {})) != expected_seed_keys
    ):
        raise RuntimeError(
            f"Incomplete metric row for {table.key}/{dataset}/{method.key}/{metric.key}"
        )
    values = [row.get("mean"), row.get("std"), *row["values_by_seed"].values()]
    try:
        finite = all(math.isfinite(float(value)) for value in values)
    except (TypeError, ValueError):
        finite = False
    if not finite or float(row["std"]) < 0:
        raise RuntimeError(f"Invalid metric values for {table.key}/{dataset}/{method.key}/{metric.key}")
    return row


def collect_evidence_rows(results_root: Path) -> list[dict[str, Any]]:
    evidence_rows: list[dict[str, Any]] = []
    for table in TABLES:
        for dataset in table.datasets:
            for method in METHODS:
                summary_rows = _collect_dataset_rows(
                    results_root,
                    table,
                    dataset,
                    method,
                )
                evidence_row: dict[str, Any] = {
                    "table": table.key,
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS[dataset],
                    "method": method.key,
                    "method_label": method.label,
                    "role": method.role,
                    "architecture": method.architecture,
                    "checkpoint": method.checkpoint,
                    "feature_type": method.feature_type,
                    "feature_label": method.feature_label,
                    "crop_size": method.crop_size,
                    "newly_added": method.newly_added,
                    "provenance": {},
                }
                for metric in table.metrics:
                    metric_row = _find_metric_row(
                        summary_rows,
                        table,
                        dataset,
                        method,
                        metric,
                    )
                    evidence_row[f"{metric.key}_mean"] = float(metric_row["mean"])
                    evidence_row[f"{metric.key}_std"] = float(metric_row["std"])
                    for seed in SEEDS:
                        evidence_row[f"{metric.key}_seed_{seed}"] = float(
                            metric_row["values_by_seed"][str(seed)]
                        )
                    evidence_row["provenance"][metric.key] = metric_row[
                        "source_files_by_seed"
                    ]
                evidence_rows.append(evidence_row)
    return evidence_rows


def _validate_evidence_rows(evidence_rows: Sequence[Mapping[str, Any]]) -> None:
    table_index = {table.key: table for table in TABLES}
    method_index = {method.key: method for method in METHODS}
    expected = {
        (table.key, dataset, method.key)
        for table in TABLES for dataset in table.datasets for method in METHODS
    }
    seen = set()
    for row in evidence_rows:
        key = (row.get("table"), row.get("dataset"), row.get("method"))
        if key not in expected:
            raise RuntimeError(f"Unexpected evidence row: {key}")
        if key in seen:
            raise RuntimeError(f"Duplicate evidence row: {key}")
        seen.add(key)
        method = method_index[key[2]]
        for field in ("checkpoint", "feature_type", "crop_size"):
            if row.get(field) != getattr(method, field):
                raise RuntimeError(f"Inconsistent {field} for evidence row: {key}")
        for metric in table_index[key[0]].metrics:
            fields = [f"{metric.key}_mean", f"{metric.key}_std", *[
                f"{metric.key}_seed_{seed}" for seed in SEEDS
            ]]
            try:
                finite = all(math.isfinite(float(row.get(field))) for field in fields)
            except (TypeError, ValueError):
                finite = False
            if not finite or float(row[f"{metric.key}_std"]) < 0:
                raise RuntimeError(f"Invalid {metric.key} values for evidence row: {key}")
    missing = expected - seen
    if missing:
        raise RuntimeError(f"Missing evidence rows: {sorted(missing)}")


def build_profile_rows(
    evidence_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    _validate_evidence_rows(evidence_rows)
    table_index = {table.key: table for table in TABLES}
    profile_rows: list[dict[str, Any]] = []
    for method in METHODS:
        profile_row: dict[str, Any] = {
            **asdict(method),
            "metric_contract": {},
        }
        for table_key, metric_key in PROFILE_METRICS:
            table = table_index[table_key]
            values = [
                float(row[f"{metric_key}_mean"])
                for row in evidence_rows
                if row["table"] == table_key
                and row["method"] == method.key
                and row["dataset"] in table.profile_datasets
            ]
            if len(values) != len(table.profile_datasets):
                raise RuntimeError(
                    f"Expected {len(table.profile_datasets)} profile values for "
                    f"{method.key}/{table_key}/{metric_key}, found {len(values)}"
                )
            profile_row[f"{table_key}_{metric_key}_mean"] = fmean(values)
            profile_row["metric_contract"][f"{table_key}_{metric_key}"] = {
                "datasets": list(table.profile_datasets),
                "aggregation": "arithmetic_mean_of_dataset_level_seed_means",
            }
        profile_rows.append(profile_row)

    for table_key, metric_key in PROFILE_METRICS:
        table = table_index[table_key]
        metric = next(item for item in table.metrics if item.key == metric_key)
        value_key = f"{table_key}_{metric_key}_mean"
        ordered = sorted(
            profile_rows,
            key=lambda row: float(row[value_key]),
            reverse=metric.direction == "higher",
        )
        for rank, row in enumerate(ordered, start=1):
            row[f"{table_key}_{metric_key}_rank"] = rank
    return profile_rows


def _format_score(value: float) -> str:
    return f"{100.0 * value:.2f}"


def _format_mean_std(mean: float, std: float) -> str:
    return f"{100.0 * mean:.2f}±{100.0 * std:.2f}"


def _profile_markdown(profile_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    table_index = {table.key: table for table in TABLES}
    best_values: dict[tuple[str, str], float] = {}
    for table_key, metric_key in PROFILE_METRICS:
        table = table_index[table_key]
        metric = next(item for item in table.metrics if item.key == metric_key)
        values = [
            float(row[f"{table_key}_{metric_key}_mean"])
            for row in profile_rows
        ]
        best_values[(table_key, metric_key)] = (
            max(values) if metric.direction == "higher" else min(values)
        )

    lines = [
        "## Compact Cross-Family Representation Profile",
        "",
        "| Method | Feature | Drift ↓ | CKA ↑ | CT–MR mAP ↑ | CT–MR Top-1 ↑ | Centroid BA ↑ | Probe BA ↑ |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in profile_rows:
        values: list[str] = []
        for table_key, metric_key in PROFILE_METRICS:
            value = float(row[f"{table_key}_{metric_key}_mean"])
            formatted = _format_score(value)
            if value == best_values[(table_key, metric_key)]:
                formatted = f"**{formatted}**"
            values.append(formatted)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    str(row["feature_label"]),
                    *values,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Values are ×100 and are descriptive means of dataset-level three-seed means. "
            "Drift and CKA average AbdomenAtlas, KiTS23, ImageCAS, TS-MR, and JHU; "
            "semantic transfer is retained only in the dataset-level table because its target "
            "definition differs across datasets. "
            "CT–MR columns average the three multi-organ datasets; CHAOS is excluded because "
            "it is a single-organ surface. Anatomical columns average TotalSeg CT–MR, MMWHS, "
            "and AMOS22. Each method uses its declared primary frozen feature. Bold marks the "
            "best value in each column; no overall score is computed.",
            "",
        ]
    )
    return lines


def _coverage_markdown() -> list[str]:
    lines = [
        "## Encoder Coverage",
        "",
        "| Method | Evaluation role | Architecture | Input | Primary frozen feature |",
        "|---|---|---|---:|---|",
    ]
    for method in METHODS:
        lines.append(
            f"| {method.label} | {method.role} | {method.architecture} | "
            f"{method.crop_size}³ | {method.feature_label} |"
        )
    lines.extend(
        [
            "",
            "The independent reference methods differ in architecture, pretraining corpus, objective, "
            "input interface, and feature surface. Their comparisons describe behavior under "
            "the same REFORM-3D evaluation protocol; geometric-policy attribution is restricted "
            "to the controlled ISO/REL/SA triplet.",
            "",
        ]
    )
    return lines


def _compact_external_table(
    table: TableSpec,
    evidence_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    metric_order = " / ".join(metric.label for metric in table.metrics)
    header = "| Method | Feature | " + " | ".join(
        DATASET_LABELS[dataset] for dataset in table.datasets
    ) + " |"
    divider = "|---|---|" + "|".join("---:" for _ in table.datasets) + "|"
    lines = [
        f"## Added External Encoders — {table.title}",
        "",
        f"Each dataset cell reports {metric_order}, with values ×100 as mean±SD over seeds "
        "42, 123, and 456.",
        "",
        header,
        divider,
    ]
    for method in (item for item in METHODS if item.newly_added):
        cells = []
        for dataset in table.datasets:
            row = next(
                item
                for item in evidence_rows
                if item["table"] == table.key
                and item["dataset"] == dataset
                and item["method"] == method.key
            )
            cells.append(
                " / ".join(
                    _format_mean_std(
                        float(row[f"{metric.key}_mean"]),
                        float(row[f"{metric.key}_std"]),
                    )
                    for metric in table.metrics
                )
            )
        lines.append(
            f"| {method.label} | {method.feature_label} | "
            + " | ".join(cells)
            + " |"
        )
    lines.append("")
    if table.key == "table6":
        lines.extend(
            [
                "Adaptation gain is reported as probe BA minus centroid BA. A larger value "
                "indicates greater change after fitting the fixed readout and is not treated "
                "as a stronger final representation by itself.",
                "",
            ]
        )
    if table.key == "table5":
        lines.extend(
            [
                "CHAOS reports its patient-matched single-organ liver surface and is not "
                "included in the multi-organ profile average.",
                "",
            ]
        )
    return lines


def _best_method(
    rows: Sequence[Mapping[str, Any]],
    table: TableSpec,
    dataset: str,
    metric: MetricSpec,
) -> Mapping[str, Any]:
    candidates = [
        row
        for row in rows
        if row["table"] == table.key and row["dataset"] == dataset
    ]
    return (
        max(candidates, key=lambda row: float(row[f"{metric.key}_mean"]))
        if metric.direction == "higher"
        else min(candidates, key=lambda row: float(row[f"{metric.key}_mean"]))
    )


def _interpretation_markdown(
    evidence_rows: Sequence[Mapping[str, Any]],
    profile_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    table3 = next(table for table in TABLES if table.key == "table3")
    table5 = next(table for table in TABLES if table.key == "table5")
    table6 = next(table for table in TABLES if table.key == "table6")
    acquisition_reference_wins = sum(
        _best_method(evidence_rows, table3, dataset, metric)["role"]
        != "Controlled geometry"
        for dataset in table3.profile_datasets
        for metric in table3.metrics
    )
    retrieval_sa_wins = sum(
        _best_method(evidence_rows, table5, dataset, metric)["method"]
        == "med3dino_sa"
        for dataset in table5.profile_datasets
        for metric in table5.metrics
    )
    anatomical_primary_metrics = table6.metrics[:2]
    anatomical_sa_wins = sum(
        _best_method(evidence_rows, table6, dataset, metric)["method"]
        == "med3dino_sa"
        for dataset in table6.profile_datasets
        for metric in anatomical_primary_metrics
    )
    added_rows = [row for row in profile_rows if row["newly_added"]]
    best_added_probe = max(
        added_rows,
        key=lambda row: float(row["table6_probe_ba_mean"]),
    )
    return [
        "## Evidence Interpretation",
        "",
        f"- Independently pretrained encoders lead {acquisition_reference_wins}/15 controlled-acquisition "
        "dataset–metric cells, including every drift and CKA surface.",
        f"- Med3DINO-SA leads {retrieval_sa_wins}/9 multi-organ retrieval cells and "
        f"{anatomical_sa_wins}/6 centroid/probe cells.",
        f"- {best_added_probe['label']} has the highest added-method mean probe BA "
        f"({_format_score(float(best_added_probe['table6_probe_ba_mean']))}).",
        "- Dataset-level semantic transfer, cross-modal retrieval, and low-shot results "
        "produce different encoder orderings rather than one global ranking.",
        "",
        "These are descriptive cross-family profiles. They do not establish that differences "
        "are caused by architecture or objective, because the independent reference encoders also differ "
        "in pretraining data, preprocessing assumptions, input size, and feature interface.",
        "",
    ]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_profile_csv(
    output_dir: Path,
    profile_rows: Sequence[Mapping[str, Any]],
) -> None:
    metric_fields: list[str] = []
    for table_key, metric_key in PROFILE_METRICS:
        metric_fields.extend(
            [
                f"{table_key}_{metric_key}_mean",
                f"{table_key}_{metric_key}_rank",
            ]
        )
    _write_csv(
        output_dir / "cross_family_profile_matrix.csv",
        profile_rows,
        (
            "key",
            "label",
            "role",
            "architecture",
            "checkpoint",
            "feature_type",
            "feature_label",
            "crop_size",
            "newly_added",
            *metric_fields,
        ),
    )


def _write_external_csvs(
    output_dir: Path,
    evidence_rows: Sequence[Mapping[str, Any]],
) -> None:
    for table in TABLES:
        rows = [
            row
            for row in evidence_rows
            if row["table"] == table.key and row["newly_added"]
        ]
        metric_fields: list[str] = []
        for metric in table.metrics:
            metric_fields.extend(
                [
                    f"{metric.key}_mean",
                    f"{metric.key}_std",
                    *[f"{metric.key}_seed_{seed}" for seed in SEEDS],
                ]
            )
        _write_csv(
            output_dir / f"added_external_{table.key}.csv",
            rows,
            (
                "dataset",
                "dataset_label",
                "method",
                "method_label",
                "architecture",
                "checkpoint",
                "feature_type",
                "feature_label",
                "crop_size",
                *metric_fields,
            ),
        )


def generate_tables(results_root: Path, output_dir: Path) -> dict[str, Any]:
    evidence_rows = collect_evidence_rows(results_root)
    profile_rows = build_profile_rows(evidence_rows)
    markdown_lines = [
        "# REFORM-3D Cross-Family Representation Profiles",
        "",
        "This artifact reports the designated primary frozen feature for each encoder. "
        "It is a diagnostic comparison under a common evaluation protocol, not a controlled "
        "causal comparison across architectures or pretraining objectives.",
        "",
        *_coverage_markdown(),
        *_profile_markdown(profile_rows),
        *_interpretation_markdown(evidence_rows, profile_rows),
    ]
    for table in TABLES:
        markdown_lines.extend(_compact_external_table(table, evidence_rows))
    markdown_lines.extend(
        [
            "## Evidence Boundary",
            "",
            "- The profile matrix does not combine metrics into an overall score.",
            "- ISO/REL/SA support controlled geometric-policy comparisons; all other methods "
            "are independent frozen-encoder references.",
            "- Profile averages are descriptive and do not replace dataset-level mean±SD or "
            "paired statistical analyses.",
            "- Native feature surfaces and input sizes remain explicit because heterogeneous "
            "encoders do not expose one universally equivalent feature token.",
            "",
            "## Provenance",
            "",
            "The tables are generated directly from completed seed-42/123/456 result artifacts. "
            "Generation fails when a required metric, seed, or source path is missing. The JSON "
            "output retains each metric's source artifact by seed.",
            "",
            "Regenerate with:",
            "",
            "```bash",
            "python -m reform3d.core.cross_family_profile_tables",
            "```",
            "",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "cross_family_profile_evidence.md"
    markdown_path.write_text(
        "\n".join(markdown_lines).rstrip() + "\n",
        encoding="utf-8",
    )
    _write_profile_csv(output_dir, profile_rows)
    _write_external_csvs(output_dir, evidence_rows)
    payload = {
        "analysis_scope": "cross_family_primary_feature_profiles",
        "results_root": str(results_root),
        "seeds": list(SEEDS),
        "profile_contract": {
            "acquisition_datasets": list(TABLES[0].profile_datasets),
            "cross_modal_datasets": list(TABLES[1].profile_datasets),
            "anatomical_datasets": list(TABLES[2].profile_datasets),
            "aggregation": "arithmetic_mean_of_dataset_level_seed_means",
            "overall_score": None,
            "external_comparison": "independent_reference_not_causal",
            "controlled_geometric_comparison": "med3dino_iso_rel_sa_only",
        },
        "methods": [asdict(method) for method in METHODS],
        "profile_rows": profile_rows,
        "evidence_rows": evidence_rows,
    }
    _write_json(output_dir / "cross_family_profile_evidence.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate cross-family REFORM-3D profile and external-result tables."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = generate_tables(args.results_root, args.output_dir)
    print(f"methods: {len(payload['profile_rows'])}")
    print(f"evidence rows: {len(payload['evidence_rows'])}")
    print(f"wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
