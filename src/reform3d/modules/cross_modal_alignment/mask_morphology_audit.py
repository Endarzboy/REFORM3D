#!/usr/bin/env python
"""Audit physical mask-erosion radii against Cross-Modal manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .config import (
    CROSS_MODAL_OUTPUTS,
    DEFAULT_MANIFEST_VARIANT,
    get_cross_modal_manifest_path,
    normalize_cross_modal_dataset_name,
    normalize_manifest_variant,
)
from .feature_extractor import (
    OrganSample,
    _binarize_mask,
    _load_binary_mask_volume,
    _load_manifest_samples,
    _load_png_mask_stack,
    _uses_png_mask_stack,
)
from .physical_mask import erode_binary_mask_mm, normalize_spacing_mm


DEFAULT_DATASETS = ("amos22_ct_mr", "mmwhs_ct_mr")
DEFAULT_RADII_MM = (1.0, 2.0)
DEFAULT_OUTPUT_DIR = CROSS_MODAL_OUTPUTS / "protocol_stability"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_DIR / "mask_morphology_viability.json"
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "mask_morphology_viability.csv"


@dataclass(frozen=True)
class MaskGroupTask:
    dataset: str
    samples: tuple[OrganSample, ...]
    radii_mm: tuple[float, ...]
    min_mask_voxels: int


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "sample_id",
        "source_case_id",
        "modality",
        "organ",
        "radius_mm",
        "original_voxels",
        "eroded_voxels",
        "retained_fraction",
        "min_mask_voxels",
        "passes_min_mask_voxels",
        "mask_path",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _manifest_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mask_group_key(sample: OrganSample) -> tuple[str, str]:
    return str(sample.mask_path), str(sample.mask_loader or "")


def _load_group_mask(sample: OrganSample) -> np.ndarray:
    if _uses_png_mask_stack(sample.mask_loader):
        return _load_png_mask_stack(sample.mask_path)
    return _load_binary_mask_volume(sample.mask_path)


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coordinates = np.where(mask)
    if coordinates[0].size == 0:
        return mask.astype(bool, copy=False)
    slices = tuple(
        slice(max(0, int(axis.min()) - 1), min(mask.shape[index], int(axis.max()) + 2))
        for index, axis in enumerate(coordinates)
    )
    return mask[slices].astype(bool, copy=False)


def _audit_mask_group(task: MaskGroupTask) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    first_sample = task.samples[0]
    try:
        raw_mask = _load_group_mask(first_sample)
    except Exception as exc:
        for sample in task.samples:
            for radius_mm in task.radii_mm:
                rows.append(
                    {
                        "dataset": task.dataset,
                        "sample_id": sample.sample_id,
                        "source_case_id": sample.source_case_id,
                        "modality": sample.modality,
                        "organ": sample.primary_organ,
                        "radius_mm": radius_mm,
                        "original_voxels": None,
                        "eroded_voxels": None,
                        "retained_fraction": None,
                        "min_mask_voxels": task.min_mask_voxels,
                        "passes_min_mask_voxels": False,
                        "mask_path": str(sample.mask_path),
                        "error": f"mask_load_failed:{exc}",
                    }
                )
        return rows

    for sample in task.samples:
        base_row = {
            "dataset": task.dataset,
            "sample_id": sample.sample_id,
            "source_case_id": sample.source_case_id,
            "modality": sample.modality,
            "organ": sample.primary_organ,
            "min_mask_voxels": task.min_mask_voxels,
            "mask_path": str(sample.mask_path),
        }
        try:
            if sample.spacing is None or len(sample.spacing) != 3:
                raise ValueError("missing_spacing")
            spacing = normalize_spacing_mm(sample.spacing)

            binary_mask = _binarize_mask(raw_mask, sample).astype(bool, copy=False)
            local_mask = _tight_mask(binary_mask)
            original_voxels = int(local_mask.sum())
            if original_voxels == 0:
                raise ValueError("empty_mask")

            for radius_mm in task.radii_mm:
                eroded_voxels = int(
                    np.count_nonzero(
                        erode_binary_mask_mm(local_mask, spacing, radius_mm)
                    )
                )
                rows.append(
                    {
                        **base_row,
                        "radius_mm": radius_mm,
                        "original_voxels": original_voxels,
                        "eroded_voxels": eroded_voxels,
                        "retained_fraction": float(eroded_voxels / original_voxels),
                        "passes_min_mask_voxels": eroded_voxels >= task.min_mask_voxels,
                        "error": None,
                    }
                )
        except Exception as exc:
            for radius_mm in task.radii_mm:
                rows.append(
                    {
                        **base_row,
                        "radius_mm": radius_mm,
                        "original_voxels": None,
                        "eroded_voxels": None,
                        "retained_fraction": None,
                        "passes_min_mask_voxels": False,
                        "error": str(exc),
                    }
                )
    return rows


def _quantile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _summarize_rows(rows: Sequence[Mapping[str, Any]], min_mask_voxels: int) -> dict[str, Any]:
    valid_rows = [row for row in rows if not row.get("error")]
    nonempty = [
        row
        for row in valid_rows
        if row.get("eroded_voxels") is not None and int(row["eroded_voxels"]) > 0
    ]
    retained = [
        float(row["retained_fraction"])
        for row in valid_rows
        if row.get("retained_fraction") is not None
    ]
    passed = [row for row in valid_rows if bool(row.get("passes_min_mask_voxels"))]
    failures = [
        {
            "sample_id": row.get("sample_id"),
            "modality": row.get("modality"),
            "organ": row.get("organ"),
            "eroded_voxels": row.get("eroded_voxels"),
            "error": row.get("error"),
        }
        for row in rows
        if row.get("error") or not bool(row.get("passes_min_mask_voxels"))
    ]
    return {
        "n_samples": len(rows),
        "n_valid": len(valid_rows),
        "n_nonempty": len(nonempty),
        "n_passing": len(passed),
        "n_failing": len(rows) - len(passed),
        "all_samples_nonempty": len(rows) > 0 and len(nonempty) == len(rows),
        "all_samples_pass": len(rows) > 0 and len(passed) == len(rows),
        "min_mask_voxels": int(min_mask_voxels),
        "retained_fraction": {
            "minimum": min(retained) if retained else None,
            "q25": _quantile(retained, 25.0),
            "median": _quantile(retained, 50.0),
            "q75": _quantile(retained, 75.0),
            "maximum": max(retained) if retained else None,
        },
        "failures": failures,
    }


def _group_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_fields: Sequence[str],
    min_mask_voxels: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in group_fields)].append(row)
    summaries = []
    for group_key, group_rows in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        summaries.append(
            {
                **dict(zip(group_fields, group_key)),
                **_summarize_rows(group_rows, min_mask_voxels),
            }
        )
    return summaries


def _build_tasks(
    dataset: str,
    samples: Sequence[OrganSample],
    radii_mm: tuple[float, ...],
    min_mask_voxels: int,
) -> list[MaskGroupTask]:
    grouped: dict[tuple[str, str], list[OrganSample]] = defaultdict(list)
    for sample in samples:
        grouped[_mask_group_key(sample)].append(sample)
    return [
        MaskGroupTask(
            dataset=dataset,
            samples=tuple(group),
            radii_mm=radii_mm,
            min_mask_voxels=min_mask_voxels,
        )
        for _, group in sorted(grouped.items())
    ]


def _run_tasks(tasks: Sequence[MaskGroupTask], workers: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if workers <= 1:
        iterator = (_audit_mask_group(task) for task in tasks)
        for group_rows in tqdm(iterator, total=len(tasks), desc="Auditing mask files", unit="file"):
            rows.extend(group_rows)
        return rows

    with ProcessPoolExecutor(max_workers=workers) as executor:
        iterator = executor.map(_audit_mask_group, tasks)
        for group_rows in tqdm(iterator, total=len(tasks), desc="Auditing mask files", unit="file"):
            rows.extend(group_rows)
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Audit physical erosion radii without extracting encoder features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--manifest-variant", default=DEFAULT_MANIFEST_VARIANT)
    parser.add_argument("--radii-mm", nargs="+", type=float, default=list(DEFAULT_RADII_MM))
    parser.add_argument("--min-mask-voxels", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args(argv)

    datasets = tuple(dict.fromkeys(normalize_cross_modal_dataset_name(name) for name in args.datasets))
    manifest_variant = normalize_manifest_variant(args.manifest_variant)
    radii_mm = tuple(sorted(set(float(value) for value in args.radii_mm)))
    if not radii_mm or any(not math.isfinite(value) or value <= 0 for value in radii_mm):
        raise ValueError("All morphology radii must be positive finite values")
    if args.min_mask_voxels <= 0:
        raise ValueError("--min-mask-voxels must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    all_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for dataset in datasets:
        manifest_path = get_cross_modal_manifest_path(dataset, "sampled", manifest_variant)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        samples = _load_manifest_samples(manifest_path)
        tasks = _build_tasks(dataset, samples, radii_mm, int(args.min_mask_voxels))
        dataset_rows = _run_tasks(tasks, int(args.workers))
        all_rows.extend(dataset_rows)
        manifests.append(
            {
                "dataset": dataset,
                "path": str(manifest_path),
                "sha256": _manifest_sha256(manifest_path),
                "n_samples": len(samples),
                "n_mask_files": len(tasks),
            }
        )

    all_rows.sort(
        key=lambda row: (
            str(row.get("dataset")),
            float(row.get("radius_mm") or 0.0),
            str(row.get("modality")),
            str(row.get("organ")),
            str(row.get("sample_id")),
        )
    )
    radius_summary = _group_summary(
        all_rows,
        group_fields=("radius_mm",),
        min_mask_voxels=int(args.min_mask_voxels),
    )
    dataset_radius_summary = _group_summary(
        all_rows,
        group_fields=("dataset", "radius_mm"),
        min_mask_voxels=int(args.min_mask_voxels),
    )
    organ_radius_summary = _group_summary(
        all_rows,
        group_fields=("dataset", "modality", "organ", "radius_mm"),
        min_mask_voxels=int(args.min_mask_voxels),
    )
    strict_passing_radii = [
        float(summary["radius_mm"])
        for summary in radius_summary
        if bool(summary["all_samples_pass"])
    ]
    nonempty_radii = [
        float(summary["radius_mm"])
        for summary in radius_summary
        if bool(summary["all_samples_nonempty"])
    ]
    strict_selected_radius = max(strict_passing_radii) if strict_passing_radii else None
    largest_all_nonempty_radius = max(nonempty_radii) if nonempty_radii else None
    selected_common_radius = largest_all_nonempty_radius
    payload = {
        "task": "cross_modal_mask_morphology_viability",
        "manifest_variant": manifest_variant,
        "datasets": list(datasets),
        "radii_mm": list(radii_mm),
        "min_mask_voxels": int(args.min_mask_voxels),
        "erosion_definition": "distance_transform_edt(mask, sampling=spacing_mm) > radius_mm",
        "strict_selection_rule": "largest tested radius for which every transformed manifest mask remains above min_mask_voxels",
        "nonempty_selection_rule": "largest tested radius for which every transformed manifest mask remains nonempty",
        "selection_rule": (
            "largest tested radius preserving a nonempty transformed mask for every "
            "already-eligible manifest sample; min_mask_voxels remains a diagnostic "
            "post-transformation threshold"
        ),
        "selected_common_radius_mm": selected_common_radius,
        "strict_selected_common_radius_mm": strict_selected_radius,
        "largest_all_nonempty_radius_mm": largest_all_nonempty_radius,
        "manifests": manifests,
        "radius_summary": radius_summary,
        "dataset_radius_summary": dataset_radius_summary,
        "organ_radius_summary": organ_radius_summary,
        "rows": all_rows,
    }
    _write_json(args.output_json, payload)
    _write_csv(args.output_csv, all_rows)
    print(
        json.dumps(
            {
                "datasets": list(datasets),
                "n_rows": len(all_rows),
                "selected_common_radius_mm": selected_common_radius,
                "strict_selected_common_radius_mm": strict_selected_radius,
                "largest_all_nonempty_radius_mm": largest_all_nonempty_radius,
                "output_json": str(args.output_json),
                "output_csv": str(args.output_csv),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
