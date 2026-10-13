#!/usr/bin/env python
"""Import cleaned MRI manifests into Acquisition Robustness manifests.

Examples:
    python -m reform3d.modules.acquisition_robustness.import_cleaned_mri_manifest \
        --input-manifest /path/to/jhu_stroke_cleaned_manifest.json \
        --dataset-slug jhu_stroke \
        --dataset-title JHU-Stroke \
        --manifest-variant original_bins

    python -m reform3d.modules.acquisition_robustness.import_cleaned_mri_manifest \
        --input-manifest /path/to/totalsegmentermri_cleaned_manifest.json \
        --dataset-slug totalsegmentermri \
        --manifest-variant coarse_bins \
        --sample-mode ratio_coverage
"""

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List

import numpy as np

try:
    from .config import (
        DEFAULT_BINNING_SCHEME,
        MANIFEST_FILE_NAMES,
        get_anisotropy_bin,
        get_bin_configs,
        get_acquisition_manifest_dir,
        normalize_manifest_variant,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    from config import (
        DEFAULT_BINNING_SCHEME,
        MANIFEST_FILE_NAMES,
        get_anisotropy_bin,
        get_bin_configs,
        get_acquisition_manifest_dir,
        normalize_manifest_variant,
    )


def load_source_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "volumes" in payload:
        return payload, payload["volumes"]
    if isinstance(payload, list):
        return {}, payload
    raise ValueError(f"Unsupported manifest format: {path}")


def resolve_path(value: str | None, source_root: Path) -> str | None:
    if not value:
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)
    return str((source_root / candidate).resolve())


def infer_case_id(entry: dict[str, Any], file_path: str, index: int) -> str:
    for key in ("case_id", "subject_id", "id"):
        if key in entry and entry[key]:
            return str(entry[key])
    return f"case_{index:05d}_{Path(file_path).stem}"


def infer_volume_modality(entry: dict[str, Any], manifest_modality: str) -> str:
    if entry.get("modality"):
        return str(entry["modality"]).lower()
    normalized = manifest_modality.lower()
    if "dwi" in normalized:
        return "dwi"
    if normalized in {"mr", "mri", "mri_dwi"}:
        return "mr"
    return normalized


def compute_ratio(entry: dict[str, Any], spacing: list[float]) -> float:
    if entry.get("anisotropy_ratio") is not None:
        return round(float(entry["anisotropy_ratio"]), 4)
    values = [float(x) for x in spacing]
    return round(max(values) / min(values), 4)


def compute_ratio_summary(volumes: List[Dict[str, Any]]) -> Dict[str, float]:
    if not volumes:
        return {}
    ratios = np.array([float(volume["anisotropy_ratio"]) for volume in volumes], dtype=float)
    return {
        "min": float(np.min(ratios)),
        "p10": float(np.percentile(ratios, 10)),
        "p25": float(np.percentile(ratios, 25)),
        "median": float(np.median(ratios)),
        "p75": float(np.percentile(ratios, 75)),
        "p90": float(np.percentile(ratios, 90)),
        "max": float(np.max(ratios)),
        "mean": float(np.mean(ratios)),
        "std": float(np.std(ratios)),
        "n_unique": int(len(np.unique(ratios))),
    }


def sample_with_ratio_coverage_records(
    volumes: List[Dict[str, Any]],
    target: int,
    seed: int,
    n_ratio_strata: int = 10,
) -> List[Dict[str, Any]]:
    if target >= len(volumes):
        return list(volumes)

    rng = np.random.default_rng(seed)
    ordered = sorted(volumes, key=lambda volume: (float(volume["anisotropy_ratio"]), volume["file_path"]))
    ratios = np.array([float(volume["anisotropy_ratio"]) for volume in ordered], dtype=float)
    ratio_min = float(ratios.min())
    ratio_max = float(ratios.max())

    if ratio_max - ratio_min < 1e-12:
        chosen_idx = rng.choice(len(ordered), size=target, replace=False)
        return [ordered[int(idx)] for idx in chosen_idx]

    n_strata = max(1, min(n_ratio_strata, target, len(ordered)))
    edges = np.linspace(ratio_min, ratio_max, n_strata + 1)
    strata: List[List[Dict[str, Any]]] = []
    for index in range(n_strata):
        left = edges[index]
        right = edges[index + 1]
        if index == n_strata - 1:
            stratum = [volume for volume in ordered if left <= float(volume["anisotropy_ratio"]) <= right]
        else:
            stratum = [volume for volume in ordered if left <= float(volume["anisotropy_ratio"]) < right]
        if stratum:
            strata.append(stratum)

    quotas = [target // len(strata)] * len(strata)
    for index in range(target % len(strata)):
        quotas[index] += 1

    selected: List[Dict[str, Any]] = []
    leftovers: List[Dict[str, Any]] = []
    for stratum, quota in zip(strata, quotas):
        if quota <= 0:
            leftovers.extend(stratum)
            continue
        if len(stratum) <= quota:
            selected.extend(stratum)
            continue
        chosen_idx = set(rng.choice(len(stratum), size=quota, replace=False).tolist())
        for idx, volume in enumerate(stratum):
            if idx in chosen_idx:
                selected.append(volume)
            else:
                leftovers.append(volume)

    remaining = target - len(selected)
    if remaining > 0 and leftovers:
        fill_idx = rng.choice(len(leftovers), size=remaining, replace=False)
        selected.extend([leftovers[int(idx)] for idx in fill_idx])

    return selected


def build_manifest_volumes(
    entries: List[Dict[str, Any]],
    source_root: Path,
    dataset_slug: str,
    manifest_modality: str,
    binning_scheme: str,
) -> List[Dict[str, Any]]:
    volumes: List[Dict[str, Any]] = []
    for index, entry in enumerate(entries):
        image_value = entry.get("image") or entry.get("image_path") or entry.get("file_path")
        mask_value = entry.get("mask") or entry.get("mask_path") or entry.get("label_path")
        if not image_value:
            raise ValueError(f"Entry {index} is missing image path")

        file_path = resolve_path(str(image_value), source_root)
        label_path = resolve_path(str(mask_value), source_root) if mask_value else None
        spacing = [float(x) for x in entry["spacing"]]
        ratio = compute_ratio(entry, spacing)
        volume = {
            "case_id": infer_case_id(entry, file_path, index),
            "file_path": file_path,
            "dataset": dataset_slug,
            "modality": infer_volume_modality(entry, manifest_modality),
            "shape": [int(x) for x in entry["shape"]],
            "spacing": spacing,
            "anisotropy_ratio": ratio,
            "anisotropy_bin": int(get_anisotropy_bin(ratio, spacing=tuple(spacing), scheme=binning_scheme)),
            "has_label": label_path is not None,
            "label_path": label_path,
        }
        for key in ("subset", "split", "source_dataset", "mask_coverage_pct"):
            if key in entry:
                volume[key] = entry[key]
        volumes.append(volume)

    return sorted(volumes, key=lambda volume: (float(volume["anisotropy_ratio"]), volume["file_path"]))


def build_sampled_manifest(
    volumes: List[Dict[str, Any]],
    sample_mode: str,
    min_per_bin: int,
    max_per_bin: int,
    seed: int,
    binning_scheme: str,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if sample_mode == "all":
        return list(volumes), {
            "min_per_bin": 0,
            "max_per_bin": len(volumes),
            "seed": seed,
            "strategy": "all_available",
        }

    sampled: List[Dict[str, Any]] = []
    for bin_config in get_bin_configs(binning_scheme):
        bin_volumes = [volume for volume in volumes if int(volume["anisotropy_bin"]) == bin_config.bin_id]
        n_bin = len(bin_volumes)
        if n_bin == 0:
            continue
        if n_bin < min_per_bin or n_bin <= max_per_bin:
            sampled.extend(bin_volumes)
        else:
            sampled.extend(sample_with_ratio_coverage_records(bin_volumes, max_per_bin, seed + bin_config.bin_id))

    sampled = sorted(sampled, key=lambda volume: (float(volume["anisotropy_ratio"]), volume["file_path"]))
    return sampled, {
        "min_per_bin": min_per_bin,
        "max_per_bin": max_per_bin,
        "seed": seed,
        "strategy": "ratio_coverage_within_bin",
        "ratio_strata_per_bin": 10,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import cleaned MRI manifests into Acquisition Robustness manifests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m reform3d.modules.acquisition_robustness.import_cleaned_mri_manifest --input-manifest /path/to/jhu_stroke_cleaned_manifest.json --dataset-slug jhu_stroke --dataset-title JHU-Stroke --manifest-variant original_bins\n"
            "  python -m reform3d.modules.acquisition_robustness.import_cleaned_mri_manifest --input-manifest /path/to/totalsegmentermri_cleaned_manifest.json --dataset-slug totalsegmentermri --manifest-variant coarse_bins --sample-mode ratio_coverage"
        ),
    )
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--dataset-slug", required=True)
    parser.add_argument("--dataset-title", default=None)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--manifest-variant", default="original_bins")
    parser.add_argument("--sample-mode", choices=["all", "ratio_coverage"], default="all")
    parser.add_argument("--min-per-bin", type=int, default=50)
    parser.add_argument("--max-per-bin", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_payload, entries = load_source_manifest(args.input_manifest)
    source_root = args.source_root or Path(source_payload.get("output_root") or args.input_manifest.parent)
    manifest_variant = normalize_manifest_variant(args.manifest_variant)
    binning_scheme = "coarse_ratio_thickness" if manifest_variant == "coarse_bins" else DEFAULT_BINNING_SCHEME
    dataset_title = args.dataset_title or source_payload.get("dataset") or args.dataset_slug
    manifest_modality = str(source_payload.get("modality", "mr"))

    volumes = build_manifest_volumes(entries, source_root, args.dataset_slug, manifest_modality, binning_scheme)
    sampled_volumes, sampling_config = build_sampled_manifest(
        volumes,
        args.sample_mode,
        args.min_per_bin,
        args.max_per_bin,
        args.seed,
        binning_scheme,
    )

    output_dir = get_acquisition_manifest_dir(args.dataset_slug, manifest_variant)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_manifest = {
        "version": "1.1",
        "description": f"Acquisition Robustness full manifest: {dataset_title} (mr, {manifest_variant})",
        "binning_scheme": binning_scheme,
        "dataset": args.dataset_slug,
        "dataset_title": dataset_title,
        "modality": "mr",
        "source_manifest": str(args.input_manifest),
        "source_root": str(source_root),
        "sampling_config": None,
        "total_volumes": len(volumes),
        "volumes": volumes,
    }

    sampled_manifest = {
        "version": "1.1",
        "description": f"Acquisition Robustness sampled manifest: {dataset_title} (mr, {manifest_variant})",
        "binning_scheme": binning_scheme,
        "dataset": args.dataset_slug,
        "dataset_title": dataset_title,
        "modality": "mr",
        "source_manifest": str(args.input_manifest),
        "source_root": str(source_root),
        "sampling_config": sampling_config,
        "total_volumes": len(sampled_volumes),
        "volumes": sampled_volumes,
        "bin_ratio_statistics": {
            str(bin_config.bin_id): compute_ratio_summary(
                [volume for volume in sampled_volumes if int(volume["anisotropy_bin"]) == bin_config.bin_id]
            )
            for bin_config in get_bin_configs(binning_scheme)
        },
    }

    meta = {
        "dataset": args.dataset_slug,
        "dataset_title": dataset_title,
        "module": "acquisition_robustness",
        "manifest_variant": manifest_variant,
        "binning_scheme": binning_scheme,
        "modality": "mr",
        "source_manifest": str(args.input_manifest),
        "source_root": str(source_root),
        "full_manifest": MANIFEST_FILE_NAMES["full"],
        "sampled_manifest": MANIFEST_FILE_NAMES["sampled"],
        "total_volumes_full": len(volumes),
        "total_volumes_sampled": len(sampled_volumes),
        "anisotropy_ratio": {
            "min": round(min(float(volume["anisotropy_ratio"]) for volume in volumes), 4),
            "max": round(max(float(volume["anisotropy_ratio"]) for volume in volumes), 4),
            "mean": round(mean(float(volume["anisotropy_ratio"]) for volume in volumes), 4),
            "median": round(median(float(volume["anisotropy_ratio"]) for volume in volumes), 4),
        },
        "sampling": sampling_config,
    }

    (output_dir / MANIFEST_FILE_NAMES["full"]).write_text(json.dumps(full_manifest, indent=2) + "\n")
    (output_dir / MANIFEST_FILE_NAMES["sampled"]).write_text(json.dumps(sampled_manifest, indent=2) + "\n")
    (output_dir / MANIFEST_FILE_NAMES["meta"]).write_text(json.dumps(meta, indent=2) + "\n")

    print(f"Imported {len(volumes)} volumes into {output_dir}")


if __name__ == "__main__":
    main()