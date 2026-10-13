#!/usr/bin/env python
"""Regenerate Acquisition Robustness anisotropy regression artifacts from cached features.

This helper avoids re-running feature extraction or the full native-spacing
pipeline. It recomputes the spacing readout using the Acquisition semantic-analysis
spacing-regression logic and writes per-checkpoint payloads to a separate JSON
filename by default.

Example:
    python -m reform3d.modules.acquisition_robustness.regenerate_anisotropy_regression_from_cache \
        --manifest totalsegmentermri/original_bins/manifest_sampled.json \
        --analysis-name totalsegmentermri
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    from .config import (
        CHECKPOINTS,
        ensure_output_directories,
        get_checkpoint_feature_dir,
        get_dataset_name_from_manifest_path,
        get_manifest_variant_from_manifest_path,
        get_output_paths,
    )
    from .semantic_analysis import (
        _select_balanced_bin_indices,
        serialize_spacing_regression_results,
        spacing_regression_probe,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    from config import (
        CHECKPOINTS,
        ensure_output_directories,
        get_checkpoint_feature_dir,
        get_dataset_name_from_manifest_path,
        get_manifest_variant_from_manifest_path,
        get_output_paths,
    )
    from semantic_analysis import (
        _select_balanced_bin_indices,
        serialize_spacing_regression_results,
        spacing_regression_probe,
    )

from reform3d.core.checkpoints import resolve_checkpoint_config

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

FEATURE_TYPES = ["cls", "avg_pool", "multilayer"]
SPACING_READOUT_KEY = "anisotropy_regression"


def _checkpoint_crop_size(checkpoint_name: str) -> int:
    return CHECKPOINTS[checkpoint_name].crop_size


def _resolve_requested_checkpoints(requested: Iterable[str]) -> List[str]:
    resolved_names: List[str] = []
    for checkpoint_name in requested:
        resolved_name = resolve_checkpoint_config(checkpoint_name).checkpoint_key
        if resolved_name not in resolved_names:
            resolved_names.append(resolved_name)
    return resolved_names


def _manifest_hash(manifest_path: Path) -> str:
    return hashlib.md5(manifest_path.read_bytes()).hexdigest()[:8]


def _normalize_path_list(values: np.ndarray) -> List[str]:
    normalized: List[str] = []
    for value in values.tolist():
        if isinstance(value, bytes):
            normalized.append(value.decode("utf-8"))
        else:
            normalized.append(str(value))
    return normalized


def _load_manifest(manifest_path: Path) -> Tuple[dict, Dict[str, Dict[str, float]]]:
    manifest = json.loads(manifest_path.read_text())
    lookup: Dict[str, Dict[str, float]] = {}
    for volume in manifest["volumes"]:
        file_path = str(volume["file_path"])
        if file_path in lookup:
            raise ValueError(f"Duplicate manifest file_path detected: {file_path}")
        lookup[file_path] = {
            "anisotropy_ratio": float(volume.get("anisotropy_ratio", 1.0)),
            "anisotropy_bin": int(volume["anisotropy_bin"]),
        }
    return manifest, lookup


def _resolve_targets(
    cache: np.lib.npyio.NpzFile,
    manifest: dict,
    manifest_lookup: Dict[str, Dict[str, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    if "file_paths" in cache.files:
        cache_paths = _normalize_path_list(cache["file_paths"])
        missing = [path for path in cache_paths if path not in manifest_lookup]
        if missing:
            raise KeyError(f"Feature cache contains {len(missing)} paths not present in manifest; first missing path: {missing[0]}")
        anisotropy_ratios = np.array(
            [manifest_lookup[path]["anisotropy_ratio"] for path in cache_paths],
            dtype=np.float64,
        )
        bin_labels = np.array(
            [manifest_lookup[path]["anisotropy_bin"] for path in cache_paths],
            dtype=np.int32,
        )
    else:
        volumes = manifest["volumes"]
        anisotropy_ratios = np.array([float(volume.get("anisotropy_ratio", 1.0)) for volume in volumes], dtype=np.float64)
        bin_labels = np.array([int(volume["anisotropy_bin"]) for volume in volumes], dtype=np.int32)

    if "bin_labels" in cache.files:
        cached_bins = np.asarray(cache["bin_labels"], dtype=np.int32)
        if cached_bins.shape == bin_labels.shape and not np.array_equal(cached_bins, bin_labels):
            logger.warning("Manifest-aligned bins differ from cached bins; using manifest-aligned bins for regeneration")

    return anisotropy_ratios, bin_labels


def _build_spacing_payload(
    features: np.ndarray,
    anisotropy_ratios: np.ndarray,
    bin_labels: np.ndarray,
    checkpoint_name: str,
    feature_type: str,
) -> Dict[str, object]:
    spacing_reg = spacing_regression_probe(
        features=features,
        anisotropy_ratios=anisotropy_ratios,
        bin_labels=bin_labels,
    )
    spacing_reg.checkpoint_name = checkpoint_name
    spacing_reg.feature_type = feature_type

    spacing_readout = serialize_spacing_regression_results(spacing_reg)
    balanced_idx, _, sample_count_per_bin = _select_balanced_bin_indices(bin_labels, random_state=42)
    if balanced_idx is not None:
        balanced_spacing_reg = spacing_regression_probe(
            features=features[balanced_idx],
            anisotropy_ratios=anisotropy_ratios[balanced_idx],
            bin_labels=bin_labels[balanced_idx],
        )
        spacing_readout["balanced_bin_sensitivity"] = {
            "sample_count_per_bin": int(sample_count_per_bin),
            **serialize_spacing_regression_results(balanced_spacing_reg),
        }

    return {
        "checkpoint": checkpoint_name,
        "feature_type": feature_type,
        "task_name": SPACING_READOUT_KEY,
        "alternate_name": "spacing_regression",
        "anisotropy_regression": spacing_readout,
        "spacing_regression": spacing_readout,
    }


def _iter_checkpoints(features_dir: Path, requested: Iterable[str] | None) -> List[str]:
    if requested:
        return _resolve_requested_checkpoints(requested)

    discovered = sorted(path.name for path in features_dir.iterdir() if path.is_dir())
    unknown = [checkpoint_name for checkpoint_name in discovered if checkpoint_name not in CHECKPOINTS]
    if unknown:
        unknown_display = ", ".join(unknown)
        raise ValueError(
            f"Unregistered checkpoint feature directories under {features_dir}: {unknown_display}. "
            "Acquisition Robustness cache regeneration only supports registry-resolved checkpoints."
        )
    return discovered


def _iter_feature_types(checkpoint_dir: Path, requested: Iterable[str] | None) -> List[str]:
    if requested:
        return list(requested)
    return [feature_type for feature_type in FEATURE_TYPES if (checkpoint_dir / feature_type).is_dir()]


def regenerate_outputs(
    manifest_path: Path,
    analysis_name: str | None,
    checkpoints: List[str] | None,
    feature_types: List[str] | None,
    output_filename: str,
) -> int:
    dataset_name = analysis_name or get_dataset_name_from_manifest_path(manifest_path)
    manifest_variant = get_manifest_variant_from_manifest_path(manifest_path)
    ensure_output_directories(dataset_name, manifest_variant)
    output_paths = get_output_paths(dataset_name, manifest_variant)
    features_dir = output_paths["features"]
    results_dir = output_paths["results"]
    manifest, manifest_lookup = _load_manifest(manifest_path)
    manifest_hash = _manifest_hash(manifest_path)
    binning_scheme = manifest.get("binning_scheme", manifest_variant)

    updated = 0
    skipped = 0
    checkpoint_names = _iter_checkpoints(features_dir, checkpoints)
    for checkpoint_name in checkpoint_names:
        checkpoint_feature_dir = features_dir / checkpoint_name
        if not checkpoint_feature_dir.is_dir():
            logger.warning("Skipping missing checkpoint feature directory: %s", checkpoint_feature_dir)
            skipped += 1
            continue

        for feature_type in _iter_feature_types(checkpoint_feature_dir, feature_types):
            crop_size = _checkpoint_crop_size(checkpoint_name)
            cache_path = get_checkpoint_feature_dir(features_dir, checkpoint_name, feature_type) / (
                f"features_c{crop_size}_{manifest_hash}.npz"
            )
            if not cache_path.exists():
                logger.warning("Skipping missing feature cache: %s", cache_path)
                skipped += 1
                continue

            with np.load(cache_path, allow_pickle=False) as cache:
                feature_key = "features" if "features" in cache.files else cache.files[0]
                features = np.asarray(cache[feature_key])
                anisotropy_ratios, bin_labels = _resolve_targets(cache, manifest, manifest_lookup)

            if features.shape[0] != len(anisotropy_ratios):
                raise ValueError(
                    f"Feature/target length mismatch for {checkpoint_name}/{feature_type}: "
                    f"{features.shape[0]} features vs {len(anisotropy_ratios)} targets"
                )

            result_payload = _build_spacing_payload(
                features=features,
                anisotropy_ratios=anisotropy_ratios,
                bin_labels=bin_labels,
                checkpoint_name=checkpoint_name,
                feature_type=feature_type,
            )
            result_payload["binning_scheme"] = binning_scheme

            result_dir = get_checkpoint_feature_dir(results_dir, checkpoint_name, feature_type)
            result_dir.mkdir(parents=True, exist_ok=True)
            result_path = result_dir / output_filename
            with open(result_path, "w") as handle:
                json.dump(result_payload, handle, indent=2)

            updated += 1
            logger.info(
                "Regenerated %s (%s): r2=%.4f",
                checkpoint_name,
                feature_type,
                result_payload["anisotropy_regression"]["r2_score"],
            )

    logger.info(
        "Finished regenerating %s artifacts for %s/%s: updated=%d skipped=%d",
        output_filename,
        dataset_name,
        manifest_variant,
        updated,
        skipped,
    )
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate Acquisition Robustness anisotropy regression outputs from cached feature npz files.",
    )
    parser.add_argument(
        "-m",
        "--manifest",
        type=Path,
        required=True,
        help="Path to the sampled manifest used to build the feature caches.",
    )
    parser.add_argument(
        "-a",
        "--analysis-name",
        type=str,
        default=None,
        help="Dataset/output namespace. Defaults to the dataset inferred from the manifest path.",
    )
    parser.add_argument(
        "-c",
        "--checkpoints",
        nargs="*",
        default=None,
        help="Optional checkpoint subset. Defaults to all checkpoint directories under the features root.",
    )
    parser.add_argument(
        "-f",
        "--feature-types",
        nargs="*",
        choices=FEATURE_TYPES,
        default=None,
        help="Optional feature-type subset. Defaults to all available feature directories per checkpoint.",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="anisotropy_regression_log_target.json",
        help="Output JSON filename to write under each checkpoint/feature directory.",
    )
    args = parser.parse_args()

    regenerate_outputs(
        manifest_path=args.manifest,
        analysis_name=args.analysis_name,
        checkpoints=args.checkpoints,
        feature_types=args.feature_types,
        output_filename=args.output_filename,
    )


if __name__ == "__main__":
    main()