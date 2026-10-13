"""Extract semantic labels from segmentation masks.

The extractor converts label volumes into compact targets used by Acquisition
Robustness readout probes. It supports organ-presence targets, foreground-volume
targets, label-id multi-label targets, and dominant-organ summaries depending on
the configured ``SemanticTargetSpec``.

Inputs:
    label_path: segmentation mask path from a manifest record.
    target_spec: semantic target definition from ``DATASET_SEMANTIC_TARGET_SPECS``.
    cache_path: optional JSON cache for repeated probe runs.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import nibabel as nib
from dataclasses import dataclass
import json
import fcntl
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

try:
    from .config import (
        SEMANTIC_LABEL_CONFIG,
        SemanticLabelConfig,
        SemanticTargetSpec,
        resolve_semantic_target_spec,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from config import (
        SEMANTIC_LABEL_CONFIG,
        SemanticLabelConfig,
        SemanticTargetSpec,
        resolve_semantic_target_spec,
    )

logger = logging.getLogger(__name__)


SEMANTIC_CACHE_FLUSH_EVERY = 128


def _compute_label_counts(label_path: Union[str, Path]) -> Dict[int, int]:
    """Load a label volume and count voxels per integer label efficiently."""
    img = nib.load(str(label_path))
    data = np.asarray(img.dataobj)
    if data.dtype.kind not in "iu" and not np.isfinite(data).all():
        raise ValueError(f"Annotation contains nonfinite values: {label_path}")

    if data.dtype.kind in "iu":
        flat = np.ravel(data)
        if flat.size == 0:
            return {}
        min_label = int(flat.min())
        if min_label >= 0:
            counts = np.bincount(flat.astype(np.int64, copy=False))
            present_labels = np.nonzero(counts)[0]
            return {int(label): int(counts[label]) for label in present_labels}

    # Fallback for non-integer masks.
    dense = img.get_fdata().astype(np.int32)
    unique_labels, counts = np.unique(dense, return_counts=True)
    return dict(zip(unique_labels.astype(int), counts.astype(int)))


def _extract_semantic_label_entry(
    vol: Dict[str, Any],
    config: "SemanticLabelConfig",
    target_spec: "SemanticTargetSpec",
) -> Tuple[str, "SemanticLabels"]:
    file_path = vol["file_path"]
    label_path = vol["label_path"]
    return file_path, extract_semantic_labels(label_path, config, target_spec=target_spec)


def _extract_labels_for_volumes(
    volumes_with_labels: List[Dict[str, Any]],
    config: "SemanticLabelConfig",
    target_spec: "SemanticTargetSpec",
    existing_results: Optional[Dict[str, "SemanticLabels"]] = None,
    cache_path: Optional[Path] = None,
    num_workers: int = 1,
    flush_every: int = SEMANTIC_CACHE_FLUSH_EVERY,
) -> Dict[str, "SemanticLabels"]:
    results = dict(existing_results or {})
    pending = [vol for vol in volumes_with_labels if vol["file_path"] not in results]

    if not pending:
        return results

    logger.info(
        "Extracting semantic labels for %d pending volumes (%d cached)",
        len(pending),
        len(results),
    )

    if num_workers <= 1 or len(pending) == 1:
        for index, vol in enumerate(tqdm(pending, desc="Extracting semantic labels"), start=1):
            file_path, labels = _extract_semantic_label_entry(vol, config, target_spec)
            results[file_path] = labels
            if cache_path is not None and index % flush_every == 0:
                _write_cached_labels(cache_path, results)
        return results

    max_workers = min(num_workers, len(pending))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_extract_semantic_label_entry, vol, config, target_spec) for vol in pending]
        for index, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Extracting semantic labels"), start=1):
            file_path, labels = future.result()
            results[file_path] = labels
            if cache_path is not None and index % flush_every == 0:
                _write_cached_labels(cache_path, results)

    return results


def _load_cached_labels(cache_path: Path) -> Dict[str, "SemanticLabels"]:
    logger.info(f"Loading cached semantic labels from {cache_path}")
    with open(cache_path) as f:
        cached = json.load(f)

    results = {}
    for fp, data in cached.items():
        results[fp] = SemanticLabels(
            organ_presence=np.array(data["organ_presence"]) if data.get("organ_presence") else None,
            dominant_organ=data.get("dominant_organ"),
            organ_volumes=data.get("organ_volumes"),
            label_path=data.get("label_path"),
            extraction_mode=data.get("extraction_mode", "multi_binary"),
            semantic_target=np.array(data["semantic_target"]) if data.get("semantic_target") is not None else None,
            foreground_voxel_count=data.get("foreground_voxel_count"),
        )
    return results


def _write_cached_labels(cache_path: Path, results: Dict[str, "SemanticLabels"]) -> None:
    cache_data = {}
    for fp, labels in results.items():
        cache_data[fp] = {
            "organ_presence": labels.organ_presence.tolist() if labels.organ_presence is not None else None,
            "dominant_organ": labels.dominant_organ,
            "organ_volumes": labels.organ_volumes,
            "label_path": labels.label_path,
            "extraction_mode": labels.extraction_mode,
            "semantic_target": labels.semantic_target.tolist() if labels.semantic_target is not None else None,
            "foreground_voxel_count": labels.foreground_voxel_count,
        }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + f".{os.getpid()}.tmp")
    with open(tmp_path, 'w') as f:
        json.dump(cache_data, f)
    os.replace(tmp_path, cache_path)
    logger.info(f"Cached semantic labels to {cache_path}")


@dataclass
class SemanticLabels:
    """Container for extracted semantic labels."""
    # Multi-label binary: shape (n_organs,) where 1 = present, 0 = absent
    organ_presence: Optional[np.ndarray] = None

    # Dominant organ: integer label ID
    dominant_organ: Optional[int] = None

    # Organ volumes for analysis
    organ_volumes: Optional[Dict[int, int]] = None

    # Metadata
    label_path: Optional[str] = None
    extraction_mode: str = "multi_binary"
    semantic_target: Optional[np.ndarray] = None
    foreground_voxel_count: Optional[int] = None


def extract_organ_presence(
    label_path: Union[str, Path],
    organ_mapping: Dict[int, str],
    min_voxels: int = 100,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Extract multi-label binary organ presence from a segmentation mask.

    Args:
        label_path: Path to segmentation NIfTI file
        organ_mapping: Dict mapping label_id -> organ_name
        min_voxels: Minimum voxels for organ to be considered present

    Returns:
        Tuple of:
        - organ_presence: Binary array of shape (n_organs,)
        - organ_volumes: Dict mapping label_id -> voxel count
    """
    try:
        label_counts = _compute_label_counts(label_path)

        # Build organ presence vector
        n_organs = len(organ_mapping)
        organ_presence = np.zeros(n_organs, dtype=np.int32)
        organ_volumes = {}

        for idx, (label_id, organ_name) in enumerate(sorted(organ_mapping.items())):
            voxel_count = label_counts.get(label_id, 0)
            organ_volumes[label_id] = int(voxel_count)
            if voxel_count >= min_voxels:
                organ_presence[idx] = 1

        return organ_presence, organ_volumes

    except Exception as e:
        raise ValueError(f"Cannot read organ annotation {label_path}: {e}") from e


def extract_dominant_organ(
    label_path: Union[str, Path],
    organ_mapping: Dict[int, str],
    exclude_background: bool = True,
) -> Tuple[int, Dict[int, int]]:
    """
    Extract the dominant (largest) organ from a segmentation mask.

    Args:
        label_path: Path to segmentation NIfTI file
        organ_mapping: Dict mapping label_id -> organ_name
        exclude_background: Whether to exclude label 0 (background)

    Returns:
        Tuple of:
        - dominant_label: Label ID of the dominant organ
        - organ_volumes: Dict mapping label_id -> voxel count
    """
    try:
        label_counts = _compute_label_counts(label_path)

        # Filter to known organs
        organ_volumes = {}
        for label_id in organ_mapping.keys():
            organ_volumes[label_id] = int(label_counts.get(label_id, 0))

        # Find dominant (excluding background if specified)
        valid_labels = {k: v for k, v in organ_volumes.items() if v > 0}
        if exclude_background:
            valid_labels = {k: v for k, v in valid_labels.items() if k != 0}

        if not valid_labels:
            return -1, organ_volumes  # No valid organs found

        dominant_label = max(valid_labels, key=valid_labels.get)
        return dominant_label, organ_volumes

    except Exception as e:
        raise ValueError(f"Cannot read dominant-organ annotation {label_path}: {e}") from e


def extract_directory_structure_presence(
    label_dir: Union[str, Path],
    segmentation_names: List[str],
    min_voxels: int = 100,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Extract a multi-label presence vector from a directory of binary masks.

    This is used for TotalSegmentator-style datasets where each anatomical
    structure is stored as its own mask file instead of as an integer-valued
    combined-label volume.
    """
    label_dir = Path(label_dir)
    presence = np.zeros(len(segmentation_names), dtype=np.int32)
    voxel_counts: Dict[str, int] = {}

    for index, name in enumerate(segmentation_names):
        mask_path = label_dir / f"{name}.nii.gz"
        if not mask_path.exists():
            voxel_counts[name] = 0
            continue

        try:
            img = nib.load(str(mask_path))
            data = img.get_fdata()
            if not np.isfinite(data).all():
                raise ValueError(f"Annotation contains nonfinite values: {mask_path}")
            voxel_count = int(np.count_nonzero(data > 0))
            voxel_counts[name] = voxel_count
            if voxel_count >= min_voxels:
                presence[index] = 1
        except Exception as e:
            raise ValueError(f"Cannot read supplied structure annotation {mask_path}: {e}") from e

    return presence, voxel_counts


def extract_foreground_voxel_count(label_path: Union[str, Path]) -> int:
    """Return the foreground voxel count for a binary or integer-valued mask."""
    try:
        img = nib.load(str(label_path))
        data = np.asarray(img.dataobj)
        if data.dtype.kind not in "iu" and not np.isfinite(data).all():
            raise ValueError(f"Annotation contains nonfinite values: {label_path}")
        return int(np.count_nonzero(data > 0))
    except Exception as e:
        raise ValueError(f"Cannot read foreground annotation {label_path}: {e}") from e


def extract_semantic_labels(
    label_path: Union[str, Path],
    config: Optional[SemanticLabelConfig] = None,
    target_spec: Optional[SemanticTargetSpec] = None,
) -> SemanticLabels:
    """
    Extract semantic labels from a segmentation mask.

    Args:
        label_path: Path to segmentation NIfTI file
        config: Label extraction configuration

    Returns:
        SemanticLabels container with extracted data
    """
    config = config or SEMANTIC_LABEL_CONFIG
    target_spec = target_spec or resolve_semantic_target_spec()
    organ_mapping = target_spec.integer_label_mapping or config.abdomenatlas_organs

    if target_spec.extraction_mode == "label_id_multilabel":
        organ_presence, organ_volumes = extract_organ_presence(
            label_path, organ_mapping, target_spec.min_voxels
        )
        return SemanticLabels(
            organ_presence=organ_presence,
            organ_volumes=organ_volumes,
            label_path=str(label_path),
            extraction_mode=target_spec.extraction_mode,
            semantic_target=organ_presence,
        )
    elif target_spec.extraction_mode == "dominant_organ":
        dominant_organ, organ_volumes = extract_dominant_organ(
            label_path, organ_mapping
        )
        return SemanticLabels(
            dominant_organ=dominant_organ,
            organ_volumes=organ_volumes,
            label_path=str(label_path),
            extraction_mode=target_spec.extraction_mode,
        )
    elif target_spec.extraction_mode == "segmentation_dir_presence":
        organ_presence, organ_volumes = extract_directory_structure_presence(
            label_path,
            target_spec.segmentation_names or [],
            min_voxels=target_spec.min_voxels,
        )
        return SemanticLabels(
            organ_presence=organ_presence,
            organ_volumes=organ_volumes,
            label_path=str(label_path),
            extraction_mode=target_spec.extraction_mode,
            semantic_target=organ_presence,
        )
    elif target_spec.extraction_mode == "foreground_volume_tertiles":
        foreground_voxel_count = extract_foreground_voxel_count(label_path)
        return SemanticLabels(
            label_path=str(label_path),
            extraction_mode=target_spec.extraction_mode,
            foreground_voxel_count=foreground_voxel_count,
        )
    else:
        raise ValueError(f"Unknown extraction mode: {target_spec.extraction_mode}")


def extract_labels_for_manifest(
    manifest_path: Union[str, Path],
    config: Optional[SemanticLabelConfig] = None,
    cache_path: Optional[Union[str, Path]] = None,
    filter_dataset: Optional[str] = None,
    target_spec: Optional[SemanticTargetSpec] = None,
    num_workers: int = 1,
    cache_flush_every: int = SEMANTIC_CACHE_FLUSH_EVERY,
) -> Dict[str, SemanticLabels]:
    """
    Extract semantic labels for all volumes in a manifest.

    Args:
        manifest_path: Path to manifest JSON
        config: Label extraction configuration
        cache_path: Optional path to cache extracted labels
        filter_dataset: Only process volumes from this dataset

    Returns:
        Dict mapping file_path -> SemanticLabels
    """
    config = config or SEMANTIC_LABEL_CONFIG

    cache_path = Path(cache_path) if cache_path else None

    with open(manifest_path) as f:
        manifest = json.load(f)

    volumes = manifest["volumes"]
    if filter_dataset:
        volumes = [v for v in volumes if v.get("dataset") == filter_dataset]

    volumes_with_labels = [v for v in volumes if v.get("has_label") and v.get("label_path")]
    expected_count = len(volumes_with_labels)
    target_spec = target_spec or resolve_semantic_target_spec(manifest.get("dataset"), manifest.get("modality"))
    expected_labels = {v["file_path"]: str(v["label_path"]) for v in volumes_with_labels}

    def compatible_cached_results():
        cached = _load_cached_labels(cache_path)
        matched = {key: value for key, value in cached.items()
                   if key in expected_labels and value.label_path == expected_labels[key]
                   and value.extraction_mode == target_spec.extraction_mode}
        for key, value in matched.items():
            mode = target_spec.extraction_mode
            if mode in {"label_id_multilabel", "segmentation_dir_presence"}:
                expected_width = (len(target_spec.segmentation_names or [])
                                  if mode == "segmentation_dir_presence"
                                  else len(target_spec.integer_label_mapping or config.abdomenatlas_organs))
                for vector in (value.organ_presence, value.semantic_target):
                    if vector is None or vector.shape != (expected_width,) or not np.isin(vector, [0, 1]).all():
                        raise ValueError(f"Invalid semantic cache target for {key}: {cache_path}")
                if not np.array_equal(value.organ_presence, value.semantic_target):
                    raise ValueError(f"Inconsistent semantic cache target for {key}: {cache_path}")
            elif mode == "foreground_volume_tertiles":
                if type(value.foreground_voxel_count) is not int or value.foreground_voxel_count < 0:
                    raise ValueError(f"Invalid semantic cache voxel count for {key}: {cache_path}")
            elif mode == "dominant_organ":
                if type(value.dominant_organ) is not int:
                    raise ValueError(f"Invalid semantic cache dominant organ for {key}: {cache_path}")
        return matched

    if cache_path and cache_path.exists():
        cached_results = compatible_cached_results()
        if len(cached_results) >= expected_count:
            logger.info("Semantic label cache hit: %d/%d volumes", len(cached_results), expected_count)
            return cached_results
        logger.info(
            "Resuming partial semantic label cache: %d/%d volumes",
            len(cached_results),
            expected_count,
        )

    if cache_path:
        lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Acquiring semantic label cache lock: {lock_path}")

        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

            results = compatible_cached_results() if cache_path.exists() else {}
            results = _extract_labels_for_volumes(
                volumes_with_labels,
                config,
                target_spec,
                existing_results=results,
                cache_path=cache_path,
                num_workers=num_workers,
                flush_every=cache_flush_every,
            )

            _write_cached_labels(cache_path, results)
            return results

    return _extract_labels_for_volumes(
        volumes_with_labels,
        config,
        target_spec,
        num_workers=num_workers,
        flush_every=cache_flush_every,
    )


def get_organ_presence_matrix(
    labels_dict: Dict[str, SemanticLabels],
    file_paths: List[str],
) -> np.ndarray:
    """
    Build organ presence matrix for a list of file paths.

    Args:
        labels_dict: Dict mapping file_path -> SemanticLabels
        file_paths: List of file paths in desired order

    Returns:
        Matrix of shape (n_samples, n_organs) with binary presence indicators
    """
    n_samples = len(file_paths)
    sample_labels = labels_dict.get(file_paths[0])
    if sample_labels is None or sample_labels.organ_presence is None:
        raise ValueError("No organ presence data available")

    n_organs = len(sample_labels.organ_presence)
    matrix = np.zeros((n_samples, n_organs), dtype=np.int32)

    for i, fp in enumerate(file_paths):
        if fp in labels_dict and labels_dict[fp].organ_presence is not None:
            matrix[i] = labels_dict[fp].organ_presence

    return matrix


def get_dominant_organ_labels(
    labels_dict: Dict[str, SemanticLabels],
    file_paths: List[str],
) -> np.ndarray:
    """
    Get dominant organ labels for a list of file paths.

    Args:
        labels_dict: Dict mapping file_path -> SemanticLabels
        file_paths: List of file paths in desired order

    Returns:
        Array of shape (n_samples,) with dominant organ label IDs
    """
    labels = np.zeros(len(file_paths), dtype=np.int32)

    for i, fp in enumerate(file_paths):
        if fp in labels_dict and labels_dict[fp].dominant_organ is not None:
            labels[i] = labels_dict[fp].dominant_organ
        else:
            labels[i] = -1  # Unknown

    return labels


# =============================================================================
# Utility functions for semantic readout
# =============================================================================

def compute_semantic_task_labels(
    manifest_path: Union[str, Path],
    mode: str = "auto",
    filter_dataset: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    num_workers: int = 1,
    cache_flush_every: int = SEMANTIC_CACHE_FLUSH_EVERY,
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Compute semantic task labels for semantic readout.

    Args:
        manifest_path: Path to manifest JSON
        mode: "multi_binary" or "dominant_organ"
        filter_dataset: Only process volumes from this dataset
        cache_dir: Directory for caching extracted labels

    Returns:
        Tuple of:
        - labels: Task labels array (n_samples, n_classes) for multi_binary
                  or (n_samples,) for dominant_organ
        - file_paths: List of file paths in same order as labels
        - metadata: Additional information (organ names, class counts, etc.)
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    dataset_name = filter_dataset or manifest.get("dataset")
    modality = manifest.get("modality")

    target_spec = resolve_semantic_target_spec(dataset_name=dataset_name, modality=modality)
    config = SemanticLabelConfig(mode=mode)

    # Cache path
    cache_path = None
    if cache_dir:
        cache_name = f"semantic_labels_{target_spec.name}"
        if dataset_name:
            cache_name += f"_{dataset_name}"
        cache_name += ".json"
        cache_path = Path(cache_dir) / cache_name

    # Extract labels
    labels_dict = extract_labels_for_manifest(
        manifest_path,
        config,
        cache_path,
        filter_dataset,
        target_spec=target_spec,
        num_workers=num_workers,
        cache_flush_every=cache_flush_every,
    )

    volumes = manifest["volumes"]
    if filter_dataset:
        volumes = [v for v in volumes if v.get("dataset") == filter_dataset]

    # Filter to volumes with labels
    volumes_with_labels = [v for v in volumes if v["file_path"] in labels_dict]
    file_paths = [v["file_path"] for v in volumes_with_labels]

    # Build label arrays
    if target_spec.extraction_mode in {"label_id_multilabel", "segmentation_dir_presence"}:
        labels = np.stack([labels_dict[fp].semantic_target for fp in file_paths]) if file_paths else np.zeros((0, 0), dtype=np.int32)
        if target_spec.extraction_mode == "label_id_multilabel":
            organ_ids = list((target_spec.integer_label_mapping or {}).keys())
            organ_names = list((target_spec.integer_label_mapping or {}).values())
        else:
            organ_ids = list(range(len(target_spec.segmentation_names or [])))
            organ_names = list(target_spec.segmentation_names or [])
        prevalences = labels.mean(axis=0) if len(labels) > 0 else np.array([])
        informative_mask = (
            (prevalences >= target_spec.min_label_prevalence)
            & (prevalences <= target_spec.max_label_prevalence)
        )

        filtered_labels = labels[:, informative_mask]
        filtered_organ_ids = [organ_id for organ_id, keep in zip(organ_ids, informative_mask) if keep]
        filtered_organ_names = [name for name, keep in zip(organ_names, informative_mask) if keep]
        excluded_labels = [
            {
                "organ_id": organ_id,
                "organ_name": name,
                "prevalence": float(prevalence),
            }
            for organ_id, name, prevalence, keep in zip(organ_ids, organ_names, prevalences, informative_mask)
            if not keep
        ]

        if filtered_labels.shape[1] == 0:
            logger.warning(
                "No informative semantic labels remain after prevalence filtering "
                f"[{target_spec.min_label_prevalence:.2f}, {target_spec.max_label_prevalence:.2f}]"
            )

        metadata = {
            "mode": target_spec.extraction_mode,
            "target_spec": target_spec.name,
            "n_organs": filtered_labels.shape[1],
            "organ_names": filtered_organ_names,
            "organ_ids": filtered_organ_ids,
            "samples_per_organ": filtered_labels.sum(axis=0).tolist(),
            "prevalence_per_organ": filtered_labels.mean(axis=0).tolist() if filtered_labels.shape[1] > 0 else [],
            "all_organ_names": organ_names,
            "all_organ_ids": organ_ids,
            "all_prevalence_per_organ": prevalences.tolist(),
            "excluded_labels": excluded_labels,
            "prevalence_filter": {
                "min": target_spec.min_label_prevalence,
                "max": target_spec.max_label_prevalence,
            },
        }
        labels = filtered_labels
    elif target_spec.extraction_mode == "foreground_volume_tertiles":
        counts = np.array([labels_dict[fp].foreground_voxel_count or 0 for fp in file_paths], dtype=np.int64)
        if len(counts) == 0:
            labels = np.zeros((0, 0), dtype=np.int32)
            metadata = {
                "mode": target_spec.extraction_mode,
                "target_spec": target_spec.name,
                "n_organs": 0,
                "organ_names": [],
                "excluded_labels": [],
            }
            return labels, file_paths, metadata

        quantile_edges = np.quantile(counts, np.linspace(0.0, 1.0, target_spec.quantile_bins + 1))
        interior_edges = quantile_edges[1:-1]
        bin_ids = np.digitize(counts, interior_edges, right=False)
        labels = np.zeros((len(counts), target_spec.quantile_bins), dtype=np.int32)
        labels[np.arange(len(counts)), bin_ids] = 1

        label_names = target_spec.semantic_label_names or [
            f"foreground_volume_q{i + 1}" for i in range(target_spec.quantile_bins)
        ]
        prevalences = labels.mean(axis=0)
        informative_mask = (
            (prevalences >= target_spec.min_label_prevalence)
            & (prevalences <= target_spec.max_label_prevalence)
        )
        filtered_labels = labels[:, informative_mask]
        filtered_names = [name for name, keep in zip(label_names, informative_mask) if keep]
        excluded_labels = [
            {
                "label_name": name,
                "prevalence": float(prevalence),
            }
            for name, prevalence, keep in zip(label_names, prevalences, informative_mask)
            if not keep
        ]
        metadata = {
            "mode": target_spec.extraction_mode,
            "target_spec": target_spec.name,
            "n_organs": filtered_labels.shape[1],
            "organ_names": filtered_names,
            "organ_ids": list(range(filtered_labels.shape[1])),
            "samples_per_organ": filtered_labels.sum(axis=0).tolist() if filtered_labels.shape[1] > 0 else [],
            "prevalence_per_organ": filtered_labels.mean(axis=0).tolist() if filtered_labels.shape[1] > 0 else [],
            "all_organ_names": label_names,
            "all_organ_ids": list(range(len(label_names))),
            "all_prevalence_per_organ": prevalences.tolist(),
            "excluded_labels": excluded_labels,
            "prevalence_filter": {
                "min": target_spec.min_label_prevalence,
                "max": target_spec.max_label_prevalence,
            },
            "foreground_volume_quantiles": [float(edge) for edge in quantile_edges.tolist()],
            "foreground_voxel_counts": counts.tolist(),
        }
        labels = filtered_labels
    else:  # dominant_organ
        labels = get_dominant_organ_labels(labels_dict, file_paths)
        unique_labels = np.unique(labels[labels >= 0])
        metadata = {
            "mode": target_spec.extraction_mode,
            "target_spec": target_spec.name,
            "n_classes": len(unique_labels),
            "class_ids": unique_labels.tolist(),
            "class_names": [config.abdomenatlas_organs.get(int(l), "unknown") for l in unique_labels],
            "samples_per_class": {int(l): int((labels == l).sum()) for l in unique_labels},
        }

    return labels, file_paths, metadata
