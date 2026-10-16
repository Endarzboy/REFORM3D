"""Generate Acquisition Robustness manifests from raw volume directories.

The builder scans registered datasets, reads NIfTI spacing and shape metadata,
computes anisotropy ratios, assigns spacing bins, and writes full/sampled/meta
manifest files. Downstream evaluation stages use those manifests as their
dataset contract.

Inputs:
    dataset config: one ``RawDatasetConfig`` from ``RAW_DATASETS``.
    binning scheme: ``original`` or another configured spacing-bin policy.
    sampling config: per-bin limits and deterministic random seed.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import nibabel as nib

try:
    from .config import (
        RAW_DATASETS,
        RAW_DATA_ROOT,
        ACQUISITION_MANIFESTS,
        MANIFEST_FILE_NAMES,
        ANISOTROPY_BINS,
        SAMPLING_CONFIG,
        ANALYSIS_CONFIGS,
        AnalysisConfig,
        DEFAULT_BINNING_SCHEME,
        DEFAULT_ANALYSIS,
        get_anisotropy_bin,
        get_binning_scheme_from_manifest,
        get_bin_configs,
        get_bin_name_map,
        get_acquisition_manifest_dir,
        normalize_manifest_variant,
        compute_anisotropy_ratio,
        ensure_directories,
        RawDatasetConfig,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from config import (
        RAW_DATASETS,
        RAW_DATA_ROOT,
        ACQUISITION_MANIFESTS,
        MANIFEST_FILE_NAMES,
        ANISOTROPY_BINS,
        SAMPLING_CONFIG,
        ANALYSIS_CONFIGS,
        AnalysisConfig,
        DEFAULT_BINNING_SCHEME,
        DEFAULT_ANALYSIS,
        get_anisotropy_bin,
        get_binning_scheme_from_manifest,
        get_bin_configs,
        get_bin_name_map,
        get_acquisition_manifest_dir,
        normalize_manifest_variant,
        compute_anisotropy_ratio,
        ensure_directories,
        RawDatasetConfig,
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Reject spacings that are numerically positive but physically implausible.
# Near-zero header values can explode anisotropy ratios and contaminate
# observational binning without representing a real acquisition geometry.
MIN_VALID_SPACING_MM = 1e-3
MAX_VALID_SPACING_MM = 100.0


def _compute_ratio_summary(volumes: List["VolumeMetadata"]) -> Dict[str, float]:
    """Summarize anisotropy-ratio spread for manifest metadata."""
    ratios = np.array([v.anisotropy_ratio for v in volumes], dtype=float)
    if len(ratios) == 0:
        return {}
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


def sample_with_ratio_coverage(
    volumes: List["VolumeMetadata"],
    target: int,
    seed: int = 42,
    n_ratio_strata: int = 10,
) -> List["VolumeMetadata"]:
    """
    Sample a bin with deliberate coverage across its internal anisotropy-ratio range.

    This keeps the bin definition unchanged but avoids letting dense regions dominate
    the sample when more volumes are available than the per-bin budget.
    """
    if target >= len(volumes):
        return list(volumes)

    rng = np.random.default_rng(seed)
    ordered = sorted(volumes, key=lambda volume: (volume.anisotropy_ratio, volume.file_path))
    ratios = np.array([volume.anisotropy_ratio for volume in ordered], dtype=float)
    ratio_min = float(ratios.min())
    ratio_max = float(ratios.max())

    if ratio_max - ratio_min < 1e-12:
        chosen_idx = rng.choice(len(ordered), size=target, replace=False)
        return [ordered[int(idx)] for idx in chosen_idx]

    n_strata = max(1, min(n_ratio_strata, target, len(ordered)))
    edges = np.linspace(ratio_min, ratio_max, n_strata + 1)
    strata: List[List[VolumeMetadata]] = []
    for index in range(n_strata):
        left = edges[index]
        right = edges[index + 1]
        if index == n_strata - 1:
            stratum = [volume for volume in ordered if left <= volume.anisotropy_ratio <= right]
        else:
            stratum = [volume for volume in ordered if left <= volume.anisotropy_ratio < right]
        if stratum:
            strata.append(stratum)

    n_active_strata = len(strata)
    quotas = [target // n_active_strata] * n_active_strata
    for index in range(target % n_active_strata):
        quotas[index] += 1

    selected: List[VolumeMetadata] = []
    leftovers: List[VolumeMetadata] = []

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


@dataclass
class VolumeMetadata:
    """Metadata for a single volume."""
    file_path: str
    dataset: str
    modality: str
    shape: Tuple[int, int, int]
    spacing: Tuple[float, float, float]
    anisotropy_ratio: float
    anisotropy_bin: int
    has_label: bool
    anisotropy_bin_original: Optional[int] = None
    label_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_nifti_metadata(
    file_path: Path,
    dataset_name: str,
    modality: str,
    has_label: bool,
    label_path: Optional[Path] = None
) -> Optional[VolumeMetadata]:
    """
    Extract metadata from a single NIfTI file.

    Args:
        file_path: Path to the NIfTI file
        dataset_name: Name of the source dataset
        modality: Imaging modality
        has_label: Whether labels are available
        label_path: Path to label file (if available)

    Returns:
        VolumeMetadata object or None if extraction fails
    """
    try:
        # Load header only (don't load full data)
        img = nib.load(str(file_path))
        header = img.header

        # Get shape
        shape = tuple(int(d) for d in img.shape[:3])

        # Get spacing from affine column norms rather than diagonal entries.
        # Diagonal entries are wrong for oblique acquisitions because rotation
        # mixes voxel axes across multiple affine components.
        affine = img.affine
        spacing = tuple(float(np.linalg.norm(affine[:3, i])) for i in range(3))

        # Validate spacing
        if any(s < MIN_VALID_SPACING_MM or s > MAX_VALID_SPACING_MM for s in spacing):
            # Fallback to pixdim
            pixdim = header.get_zooms()[:3]
            if len(pixdim) >= 3 and all(MIN_VALID_SPACING_MM <= p <= MAX_VALID_SPACING_MM for p in pixdim):
                spacing = tuple(float(p) for p in pixdim)
            else:
                logger.warning(f"Invalid spacing for {file_path}: {spacing}")
                return None

        # Compute anisotropy ratio using global max/min formula
        # ratio = max(sx, sy, sz) / min(sx, sy, sz)
        # This is orientation-agnostic and robust to axis ordering
        anisotropy_ratio = compute_anisotropy_ratio(spacing)
        anisotropy_bin = get_anisotropy_bin(anisotropy_ratio)

        # Resolve label path
        resolved_label_path = None
        if has_label and label_path:
            resolved_label_path = str(label_path)

        return VolumeMetadata(
            file_path=str(file_path),
            dataset=dataset_name,
            modality=modality,
            shape=shape,
            spacing=spacing,
            anisotropy_ratio=round(anisotropy_ratio, 4),
            anisotropy_bin=anisotropy_bin,
            has_label=has_label,
            label_path=resolved_label_path
        )

    except Exception as e:
        logger.warning(f"Failed to process {file_path}: {e}")
        return None


def validate_volume_integrity(
    file_path: Path,
    min_variance: float = 1e-6,
    max_constant_ratio: float = 0.99
) -> Tuple[bool, str]:
    """
    Validate a NIfTI volume for data integrity issues that could cause
    training/inference failures (e.g., constant intensity causing divide-by-zero).

    Args:
        file_path: Path to the NIfTI file
        min_variance: Minimum intensity variance required
        max_constant_ratio: Maximum fraction of voxels that can be a single value

    Returns:
        (is_valid, reason) - True if valid, False with reason if invalid
    """
    try:
        # 1. Check file exists and is readable
        if not file_path.exists():
            return False, "File does not exist"

        # 2. Try to load with nibabel (validates header)
        try:
            img = nib.load(str(file_path))
        except Exception as e:
            return False, f"Failed to load NIfTI: {e}"

        # 3. Validate shape
        shape = img.shape[:3]
        if len(shape) < 3:
            return False, f"Invalid shape: {shape}"
        if any(d <= 0 for d in shape):
            return False, f"Invalid dimensions: {shape}"
        if any(d > 2048 for d in shape):  # Sanity check
            return False, f"Unreasonably large dimensions: {shape}"

        # 4. Load data and check for constant intensity
        # Sample a subset for efficiency (don't load full volume for large files)
        try:
            data = img.get_fdata()
        except Exception as e:
            return False, f"Failed to read data: {e}"

        # 5. Check for NaN/Inf values
        if np.any(np.isnan(data)):
            return False, "Contains NaN values"
        if np.any(np.isinf(data)):
            return False, "Contains Inf values"

        # 6. Check intensity variance (avoid divide-by-zero in normalization)
        variance = np.var(data)
        if variance < min_variance:
            return False, f"Near-constant intensity (variance={variance:.2e})"

        # 7. Check if dominated by a single value
        unique, counts = np.unique(data, return_counts=True)
        max_ratio = counts.max() / data.size
        if max_ratio > max_constant_ratio:
            return False, f"Single value dominates ({max_ratio:.1%} of voxels)"

        return True, "OK"

    except Exception as e:
        return False, f"Validation error: {e}"


def find_label_path(
    image_path: Path,
    label_root: Optional[Path],
    dataset_name: str
) -> Optional[Path]:
    """
    Find the corresponding label file for an image.

    Different datasets have different label naming conventions:
    - AbdomenAtlas: BDMAP_XXX/combined_labels.nii.gz (co-located)
    - autopet: lbl/{name}.nii.gz (remove _000X suffix)
    - HECKTOR25: {subject}/{subject}.nii.gz (remove __CT/__PT suffix)
    - ImageCAS: {id}.label.nii.gz (co-located with .img.nii.gz)
    - JHU Stroke preprocessed: masks/{name}_mask.nii.gz for images/{name}_dwi.nii.gz
    - KiTS23: case_xxxxx/segmentation.nii.gz
    - TotalSegmenter CT/MRI: per-case segmentations/ directory
    """
    image_name = image_path.name

    # =====================================================================
    # AbdomenAtlas: co-located combined_labels.nii.gz
    # =====================================================================
    if dataset_name == "abdomenatlas":
        label_path = image_path.parent / "combined_labels.nii.gz"
        return label_path if label_path.exists() else None

    # =====================================================================
    # ImageCAS: {id}.label.nii.gz co-located with {id}.img.nii.gz
    # =====================================================================
    if dataset_name == "imagecas":
        # image: 100.img.nii.gz -> label: 100.label.nii.gz
        label_name = image_name.replace(".img.nii.gz", ".label.nii.gz")
        label_path = image_path.parent / label_name
        return label_path if label_path.exists() else None

    if dataset_name == "kits23":
        if label_root is not None:
            label_name = image_name.replace("_ct.nii.gz", "_mask.nii.gz")
            label_path = label_root / label_name
            if label_path.exists():
                return label_path
        label_path = image_path.parent / "segmentation.nii.gz"
        return label_path if label_path.exists() else None

    # =====================================================================
    # TotalSegmentator CT / TotalSegmenterMRI: per-case segmentations/ directory
    # =====================================================================
    if dataset_name in {"totalsegmenter_ct", "totalsegmentermri"}:
        label_path = image_path.parent / "segmentations"
        return label_path if label_path.exists() else None

    # =====================================================================
    # JHU Stroke preprocessed: images/{name}_dwi.nii.gz -> masks/{name}_mask.nii.gz
    # =====================================================================
    if dataset_name == "jhu_stroke":
        if label_root is None or not image_name.endswith("_dwi.nii.gz"):
            return None
        label_name = image_name.replace("_dwi.nii.gz", "_mask.nii.gz")
        label_path = label_root / label_name
        return label_path if label_path.exists() else None

    # =====================================================================
    # HECKTOR25: {subject}__CT.nii.gz -> {subject}.nii.gz
    # =====================================================================
    if dataset_name == "hecktor25":
        # image: Task_1/{subject}/{subject}__CT.nii.gz
        # label: Task_1/{subject}/{subject}.nii.gz
        label_name = image_name.replace("__CT.nii.gz", ".nii.gz").replace("__PT.nii.gz", ".nii.gz")
        label_path = image_path.parent / label_name
        return label_path if label_path.exists() else None

    # =====================================================================
    # autopet: {name}_0000.nii.gz -> lbl/{name}.nii.gz
    # =====================================================================
    if dataset_name == "autopet":
        if label_root is None:
            return None
        # Remove _0000 or _0001 channel suffix
        import re
        label_name = re.sub(r'_000[0-9]\.nii\.gz$', '.nii.gz', image_name)
        label_path = label_root / label_name
        return label_path if label_path.exists() else None

    # =====================================================================
    # Fallback: Try common patterns
    # =====================================================================
    if label_root is None:
        return None

    patterns_to_try = [
        label_root / image_name,
        label_root / image_name.replace("_0000.nii.gz", ".nii.gz"),
        label_root / image_name.replace(".nii.gz", "_seg.nii.gz"),
    ]

    for pattern in patterns_to_try:
        if pattern.exists():
            return pattern

    return None


def scan_dataset(
    config: RawDatasetConfig,
    validate_integrity: bool = True,
    force_validate: bool = False
) -> List[VolumeMetadata]:
    """
    Scan a single dataset and extract metadata for all volumes.

    Args:
        config: Dataset configuration
        validate_integrity: If True, validate volume data integrity to filter
                           out corrupted files (e.g., constant intensity).
                           Automatically skipped for datasets marked as is_validated=True.
        force_validate: If True, run validation even for pre-validated datasets.
                       Use this when adding new files to a validated dataset.

    Returns:
        List of VolumeMetadata objects
    """
    logger.info(f"Scanning dataset: {config.name}")

    # Determine if validation should run
    # Skip validation for pre-validated datasets unless forced
    should_validate = validate_integrity and (force_validate or not config.is_validated)
    if config.is_validated and not force_validate:
        logger.info(f"  Dataset '{config.name}' is pre-validated, skipping integrity checks")
    elif should_validate:
        logger.info(f"  Will validate volume integrity for '{config.name}'")

    if not config.path.exists():
        logger.error(f"Dataset path does not exist: {config.path}")
        return []

    # Find all NIfTI files
    nifti_files = []

    # Handle different directory structures
    if config.name == "jhu_stroke":
        if config.path.name == "images":
            nifti_files = list(config.path.glob(config.file_pattern))
        else:
            # Alternate raw JHU Stroke layout: datasetXX/raw_data/sub-XXX/anat/*.nii.gz
            for subdir in config.path.iterdir():
                if subdir.is_dir():
                    nifti_files.extend(subdir.glob(f"**/{config.file_pattern}"))
    elif config.name == "imagecas":
        # ImageCAS has range subdirectories (1-200, 201-400, etc.)
        for subdir in config.path.iterdir():
            if subdir.is_dir():
                nifti_files.extend(subdir.glob(f"**/{config.file_pattern}"))
    elif config.name == "hecktor25":
        # HECKTOR25: Only Task_1 has labels (Task_2 has no masks)
        # Only include CT files (not PT or RTDOSE)
        task_path = config.path / "Task_1"
        if task_path.exists():
            for ct_file in task_path.glob("**/*__CT.nii.gz"):
                nifti_files.append(ct_file)
    elif config.name == "abdomenatlas":
        # AbdomenAtlas has structure: data/BDMAP_XXXXXXXX/ct.nii.gz
        # Each subject directory contains ct.nii.gz and combined_labels.nii.gz
        for subdir in config.path.iterdir():
            if subdir.is_dir() and subdir.name.startswith("BDMAP_"):
                ct_file = subdir / "ct.nii.gz"
                if ct_file.exists():
                    nifti_files.append(ct_file)
    else:
        # Standard flat directory
        nifti_files = list(config.path.glob(config.file_pattern))
        # Also check subdirectories
        nifti_files.extend(config.path.glob(f"**/{config.file_pattern}"))

    # Keep record order and floating-point summaries stable across processes.
    nifti_files = sorted(set(nifti_files))

    logger.info(f"Found {len(nifti_files)} NIfTI files in {config.name}")

    # Extract metadata for each file
    results = []
    skipped_validation = 0
    for file_path in tqdm(nifti_files, desc=f"Processing {config.name}"):
        # Skip known label/segmentation artifacts while preserving image files
        # from datasets that use nested image/label directory structures.
        file_name_lower = file_path.name.lower()
        if file_name_lower.endswith(("_label.nii.gz", ".label.nii.gz", "_seg.nii.gz", "segmentation.nii.gz")):
            if "imagesTr" not in str(file_path):
                continue

        # Validate volume integrity if needed (skipped for pre-validated datasets)
        if should_validate:
            is_valid, reason = validate_volume_integrity(file_path)
            if not is_valid:
                logger.warning(f"Skipping {file_path.name}: {reason}")
                skipped_validation += 1
                continue

        # Find label path
        label_path = None
        if config.has_labels:
            label_path = find_label_path(file_path, config.label_path, config.name)

        metadata = extract_nifti_metadata(
            file_path=file_path,
            dataset_name=config.name,
            modality=config.modality,
            has_label=config.has_labels and label_path is not None,
            label_path=label_path
        )

        if metadata:
            results.append(metadata)

    if skipped_validation > 0:
        logger.info(f"Skipped {skipped_validation} files due to validation failures in {config.name}")
    logger.info(f"Successfully processed {len(results)} volumes from {config.name}")
    return results


def build_full_manifest(
    datasets: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    validate_integrity: bool = True,
    force_validate: bool = False
) -> List[VolumeMetadata]:
    """
    Build manifest for all (or specified) raw datasets.

    Args:
        datasets: List of dataset names to include (None = all)
        save_path: Path to save manifest JSON
        validate_integrity: If True, validate volume data integrity.
                           Automatically skipped for datasets with is_validated=True.
        force_validate: If True, run validation even for pre-validated datasets.

    Returns:
        List of all VolumeMetadata objects
    """
    ensure_directories()

    # Select datasets (filter by is_enabled unless explicitly specified)
    if datasets is None:
        # Default: only include enabled datasets
        datasets_to_scan = {k: v for k, v in RAW_DATASETS.items() if v.is_enabled}
    else:
        # Explicit list: include regardless of is_enabled flag
        datasets_to_scan = {k: v for k, v in RAW_DATASETS.items() if k in datasets}

    # Scan all datasets
    all_metadata = []
    for name, config in datasets_to_scan.items():
        metadata = scan_dataset(
            config,
            validate_integrity=validate_integrity,
            force_validate=force_validate
        )
        all_metadata.extend(metadata)

    logger.info(f"Total volumes: {len(all_metadata)}")

    # Compute statistics
    print_manifest_statistics(all_metadata)

    # Save manifest
    if save_path is None:
        save_path = ACQUISITION_MANIFESTS / "acquisition_raw_data_manifest.json"

    save_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_data = {
        "version": "1.0",
        "description": "Acquisition robustness raw data manifest",
        "total_volumes": len(all_metadata),
        "volumes": [m.to_dict() for m in all_metadata]
    }

    with open(save_path, "w") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info(f"Manifest saved to: {save_path}")

    return all_metadata


def print_manifest_statistics(metadata: List[VolumeMetadata]):
    """Print detailed statistics about the manifest."""
    print("\n" + "=" * 60)
    print("MANIFEST STATISTICS")
    print("=" * 60)

    # Overall counts
    print(f"\nTotal volumes: {len(metadata)}")

    # By dataset
    print("\n--- By Dataset ---")
    dataset_counts = {}
    for m in metadata:
        dataset_counts[m.dataset] = dataset_counts.get(m.dataset, 0) + 1
    for ds, count in sorted(dataset_counts.items(), key=lambda x: -x[1]):
        print(f"  {ds}: {count}")

    # By modality
    print("\n--- By Modality ---")
    modality_counts = {}
    for m in metadata:
        modality_counts[m.modality] = modality_counts.get(m.modality, 0) + 1
    for mod, count in sorted(modality_counts.items(), key=lambda x: -x[1]):
        print(f"  {mod}: {count}")

    # By anisotropy bin
    print("\n--- By Anisotropy Bin ---")
    bin_counts = {b.bin_id: 0 for b in ANISOTROPY_BINS}
    for m in metadata:
        bin_counts[m.anisotropy_bin] = bin_counts.get(m.anisotropy_bin, 0) + 1

    for bin_config in ANISOTROPY_BINS:
        count = bin_counts[bin_config.bin_id]
        pct = 100 * count / len(metadata) if metadata else 0
        print(f"  Bin {bin_config.bin_id} ({bin_config.name}): {count} ({pct:.1f}%)")

    # Anisotropy ratio statistics
    print("\n--- Anisotropy Ratio Statistics ---")
    ratios = [m.anisotropy_ratio for m in metadata]
    if ratios:
        print(f"  Min: {min(ratios):.3f}")
        print(f"  Max: {max(ratios):.3f}")
        print(f"  Mean: {np.mean(ratios):.3f}")
        print(f"  Median: {np.median(ratios):.3f}")

    # By modality and bin (cross-tabulation)
    print("\n--- Modality × Anisotropy Bin ---")
    cross_tab = {}
    for m in metadata:
        key = (m.modality, m.anisotropy_bin)
        cross_tab[key] = cross_tab.get(key, 0) + 1

    # Print as table
    modalities = sorted(set(m.modality for m in metadata))
    bins = sorted(set(m.anisotropy_bin for m in metadata))

    header = "Modality\t" + "\t".join(f"Bin {b}" for b in bins) + "\tTotal"
    print(f"  {header}")
    for mod in modalities:
        row = [cross_tab.get((mod, b), 0) for b in bins]
        total = sum(row)
        row_str = "\t".join(str(c) for c in row)
        print(f"  {mod}\t\t{row_str}\t{total}")

    # Label availability
    print("\n--- Label Availability ---")
    labeled = sum(1 for m in metadata if m.has_label)
    print(f"  With labels: {labeled} ({100*labeled/len(metadata):.1f}%)")
    print(f"  Without labels: {len(metadata) - labeled}")


def build_single_dataset_manifest(
    dataset_name: str,
    output_suffix: str,
    binning_scheme: str = DEFAULT_BINNING_SCHEME,
    validate_integrity: bool = False,
    force_validate: bool = False,
    min_per_bin: int = 500,
    max_per_bin: int = 1000,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> List[VolumeMetadata]:
    """Build full and sampled manifests for one dataset.

    This single-dataset acquisition manifest builder scans one
    raw dataset, optionally remaps bins under the requested binning scheme,
    performs within-bin sampling with ratio-coverage stratification, and writes
    the `manifest_full.json`, `manifest_sampled.json`, and
    `manifest_meta.json` files to the acquisition manifest directory.
    """
    ensure_directories()
    np.random.seed(seed)

    manifest_variant = normalize_manifest_variant(binning_scheme)
    resolved_binning_scheme = (
        "coarse_ratio_thickness" if manifest_variant == "coarse_bins" else DEFAULT_BINNING_SCHEME
    )
    bin_configs = get_bin_configs(resolved_binning_scheme)

    output_dir = output_dir or get_acquisition_manifest_dir(output_suffix, manifest_variant)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset_name not in RAW_DATASETS:
        available = list(RAW_DATASETS.keys())
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")

    dataset_config = RAW_DATASETS[dataset_name]

    logger.info(f"\n{'='*70}")
    logger.info(f"BUILDING SINGLE-DATASET MANIFEST: {dataset_name}")
    logger.info(f"Modality: {dataset_config.modality}")
    logger.info(f"Binning scheme: {resolved_binning_scheme} -> {manifest_variant}")
    logger.info(f"Has labels: {dataset_config.has_labels}")
    logger.info(f"Is pre-validated: {dataset_config.is_validated}")
    logger.info(f"{'='*70}")

    logger.info("\n--- Step 1: Scanning dataset ---")
    all_volumes = scan_dataset(
        dataset_config,
        validate_integrity=validate_integrity,
        force_validate=force_validate,
    )

    if not all_volumes:
        logger.error(f"No volumes found in {dataset_name}")
        return []

    for volume in all_volumes:
        original_bin = volume.anisotropy_bin
        recomputed_bin = get_anisotropy_bin(
            volume.anisotropy_ratio,
            spacing=tuple(volume.spacing),
            scheme=resolved_binning_scheme,
        )
        if resolved_binning_scheme != DEFAULT_BINNING_SCHEME:
            volume.anisotropy_bin_original = original_bin
        volume.anisotropy_bin = recomputed_bin

    logger.info(f"Found {len(all_volumes)} volumes total")

    logger.info("\n--- Step 2: Pre-sampling Statistics ---")
    print_manifest_statistics(all_volumes)

    logger.info(f"\n--- Step 3: Sampling (min={min_per_bin}, max={max_per_bin}, seed={seed}) ---")
    sampled_volumes: List[VolumeMetadata] = []

    for bin_config in bin_configs:
        bin_id = bin_config.bin_id
        bin_name = bin_config.name
        bin_volumes = [v for v in all_volumes if v.anisotropy_bin == bin_id]
        n_bin = len(bin_volumes)

        logger.info(f"\nBin {bin_id} ({bin_name}): {n_bin} volumes available")

        if n_bin == 0:
            logger.warning(f"  No volumes in bin {bin_id} - skipping")
            continue

        if n_bin < min_per_bin:
            logger.warning(f"  Only {n_bin} < {min_per_bin} min - using all available")
            sampled_volumes.extend(bin_volumes)
        elif n_bin <= max_per_bin:
            logger.info(f"  {n_bin} <= {max_per_bin} max - using all available")
            sampled_volumes.extend(bin_volumes)
        else:
            selected = sample_with_ratio_coverage(bin_volumes, max_per_bin, seed=seed + bin_id)
            sampled_volumes.extend(selected)
            logger.info(f"  Sampled {max_per_bin} from {n_bin} with ratio-coverage stratification")
            logger.info(f"  Full-bin ratio spread: {_compute_ratio_summary(bin_volumes)}")
            logger.info(f"  Sampled ratio spread: {_compute_ratio_summary(selected)}")

    logger.info(f"\nTotal sampled: {len(sampled_volumes)}")

    logger.info("\n--- Step 4: Post-sampling Statistics ---")
    print_manifest_statistics(sampled_volumes)

    logger.info("\n--- Step 5: Saving Manifests ---")
    full_manifest_path = output_dir / MANIFEST_FILE_NAMES["full"]
    full_manifest = {
        "version": "1.1",
        "description": f"Acquisition robustness full manifest: {dataset_name} ({dataset_config.modality}, {manifest_variant})",
        "binning_scheme": resolved_binning_scheme,
        "dataset": dataset_name,
        "modality": dataset_config.modality,
        "sampling_config": None,
        "total_volumes": len(all_volumes),
        "volumes": [v.to_dict() for v in all_volumes],
    }
    with open(full_manifest_path, "w") as f:
        json.dump(full_manifest, f, indent=2)
    logger.info(f"Full manifest saved: {full_manifest_path}")

    sampled_manifest_path = output_dir / MANIFEST_FILE_NAMES["sampled"]
    sampled_manifest = {
        "version": "1.1",
        "description": f"Acquisition robustness sampled manifest: {dataset_name} ({dataset_config.modality}, {manifest_variant})",
        "binning_scheme": resolved_binning_scheme,
        "dataset": dataset_name,
        "modality": dataset_config.modality,
        "sampling_config": {
            "min_per_bin": min_per_bin,
            "max_per_bin": max_per_bin,
            "seed": seed,
            "strategy": "ratio_coverage_within_bin",
            "ratio_strata_per_bin": 10,
        },
        "total_volumes": len(sampled_volumes),
        "volumes": [v.to_dict() for v in sampled_volumes],
        "bin_ratio_statistics": {
            str(bin_config.bin_id): _compute_ratio_summary(
                [v for v in sampled_volumes if v.anisotropy_bin == bin_config.bin_id]
            )
            for bin_config in bin_configs
        },
    }
    with open(sampled_manifest_path, "w") as f:
        json.dump(sampled_manifest, f, indent=2)
    logger.info(f"Sampled manifest saved: {sampled_manifest_path}")

    manifest_meta_path = output_dir / MANIFEST_FILE_NAMES["meta"]
    manifest_meta = {
        "dataset": dataset_name,
        "module": "acquisition_robustness",
        "manifest_variant": manifest_variant,
        "binning_scheme": resolved_binning_scheme,
        "modality": dataset_config.modality,
        "full_manifest": MANIFEST_FILE_NAMES["full"],
        "sampled_manifest": MANIFEST_FILE_NAMES["sampled"],
        "total_volumes_full": len(all_volumes),
        "total_volumes_sampled": len(sampled_volumes),
        "sampling": {
            "min_per_bin": min_per_bin,
            "max_per_bin": max_per_bin,
            "seed": seed,
            "strategy": "ratio_coverage_within_bin",
        },
    }
    with open(manifest_meta_path, "w") as f:
        json.dump(manifest_meta, f, indent=2)
    logger.info(f"Manifest metadata saved: {manifest_meta_path}")

    logger.info(f"\n{'='*70}")
    logger.info("MANIFEST GENERATION COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Modality: {dataset_config.modality}")
    logger.info(f"Manifest variant: {manifest_variant}")
    logger.info(f"Full manifest: {len(all_volumes)} volumes → {full_manifest_path}")
    logger.info(f"Sampled manifest: {len(sampled_volumes)} volumes → {sampled_manifest_path}")
    logger.info("\nSampled bin distribution:")
    for bin_config in bin_configs:
        count = sum(1 for v in sampled_volumes if v.anisotropy_bin == bin_config.bin_id)
        logger.info(f"  Bin {bin_config.bin_id} ({bin_config.name}): {count}")

    return sampled_volumes


def sample_for_representation_geometry(
    manifest_path: Path,
    output_path: Optional[Path] = None,
    min_per_bin: int = 500,
    max_per_bin: int = 1000,
    seed: int = 42
) -> List[VolumeMetadata]:
    """
    Sample volumes for representation geometry analysis.

    Stratified sampling by anisotropy bin and modality.

    Args:
        manifest_path: Path to full manifest
        output_path: Path to save sampled manifest
        min_per_bin: Minimum samples per bin
        max_per_bin: Maximum samples per bin
        seed: Random seed

    Returns:
        List of sampled VolumeMetadata objects
    """
    np.random.seed(seed)

    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)

    binning_scheme = get_binning_scheme_from_manifest(manifest)
    bin_configs = get_bin_configs(binning_scheme)

    volumes = [VolumeMetadata(**v) for v in manifest["volumes"]]

    # Group by bin and modality
    bin_modality_groups = {}
    for v in volumes:
        key = (v.anisotropy_bin, v.modality)
        if key not in bin_modality_groups:
            bin_modality_groups[key] = []
        bin_modality_groups[key].append(v)

    # Calculate sampling strategy
    print("\n" + "=" * 60)
    print("SAMPLING FOR REPRESENTATION GEOMETRY")
    print("=" * 60)

    sampled = []

    for bin_config in bin_configs:
        bin_id = bin_config.bin_id
        print(f"\n--- Bin {bin_id} ({bin_config.name}) ---")

        # Get all volumes in this bin
        bin_volumes = [v for v in volumes if v.anisotropy_bin == bin_id]

        if len(bin_volumes) < min_per_bin:
            print(f"  WARNING: Only {len(bin_volumes)} volumes available (< {min_per_bin})")
            # Include all available
            sampled.extend(bin_volumes)
            print(f"  Sampled: {len(bin_volumes)} (all available)")
        else:
            # Sample up to max_per_bin, stratified by modality and diversified by ratio coverage
            modalities_in_bin = {}
            for v in bin_volumes:
                if v.modality not in modalities_in_bin:
                    modalities_in_bin[v.modality] = []
                modalities_in_bin[v.modality].append(v)

            # Calculate samples per modality (proportional)
            target = min(max_per_bin, len(bin_volumes))

            for mod, mod_volumes in modalities_in_bin.items():
                # Proportional allocation
                mod_target = int(target * len(mod_volumes) / len(bin_volumes))
                mod_target = max(1, mod_target)  # At least 1

                if len(mod_volumes) <= mod_target:
                    sampled.extend(mod_volumes)
                    print(f"  {mod}: {len(mod_volumes)} (all)")
                else:
                    selected = sample_with_ratio_coverage(mod_volumes, mod_target, seed=seed + bin_id)
                    sampled.extend(selected)
                    print(f"  {mod}: {len(selected)} (sampled from {len(mod_volumes)})")

    print(f"\nTotal sampled: {len(sampled)}")

    # Save sampled manifest
    if output_path is None:
        output_path = ACQUISITION_MANIFESTS / "representation_geometry_sampled.json"

    sampled_data = {
        "version": "1.0",
        "description": "Acquisition robustness sampled manifest for representation geometry",
        "binning_scheme": binning_scheme,
        "sampling_config": {
            "min_per_bin": min_per_bin,
            "max_per_bin": max_per_bin,
            "seed": seed
        },
        "total_volumes": len(sampled),
        "volumes": [v.to_dict() for v in sampled]
    }

    with open(output_path, "w") as f:
        json.dump(sampled_data, f, indent=2)

    print(f"\nSampled manifest saved to: {output_path}")

    return sampled


def sample_for_analysis(
    manifest_path: Path,
    analysis_config: AnalysisConfig,
    output_path: Optional[Path] = None,
    seed: int = 42
) -> List[VolumeMetadata]:
    """
    Sample volumes for a specific analysis configuration.

    This generalized function handles modality and bin filtering
    to create scientifically valid analysis subsets.

    Args:
        manifest_path: Path to full manifest
        analysis_config: AnalysisConfig specifying modality/bin constraints
        output_path: Path to save sampled manifest
        seed: Random seed

    Returns:
        List of sampled VolumeMetadata objects
    """
    np.random.seed(seed)

    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)

    binning_scheme = get_binning_scheme_from_manifest(manifest)
    bin_configs = get_bin_configs(binning_scheme)
    bin_names = get_bin_name_map(binning_scheme)

    volumes = [VolumeMetadata(**v) for v in manifest["volumes"]]

    print("\n" + "=" * 70)
    print(f"SAMPLING FOR ANALYSIS: {analysis_config.name}")
    print(f"Description: {analysis_config.description}")
    print("=" * 70)

    # Filter by allowed modalities
    if analysis_config.allowed_modalities:
        filtered_volumes = [v for v in volumes if v.modality in analysis_config.allowed_modalities]
        print(f"\nModality filter: {analysis_config.allowed_modalities}")
        print(f"  Before: {len(volumes)} volumes")
        print(f"  After:  {len(filtered_volumes)} volumes")
        volumes = filtered_volumes

    # Filter by allowed bins
    if analysis_config.allowed_bins:
        filtered_volumes = [v for v in volumes if v.anisotropy_bin in analysis_config.allowed_bins]
        print(f"\nBin filter: {analysis_config.allowed_bins}")
        print(f"  Before: {len(volumes)} volumes")
        print(f"  After:  {len(filtered_volumes)} volumes")
        volumes = filtered_volumes

    if len(volumes) == 0:
        print("\nERROR: No volumes match the filter criteria!")
        return []

    # Group by bin and modality
    print("\n--- Distribution after filtering ---")
    bin_modality_counts = {}
    for v in volumes:
        key = (v.anisotropy_bin, v.modality)
        bin_modality_counts[key] = bin_modality_counts.get(key, 0) + 1

    for (bin_id, modality), count in sorted(bin_modality_counts.items()):
        bin_name = bin_names.get(bin_id, f"bin_{bin_id}")
        print(f"  Bin {bin_id} ({bin_name}) - {modality}: {count}")

    # Sample within each allowed bin
    sampled = []
    min_per_bin = analysis_config.min_per_bin
    max_per_bin = analysis_config.max_per_bin

    for bin_id in analysis_config.allowed_bins if analysis_config.allowed_bins else [0, 1, 2]:
        bin_config = next((b for b in bin_configs if b.bin_id == bin_id), None)
        bin_name = bin_config.name if bin_config else f"bin_{bin_id}"

        print(f"\n--- Sampling Bin {bin_id} ({bin_name}) ---")

        # Get all volumes in this bin
        bin_volumes = [v for v in volumes if v.anisotropy_bin == bin_id]

        if len(bin_volumes) == 0:
            print(f"  No volumes in this bin")
            continue

        if len(bin_volumes) < min_per_bin:
            print(f"  WARNING: Only {len(bin_volumes)} volumes (< {min_per_bin} min)")
            print(f"  → Including all available")
            sampled.extend(bin_volumes)
        else:
            # Sample up to max_per_bin, stratified by modality and diversified by ratio coverage
            modalities_in_bin = {}
            for v in bin_volumes:
                if v.modality not in modalities_in_bin:
                    modalities_in_bin[v.modality] = []
                modalities_in_bin[v.modality].append(v)

            target = min(max_per_bin, len(bin_volumes))
            bin_sampled = []

            for mod, mod_volumes in modalities_in_bin.items():
                # Proportional allocation
                mod_target = int(target * len(mod_volumes) / len(bin_volumes))
                mod_target = max(1, mod_target)  # At least 1

                if len(mod_volumes) <= mod_target:
                    bin_sampled.extend(mod_volumes)
                    print(f"  {mod}: {len(mod_volumes)} (all)")
                else:
                    selected = sample_with_ratio_coverage(mod_volumes, mod_target, seed=seed + bin_id)
                    bin_sampled.extend(selected)
                    print(f"  {mod}: {len(selected)} (sampled from {len(mod_volumes)})")

            sampled.extend(bin_sampled)

    print(f"\n{'='*70}")
    print(f"TOTAL SAMPLED: {len(sampled)}")

    # Summary by bin
    print("\n--- Final Sample Distribution ---")
    for bin_id in sorted(set(v.anisotropy_bin for v in sampled)):
        bin_name = bin_names.get(bin_id, f"bin_{bin_id}")
        count = sum(1 for v in sampled if v.anisotropy_bin == bin_id)
        print(f"  Bin {bin_id} ({bin_name}): {count}")

    # Summary by modality
    print("\n--- Final Sample by Modality ---")
    for mod in sorted(set(v.modality for v in sampled)):
        count = sum(1 for v in sampled if v.modality == mod)
        print(f"  {mod}: {count}")

    # Save sampled manifest
    if output_path is None:
        output_path = ACQUISITION_MANIFESTS / f"acquisition{analysis_config.manifest_suffix}_sampled.json"

    sampled_data = {
        "version": "1.0",
        "description": f"Acquisition Robustness manifest: {analysis_config.description}",
        "binning_scheme": binning_scheme,
        "analysis_config": {
            "name": analysis_config.name,
            "allowed_modalities": analysis_config.allowed_modalities,
            "allowed_bins": analysis_config.allowed_bins,
            "min_per_bin": min_per_bin,
            "max_per_bin": max_per_bin,
            "seed": seed
        },
        "total_volumes": len(sampled),
        "volumes": [v.to_dict() for v in sampled]
    }

    with open(output_path, "w") as f:
        json.dump(sampled_data, f, indent=2)

    print(f"\nManifest saved to: {output_path}")
    print("=" * 70)

    return sampled


def generate_all_analysis_manifests(
    manifest_path: Path,
    seed: int = 42
) -> Dict[str, List[VolumeMetadata]]:
    """
    Generate manifests for all defined analysis configurations.

    Named presets are optional analysis hooks. The primary workflow
    builds dataset-scoped manifests directly.

    Args:
        manifest_path: Path to full raw manifest
        seed: Random seed

    Returns:
        Dictionary mapping analysis name to sampled volumes
    """
    results = {}

    if not ANALYSIS_CONFIGS:
        print("No named analysis presets are configured; use dataset-scoped manifests.")
        return results

    for name, config in ANALYSIS_CONFIGS.items():
        print(f"\n{'#'*70}")
        print(f"# Processing: {name}")
        print(f"{'#'*70}")

        sampled = sample_for_analysis(
            manifest_path=manifest_path,
            analysis_config=config,
            seed=seed
        )
        results[name] = sampled

    # Print summary
    print("\n" + "=" * 70)
    print("ANALYSIS MANIFEST GENERATION COMPLETE")
    print("=" * 70)

    for name, volumes in results.items():
        config = ANALYSIS_CONFIGS[name]
        print(f"\n{name} ({config.description}):")
        print(f"  Total volumes: {len(volumes)}")
        if volumes:
            bins = sorted(set(v.anisotropy_bin for v in volumes))
            mods = sorted(set(v.modality for v in volumes))
            print(f"  Bins: {bins}")
            print(f"  Modalities: {mods}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build acquisition robustness manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python manifest_generation.py --datasets abdomenatlas kits23 --output acquisition_raw_data_manifest.json\n"
            "  python manifest_generation.py --analysis-only all --output acquisition_raw_data_manifest.json"
        ),
    )
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Specific datasets to include (default: all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for manifest")
    parser.add_argument("--sample", action="store_true",
                        help="Also generate a sampled representation-analysis manifest")
    parser.add_argument("--min-per-bin", type=int, default=500,
                        help="Minimum samples per bin for representation geometry")
    parser.add_argument("--max-per-bin", type=int, default=1000,
                        help="Maximum samples per bin for representation geometry")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip integrity validation entirely (not recommended)")
    parser.add_argument("--force-validate", action="store_true",
                        help="Force validation even for pre-validated datasets (use when adding new files)")

    # Optional named analysis presets. The primary path is dataset-scoped.
    parser.add_argument("--analysis", type=str, nargs="+",
                        choices=list(ANALYSIS_CONFIGS.keys()) + ["all"],
                        help="Generate named analysis manifests. No named presets are currently configured; 'all' selects every configured preset.")
    parser.add_argument("--analysis-only", type=str, nargs="+",
                        choices=list(ANALYSIS_CONFIGS.keys()) + ["all"],
                        help="Only generate named analysis manifests (skip full manifest build). Requires existing raw manifest.")

    args = parser.parse_args()

    # Handle --analysis-only flag (skip manifest building)
    if args.analysis_only:
        manifest_path = Path(args.output) if args.output else (ACQUISITION_MANIFESTS / "acquisition_raw_data_manifest.json")

        if not manifest_path.exists():
            print(f"ERROR: Raw manifest not found at {manifest_path}")
            print("Run without --analysis-only first to build the raw manifest.")
            exit(1)

        analyses = list(ANALYSIS_CONFIGS.keys()) if "all" in args.analysis_only else args.analysis_only
        if not analyses:
            print("No named analysis presets are configured; use dataset-scoped manifests.")

        for analysis_name in analyses:
            config = ANALYSIS_CONFIGS[analysis_name]
            sample_for_analysis(
                manifest_path=manifest_path,
                analysis_config=config,
                seed=42
            )
    else:
        # Build full manifest
        output_path = Path(args.output) if args.output else None
        metadata = build_full_manifest(
            datasets=args.datasets,
            save_path=output_path,
            validate_integrity=not args.no_validate,
            force_validate=args.force_validate
        )

        # Optionally sample for representation analysis.
        if args.sample:
            manifest_path = output_path or (ACQUISITION_MANIFESTS / "acquisition_raw_data_manifest.json")
            sample_for_representation_geometry(
                manifest_path=manifest_path,
                min_per_bin=args.min_per_bin,
                max_per_bin=args.max_per_bin
            )

        # Generate named analysis manifests if requested.
        if args.analysis:
            manifest_path = output_path or (ACQUISITION_MANIFESTS / "acquisition_raw_data_manifest.json")

            analyses = list(ANALYSIS_CONFIGS.keys()) if "all" in args.analysis else args.analysis
            if not analyses:
                print("No named analysis presets are configured; use dataset-scoped manifests.")

            for analysis_name in analyses:
                config = ANALYSIS_CONFIGS[analysis_name]
                sample_for_analysis(
                    manifest_path=manifest_path,
                    analysis_config=config,
                    seed=42
                )
