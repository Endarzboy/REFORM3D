"""Configuration for Acquisition Robustness evaluation.

This module defines dataset roots, checkpoint access, manifest/result paths,
spacing-bin policies, semantic targets, controlled perturbation settings, and
small helper functions used by the Acquisition Robustness pipeline.

Typical inputs are environment variables such as ``MED3DINO_EXTERNAL_DATA_ROOT``,
dataset keys from ``RAW_DATASETS``, checkpoint keys from the shared registry,
and manifest variants such as ``original_bins`` or ``coarse_bins``.
"""

from pathlib import Path
from reform3d.core.layout import active_output_paths
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple, Optional
import os
import hashlib
import json

from reform3d.core.checkpoints import CHECKPOINTS, CheckpointConfig
from reform3d.core.paths import (
    DOWNSTREAM_DATASET_ROOT,
    REFORM3D_DATASET_ROOT,
    PROJECT_ROOT,
    get_reform3d_cache_root,
    get_reform3d_module_manifest_roots,
    get_reform3d_module_manifest_root,
    get_reform3d_module_results_root,
    resolve_reform3d_module_manifest_path,
)
from reform3d.core.seed_context import apply_active_seed_to_path

# =============================================================================
# BASE PATHS
# =============================================================================

# Raw data location (locally downloaded, unprocessed)
RAW_DATA_ROOT = REFORM3D_DATASET_ROOT / "datasets"
OOD_DATA_ROOT = REFORM3D_DATASET_ROOT / "ood_datasets"
DOWNSTREAM_OTHERS_ROOT = DOWNSTREAM_DATASET_ROOT / "others"

# Downstream evaluation data (preprocessed).
DOWNSTREAM_DATASETS = DOWNSTREAM_DATASET_ROOT
DOWNSTREAM_SEGMENTATION_ROOT = DOWNSTREAM_DATASETS / "segmentation"
KITS23_DATA_ROOT = DOWNSTREAM_SEGMENTATION_ROOT / "KiTS23"
KITS23_RAW_ROOT = KITS23_DATA_ROOT / "images"

# Acquisition robustness outputs
ACQUISITION_ROOT = Path(__file__).resolve().parent
ACQUISITION_MODULE_NAME = "acquisition_robustness"
ACQUISITION_OUTPUTS = get_reform3d_module_results_root(ACQUISITION_MODULE_NAME)
ACQUISITION_MANIFESTS = get_reform3d_module_manifest_root(ACQUISITION_MODULE_NAME)
ACQUISITION_CACHES = get_reform3d_cache_root()
DEFAULT_ACQUISITION_DATASET = "abdomenatlas"
DEFAULT_BINNING_SCHEME = "original"
DEFAULT_MANIFEST_VARIANT = "original_bins"

MANIFEST_VARIANT_ALIASES = {
    "original": "original_bins",
    "original_bin": "original_bins",
    "original_bins": "original_bins",
    "coarse": "coarse_bins",
    "coarse_bin": "coarse_bins",
    "coarse_bins": "coarse_bins",
    "coarse_ratio_thickness": "coarse_bins",
    "coarse_ratio_thickness_bins": "coarse_bins",
}

MANIFEST_FILE_NAMES = {
    "full": "manifest_full.json",
    "sampled": "manifest_sampled.json",
    "meta": "manifest_meta.json",
}


def normalize_manifest_variant(manifest_variant: str = DEFAULT_MANIFEST_VARIANT) -> str:
    """Normalize manifest variant names to configured directory names."""
    return MANIFEST_VARIANT_ALIASES.get(manifest_variant, manifest_variant)


def get_acquisition_manifest_dir(
    dataset_name: str = DEFAULT_ACQUISITION_DATASET,
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> Path:
    """Return the dataset and manifest-variant directory for acquisition manifests."""
    return apply_active_seed_to_path(ACQUISITION_MANIFESTS / dataset_name / normalize_manifest_variant(manifest_variant))


def get_acquisition_manifest_path(
    dataset_name: str = DEFAULT_ACQUISITION_DATASET,
    manifest_kind: str = "sampled",
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> Path:
    """Return the acquisition manifest path for the requested dataset and variant."""
    if manifest_kind not in MANIFEST_FILE_NAMES:
        raise ValueError(f"Unknown manifest kind: {manifest_kind}")
    return get_acquisition_manifest_dir(dataset_name, manifest_variant) / MANIFEST_FILE_NAMES[manifest_kind]


def resolve_acquisition_manifest_path(manifest_path: Path | str) -> Path:
    return resolve_reform3d_module_manifest_path(ACQUISITION_MODULE_NAME, manifest_path)


def get_cache_root(
    dataset_name: str = DEFAULT_ACQUISITION_DATASET,
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> Path:
    """Return the cache root for perturbation caches."""
    return apply_active_seed_to_path(
        ACQUISITION_CACHES / ACQUISITION_MODULE_NAME / dataset_name / normalize_manifest_variant(manifest_variant)
    )

# =============================================================================
# RAW DATASET REGISTRY
# =============================================================================
# These datasets are used for representation geometry and controlled perturbation analyses.
# Status: VERIFIED paths from local exploration.

@dataclass
class RawDatasetConfig:
    """Configuration for a raw (unprocessed) dataset."""
    name: str
    modality: str
    path: Path
    has_labels: bool
    label_path: Optional[Path] = None
    file_pattern: str = "*.nii.gz"
    notes: str = ""
    is_validated: bool = False  # True = corrupted files already removed, skip validation
    is_enabled: bool = True  # False = skip this dataset in manifest building


RAW_DATASETS: Dict[str, RawDatasetConfig] = {
    # MRI Datasets
    "jhu_stroke": RawDatasetConfig(
        name="jhu_stroke",
        modality="mr",
        path=RAW_DATA_ROOT / "mri" / "JHUS_Stroke_preprocessed" / "images",
        has_labels=True,
        label_path=RAW_DATA_ROOT / "mri" / "JHUS_Stroke_preprocessed" / "masks",
        file_pattern="*_dwi.nii.gz",
        notes="Preprocessed JHUS DWI volumes with paired stroke masks, full anisotropy spectrum (0.92-13.47).",
        is_enabled=False,  # Disabled: needs validation, enable when ready
    ),

    # PET/CT Datasets
    "autopet": RawDatasetConfig(
        name="autopet",
        modality="pet_ct",
        path=RAW_DATA_ROOT / "ct_pet" / "autoPET_v1.1" / "imagesTr",
        has_labels=True,
        label_path=RAW_DATA_ROOT / "ct_pet" / "autoPET_v1.1" / "lbl",
        file_pattern="*.nii.gz",
        notes="~2,597 volumes, near-isotropic (0.49-1.47)",
        is_validated=True,  # Cleaned 2024-03: 1 corrupted file removed
        is_enabled=False,
    ),
    "hecktor25": RawDatasetConfig(
        name="hecktor25",
        modality="pet_ct",
        path=RAW_DATA_ROOT / "ct_pet" / "hecktor25",
        has_labels=True,
        file_pattern="*.nii.gz",
        notes="PET/CT for head-neck cancer",
        is_validated=True,  # Cleaned 2024-03: 2 corrupted files removed
        is_enabled=False,
    ),

    # CTA Dataset
    "imagecas": RawDatasetConfig(
        name="imagecas",
        modality="cta",
        path=RAW_DATA_ROOT / "cta" / "imagecas",
        has_labels=True,  # Coronary segmentation labels (co-located .label.nii.gz)
        label_path=None,  # Labels found via filename replacement (.img -> .label)
        file_pattern="*.img.nii.gz",  # Only match image files, not labels
        notes="~1,000 volumes, mild anisotropy (1.08-1.73)",
        is_validated=True  # Cleaned 2024-03: no corrupted files found
    ),

    # Abdominal CT
    "abdomenatlas": RawDatasetConfig(
        name="abdomenatlas",
        modality="ct",
        path=RAW_DATA_ROOT / "ct" / "abdomenatlas1" / "AbdomenAtlas1.1Mini" / "data",
        has_labels=True,
        label_path=None,  # Labels are co-located with images (combined_labels.nii.gz)
        file_pattern="ct.nii.gz",  # Each subject has ct.nii.gz
        notes="AbdomenAtlas 1.1 Mini: ~9262 volumes, comprehensive abdominal CT with multi-organ segmentation",
        is_validated=True  # Cleaned 2024-03: no corrupted files found
    ),
    "kits23": RawDatasetConfig(
        name="kits23",
        modality="ct",
        path=KITS23_RAW_ROOT,
        has_labels=True,
        label_path=KITS23_DATA_ROOT / "masks",
        file_pattern="*_ct.nii.gz",
        notes="KiTS23 kidney CT with image/mask folders and case-level kidney, tumor, and cyst labels.",
        is_validated=False,
    ),
    "totalsegmenter_ct": RawDatasetConfig(
        name="totalsegmenter_ct",
        modality="ct",
        path=DOWNSTREAM_SEGMENTATION_ROOT / "totalsegmenter_ct",
        has_labels=True,
        label_path=None,  # Per-case segmentations directory resolved from image path.
        file_pattern="ct.nii.gz",
        notes="TotalSegmentator CT with per-structure segmentation directory; controlled acquisition robustness candidate.",
        is_validated=False,
        is_enabled=False,
    ),
    "totalsegmentermri": RawDatasetConfig(
        name="totalsegmentermri",
        modality="mr",
        path=OOD_DATA_ROOT / "mri" / "TotalSegmenterMRI",
        has_labels=True,
        label_path=None,  # Per-case segmentations directory resolved from image path.
        file_pattern="mri.nii.gz",
        notes="TotalSegmenterMRI with per-structure segmentation directory; MR controlled acquisition robustness candidate.",
        is_validated=False,
        is_enabled=False,
    ),
}


# =============================================================================
# DOWNSTREAM DATASET REGISTRY
# =============================================================================
# These datasets have labels and can support spacing-stratified readout analysis.
# Status: VERIFIED paths from local exploration.

@dataclass
class DownstreamDatasetConfig:
    """Configuration for a preprocessed downstream dataset."""
    name: str
    task: str  # "segmentation" or "classification"
    modality: str
    num_classes: int
    datalist_path: Path
    data_root: Path
    metrics_class: str  # Class name from dinov2/eval/segmentation_3d/metrics.py
    spacing_stratifiable: bool = True  # Whether spacing stratification makes sense
    notes: str = ""


DOWNSTREAM_DATASETS_CONFIG: Dict[str, DownstreamDatasetConfig] = {
    "btcv": DownstreamDatasetConfig(
        name="btcv",
        task="segmentation",
        modality="ct",
        num_classes=14,
        datalist_path=DOWNSTREAM_SEGMENTATION_ROOT / "BTCV_100_datalist.json",
        data_root=DOWNSTREAM_SEGMENTATION_ROOT / "BTCV",
        metrics_class="BTCVMetrics",
        spacing_stratifiable=True,  # UNKNOWN: Need to verify spacing distribution
        notes="14-class abdominal CT segmentation"
    ),
    "brats2023": DownstreamDatasetConfig(
        name="brats2023",
        task="segmentation",
        modality="mr",
        num_classes=3,  # TC, WT, ET (multi-label)
        datalist_path=DOWNSTREAM_SEGMENTATION_ROOT / "BraTS_100_datalist_3dinov2.json",
        data_root=DOWNSTREAM_SEGMENTATION_ROOT / "BraTS2023",
        metrics_class="BraTSMetrics",
        spacing_stratifiable=True,  # UNKNOWN: Need to verify spacing distribution
        notes="Multi-label brain tumor segmentation (4-channel input)"
    ),
    "amos22": DownstreamDatasetConfig(
        name="amos22",
        task="segmentation",
        modality="ct_mr",  # Both CT and MR available
        num_classes=16,
        datalist_path=DOWNSTREAM_SEGMENTATION_ROOT / "AMOS22_100_datalist.json",
        data_root=DOWNSTREAM_OTHERS_ROOT / "amos22" / "amos22",
        metrics_class="AMOS22Metrics",
        spacing_stratifiable=True,  # CONFIRMED: Full anisotropy spectrum
        notes="16-class abdominal segmentation, CT and MR"
    ),
    "isles22": DownstreamDatasetConfig(
        name="isles22",
        task="segmentation",
        modality="mr",
        num_classes=2,  # Background + lesion (binary)
        datalist_path=DOWNSTREAM_SEGMENTATION_ROOT / "ISLES22_100_datalist.json",
        data_root=DOWNSTREAM_SEGMENTATION_ROOT / "ISLES22",
        metrics_class="ISLES22Metrics",
        spacing_stratifiable=True,  # UNKNOWN: Need to verify spacing distribution
        notes="Binary stroke lesion segmentation (DWI/ADC)"
    ),
    "la_seg": DownstreamDatasetConfig(
        name="la_seg",
        task="segmentation",
        modality="mr",
        num_classes=2,
        datalist_path=DOWNSTREAM_SEGMENTATION_ROOT / "LA-SEG_100_datalist.json",
        data_root=DOWNSTREAM_SEGMENTATION_ROOT / "Left_Atrium",
        metrics_class="LASEGMetrics",
        spacing_stratifiable=True,  # UNKNOWN: Need to verify
        notes="Binary left atrium segmentation (LGE cardiac MRI)"
    ),
    "tdsc_abus": DownstreamDatasetConfig(
        name="tdsc_abus",
        task="segmentation",
        modality="us",  # Ultrasound
        num_classes=2,
        datalist_path=DOWNSTREAM_SEGMENTATION_ROOT / "TDSC-ABUS_100_datalist.json",
        data_root=DOWNSTREAM_SEGMENTATION_ROOT / "TDSC_ABUS2023",
        metrics_class="LASEGMetrics",  # Binary segmentation, same as LA-SEG
        spacing_stratifiable=False,  # Ultrasound has different physics
        notes="Binary breast tumor segmentation (automated breast ultrasound)"
    ),
    "kits23": DownstreamDatasetConfig(
        name="kits23",
        task="segmentation",
        modality="ct",
        num_classes=4,
        datalist_path=DOWNSTREAM_SEGMENTATION_ROOT / "KiTS23_100_datalist_validated.json",
        data_root=KITS23_DATA_ROOT,
        metrics_class="KiTS23Metrics",
        spacing_stratifiable=True,
        notes="Kidney, renal tumor, and cyst CT segmentation"
    ),
    "lits": DownstreamDatasetConfig(
        name="lits",
        task="segmentation",
        modality="ct",
        num_classes=3,
        datalist_path=DOWNSTREAM_SEGMENTATION_ROOT / "LiTS_100_datalist.json",
        data_root=REFORM3D_DATASET_ROOT / "datasets" / "ct" / "LiTS" / "data",
        metrics_class="LiTSMetrics",
        spacing_stratifiable=True,
        notes="Liver and tumor CT segmentation"
    ),
    "word": DownstreamDatasetConfig(
        name="word",
        task="segmentation",
        modality="ct",
        num_classes=17,
        datalist_path=DOWNSTREAM_SEGMENTATION_ROOT / "WORD_100_datalist.json",
        data_root=DOWNSTREAM_SEGMENTATION_ROOT / "WORD",
        metrics_class="WORDMetrics",
        spacing_stratifiable=True,
        notes="Whole abdomen organ CT segmentation"
    ),
    "totalsegmenter_ct": DownstreamDatasetConfig(
        name="totalsegmenter_ct",
        task="segmentation",
        modality="ct",
        num_classes=105,
        datalist_path=DOWNSTREAM_SEGMENTATION_ROOT / "TotalSegmenterCT_100_datalist.json",
        data_root=DOWNSTREAM_SEGMENTATION_ROOT / "totalsegmenter_ct",
        metrics_class="TotalSegmenterCTMetrics",
        spacing_stratifiable=True,
        notes="Whole-body CT segmentation with combined 104-organ labels"
    ),
}


# =============================================================================
# CHECKPOINT REGISTRY
# =============================================================================
# Shared checkpoint registry imported through the REFORM-3D core.


# =============================================================================
# ANISOTROPY BIN CONFIGURATION
# =============================================================================

@dataclass
class AnisotropyBinConfig:
    """Configuration for anisotropy ratio stratification."""
    bin_id: int
    name: str
    ratio_min: float
    ratio_max: float
    description: str


COARSE_ANISOTROPY_BINS: List[AnisotropyBinConfig] = [
    AnisotropyBinConfig(
        bin_id=0,
        name="coarse_low_ratio_thin_slice",
        ratio_min=1.0,
        ratio_max=2.0,
        description="Low anisotropy and thin-slice: ratio < 2.0 and max spacing < 2.0 mm",
    ),
    AnisotropyBinConfig(
        bin_id=1,
        name="coarse_intermediate_ratio_thickness",
        ratio_min=2.0,
        ratio_max=4.0,
        description="Intermediate ratio/thickness: not bin 0, with ratio < 4.0 and max spacing < 4.0 mm",
    ),
    AnisotropyBinConfig(
        bin_id=2,
        name="coarse_high_ratio_or_thick_slice",
        ratio_min=4.0,
        ratio_max=float('inf'),
        description="High anisotropy or thick-slice: ratio >= 4.0 or max spacing >= 4.0 mm",
    ),
]


# Anisotropy ratio formula: max(sx, sy, sz) / min(sx, sy, sz)
# This is orientation-agnostic and captures anisotropy across all axes.
#
# 3-bin scheme for spacing robustness evaluation:
# - Near-Isotropic: 1.0 ≤ ratio < 1.5 (nearly cubic voxels)
# - Moderately Anisotropic: 1.5 ≤ ratio < 3.0 (common clinical CT/MRI)
# - Highly Anisotropic: ratio ≥ 3.0 (thick-slice acquisitions)
#
# These thresholds are designed to be:
# - Clinically meaningful (1.5x and 3x are natural boundaries)
# - Dataset-agnostic (work with any modality/anatomy)
# - Sensitive to common anisotropy patterns in medical imaging
ANISOTROPY_BINS: List[AnisotropyBinConfig] = [
    AnisotropyBinConfig(
        bin_id=0,
        name="near_isotropic",
        ratio_min=1.0,  # min possible ratio is 1.0
        ratio_max=1.5,
        description="Near-isotropic voxels (1.0 ≤ ratio < 1.5)"
    ),
    AnisotropyBinConfig(
        bin_id=1,
        name="moderately_anisotropic",
        ratio_min=1.5,
        ratio_max=3.0,
        description="Moderate anisotropy (1.5 ≤ ratio < 3.0)"
    ),
    AnisotropyBinConfig(
        bin_id=2,
        name="highly_anisotropic",
        ratio_min=3.0,
        ratio_max=float('inf'),
        description="Highly anisotropic (ratio ≥ 3.0)"
    ),
]


def compute_anisotropy_ratio(spacing: Tuple[float, float, float]) -> float:
    """
    Compute global anisotropy ratio from voxel spacing.

    Formula: max(sx, sy, sz) / min(sx, sy, sz)

    This is orientation-agnostic and robust to:
    - Different axis ordering conventions
    - Non-standard slice orientations
    - Volumes reoriented during preprocessing

    Args:
        spacing: Tuple of (sx, sy, sz) voxel spacings in mm

    Returns:
        Anisotropy ratio (≥ 1.0, where 1.0 = perfectly isotropic)
    """
    sx, sy, sz = spacing
    return max(sx, sy, sz) / min(sx, sy, sz)


def compute_slice_thickness_proxy(spacing: Tuple[float, float, float]) -> float:
    """
    Compute an orientation-agnostic proxy for slice thickness.

    We use the largest voxel spacing as a conservative surrogate for the thick axis.
    This keeps the rule dataset-agnostic without assuming a fixed slice dimension.
    """
    return max(float(s) for s in spacing)


def get_bin_configs(scheme: str = "original") -> List[AnisotropyBinConfig]:
    """Return bin definitions for the requested scheme."""
    if not scheme:
        scheme = DEFAULT_BINNING_SCHEME
    if scheme == "original":
        return ANISOTROPY_BINS
    if scheme in {"coarse", "coarse_ratio_thickness"}:
        return COARSE_ANISOTROPY_BINS
    raise ValueError(f"Unknown binning scheme: {scheme}")


def get_binning_scheme_from_manifest(manifest_data: Optional[Dict[str, Any]]) -> str:
    """Resolve binning scheme from manifest metadata."""
    if not manifest_data:
        return DEFAULT_BINNING_SCHEME
    return manifest_data.get("binning_scheme", DEFAULT_BINNING_SCHEME)


def get_bin_name_map(scheme: str = DEFAULT_BINNING_SCHEME) -> Dict[int, str]:
    """Return a stable bin-id to name mapping for the requested scheme."""
    return {bin_config.bin_id: bin_config.name for bin_config in get_bin_configs(scheme)}


def get_bin_description_map(scheme: str = DEFAULT_BINNING_SCHEME) -> Dict[int, str]:
    """Return a stable bin-id to description mapping for the requested scheme."""
    return {bin_config.bin_id: bin_config.description for bin_config in get_bin_configs(scheme)}


def get_anisotropy_bin(
    ratio: float,
    spacing: Optional[Tuple[float, float, float]] = None,
    scheme: str = "original",
) -> int:
    """Assign a volume to an anisotropy bin using the requested binning scheme."""
    if scheme in {"coarse", "coarse_ratio_thickness"}:
        if spacing is None:
            raise ValueError("spacing is required for coarse binning")
        thickness = compute_slice_thickness_proxy(spacing)
        if ratio < 2.0 and thickness < 2.0:
            return 0
        if ratio < 4.0 and thickness < 4.0:
            return 1
        return 2

    for bin_config in ANISOTROPY_BINS:
        if bin_config.ratio_min <= ratio < bin_config.ratio_max:
            return bin_config.bin_id
    return len(ANISOTROPY_BINS) - 1  # Default to highest bin


# =============================================================================
# SAMPLING CONFIGURATION
# =============================================================================

@dataclass
class SamplingConfig:
    """Configuration for representation-geometry sample selection."""
    min_samples_per_bin: int = 500
    max_samples_per_bin: int = 1000
    stratify_by_modality: bool = True
    random_seed: int = 42

    # Modality handling for low-sample cases
    # If a modality has < min_samples_per_bin in a bin, either:
    # - "include_all": Use all available samples
    # - "exclude": Exclude the modality from that bin
    # - "merge_bins": Merge adjacent bins for the modality
    low_sample_strategy: str = "include_all"


SAMPLING_CONFIG = SamplingConfig()


# =============================================================================
# ANALYSIS PRESET CONFIGURATION
# =============================================================================
# The Acquisition Robustness workflow is dataset-scoped. Modality-filter presets
# are intentionally not exposed here because they mixed experimental naming with
# manifest construction. Use explicit dataset manifests instead, e.g. AbdomenAtlas
# for the acquisition-robustness reproduction target.

@dataclass
class AnalysisConfig:
    """Configuration for a specific analysis preset."""
    name: str
    description: str

    # Modality filtering
    allowed_modalities: List[str]  # Empty = all modalities

    # Bin filtering
    allowed_bins: List[int]  # Empty = all bins

    # Sampling parameters
    min_per_bin: int = 500
    max_per_bin: int = 1000

    # Output manifest name
    manifest_suffix: str = ""

    # Priority for reporting (lower = higher priority)
    priority: int = 0


ANALYSIS_CONFIGS: Dict[str, AnalysisConfig] = {}

# Default dataset-scoped analysis anchor.
DEFAULT_ANALYSIS = DEFAULT_ACQUISITION_DATASET


# =============================================================================
# SEMANTIC LABEL CONFIGURATION
# =============================================================================
# Configuration for semantic probing labels derived from segmentation masks.

@dataclass
class SemanticLabelConfig:
    """Configuration for semantic label extraction from segmentation masks."""
    # Label extraction mode
    # - "multi_binary": Multi-label binary (organ presence/absence)
    # - "dominant_organ": Single-label (organ with largest volume)
    mode: str = "multi_binary"

    # Minimum voxel count for organ presence (for multi_binary mode)
    min_organ_voxels: int = 100

    # Keep only labels with enough variation to support a meaningful probe.
    # Labels that are nearly always absent or nearly always present inflate
    # plain accuracy and are not informative for single-dataset acquisition robustness.
    min_label_prevalence: float = 0.05
    max_label_prevalence: float = 0.95

    # AbdomenAtlas organ mapping (label_id -> organ_name)
    # These are the 25 anatomical structures in AbdomenAtlas 1.1
    abdomenatlas_organs: Dict[int, str] = None

    def __post_init__(self):
        if self.abdomenatlas_organs is None:
            self.abdomenatlas_organs = {
                1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder",
                5: "liver", 6: "stomach", 7: "pancreas", 8: "adrenal_right",
                9: "adrenal_left", 10: "lung_upper_left", 11: "lung_lower_left",
                12: "lung_upper_right", 13: "lung_middle_right", 14: "lung_lower_right",
                15: "esophagus", 16: "trachea", 17: "thyroid", 18: "small_bowel",
                19: "duodenum", 20: "colon", 21: "urinary_bladder", 22: "prostate",
                23: "kidney_cyst_left", 24: "kidney_cyst_right", 25: "aorta"
            }


SEMANTIC_LABEL_CONFIG = SemanticLabelConfig()


@dataclass(frozen=True)
class SemanticTargetSpec:
    """Dataset-aware semantic target definition for spacing readouts and transfer.

    The extraction mode determines how semantic targets are derived from the
    label source associated with each case. Modes currently supported are:

    1. `label_id_multilabel`: one multi-label vector from an integer-valued mask.
    2. `segmentation_dir_presence`: one multi-label vector from a directory of
       binary structure masks.
    3. `foreground_volume_tertiles`: a one-hot target derived from foreground
         volume quantiles for single-structure binary masks.
    """

    name: str
    extraction_mode: str
    description: str
    integer_label_mapping: Optional[Dict[int, str]] = None
    segmentation_names: Optional[List[str]] = None
    semantic_label_names: Optional[List[str]] = None
    quantile_bins: int = 3
    min_voxels: int = 100
    min_label_prevalence: float = 0.05
    max_label_prevalence: float = 0.95


TOTALSEGMENTER_SHARED_STRUCTURES: List[str] = [
    "spleen",
    "kidney_left",
    "kidney_right",
    "gallbladder",
    "liver",
    "stomach",
    "pancreas",
    "small_bowel",
    "colon",
    "urinary_bladder",
    "aorta",
    "inferior_vena_cava",
]


FOREGROUND_VOLUME_TERTILES_SPEC = SemanticTargetSpec(
    name="foreground_volume_tertiles",
    extraction_mode="foreground_volume_tertiles",
    description="One-hot semantic target derived from foreground-volume tertiles for binary masks.",
    quantile_bins=3,
)


HECKTOR25_TARGETS: Dict[int, str] = {
    1: "primary_tumor",
    2: "metastatic_lymph_nodes",
}


KITS23_TARGETS: Dict[int, str] = {
    1: "kidney",
    2: "tumor",
    3: "cyst",
}


IMAGECAS_CORONARY_BURDEN_SPEC = SemanticTargetSpec(
    name="imagecas_coronary_burden_tertiles",
    extraction_mode="foreground_volume_tertiles",
    description="Dataset-specific coronary-tree burden strata derived from coronary foreground volume tertiles.",
    semantic_label_names=[
        "coronary_tree_burden_low",
        "coronary_tree_burden_mid",
        "coronary_tree_burden_high",
    ],
    quantile_bins=3,
)


DATASET_SEMANTIC_TARGET_SPECS: Dict[str, SemanticTargetSpec] = {
    "abdomenatlas": SemanticTargetSpec(
        name="abdomenatlas_multi_organ_presence",
        extraction_mode="label_id_multilabel",
        description="Multi-organ presence targets from the AbdomenAtlas combined-label mask.",
        integer_label_mapping=SEMANTIC_LABEL_CONFIG.abdomenatlas_organs,
        min_voxels=SEMANTIC_LABEL_CONFIG.min_organ_voxels,
        min_label_prevalence=SEMANTIC_LABEL_CONFIG.min_label_prevalence,
        max_label_prevalence=SEMANTIC_LABEL_CONFIG.max_label_prevalence,
    ),
    "totalsegmenter_ct": SemanticTargetSpec(
        name="totalsegmenter_shared_structure_presence",
        extraction_mode="segmentation_dir_presence",
        description="Shared-structure presence targets derived from the TotalSegmentator per-structure directory.",
        segmentation_names=TOTALSEGMENTER_SHARED_STRUCTURES,
        min_voxels=100,
    ),
    "totalsegmentermri": SemanticTargetSpec(
        name="totalsegmentermri_shared_structure_presence",
        extraction_mode="segmentation_dir_presence",
        description="Shared-structure presence targets derived from the TotalSegmenterMRI per-structure directory.",
        segmentation_names=TOTALSEGMENTER_SHARED_STRUCTURES,
        min_voxels=100,
    ),
    "hecktor25": SemanticTargetSpec(
        name="hecktor25_lesion_presence",
        extraction_mode="label_id_multilabel",
        description="Primary-tumor and nodal-lesion presence targets from the HECKTOR Task 1 mask.",
        integer_label_mapping=HECKTOR25_TARGETS,
        min_voxels=100,
        min_label_prevalence=SEMANTIC_LABEL_CONFIG.min_label_prevalence,
        max_label_prevalence=SEMANTIC_LABEL_CONFIG.max_label_prevalence,
    ),
    "jhu_stroke": FOREGROUND_VOLUME_TERTILES_SPEC,
    "kits23": SemanticTargetSpec(
        name="kits23_kidney_tumor_cyst_presence",
        extraction_mode="label_id_multilabel",
        description="Kidney, tumor, and cyst presence targets from the KiTS23 case segmentation.",
        integer_label_mapping=KITS23_TARGETS,
        min_voxels=100,
        min_label_prevalence=SEMANTIC_LABEL_CONFIG.min_label_prevalence,
        max_label_prevalence=SEMANTIC_LABEL_CONFIG.max_label_prevalence,
    ),
    "imagecas": IMAGECAS_CORONARY_BURDEN_SPEC,
}


MODALITY_SEMANTIC_TARGET_SPECS: Dict[str, SemanticTargetSpec] = {
    "ct": FOREGROUND_VOLUME_TERTILES_SPEC,
    "mr": FOREGROUND_VOLUME_TERTILES_SPEC,
    "cta": FOREGROUND_VOLUME_TERTILES_SPEC,
    "pet_ct": FOREGROUND_VOLUME_TERTILES_SPEC,
}


def resolve_semantic_target_spec(
    dataset_name: Optional[str] = None,
    modality: Optional[str] = None,
) -> SemanticTargetSpec:
    """Resolve the semantic target definition for a dataset or modality.

    Dataset-specific specifications take precedence over modality defaults.
    The fallback target for datasets without explicit structure semantics is a
    foreground-volume tertile target, which keeps the semantic claim narrow and
    target-specific for binary-mask datasets.
    """

    if dataset_name and dataset_name in DATASET_SEMANTIC_TARGET_SPECS:
        return DATASET_SEMANTIC_TARGET_SPECS[dataset_name]
    if modality and modality in MODALITY_SEMANTIC_TARGET_SPECS:
        return MODALITY_SEMANTIC_TARGET_SPECS[modality]
    return FOREGROUND_VOLUME_TERTILES_SPEC


# =============================================================================
# CONTROLLED PERTURBATION CONFIGURATION
# =============================================================================
# Configuration for synthetic spacing perturbation experiments.
# These isolate spacing as the causal factor by resampling the same volume
# to different target spacings.

@dataclass
class ControlledPerturbationConfig:
    """Configuration for controlled spacing perturbation."""
    protocol_name: str = "fixed_absolute"

    # Target spacings for resampling (mm)
    # Each volume will be resampled to these target spacings
    target_spacings: List[Tuple[float, float, float]] = None

    # MR protocol: keep native in-plane spacing and vary only through-plane spacing.
    target_z_spacings: Optional[List[float]] = None

    # Source: use volumes from the configured native-observational source bin
    source_bin: int = 0

    # Number of source volumes to use (None = all available)
    n_source_volumes: Optional[int] = None

    # Resampling interpolation mode
    interpolation_mode: str = "trilinear"  # or "nearest" for labels

    # Whether to anti-alias before downsampling
    anti_alias: bool = True

    # How source orientation is standardized before controlled perturbation.
    # "spacing_permutation" matches pretraining/eval: largest-spacing axis becomes depth.
    # "identity" keeps loaded voxel axes and only crops foreground.
    orientation_policy: str = "spacing_permutation"

    # Optional fixed physical field-of-view crop after resampling, before
    # resizing back to checkpoint input size. None preserves the legacy center
    # crop/resample/resize pipeline.
    physical_fov_mm: Optional[Tuple[float, float, float]] = None

    # Optional deterministic non-spacing perturbations added to the same
    # controlled variant bundle. Names are resolved in
    # perturbation_robustness_analysis.py.
    extra_perturbations: Optional[List[str]] = None

    extra_perturbation_seed: int = 17

    def __post_init__(self):
        self.orientation_policy = str(self.orientation_policy or "spacing_permutation").strip().lower()
        if self.orientation_policy not in {"spacing_permutation", "identity"}:
            raise ValueError(f"Unknown orientation policy: {self.orientation_policy}")
        if self.physical_fov_mm is not None:
            if len(self.physical_fov_mm) != 3:
                raise ValueError("physical_fov_mm must contain exactly three values")
            self.physical_fov_mm = tuple(float(value) for value in self.physical_fov_mm)
        if self.extra_perturbations is not None:
            self.extra_perturbations = [
                str(name).strip().lower().replace("-", "_")
                for name in self.extra_perturbations
                if str(name).strip()
            ]
        if self.protocol_name == "fixed_absolute" and self.target_spacings is None:
            # Default target spacings representing the 3 anisotropy regimes
            self.target_spacings = [
                (1.0, 1.0, 1.0),  # Isotropic (ratio = 1.0)
                (1.0, 1.0, 3.0),  # Moderate anisotropy (ratio = 3.0)
                (1.0, 1.0, 5.0),  # High anisotropy (ratio = 5.0)
            ]
        if self.protocol_name == "native_inplane_z" and self.target_z_spacings is None:
            self.target_z_spacings = [2.0, 4.0, 6.0]

    def resolve_variant_specs(
        self,
        source_spacing: Optional[Tuple[float, float, float]] = None,
    ) -> List[Tuple[str, Tuple[float, float, float]]]:
        """Resolve named target spacing variants for a source volume."""
        if self.protocol_name == "fixed_absolute":
            return [
                (f"{sx:.1f}x{sy:.1f}x{sz:.1f}", (sx, sy, sz))
                for sx, sy, sz in self.target_spacings
            ]

        if self.protocol_name == "native_inplane_z":
            if source_spacing is None:
                raise ValueError("source_spacing is required for native_inplane_z perturbation")
            sx, sy, _ = [float(value) for value in source_spacing]
            return [
                (f"native_xy_z{z:.1f}", (sx, sy, float(z)))
                for z in self.target_z_spacings
            ]

        raise ValueError(f"Unknown perturbation protocol: {self.protocol_name}")

    def reference_variant_name(self, source_spacing: Optional[Tuple[float, float, float]] = None) -> str:
        """Return the baseline variant name used for representation drift."""
        return self.resolve_variant_specs(source_spacing)[0][0]

    def cache_signature(self) -> str:
        """Identify per-volume geometry without rounding physical protocol values."""
        protocol = asdict(self)
        protocol.pop("n_source_volumes")
        encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return "protocol_v3_" + hashlib.sha256(encoded.encode()).hexdigest()

    def describe_targets(self) -> List[str]:
        """Return human-readable target descriptions for logging/serialization."""
        if self.protocol_name == "fixed_absolute":
            return [f"{sx:.1f}x{sy:.1f}x{sz:.1f}" for sx, sy, sz in self.target_spacings]
        if self.protocol_name == "native_inplane_z":
            return [f"native_xy_z{z:.1f}" for z in self.target_z_spacings]
        return [self.protocol_name]


CONTROLLED_PERTURBATION_CONFIG = ControlledPerturbationConfig()
MRI_CONTROLLED_PERTURBATION_CONFIG = ControlledPerturbationConfig(
    protocol_name="native_inplane_z",
    target_spacings=None,
    target_z_spacings=[2.0, 4.0, 6.0],
    source_bin=1,
)


def get_controlled_perturbation_config(modality: str = "ct") -> ControlledPerturbationConfig:
    """Return the controlled perturbation protocol for a modality."""
    if str(modality or "ct").lower() == "mr":
        return MRI_CONTROLLED_PERTURBATION_CONFIG
    return CONTROLLED_PERTURBATION_CONFIG


def get_controlled_perturbation_config_from_manifest(
    manifest_data: Dict[str, Any],
) -> ControlledPerturbationConfig:
    """Return the controlled perturbation protocol for a manifest payload."""
    return get_controlled_perturbation_config(str(manifest_data.get("modality", "ct")))


# =============================================================================
# CROSS-BIN TRANSFER CONFIGURATION
# =============================================================================
# Configuration for cross-bin transfer experiments in spacing readouts.

@dataclass
class CrossBinTransferConfig:
    """Configuration for cross-bin transfer probing experiments."""
    # Datasets allowed for cross-bin transfer (must span multiple bins)
    # Only datasets with sufficient samples in multiple bins are valid
    allowed_datasets: List[str] = None

    # Minimum samples required per bin for a dataset to be included
    min_samples_per_bin: int = 50

    # Probing tasks to evaluate
    # - "domain": Dataset/subdomain classification
    # - "semantic": Organ presence (multi-label binary)
    probing_tasks: List[str] = None

    def __post_init__(self):
        if self.allowed_datasets is None:
            # Only AbdomenAtlas spans all bins with sufficient samples
            self.allowed_datasets = ["abdomenatlas"]
        if self.probing_tasks is None:
            self.probing_tasks = ["semantic"]


CROSS_BIN_TRANSFER_CONFIG = CrossBinTransferConfig()


# =============================================================================
# PREPROCESSING CONFIGURATION
# =============================================================================
# Follows pretraining preprocessing exactly.

@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""
    # Intensity normalization (percentile-based)
    # Match the verified Med3DINO training pipeline in train3d.py.
    intensity_percentile_lower: float = 0.05
    intensity_percentile_upper: float = 99.95
    intensity_output_min: float = -1.0
    intensity_output_max: float = 1.0

    # Cropping
    crop_foreground: bool = True
    foreground_threshold: float = -1.0  # Training uses select_fn=lambda x: x > -1

    # Evaluation should fail loudly if a sample cannot be loaded.
    fail_on_load_error: bool = True

    # Target crop size (matches checkpoint)
    crop_size: int = 96

    # Data format
    output_dtype: str = "float32"


PREPROCESSING_CONFIG = PreprocessingConfig()


# =============================================================================
# FEATURE EXTRACTION CONFIGURATION
# =============================================================================

@dataclass
class FeatureExtractionConfig:
    """Configuration for feature extraction from ViT encoder."""
    # Global representation methods
    use_cls_token: bool = True
    use_avg_pooled_patches: bool = True  # Aggregate patch tokens via spatial average pooling

    # Multi-scale analysis
    use_multilayer: bool = True
    layer_indices: Tuple[int, ...] = (-4, -3, -2, -1)  # Last 4 blocks

    # Patch tokens (for CKA on local representations)
    extract_patch_tokens: bool = True

    # Batch processing
    batch_size: int = 4
    num_workers: int = 4

    # Output format
    save_format: str = "pt"  # "pt" (PyTorch) or "npy" (NumPy)


FEATURE_EXTRACTION_CONFIG = FeatureExtractionConfig()


# =============================================================================
# METRICS CONFIGURATION
# =============================================================================
# Metrics for readout evaluation.

@dataclass
class MetricsConfig:
    """Configuration for evaluation metrics."""
    # Segmentation metrics
    compute_dice: bool = True
    compute_hd95: bool = True
    compute_asd: bool = True
    compute_nsd: bool = True
    nsd_tolerance_mm: float = 2.0

    # Classification metrics (if applicable)
    compute_accuracy: bool = True
    compute_auroc: bool = True
    compute_f1: bool = True


METRICS_CONFIG = MetricsConfig()


# =============================================================================
# EVALUATION SEEDS (for reproducibility)
# =============================================================================
# Run all probing experiments with multiple seeds and report mean ± std
EVALUATION_SEEDS: List[int] = [42, 123, 456]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_directories():
    """Create shared acquisition robustness directories."""
    directories = [
        ACQUISITION_OUTPUTS,
        ACQUISITION_MANIFESTS,
        ACQUISITION_CACHES,
    ]
    for d in directories:
        d.mkdir(parents=True, exist_ok=True)


def validate_checkpoint_paths() -> Dict[str, bool]:
    """Validate that all checkpoint files exist."""
    results = {}
    for name, config in CHECKPOINTS.items():
        results[name] = config.path.exists()
    return results


def validate_raw_dataset_paths() -> Dict[str, bool]:
    """Validate that all raw dataset directories exist."""
    results = {}
    for name, config in RAW_DATASETS.items():
        results[name] = config.path.exists()
    return results


def get_modality_abbreviation(modality: str) -> str:
    """Get standardized modality abbreviation."""
    mapping = {
        "ct": "CT",
        "mr": "MR",
        "pet": "PET",
        "pet_ct": "PET-CT",
        "cta": "CTA",
        "cbct": "CBCT",
        "us": "US",
        "oct": "OCT",
    }
    return mapping.get(modality.lower(), modality.upper())


# =============================================================================
# DATASET-SPECIFIC OUTPUT PATH FUNCTIONS
# =============================================================================
# Functions to generate output paths for dataset-specific experiments.
# Example: abdomenatlas/original_bins/results/, etc.

def get_output_paths(
    dataset_name: str = DEFAULT_ACQUISITION_DATASET,
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> Dict[str, Path]:
    """
    Get output path dictionary for a specific dataset and manifest variant.

    Args:
        dataset_name: Dataset name (e.g., "abdomenatlas", "kits23")
        manifest_variant: Manifest variant directory (e.g., "original_bins", "coarse_bins")

    Returns:
        Dictionary with keys: "root", "results", "figures", "features", "logs", "cache_root", "manifests"
    """
    manifest_variant = normalize_manifest_variant(manifest_variant)
    active = active_output_paths("acquisition_robustness")
    if active is not None:
        return active
    root = apply_active_seed_to_path(ACQUISITION_OUTPUTS / dataset_name / manifest_variant)
    return {
        "root": root,
        "results": root / "results",
        "figures": root / "figures",
        "features": root / "features",
        "logs": root / "logs",
        "cache_root": get_cache_root(dataset_name, manifest_variant),
        "manifests": get_acquisition_manifest_dir(dataset_name, manifest_variant),
    }


def get_checkpoint_feature_dir(base_dir: Path, checkpoint_name: str, feature_type: str) -> Path:
    """Return the nested directory for a checkpoint and feature family."""
    return Path(base_dir) / checkpoint_name / feature_type


def get_summary_feature_dir(base_dir: Path, feature_type: str) -> Path:
    """Return the nested summary directory for a feature family."""
    return Path(base_dir) / "summaries" / feature_type


def get_dataset_name_from_manifest_path(
    manifest_path: Path,
    fallback: str = DEFAULT_ACQUISITION_DATASET,
) -> str:
    """Infer the dataset name from a manifest path under the shared manifest root."""
    manifest_path = Path(manifest_path)

    for root in get_reform3d_module_manifest_roots(ACQUISITION_MODULE_NAME):
        try:
            relative_path = manifest_path.resolve().relative_to(root.resolve())
        except ValueError:
            continue
        if len(relative_path.parts) >= 1:
            return relative_path.parts[0]
    return fallback


def get_manifest_variant_from_manifest_path(
    manifest_path: Path,
    fallback: str = DEFAULT_MANIFEST_VARIANT,
) -> str:
    """Infer the manifest variant from a manifest path under the shared manifest root."""
    manifest_path = Path(manifest_path)

    for root in get_reform3d_module_manifest_roots(ACQUISITION_MODULE_NAME):
        try:
            relative_path = manifest_path.resolve().relative_to(root.resolve())
        except ValueError:
            continue
        if len(relative_path.parts) >= 2:
            return normalize_manifest_variant(relative_path.parts[1])
    return normalize_manifest_variant(fallback)


def ensure_output_directories(
    dataset_name: str = DEFAULT_ACQUISITION_DATASET,
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
):
    """Create all necessary output directories for a dataset and manifest variant."""
    paths = get_output_paths(dataset_name, manifest_variant)
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Usage: python config.py
    # Prints a quick validation summary for checkpoints, raw datasets, and bin definitions.
    # Quick validation
    print("=" * 60)
    print("Acquisition Robustness Configuration Validation")
    print("=" * 60)

    print("\n--- Checkpoint Paths ---")
    for name, exists in validate_checkpoint_paths().items():
        status = "✓" if exists else "✗"
        print(f"  [{status}] {name}: {CHECKPOINTS[name].path}")

    print("\n--- Raw Dataset Paths ---")
    for name, exists in validate_raw_dataset_paths().items():
        status = "✓" if exists else "✗"
        print(f"  [{status}] {name}: {RAW_DATASETS[name].path}")

    print("\n--- Anisotropy Bins ---")
    for bin_config in ANISOTROPY_BINS:
        print(f"  Bin {bin_config.bin_id} ({bin_config.name}): {bin_config.description}")

    print("\n--- Coarse Ratio+Thickness Bins ---")
    for bin_config in COARSE_ANISOTROPY_BINS:
        print(f"  Bin {bin_config.bin_id} ({bin_config.name}): {bin_config.description}")
