"""Configuration and path resolution for the readout-sensitivity analysis.

All paths are resolved from the shared REFORM-3D result root so the analysis
reuses completed manifests and frozen-feature artifacts without recomputation.
"""

from __future__ import annotations

from pathlib import Path

from reform3d.core.paths import get_reform3d_module_results_root
from reform3d.core.seed_context import apply_seed_to_path
from reform3d.modules.anatomical_generalization.config import (
    get_anatomical_manifest_path,
)

# --- Scope -----------------------------------------------------------------
CHECKPOINTS = ("med3dino_iso_96", "med3dino_rel_96", "med3dino_sa_96")
# Short labels used for ordering/paired reporting (ISO/REL/SA).
CHECKPOINT_LABELS = {
    "med3dino_iso_96": "ISO",
    "med3dino_rel_96": "REL",
    "med3dino_sa_96": "SA",
}
FEATURE_TYPES = ("cls", "avg_pool")
SEEDS = (42, 123, 456)
READOUTS = ("logistic", "nearest_centroid")
LOGISTIC_C_VALUES = (0.1, 1.0, 10.0)

# Annotation-derived semantic-target datasets (avoid weaker volume-tertile ones).
SEMANTIC_DATASETS = ("abdomenatlas", "kits23", "totalsegmentermri")
ANATOMICAL_DATASETS = ("totalsegmenter_ct_mr_anchor", "mmwhs_ct_mr", "amos22_ct_mr")

SEMANTIC_MANIFEST_VARIANT = "original_bins"
ANATOMICAL_MANIFEST_VARIANT = "core"

# Fixed protocol contract mirrored from the production pipelines.
SEMANTIC_N_CV_SPLITS = 5
SEMANTIC_LOGISTIC_MAX_ITER = 1000
ANATOMICAL_LOGISTIC_MAX_ITER = 2000
SEMANTIC_MIN_CLASS_COUNT = 2

# Anatomical low-shot protocol (matches the production few-shot invocation).
ANATOMICAL_REQUIRED_MODALITIES = ("ct", "mr")
ANATOMICAL_SUPPORT_PER_MODALITY = 2
ANATOMICAL_QUERY_PER_MODALITY = None
ANATOMICAL_MIN_SAMPLES_PER_MODALITY = 5

# --- Paired-comparison + bootstrap contract --------------------------------
# Ordered so SA-REL is the primary comparison; all three are Bonferroni-adjusted.
PAIRED_COMPARISONS = ("REL-ISO", "SA-ISO", "SA-REL")
BONFERRONI_FAMILY_SIZE = len(PAIRED_COMPARISONS)
DEFAULT_BOOTSTRAP_RESAMPLES = 50000
DEFAULT_BOOTSTRAP_SEED = 2026
CONFIDENCE_LEVEL = 0.95

# --- Output roots ----------------------------------------------------------
MODULE_NAME = "readout_sensitivity"
RESULTS_ROOT = get_reform3d_module_results_root(MODULE_NAME)
SEMANTIC_ROOT = RESULTS_ROOT / "semantic_transfer"
ANATOMICAL_ROOT = RESULTS_ROOT / "low_shot_transfer"
SUMMARY_ROOT = RESULTS_ROOT / "summary"

LOG_ROOT = RESULTS_ROOT / "logs"

# Acquisition-robustness result root holds the controlled embedding bundles.
_ACQUISITION_RESULTS_ROOT = get_reform3d_module_results_root("acquisition_robustness")

CONTROLLED_EMBEDDING_BUNDLE_NAME = "controlled_embedding_bundle.npz"
MATCHED_SEMANTIC_TRANSFER_NAME = "matched_semantic_transfer.json"
ANATOMICAL_RESULT_NAME = "anatomical_generalization.json"


def controlled_bundle_path(
    dataset: str,
    checkpoint: str,
    feature_type: str,
    seed: int,
    manifest_variant: str = SEMANTIC_MANIFEST_VARIANT,
) -> Path:
    """Return the frozen controlled-embedding bundle for one semantic cell."""
    seeded = apply_seed_to_path(
        _ACQUISITION_RESULTS_ROOT / dataset / manifest_variant, seed
    )
    return seeded / "results" / checkpoint / feature_type / CONTROLLED_EMBEDDING_BUNDLE_NAME


def stored_semantic_transfer_path(
    dataset: str,
    checkpoint: str,
    feature_type: str,
    seed: int,
    manifest_variant: str = SEMANTIC_MANIFEST_VARIANT,
) -> Path:
    """Return the stored (production) matched_semantic_transfer.json for one cell."""
    seeded = apply_seed_to_path(
        _ACQUISITION_RESULTS_ROOT / dataset / manifest_variant, seed
    )
    return seeded / "results" / checkpoint / feature_type / MATCHED_SEMANTIC_TRANSFER_NAME


def semantic_acquisition_manifest_path(
    dataset: str,
    seed: int,
    manifest_variant: str = SEMANTIC_MANIFEST_VARIANT,
) -> Path:
    """Return the seeded acquisition manifest used to derive semantic targets."""
    from reform3d.modules.acquisition_robustness.config import (
        ACQUISITION_MANIFESTS,
        normalize_manifest_variant,
    )

    unseeded_dir = ACQUISITION_MANIFESTS / dataset / normalize_manifest_variant(manifest_variant)
    return apply_seed_to_path(unseeded_dir, seed) / "manifest_sampled.json"


def semantic_label_cache_dir(
    dataset: str,
    seed: int,
    manifest_variant: str = SEMANTIC_MANIFEST_VARIANT,
) -> Path:
    """Return the seeded acquisition results dir holding the semantic-label cache."""
    seeded = apply_seed_to_path(
        _ACQUISITION_RESULTS_ROOT / dataset / manifest_variant, seed
    )
    return seeded / "results"


def anatomical_manifest_path(
    dataset: str,
    manifest_variant: str = ANATOMICAL_MANIFEST_VARIANT,
) -> Path:
    """Return the anatomical sampled manifest (shared across seeds)."""
    return get_anatomical_manifest_path(dataset, "sampled", manifest_variant)


def anatomical_embeddings_path(
    dataset: str,
    checkpoint: str,
    feature_type: str,
    seed: int,
    manifest_variant: str = ANATOMICAL_MANIFEST_VARIANT,
) -> Path:
    """Return the frozen cross-modal embedding bundle reused for anatomical transfer.

    The seed subdirectory is stamped on the ``<dataset>/<variant>`` directory,
    matching the production Cross-Modal layout, then the feature tail is appended.
    """
    from reform3d.modules.anatomical_generalization.config import (
        CROSS_MODAL_OUTPUTS,
        normalize_anatomical_dataset_name,
        normalize_feature_type,
        normalize_manifest_variant,
        FEATURE_EMBEDDING_FILE_NAMES,
    )

    feature_type_n = normalize_feature_type(feature_type)
    dataset_n = normalize_anatomical_dataset_name(dataset)
    seeded_base = apply_seed_to_path(
        CROSS_MODAL_OUTPUTS / dataset_n / normalize_manifest_variant(manifest_variant),
        seed,
    )
    return (
        seeded_base
        / "features"
        / checkpoint
        / feature_type_n
        / FEATURE_EMBEDDING_FILE_NAMES[feature_type_n]
    )
