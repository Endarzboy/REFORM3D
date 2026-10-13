"""Path and artifact helpers for Anatomical Generalization.

This module centralizes the filesystem contract between Anatomical
Generalization and its upstream Cross-Modal organ-level manifest and
feature-bundle sources.

It should be the first place to update when Anatomical Generalization manifests,
feature bundle locations, or output naming conventions change.

Required identifiers for most helpers:
        - dataset_name: Anatomical Generalization dataset namespace, for example
            ``totalsegmenter_ct_mr_anchor`` or ``mmwhs_ct_mr``.
        - manifest_variant: usually ``core``.
        - checkpoint_name: a registered checkpoint key such as ``med3dino_rel_96``.
                - feature_type: standard Med3DINO labels ``cls``, ``avg_pool``, or ``multilayer``;
                    external-method single-bundle evaluations may also use method-specific labels.

Examples:
        >>> get_anatomical_manifest_path("totalsegmenter_ct_mr_anchor", "sampled", "core")
        PosixPath('.../reform3d_manifests/anatomical_generalization/totalsegmenter_ct_mr_anchor/core/manifest_sampled.json')

        >>> get_cross_modal_feature_npz_path("totalsegmenter_ct_mr_anchor", "core", "med3dino_rel_96", "cls")
        PosixPath('.../reform3d_results/cross_modal_alignment/totalsegmenter_ct_mr_anchor/core/features/med3dino_rel_96/cls/organ_cls_embeddings.npz')
"""

from pathlib import Path
from reform3d.core.layout import active_output_paths
from typing import Dict

from reform3d.core.checkpoints import FULL_CHECKPOINT_ORDER
from reform3d.core.paths import (
    get_reform3d_module_manifest_roots,
    get_reform3d_module_manifest_root,
    get_reform3d_module_result_root,
    resolve_reform3d_module_manifest_path,
    resolve_reform3d_module_result_path,
)
from reform3d.core.seed_context import apply_active_seed_to_path


ANATOMICAL_MODULE_NAME = "anatomical_generalization"
CROSS_MODAL_MODULE_NAME = "cross_modal_alignment"

ANATOMICAL_ROOT = Path(__file__).resolve().parent
ANATOMICAL_OUTPUTS = get_reform3d_module_result_root(ANATOMICAL_MODULE_NAME)
ANATOMICAL_MANIFESTS = get_reform3d_module_manifest_root(ANATOMICAL_MODULE_NAME)
CROSS_MODAL_OUTPUTS = get_reform3d_module_result_root(CROSS_MODAL_MODULE_NAME)
CROSS_MODAL_MANIFESTS = get_reform3d_module_manifest_root(CROSS_MODAL_MODULE_NAME)

TOTALSEGMENTER_ANATOMICAL_DATASET = "totalsegmenter_ct_mr_anchor"
MMWHS_ANATOMICAL_DATASET = "mmwhs_ct_mr"
AMOS22_ANATOMICAL_DATASET = "amos22_ct_mr"
ANATOMICAL_DATASETS = (
    TOTALSEGMENTER_ANATOMICAL_DATASET,
    MMWHS_ANATOMICAL_DATASET,
    AMOS22_ANATOMICAL_DATASET,
)
DATASET_NAME_ALIASES = {
    "amos_ct_mr": AMOS22_ANATOMICAL_DATASET,
    "amos22": AMOS22_ANATOMICAL_DATASET,
}

DEFAULT_ANATOMICAL_DATASET = AMOS22_ANATOMICAL_DATASET
DEFAULT_MANIFEST_VARIANT = "core"
DEFAULT_REQUIRED_MODALITIES = ("ct", "mr")
DEFAULT_FEATURE_TYPE = "cls"
DEFAULT_MIN_HOLDOUT_ORGANS = 2
DEFAULT_MIN_SAMPLES_PER_MODALITY = 5
DEFAULT_FEW_SHOT_SUPPORT_PER_MODALITY = 2
DEFAULT_FEW_SHOT_QUERY_PER_MODALITY = None
DEFAULT_FEW_SHOT_SEEDS = (42, 123, 456)

FEATURE_TYPES = ("cls", "avg_pool", "multilayer")
EXTERNAL_FEATURE_TYPES = ("projected", "pooled_multiscale", "final_stage_pool")
FEATURE_TYPE_ALIASES = {
    "cls": "cls",
    "avg_pool": "avg_pool",
    "avgpool": "avg_pool",
    "avg-pool": "avg_pool",
    "multilayer": "multilayer",
    "multi_layer": "multilayer",
    "multi-layer": "multilayer",
    "projected": "projected",
    "projection": "projected",
    "projector": "projected",
    "pooled_multiscale": "pooled_multiscale",
    "pooled-multiscale": "pooled_multiscale",
    "stagewise_pool": "pooled_multiscale",
    "stagewise-pool": "pooled_multiscale",
    "stage_wise_pool": "pooled_multiscale",
    "stage-wise-pool": "pooled_multiscale",
    "final_stage_pool": "final_stage_pool",
    "final-stage-pool": "final_stage_pool",
    "final_pool": "final_stage_pool",
}
FEATURE_EMBEDDING_FILE_NAMES = {
    "cls": "organ_cls_embeddings.npz",
    "avg_pool": "organ_avg_pool_embeddings.npz",
    "multilayer": "organ_multilayer_embeddings.npz",
    "projected": "organ_projected_embeddings.npz",
    "pooled_multiscale": "organ_pooled_multiscale_embeddings.npz",
    "final_stage_pool": "organ_final_stage_pool_embeddings.npz",
}
MANIFEST_VARIANT_ALIASES = {
    "core": "core",
    "default": "core",
    "shared_organs": "core",
    "paired": "paired",
    "paired_core": "paired",
}
MANIFEST_FILE_NAMES = {
    "full": "manifest_full.json",
    "sampled": "manifest_sampled.json",
    "meta": "manifest_meta.json",
}

RESULT_JSON_NAME = "anatomical_generalization.json"
SUMMARY_JSON_NAME = "anatomical_generalization_summary.json"
SUMMARY_CSV_NAME = "anatomical_generalization_summary.csv"
FEATURE_ADAPTER_DIR_NAME = "feature_adapter"
DEFAULT_CHECKPOINT_POLICY = "full_checkpoint_slate"

DEFAULT_CHECKPOINTS = list(FULL_CHECKPOINT_ORDER)


def normalize_feature_type(feature_type: str = DEFAULT_FEATURE_TYPE) -> str:
    """Normalize an Anatomical Generalization feature-family label.

    Required parameter:
        feature_type: one of the standard labels or a supported external-method label.

    Example:
        >>> normalize_feature_type("avg-pool")
        'avg_pool'
    """
    normalized = FEATURE_TYPE_ALIASES.get(str(feature_type).strip().lower())
    if normalized is None:
        raise ValueError(f"Unknown feature type: {feature_type}")
    return normalized


def normalize_manifest_variant(manifest_variant: str = DEFAULT_MANIFEST_VARIANT) -> str:
    """Normalize a manifest variant label.

    Required parameter:
        manifest_variant: typically ``core`` or an alias such as ``default``.

    Example:
        >>> normalize_manifest_variant("shared_organs")
        'core'
    """
    return MANIFEST_VARIANT_ALIASES.get(str(manifest_variant).strip().lower(), manifest_variant)


def normalize_anatomical_dataset_name(dataset_name: str = DEFAULT_ANATOMICAL_DATASET) -> str:
    normalized = str(dataset_name).strip().lower()
    return DATASET_NAME_ALIASES.get(normalized, normalized)


def get_anatomical_manifest_dir(
    dataset_name: str = DEFAULT_ANATOMICAL_DATASET,
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> Path:
    """Return the Anatomical Generalization manifest directory.

    Required parameters:
        dataset_name: Anatomical Generalization dataset namespace.
        manifest_variant: manifest variant label, usually ``core``.
    """
    return ANATOMICAL_MANIFESTS / normalize_anatomical_dataset_name(dataset_name) / normalize_manifest_variant(manifest_variant)


def get_anatomical_manifest_path(
    dataset_name: str = DEFAULT_ANATOMICAL_DATASET,
    manifest_kind: str = "sampled",
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> Path:
    """Return one Anatomical Generalization manifest path.

    Required parameters:
        dataset_name: Anatomical Generalization dataset namespace.
        manifest_kind: one of ``full``, ``sampled``, or ``meta``.
        manifest_variant: manifest variant label, usually ``core``.

    Example:
        >>> get_anatomical_manifest_path("mmwhs_ct_mr", "meta", "core")
    """
    if manifest_kind not in MANIFEST_FILE_NAMES:
        raise ValueError(f"Unknown manifest kind: {manifest_kind}")
    dataset_name = normalize_anatomical_dataset_name(dataset_name)
    return resolve_reform3d_module_manifest_path(
        ANATOMICAL_MODULE_NAME,
        Path(dataset_name) / normalize_manifest_variant(manifest_variant) / MANIFEST_FILE_NAMES[manifest_kind],
    )


def get_cross_modal_manifest_path(
    dataset_name: str = DEFAULT_ANATOMICAL_DATASET,
    manifest_kind: str = "sampled",
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> Path:
    """Return the upstream Cross-Modal organ-manifest path used by Anatomical Generalization.

    Required parameters are the same as ``get_anatomical_manifest_path``.
    """
    if manifest_kind not in MANIFEST_FILE_NAMES:
        raise ValueError(f"Unknown manifest kind: {manifest_kind}")
    dataset_name = normalize_anatomical_dataset_name(dataset_name)
    return resolve_reform3d_module_manifest_path(
        CROSS_MODAL_MODULE_NAME,
        Path(dataset_name) / normalize_manifest_variant(manifest_variant) / MANIFEST_FILE_NAMES[manifest_kind],
    )


def get_output_paths(
    dataset_name: str = DEFAULT_ANATOMICAL_DATASET,
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> Dict[str, Path]:
    """Return the Anatomical Generalization output directories for one dataset surface.

    Required parameters:
        dataset_name: Anatomical Generalization dataset namespace.
        manifest_variant: manifest variant label, usually ``core``.
    """
    active = active_output_paths("anatomical_generalization")
    if active is not None:
        return active
    dataset_name = normalize_anatomical_dataset_name(dataset_name)
    manifest_variant = normalize_manifest_variant(manifest_variant)
    root = apply_active_seed_to_path(ANATOMICAL_OUTPUTS / dataset_name / manifest_variant)
    return {
        "root": root,
        "results": root / "results",
        "figures": root / "figures",
        "logs": root / "logs",
        "manifests": get_anatomical_manifest_dir(dataset_name, manifest_variant),
        "source_cross_modal_root": apply_active_seed_to_path(CROSS_MODAL_OUTPUTS / dataset_name / manifest_variant),
    }


def get_cross_modal_feature_npz_path(
    dataset_name: str,
    manifest_variant: str,
    checkpoint_name: str,
    feature_type: str = DEFAULT_FEATURE_TYPE,
) -> Path:
    """Return the expected upstream embedding bundle path for Anatomical Generalization.

    Required parameters:
        dataset_name: Anatomical Generalization dataset namespace.
        manifest_variant: manifest variant label, usually ``core``.
        checkpoint_name: checkpoint key such as ``med3dino_rel_96``.
        feature_type: one of the standard labels or a supported external-method label.

        Cross-Modal Alignment owns the standard organ-level embedding namespace.
    """
    feature_type = normalize_feature_type(feature_type)
    dataset_name = normalize_anatomical_dataset_name(dataset_name)
    return (
        apply_active_seed_to_path(CROSS_MODAL_OUTPUTS / dataset_name / normalize_manifest_variant(manifest_variant))
        / "features"
        / checkpoint_name
        / feature_type
        / FEATURE_EMBEDDING_FILE_NAMES[feature_type]
    )


def get_cross_modal_extension_feature_npz_path(
    dataset_name: str,
    manifest_variant: str,
    checkpoint_name: str,
    feature_type: str = DEFAULT_FEATURE_TYPE,
    surface: str = "baseline_full_bbox_context",
) -> Path:
    feature_type = normalize_feature_type(feature_type)
    dataset_name = normalize_anatomical_dataset_name(dataset_name)
    return (
        apply_active_seed_to_path(CROSS_MODAL_OUTPUTS / dataset_name / normalize_manifest_variant(manifest_variant))
        / "evaluation_extensions"
        / "features"
        / surface
        / checkpoint_name
        / feature_type
        / FEATURE_EMBEDDING_FILE_NAMES[feature_type]
    )


def get_cross_modal_feature_npz_candidates(
    dataset_name: str,
    manifest_variant: str,
    checkpoint_name: str,
    feature_type: str = DEFAULT_FEATURE_TYPE,
    extension_surface: str | None = "baseline_full_bbox_context",
) -> tuple[Path, ...]:
    candidates: list[Path] = []
    if extension_surface:
        candidates.append(
            get_cross_modal_extension_feature_npz_path(
                dataset_name,
                manifest_variant,
                checkpoint_name,
                feature_type,
                extension_surface,
            )
        )
    candidates.append(get_cross_modal_feature_npz_path(dataset_name, manifest_variant, checkpoint_name, feature_type))
    return tuple(dict.fromkeys(candidates))


def resolve_cross_modal_feature_npz_path(
    dataset_name: str,
    manifest_variant: str,
    checkpoint_name: str,
    feature_type: str = DEFAULT_FEATURE_TYPE,
    extension_surface: str | None = "baseline_full_bbox_context",
) -> Path:
    candidates = get_cross_modal_feature_npz_candidates(
        dataset_name,
        manifest_variant,
        checkpoint_name,
        feature_type,
        extension_surface,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def get_checkpoint_metrics_dir(base_dir: Path, checkpoint_name: str, feature_type: str = DEFAULT_FEATURE_TYPE) -> Path:
    """Return the result directory for one checkpoint and feature family."""
    feature_type = normalize_feature_type(feature_type)
    checkpoint_dir = Path(base_dir) / checkpoint_name
    return checkpoint_dir / feature_type


def get_checkpoint_metrics_path(base_dir: Path, checkpoint_name: str, feature_type: str = DEFAULT_FEATURE_TYPE) -> Path:
    """Return the Anatomical Generalization per-checkpoint result JSON path."""
    return get_checkpoint_metrics_dir(base_dir, checkpoint_name, feature_type) / RESULT_JSON_NAME


def get_feature_adapter_metrics_dir(
    base_dir: Path,
    checkpoint_name: str,
    feature_type: str,
    adapter_name: str,
    source_key: str,
) -> Path:
    """Return the isolated result directory for one frozen-feature adapter."""
    return (
        get_checkpoint_metrics_dir(base_dir, checkpoint_name, feature_type)
        / FEATURE_ADAPTER_DIR_NAME
        / str(adapter_name)
        / str(source_key)
    )


def get_feature_adapter_metrics_path(
    base_dir: Path,
    checkpoint_name: str,
    feature_type: str,
    adapter_name: str,
    source_key: str,
) -> Path:
    """Return the Anatomical result JSON path for one frozen-feature adapter."""
    return get_feature_adapter_metrics_dir(base_dir, checkpoint_name, feature_type, adapter_name, source_key) / RESULT_JSON_NAME


def get_summary_json_path(base_dir: Path) -> Path:
    """Return the batch summary JSON path."""
    return Path(base_dir) / SUMMARY_JSON_NAME


def get_summary_csv_path(base_dir: Path) -> Path:
    """Return the batch summary CSV path."""
    return Path(base_dir) / SUMMARY_CSV_NAME


def get_dataset_name_from_manifest_path(
    manifest_path: Path,
    fallback: str = DEFAULT_ANATOMICAL_DATASET,
) -> str:
    """Infer the dataset namespace from a Cross-Modal or Anatomical manifest path.

    Required parameter:
        manifest_path: absolute or relative path under a configured manifest
        roots.
    """
    manifest_path = Path(manifest_path)
    candidate_roots = (
        *get_reform3d_module_manifest_roots(ANATOMICAL_MODULE_NAME),
        *get_reform3d_module_manifest_roots(CROSS_MODAL_MODULE_NAME),
    )
    for root in candidate_roots:
        try:
            relative_path = manifest_path.resolve().relative_to(root)
        except ValueError:
            continue
        if len(relative_path.parts) >= 1:
            return normalize_anatomical_dataset_name(relative_path.parts[0])
    return normalize_anatomical_dataset_name(fallback)


def get_manifest_variant_from_manifest_path(
    manifest_path: Path,
    fallback: str = DEFAULT_MANIFEST_VARIANT,
) -> str:
    """Infer the manifest variant from a Cross-Modal or Anatomical manifest path."""
    manifest_path = Path(manifest_path)
    candidate_roots = (
        *get_reform3d_module_manifest_roots(ANATOMICAL_MODULE_NAME),
        *get_reform3d_module_manifest_roots(CROSS_MODAL_MODULE_NAME),
    )
    for root in candidate_roots:
        try:
            relative_path = manifest_path.resolve().relative_to(root)
        except ValueError:
            continue
        if len(relative_path.parts) >= 2:
            return normalize_manifest_variant(relative_path.parts[1])
    return normalize_manifest_variant(fallback)


def ensure_output_directories(
    dataset_name: str = DEFAULT_ANATOMICAL_DATASET,
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> None:
    """Create the Anatomical Generalization output directories if they do not exist.

    Required parameters:
        dataset_name: Anatomical Generalization dataset namespace.
        manifest_variant: manifest variant label, usually ``core``.

    Example:
        >>> ensure_output_directories("totalsegmenter_ct_mr_anchor", "core")
    """
    paths = get_output_paths(dataset_name, manifest_variant)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
