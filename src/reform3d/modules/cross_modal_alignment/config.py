"""Configuration and path helpers for Cross-Modal Alignment.

This module defines supported CT/MR dataset surfaces, manifest roots, cache
roots, result layouts, feature-bundle names, and small normalization helpers.
Pipeline modules use these functions to resolve where manifests, organ crop
caches, embedding bundles, and metric summaries are stored.

Inputs:
    dataset_name: ``amos22_ct_mr``, ``mmwhs_ct_mr``, ``totalsegmenter_ct_mr_anchor``,
        or ``chaos_ct_mr``.
    manifest_variant: usually ``core``.
    feature_type: standard Med3DINO labels ``cls``, ``avg_pool``, or ``multilayer``;
        external-method single-bundle evaluations may also use method-specific labels.

Example:
    >>> get_cross_modal_manifest_path("amos22_ct_mr", "sampled", "core")
"""

from pathlib import Path
from reform3d.core.layout import active_output_paths
from typing import Dict, Tuple

from reform3d.core.paths import (
    PROJECT_ROOT,
    get_reform3d_cache_root,
    get_reform3d_module_manifest_roots,
    get_reform3d_module_manifest_root,
    get_reform3d_module_result_root,
    get_reform3d_module_result_roots,
    resolve_reform3d_module_manifest_path,
)
from reform3d.core.seed_context import apply_active_seed_to_path

CROSS_MODAL_MODULE_NAME = "cross_modal_alignment"
CROSS_MODAL_ROOT = Path(__file__).resolve().parent
CROSS_MODAL_OUTPUTS = get_reform3d_module_result_root(CROSS_MODAL_MODULE_NAME)
CROSS_MODAL_MANIFESTS = get_reform3d_module_manifest_root(CROSS_MODAL_MODULE_NAME)
CROSS_MODAL_CACHES = get_reform3d_cache_root()

TOTALSEGMENTER_CROSS_MODAL_DATASET = "totalsegmenter_ct_mr_anchor"
MMWHS_CROSS_MODAL_DATASET = "mmwhs_ct_mr"
AMOS22_CROSS_MODAL_DATASET = "amos22_ct_mr"
CHAOS_CROSS_MODAL_DATASET = "chaos_ct_mr"
CT_PET_CROSS_MODAL_DATASET = "ct_pet_alignment"
CROSS_MODAL_DATASETS = (
    TOTALSEGMENTER_CROSS_MODAL_DATASET,
    MMWHS_CROSS_MODAL_DATASET,
    AMOS22_CROSS_MODAL_DATASET,
    CHAOS_CROSS_MODAL_DATASET,
    CT_PET_CROSS_MODAL_DATASET,
)
DATASET_NAME_ALIASES = {
    "amos_ct_mr": AMOS22_CROSS_MODAL_DATASET,
    "amos22": AMOS22_CROSS_MODAL_DATASET,
    "ct_pet": CT_PET_CROSS_MODAL_DATASET,
    "pet_ct": CT_PET_CROSS_MODAL_DATASET,
    "autopet_hecktor_ct_pet": CT_PET_CROSS_MODAL_DATASET,
}

DEFAULT_CROSS_MODAL_DATASET = AMOS22_CROSS_MODAL_DATASET
DEFAULT_MANIFEST_VARIANT = "core"
DEFAULT_REQUIRED_MODALITIES = ("ct", "mr")
DEFAULT_FEATURE_TYPE = "cls"
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
PRIMARY_METRICS_FILE_NAME = "primary_metrics.json"
CHECKPOINT_COMPARISON_JSON_BY_FEATURE = {
    "cls": "checkpoint_comparison.json",
    "avg_pool": "checkpoint_comparison_avg_pool.json",
    "multilayer": "checkpoint_comparison_multilayer.json",
}
CHECKPOINT_COMPARISON_CSV_BY_FEATURE = {
    "cls": "checkpoint_comparison.csv",
    "avg_pool": "checkpoint_comparison_avg_pool.csv",
    "multilayer": "checkpoint_comparison_multilayer.csv",
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


def normalize_feature_type(feature_type: str = DEFAULT_FEATURE_TYPE) -> str:
    normalized = FEATURE_TYPE_ALIASES.get(str(feature_type).strip().lower())
    if normalized is None:
        raise ValueError(f"Unknown feature type: {feature_type}")
    return normalized


def normalize_manifest_variant(manifest_variant: str = DEFAULT_MANIFEST_VARIANT) -> str:
    return MANIFEST_VARIANT_ALIASES.get(manifest_variant, manifest_variant)


def normalize_cross_modal_dataset_name(dataset_name: str = DEFAULT_CROSS_MODAL_DATASET) -> str:
    normalized = str(dataset_name).strip().lower()
    return DATASET_NAME_ALIASES.get(normalized, normalized)


def get_cross_modal_manifest_dir(
    dataset_name: str = DEFAULT_CROSS_MODAL_DATASET,
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> Path:
    return CROSS_MODAL_MANIFESTS / normalize_cross_modal_dataset_name(dataset_name) / normalize_manifest_variant(manifest_variant)


def get_cross_modal_manifest_path(
    dataset_name: str = DEFAULT_CROSS_MODAL_DATASET,
    manifest_kind: str = "sampled",
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> Path:
    if manifest_kind not in MANIFEST_FILE_NAMES:
        raise ValueError(f"Unknown manifest kind: {manifest_kind}")
    dataset_name = normalize_cross_modal_dataset_name(dataset_name)
    return resolve_reform3d_module_manifest_path(
        CROSS_MODAL_MODULE_NAME,
        Path(dataset_name) / normalize_manifest_variant(manifest_variant) / MANIFEST_FILE_NAMES[manifest_kind],
    )


def resolve_cross_modal_manifest_path(manifest_path: Path | str) -> Path:
    return resolve_reform3d_module_manifest_path(CROSS_MODAL_MODULE_NAME, manifest_path)


def get_cache_root(
    dataset_name: str = DEFAULT_CROSS_MODAL_DATASET,
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> Path:
    return CROSS_MODAL_CACHES / CROSS_MODAL_MODULE_NAME / normalize_cross_modal_dataset_name(dataset_name) / normalize_manifest_variant(manifest_variant)


def get_crop_cache_dir(
    dataset_name: str = DEFAULT_CROSS_MODAL_DATASET,
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
    crop_size: int = 96,
) -> Path:
    return get_cache_root(dataset_name, manifest_variant) / f"crop{int(crop_size)}" / "organ_crop_cache"


def get_output_paths(
    dataset_name: str = DEFAULT_CROSS_MODAL_DATASET,
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> Dict[str, Path]:
    active = active_output_paths("cross_modal_alignment")
    if active is not None:
        return active
    dataset_name = normalize_cross_modal_dataset_name(dataset_name)
    manifest_variant = normalize_manifest_variant(manifest_variant)
    root = apply_active_seed_to_path(CROSS_MODAL_OUTPUTS / dataset_name / manifest_variant)
    return {
        "root": root,
        "results": root / "results",
        "figures": root / "figures",
        "features": root / "features",
        "logs": root / "logs",
        "cache_root": get_cache_root(dataset_name, manifest_variant),
        "crop_cache_root": get_cache_root(dataset_name, manifest_variant),
        "manifests": get_cross_modal_manifest_dir(dataset_name, manifest_variant),
    }


def get_output_root_candidates(
    dataset_name: str = DEFAULT_CROSS_MODAL_DATASET,
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> tuple[Path, ...]:
    dataset_name = normalize_cross_modal_dataset_name(dataset_name)
    manifest_variant = normalize_manifest_variant(manifest_variant)
    candidates = tuple(
        root / dataset_name / manifest_variant
        for root in get_reform3d_module_result_roots(CROSS_MODAL_MODULE_NAME)
    )
    active = apply_active_seed_to_path(CROSS_MODAL_OUTPUTS / dataset_name / manifest_variant)
    return tuple(dict.fromkeys((active, *candidates)))


def get_feature_root_candidates(
    dataset_name: str = DEFAULT_CROSS_MODAL_DATASET,
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> tuple[Path, ...]:
    return tuple(root / "features" for root in get_output_root_candidates(dataset_name, manifest_variant))


def get_results_root_candidates(
    dataset_name: str = DEFAULT_CROSS_MODAL_DATASET,
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> tuple[Path, ...]:
    return tuple(root / "results" for root in get_output_root_candidates(dataset_name, manifest_variant))


def get_existing_feature_root(
    dataset_name: str = DEFAULT_CROSS_MODAL_DATASET,
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> Path:
    for path in get_feature_root_candidates(dataset_name, manifest_variant):
        if path.exists():
            return path
    return get_output_paths(dataset_name, manifest_variant)["features"]


def get_existing_results_root(
    dataset_name: str = DEFAULT_CROSS_MODAL_DATASET,
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> Path:
    for path in get_results_root_candidates(dataset_name, manifest_variant):
        if path.exists():
            return path
    return get_output_paths(dataset_name, manifest_variant)["results"]


def get_checkpoint_feature_dir(base_dir: Path, checkpoint_name: str, feature_type: str) -> Path:
    return Path(base_dir) / checkpoint_name / normalize_feature_type(feature_type)


def get_feature_embedding_file_name(feature_type: str = DEFAULT_FEATURE_TYPE) -> str:
    feature_type = normalize_feature_type(feature_type)
    return FEATURE_EMBEDDING_FILE_NAMES[feature_type]


def get_feature_summary_file_name(feature_type: str = DEFAULT_FEATURE_TYPE) -> str:
    embedding_name = get_feature_embedding_file_name(feature_type)
    stem, suffix = embedding_name.rsplit(".", 1)
    return f"{stem}_summary.{suffix.replace('npz', 'json')}"


def get_checkpoint_feature_npz_path(base_dir: Path, checkpoint_name: str, feature_type: str) -> Path:
    feature_dir = get_checkpoint_feature_dir(base_dir, checkpoint_name, feature_type)
    return feature_dir / get_feature_embedding_file_name(feature_type)


def get_checkpoint_metrics_dir(base_dir: Path, checkpoint_name: str, feature_type: str = DEFAULT_FEATURE_TYPE) -> Path:
    feature_type = normalize_feature_type(feature_type)
    checkpoint_dir = Path(base_dir) / checkpoint_name
    if feature_type == "cls":
        return checkpoint_dir
    return checkpoint_dir / feature_type


def get_checkpoint_metrics_path(base_dir: Path, checkpoint_name: str, feature_type: str = DEFAULT_FEATURE_TYPE) -> Path:
    return get_checkpoint_metrics_dir(base_dir, checkpoint_name, feature_type) / PRIMARY_METRICS_FILE_NAME


def resolve_existing_checkpoint_feature_npz_path(
    dataset_name: str,
    manifest_variant: str,
    checkpoint_name: str,
    feature_type: str = DEFAULT_FEATURE_TYPE,
) -> Path:
    for base_dir in get_feature_root_candidates(dataset_name, manifest_variant):
        candidate = get_checkpoint_feature_npz_path(base_dir, checkpoint_name, feature_type)
        if candidate.exists():
            return candidate
    return get_checkpoint_feature_npz_path(
        get_output_paths(dataset_name, manifest_variant)["features"],
        checkpoint_name,
        feature_type,
    )


def resolve_existing_checkpoint_metrics_path(
    dataset_name: str,
    manifest_variant: str,
    checkpoint_name: str,
    feature_type: str = DEFAULT_FEATURE_TYPE,
) -> Path:
    for base_dir in get_results_root_candidates(dataset_name, manifest_variant):
        candidate = get_checkpoint_metrics_path(base_dir, checkpoint_name, feature_type)
        if candidate.exists():
            return candidate
    return get_checkpoint_metrics_path(
        get_output_paths(dataset_name, manifest_variant)["results"],
        checkpoint_name,
        feature_type,
    )


def get_checkpoint_comparison_json_path(base_dir: Path, feature_type: str = DEFAULT_FEATURE_TYPE) -> Path:
    feature_type = normalize_feature_type(feature_type)
    file_name = CHECKPOINT_COMPARISON_JSON_BY_FEATURE.get(feature_type, f"checkpoint_comparison_{feature_type}.json")
    return Path(base_dir) / file_name


def get_checkpoint_comparison_csv_path(base_dir: Path, feature_type: str = DEFAULT_FEATURE_TYPE) -> Path:
    feature_type = normalize_feature_type(feature_type)
    file_name = CHECKPOINT_COMPARISON_CSV_BY_FEATURE.get(feature_type, f"checkpoint_comparison_{feature_type}.csv")
    return Path(base_dir) / file_name


def get_dataset_name_from_manifest_path(
    manifest_path: Path,
    fallback: str = DEFAULT_CROSS_MODAL_DATASET,
) -> str:
    manifest_path = Path(manifest_path)
    for root in get_reform3d_module_manifest_roots(CROSS_MODAL_MODULE_NAME):
        try:
            relative_path = manifest_path.resolve().relative_to(root.resolve())
        except ValueError:
            continue
        if len(relative_path.parts) >= 1:
            return normalize_cross_modal_dataset_name(relative_path.parts[0])
    return normalize_cross_modal_dataset_name(fallback)


def get_manifest_variant_from_manifest_path(
    manifest_path: Path,
    fallback: str = DEFAULT_MANIFEST_VARIANT,
) -> str:
    manifest_path = Path(manifest_path)
    for root in get_reform3d_module_manifest_roots(CROSS_MODAL_MODULE_NAME):
        try:
            relative_path = manifest_path.resolve().relative_to(root.resolve())
        except ValueError:
            continue
        if len(relative_path.parts) >= 2:
            return normalize_manifest_variant(relative_path.parts[1])
    return normalize_manifest_variant(fallback)


def ensure_output_directories(
    dataset_name: str = DEFAULT_CROSS_MODAL_DATASET,
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> None:
    paths = get_output_paths(dataset_name, manifest_variant)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
