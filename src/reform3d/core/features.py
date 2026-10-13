"""Shared feature-bundle helpers for REFORM-3D evaluation.

The helpers normalize feature-type names and resolve standard embedding bundle
paths for modules that consume Cross-Modal organ features.

Example:
    >>> get_feature_embedding_file_name("avg-pool")
    'organ_avg_pool_embeddings.npz'
"""

from __future__ import annotations

from pathlib import Path

from .paths import (
    get_reform3d_module_result_roots,
    get_reform3d_module_results_root,
    resolve_reform3d_module_result_path,
)


DEFAULT_FEATURE_TYPE = "cls"
FEATURE_TYPES = ("cls", "avg_pool", "multilayer")
FEATURE_TYPE_ALIASES = {
    "cls": "cls",
    "avg_pool": "avg_pool",
    "avgpool": "avg_pool",
    "avg-pool": "avg_pool",
    "multilayer": "multilayer",
    "multi_layer": "multilayer",
    "multi-layer": "multilayer",
}
FEATURE_EMBEDDING_FILE_NAMES = {
    "cls": "organ_cls_embeddings.npz",
    "avg_pool": "organ_avg_pool_embeddings.npz",
    "multilayer": "organ_multilayer_embeddings.npz",
}
CROSS_MODAL_MODULE_NAME = "cross_modal_alignment"
CROSS_MODAL_OUTPUTS = get_reform3d_module_results_root(CROSS_MODAL_MODULE_NAME)


def normalize_feature_type(feature_type: str = DEFAULT_FEATURE_TYPE) -> str:
    normalized = FEATURE_TYPE_ALIASES.get(str(feature_type).strip().lower())
    if normalized is None:
        raise ValueError(f"Unknown feature type: {feature_type}")
    return normalized


def normalize_manifest_variant(manifest_variant: str = "core") -> str:
    aliases = {
        "core": "core",
        "default": "core",
        "shared_organs": "core",
        "paired": "paired",
        "paired_core": "paired",
    }
    return aliases.get(str(manifest_variant).strip().lower(), manifest_variant)


def get_output_root_candidates(
    dataset_name: str,
    manifest_variant: str = "core",
) -> tuple[Path, ...]:
    manifest_variant = normalize_manifest_variant(manifest_variant)
    return tuple(
        root / dataset_name / manifest_variant
        for root in get_reform3d_module_result_roots(CROSS_MODAL_MODULE_NAME)
    )


def get_feature_root_candidates(
    dataset_name: str,
    manifest_variant: str = "core",
) -> tuple[Path, ...]:
    return tuple(root / "features" for root in get_output_root_candidates(dataset_name, manifest_variant))


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
        CROSS_MODAL_OUTPUTS / dataset_name / normalize_manifest_variant(manifest_variant) / "features",
        checkpoint_name,
        feature_type,
    )


def get_upstream_feature_npz_path(
    dataset_name: str,
    manifest_variant: str,
    checkpoint_name: str,
    feature_type: str = DEFAULT_FEATURE_TYPE,
) -> Path:
    feature_type = normalize_feature_type(feature_type)
    return resolve_reform3d_module_result_path(
        CROSS_MODAL_MODULE_NAME,
        None,
        Path(dataset_name)
        / normalize_manifest_variant(manifest_variant)
        / "features"
        / checkpoint_name
        / feature_type
        / FEATURE_EMBEDDING_FILE_NAMES[feature_type],
    )


def load_organ_feature_bundle(path, expected_ids, *, dtype=None, exact=False):
    """Load finite, uniquely identified embeddings with complete cohort coverage."""
    import numpy as np
    expected = list(map(str, expected_ids))
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("Expected sample IDs must be nonempty and unique")
    with np.load(path, allow_pickle=True) as payload:
        if "features" not in payload:
            raise ValueError(f"Embedding bundle {path} requires features")
        features = np.asarray(payload["features"])
        if features.ndim != 2 or min(features.shape) < 1:
            raise ValueError(f"Features must be a nonempty [samples, dimensions] matrix: {features.shape}")
        if not np.issubdtype(features.dtype, np.number) or np.iscomplexobj(features) or not np.isfinite(features).all():
            raise ValueError("Features must contain finite real numbers")
        if dtype is not None:
            with np.errstate(over="ignore", invalid="ignore"):
                features = features.astype(dtype, copy=False)
            if not np.isfinite(features).all():
                raise ValueError("Features overflow the requested numeric dtype")
        if "sample_ids" in payload:
            raw_ids = payload["sample_ids"]
            if raw_ids.ndim != 1:
                raise ValueError("sample_ids must be one-dimensional")
            ids = list(map(str, raw_ids.tolist()))
        elif exact:
            raise ValueError("Public feature bundles require explicit sample_ids")
        else:
            ids = expected
    if len(ids) != features.shape[0] or len(set(ids)) != len(ids) or any(not value for value in ids):
        raise ValueError("Feature row count and unique sample IDs must agree")
    missing = set(expected) - set(ids)
    extra = set(ids) - set(expected)
    if missing or (exact and extra):
        raise ValueError(f"Embedding cohort mismatch: missing={sorted(missing)}, unexpected={sorted(extra) if exact else []}")
    return {sample_id: features[index] for index, sample_id in enumerate(ids)}


__all__ = [
    "load_organ_feature_bundle",
    "DEFAULT_FEATURE_TYPE",
    "FEATURE_EMBEDDING_FILE_NAMES",
    "FEATURE_TYPES",
    "get_checkpoint_feature_dir",
    "get_checkpoint_feature_npz_path",
    "get_feature_embedding_file_name",
    "get_feature_root_candidates",
    "get_feature_summary_file_name",
    "get_upstream_feature_npz_path",
    "normalize_feature_type",
    "resolve_existing_checkpoint_feature_npz_path",
]
