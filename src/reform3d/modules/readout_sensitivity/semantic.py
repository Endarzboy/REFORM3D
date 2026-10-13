"""Semantic-transfer readout-sensitivity for one (dataset, checkpoint, feature, seed).

Reuses the frozen ``controlled_embedding_bundle.npz`` features and the production
semantic-target derivation. For every readout variant it recomputes the full
inter-variant transfer matrix over the shared folds and records the six directed
off-diagonal transfers plus the six-direction mean.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from reform3d.modules.acquisition_robustness.perturbation_robustness_analysis import (
    _build_matched_split_indices,
)
from reform3d.modules.acquisition_robustness.semantic_labels import (
    compute_semantic_task_labels,
)

from . import config
from .readouts import LOGISTIC, transfer_once


def _scalar_text(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        if value.size == 1:
            return str(value.reshape(-1)[0])
    return str(value)


def load_controlled_bundle(bundle_path: Path) -> Dict[str, Any]:
    """Load features-by-variant, variant order, reference, and case ids."""
    with np.load(bundle_path, allow_pickle=True) as bundle:
        feature_keys = sorted(key for key in bundle.files if key.startswith("features__"))
        features_by_variant = {
            key.replace("features__", "", 1): np.asarray(bundle[key], dtype=np.float32)
            for key in feature_keys
        }
        if "variant_names" in bundle.files:
            variant_order = [_scalar_text(v) for v in np.asarray(bundle["variant_names"]).tolist()]
        elif "spacing_names" in bundle.files:
            variant_order = [_scalar_text(v) for v in np.asarray(bundle["spacing_names"]).tolist()]
        else:
            variant_order = sorted(features_by_variant)
        if "reference_variant_name" in bundle.files:
            reference_variant = _scalar_text(bundle["reference_variant_name"])
        elif "reference_spacing_name" in bundle.files:
            reference_variant = _scalar_text(bundle["reference_spacing_name"])
        else:
            reference_variant = variant_order[0]
        case_ids = (
            [_scalar_text(v) for v in np.asarray(bundle["case_ids"]).tolist()]
            if "case_ids" in bundle.files
            else []
        )
    return {
        "features_by_variant": features_by_variant,
        "variant_order": [v for v in variant_order if v in features_by_variant],
        "reference_variant": reference_variant,
        "case_ids": case_ids,
    }


@lru_cache(maxsize=None)
def _semantic_labels_full(dataset: str, seed: int) -> Tuple[np.ndarray, Tuple[str, ...], Tuple[str, ...]]:
    """Compute (labels_full, paths, label_names) via the production derivation.

    Cached per (dataset, seed) because the derivation reads the shared label
    cache. Deterministic; independent of the readout.
    """
    manifest_path = config.semantic_acquisition_manifest_path(dataset, seed)
    labels_full, paths, metadata = compute_semantic_task_labels(
        manifest_path,
        mode="auto",
        filter_dataset=dataset,
        cache_dir=config.semantic_label_cache_dir(dataset, seed),
    )
    label_names = tuple(metadata.get("organ_names", []) if metadata else [])
    return np.asarray(labels_full), tuple(str(p) for p in paths), label_names


def align_semantic_labels(
    dataset: str,
    seed: int,
    case_ids: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Align production semantic targets to the bundle case order.

    Returns (aligned_labels, valid_positions, label_names). ``valid_positions``
    indexes the bundle rows that carry semantic labels, in bundle order.
    """
    labels_full, paths, label_names = _semantic_labels_full(dataset, int(seed))
    if labels_full.ndim != 2 or labels_full.shape[1] == 0:
        raise ValueError(f"No informative semantic labels for {dataset} seed {seed}")
    def case_key(path: str) -> str:
        if dataset == "kits23":
            match = re.search(r"(case_\d+)", str(path))
            if match is None:
                raise ValueError(f"Cannot derive KiTS23 case ID from path: {path}")
            return match.group(1)
        return str(path)

    path_to_index: Dict[str, int] = {}
    for index, path in enumerate(paths):
        key = case_key(path)
        if key in path_to_index:
            raise ValueError(f"Duplicate semantic case key for {dataset}: {key}")
        path_to_index[key] = index
    valid_positions: List[int] = []
    aligned: List[np.ndarray] = []
    for position, case_id in enumerate(case_ids):
        index = path_to_index.get(case_key(str(case_id)))
        if index is None:
            continue
        valid_positions.append(position)
        aligned.append(labels_full[index])
    if len(aligned) < 10:
        raise ValueError(
            f"Insufficient aligned semantic cases for {dataset} seed {seed}: {len(aligned)}"
        )
    return np.stack(aligned), np.asarray(valid_positions, dtype=np.int64), list(label_names)


def compute_transfer_for_readout(
    features_by_variant: Dict[str, np.ndarray],
    semantic_labels: np.ndarray,
    variant_order: Sequence[str],
    fold_indices: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    readout: str,
    logistic_c: float,
    random_state: int,
) -> Dict[str, Any]:
    """Full inter-variant transfer matrix + directed transfers for one readout."""
    variant_names = [v for v in variant_order if v in features_by_variant]
    n_labels = int(semantic_labels.shape[1])
    transfer_matrix: Dict[str, Dict[str, float]] = {name: {} for name in variant_names}
    per_transfer: Dict[str, Dict[str, Any]] = {}

    for train_variant in variant_names:
        for test_variant in variant_names:
            fold_scores: List[float] = []
            label_lists: Dict[int, List[float]] = {idx: [] for idx in range(n_labels)}
            for train_idx, test_idx in fold_indices:
                score, label_scores = transfer_once(
                    features_by_variant[train_variant][train_idx],
                    semantic_labels[train_idx],
                    features_by_variant[test_variant][test_idx],
                    semantic_labels[test_idx],
                    readout=readout,
                    logistic_c=logistic_c if logistic_c is not None else 1.0,
                    random_state=random_state,
                    max_iter=config.SEMANTIC_LOGISTIC_MAX_ITER,
                    min_class_count=config.SEMANTIC_MIN_CLASS_COUNT,
                )
                if score > 0:
                    fold_scores.append(score)
                for label_idx, label_score in label_scores.items():
                    label_lists[label_idx].append(label_score)
            value = float(np.mean(fold_scores)) if fold_scores else 0.0
            transfer_matrix[train_variant][test_variant] = value
            if train_variant != test_variant:
                per_transfer[f"{train_variant}->{test_variant}"] = {
                    "train_variant": train_variant,
                    "test_variant": test_variant,
                    "value": value,
                    "fold_scores": [float(s) for s in fold_scores],
                    "per_label_mean": {
                        str(label_idx): float(np.mean(scores))
                        for label_idx, scores in label_lists.items()
                        if scores
                    },
                }

    directed = [
        transfer_matrix[train][test]
        for train in variant_names
        for test in variant_names
        if train != test
    ]
    in_variant = {variant: transfer_matrix[variant][variant] for variant in variant_names}
    six_direction_mean = float(np.mean(directed)) if directed else 0.0
    in_variant_mean = float(np.mean(list(in_variant.values()))) if in_variant else 0.0
    return {
        "transfer_matrix": transfer_matrix,
        "per_transfer": per_transfer,
        "in_variant_accuracies": in_variant,
        "six_direction_mean": six_direction_mean,
        "n_directed_transfers": len(directed),
        "transfer_gap": in_variant_mean - six_direction_mean,
    }


def evaluate_semantic_cell(
    dataset: str,
    checkpoint: str,
    feature_type: str,
    seed: int,
    logistic_c_values: Sequence[float] = config.LOGISTIC_C_VALUES,
) -> Dict[str, Any]:
    """Evaluate all readout variants for one semantic (dataset, ckpt, feat, seed) cell."""
    bundle_path = config.controlled_bundle_path(dataset, checkpoint, feature_type, seed)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Controlled bundle not found: {bundle_path}")
    bundle = load_controlled_bundle(bundle_path)
    features_by_variant = bundle["features_by_variant"]
    variant_order = bundle["variant_order"]
    case_ids = bundle["case_ids"]

    for name, features in features_by_variant.items():
        if not np.isfinite(features).all():
            raise ValueError(f"Non-finite features in {bundle_path} variant {name}")

    semantic_labels, valid_positions, label_names = align_semantic_labels(dataset, seed, case_ids)
    filtered = {v: f[valid_positions] for v, f in features_by_variant.items()}
    aligned_case_ids = [str(case_ids[int(p)]) for p in valid_positions]

    fold_indices = _build_matched_split_indices(
        semantic_labels, n_splits=config.SEMANTIC_N_CV_SPLITS, random_state=int(seed)
    )
    fold_sizes = [[int(len(tr)), int(len(te))] for tr, te in fold_indices]

    readouts: Dict[str, Any] = {}
    for logistic_c in logistic_c_values:
        key = f"logistic_C{logistic_c:g}"
        readouts[key] = compute_transfer_for_readout(
            filtered, semantic_labels, variant_order, fold_indices,
            readout=LOGISTIC, logistic_c=float(logistic_c), random_state=int(seed),
        )
    readouts["nearest_centroid"] = compute_transfer_for_readout(
        filtered, semantic_labels, variant_order, fold_indices,
        readout="nearest_centroid", logistic_c=1.0, random_state=int(seed),
    )

    return {
        "dataset": dataset,
        "checkpoint": checkpoint,
        "feature_type": feature_type,
        "seed": int(seed),
        "bundle_path": str(bundle_path),
        "variant_order": list(variant_order),
        "reference_variant": bundle["reference_variant"],
        "label_names": label_names,
        "n_labels": int(semantic_labels.shape[1]),
        "n_semantic_cases": int(semantic_labels.shape[0]),
        "n_cv_splits": config.SEMANTIC_N_CV_SPLITS,
        "fold_sizes": fold_sizes,
        "sample_ids": aligned_case_ids,
        "readouts": readouts,
    }


def production_semantic_baseline(
    dataset: str,
    checkpoint: str,
    feature_type: str,
    seed: int,
) -> float:
    """Freshly recompute the production logistic (C=1) six-direction mean.

    Independent code path (``compute_matched_semantic_transfer``) used only to
    gate the readout-sensitivity C=1 result. Anchored to current production
    because the stored JSON artifacts are stale (see module docs).
    """
    from reform3d.modules.acquisition_robustness.perturbation_robustness_analysis import (
        compute_matched_semantic_transfer,
    )

    bundle = load_controlled_bundle(
        config.controlled_bundle_path(dataset, checkpoint, feature_type, seed)
    )
    semantic_labels, valid_positions, label_names = align_semantic_labels(
        dataset, seed, bundle["case_ids"]
    )
    filtered = {v: f[valid_positions] for v, f in bundle["features_by_variant"].items()}
    result = compute_matched_semantic_transfer(
        filtered,
        semantic_labels,
        label_names,
        checkpoint,
        feature_type,
        n_cv_splits=config.SEMANTIC_N_CV_SPLITS,
        random_state=int(seed),
    )
    return float(result["cross_variant_accuracy"])
