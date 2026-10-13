#!/usr/bin/env python
"""Post-hoc uncertainty analysis for Acquisition Robustness controlled perturbations.

This helper consumes existing ``controlled_embedding_bundle.npz``
artifacts and does not rerun feature extraction. It reports case-level compact
drift mean/standard deviation, case-bootstrap CKA confidence intervals, and
matched semantic-transfer uncertainty recomputed from saved bundle features and
the configured semantic targets.

Examples:
    python -m reform3d.modules.acquisition_robustness.controlled_uncertainty_analysis \
        --analysis-names abdomenatlas kits23 imagecas totalsegmentermri jhu_stroke \
        --feature-types cls avg_pool multilayer \
        --bootstrap-resamples 1000

    python -m reform3d.modules.acquisition_robustness.controlled_uncertainty_analysis \
        --analysis-names imagecas \
        --output-json /tmp/acquisition_controlled_uncertainty_imagecas.json
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is present in the project env.
    def tqdm(iterable, **_: Any):  # type: ignore[no-redef]
        return iterable

try:
    from .config import DEFAULT_MANIFEST_VARIANT, ACQUISITION_OUTPUTS, get_cache_root, get_acquisition_manifest_path
except ImportError:  # pragma: no cover - supports direct script execution.
    from config import DEFAULT_MANIFEST_VARIANT, ACQUISITION_OUTPUTS, get_cache_root, get_acquisition_manifest_path


def _compute_semantic_task_labels(*args: Any, **kwargs: Any) -> tuple[np.ndarray, list[str], Dict[str, Any]]:
    try:
        from .semantic_labels import compute_semantic_task_labels
    except ImportError:  # pragma: no cover - supports direct script execution.
        from semantic_labels import compute_semantic_task_labels
    return compute_semantic_task_labels(*args, **kwargs)


ACQUISITION_ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = ACQUISITION_OUTPUTS
DEFAULT_OUTPUT_DIR = OUTPUTS_ROOT / "statistical_rigor"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_DIR / "acquisition_controlled_uncertainty.json"
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "acquisition_controlled_uncertainty.csv"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "analysis_name",
        "checkpoint_name",
        "feature_type",
        "status",
        "n_cases",
        "reference_variant",
        "compact_drift_mean",
        "compact_drift_std",
        "compact_cka_mean",
        "compact_cka_ci_lower",
        "compact_cka_ci_upper",
        "compact_cka_ci_status",
        "compact_semantic_transfer_mean",
        "compact_semantic_transfer_bootstrap_std",
        "compact_semantic_transfer_ci_lower",
        "compact_semantic_transfer_ci_upper",
        "compact_semantic_transfer_ci_status",
        "n_semantic_cases",
        "n_semantic_labels",
        "semantic_manifest_path",
        "semantic_transfer_bootstrap_method",
        "semantic_transfer_bootstrap_valid_resamples",
        "cka_method",
        "bootstrap_resamples",
        "skip_reason",
        "metric_json",
        "bundle_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _scalar_text(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        if value.size == 1:
            return str(value.reshape(-1)[0])
    return str(value)


def _rowwise_cosine_distance(source_features: np.ndarray, target_features: np.ndarray) -> np.ndarray:
    source = np.asarray(source_features, dtype=np.float64)
    target = np.asarray(target_features, dtype=np.float64)
    source_norm = np.clip(np.linalg.norm(source, axis=1), a_min=1e-12, a_max=None)
    target_norm = np.clip(np.linalg.norm(target, axis=1), a_min=1e-12, a_max=None)
    return 1.0 - (np.sum(source * target, axis=1) / (source_norm * target_norm))


def _center_features(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    return values - values.mean(axis=0, keepdims=True)


def _linear_cka_gram(source_features: np.ndarray, target_features: np.ndarray) -> float:
    source_centered = _center_features(source_features)
    target_centered = _center_features(target_features)
    numerator = np.linalg.norm(source_centered @ target_centered.T, ord="fro") ** 2
    source_scale = np.linalg.norm(source_centered @ source_centered.T, ord="fro")
    target_scale = np.linalg.norm(target_centered @ target_centered.T, ord="fro")
    denominator = source_scale * target_scale
    if denominator <= 1e-12:
        return 0.0
    return float(numerator / denominator)


def _linear_cka_feature(source_features: np.ndarray, target_features: np.ndarray) -> float:
    source_centered = _center_features(source_features)
    target_centered = _center_features(target_features)
    source_covariance = source_centered.T @ source_centered
    target_covariance = target_centered.T @ target_centered
    numerator = float(np.sum(source_covariance * target_covariance))
    source_scale = np.linalg.norm(source_covariance, ord="fro")
    target_scale = np.linalg.norm(target_covariance, ord="fro")
    denominator = source_scale * target_scale
    if denominator <= 1e-12:
        return 0.0
    return float(numerator / denominator)


def _linear_cka(source_features: np.ndarray, target_features: np.ndarray, method: str = "auto") -> float:
    if method == "gram":
        return _linear_cka_gram(source_features, target_features)
    if method == "feature":
        return _linear_cka_feature(source_features, target_features)
    if method == "auto":
        source = np.asarray(source_features)
        target = np.asarray(target_features)
        n_cases = max(int(source.shape[0]), int(target.shape[0]))
        feature_dim = max(int(source.shape[1]), int(target.shape[1]))
        if feature_dim <= n_cases:
            return _linear_cka_feature(source_features, target_features)
        return _linear_cka_gram(source_features, target_features)
    raise ValueError(f"Unsupported CKA method: {method}")


def _compact_cka(
    features_by_variant: Dict[str, np.ndarray],
    variant_order: Sequence[str],
    *,
    method: str = "auto",
) -> float | None:
    values = []
    for source_index, source_variant in enumerate(variant_order):
        for target_variant in variant_order[source_index + 1 :]:
            if source_variant not in features_by_variant or target_variant not in features_by_variant:
                continue
            values.append(_linear_cka(features_by_variant[source_variant], features_by_variant[target_variant], method=method))
    if not values:
        return None
    return float(np.mean(values))


def _compact_cka_from_payload(payload: Dict[str, Any], variant_order: Sequence[str]) -> float | None:
    matrix = payload.get("cka_matrix") or {}
    values = []
    for source_index, source_variant in enumerate(variant_order):
        for target_variant in variant_order[source_index + 1 :]:
            row = matrix.get(source_variant) or {}
            value = row.get(target_variant)
            if value is not None:
                values.append(float(value))
    if not values:
        return None
    return float(np.mean(values))


def _bootstrap_compact_cka(
    features_by_variant: Dict[str, np.ndarray],
    variant_order: Sequence[str],
    bootstrap_resamples: int,
    seed: int,
    method: str,
) -> Dict[str, Any]:
    compact_mean = _compact_cka(features_by_variant, variant_order, method=method)
    if compact_mean is None:
        return {"mean": None, "ci_lower": None, "ci_upper": None}
    first_variant = next(iter(features_by_variant.values()))
    n_cases = int(first_variant.shape[0])
    if bootstrap_resamples <= 0 or n_cases <= 1:
        return {"mean": compact_mean, "ci_lower": None, "ci_upper": None}

    rng = np.random.default_rng(int(seed))
    bootstrap_values = np.empty(int(bootstrap_resamples), dtype=np.float64)
    for resample_index in range(int(bootstrap_resamples)):
        indices = rng.integers(0, n_cases, size=n_cases)
        resampled = {variant: features[indices] for variant, features in features_by_variant.items()}
        value = _compact_cka(resampled, variant_order, method=method)
        bootstrap_values[resample_index] = np.nan if value is None else float(value)
    bootstrap_values = bootstrap_values[np.isfinite(bootstrap_values)]
    if bootstrap_values.size == 0:
        return {"mean": compact_mean, "ci_lower": None, "ci_upper": None}
    return {
        "mean": compact_mean,
        "ci_lower": float(np.percentile(bootstrap_values, 2.5)),
        "ci_upper": float(np.percentile(bootstrap_values, 97.5)),
    }


def _build_matched_split_indices(
    semantic_labels: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build shared case-disjoint folds for matched semantic evaluation."""
    n_samples = len(semantic_labels)
    effective_splits = min(int(n_splits), int(n_samples))
    if effective_splits < 2:
        raise ValueError("Matched semantic evaluation requires at least two samples")

    stratify_labels = None
    if semantic_labels.ndim == 2 and semantic_labels.shape[1] == 1:
        stratify_labels = semantic_labels[:, 0]
    elif semantic_labels.ndim == 2 and np.all(semantic_labels.sum(axis=1) == 1):
        stratify_labels = np.argmax(semantic_labels, axis=1)

    if stratify_labels is not None:
        unique_classes, counts = np.unique(stratify_labels, return_counts=True)
        if len(unique_classes) > 1 and counts.min() >= effective_splits:
            splitter = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)
            return [
                (train_idx, test_idx)
                for train_idx, test_idx in splitter.split(np.zeros(n_samples), stratify_labels)
            ]

    splitter = KFold(n_splits=effective_splits, shuffle=True, random_state=random_state)
    return [(train_idx, test_idx) for train_idx, test_idx in splitter.split(np.zeros(n_samples))]


def _train_and_evaluate_multilabel_transfer_once(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    random_state: int = 42,
    min_class_count: int = 2,
) -> tuple[float, Dict[int, float]]:
    """Train one classifier per label on one variant and evaluate on another."""
    if train_labels.ndim != 2 or test_labels.ndim != 2 or train_labels.shape[1] == 0:
        return 0.0, {}

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)

    per_label_scores: Dict[int, float] = {}
    valid_scores: list[float] = []
    for label_idx in range(train_labels.shape[1]):
        y_train = train_labels[:, label_idx]
        y_test = test_labels[:, label_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        pos_train = int(y_train.sum())
        neg_train = int(len(y_train) - pos_train)
        pos_test = int(y_test.sum())
        neg_test = int(len(y_test) - pos_test)
        if min(pos_train, neg_train, pos_test, neg_test) < min_class_count:
            continue

        clf = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state,
            n_jobs=1,
        )
        try:
            clf.fit(train_scaled, y_train)
            predictions = clf.predict(test_scaled)
            score = float(balanced_accuracy_score(y_test, predictions))
            per_label_scores[label_idx] = score
            valid_scores.append(score)
        except Exception:
            continue

    if not valid_scores:
        return 0.0, {}
    return float(np.mean(valid_scores)), per_label_scores


def _compute_matched_semantic_transfer(
    features_by_variant: Dict[str, np.ndarray],
    semantic_labels: np.ndarray,
    variant_order: Sequence[str],
    fold_indices: Sequence[tuple[np.ndarray, np.ndarray]],
    random_state: int = 42,
) -> Dict[str, Any]:
    """Recompute matched semantic transfer on aligned bundle features."""
    variant_names = [variant for variant in variant_order if variant in features_by_variant]
    transfer_matrix: Dict[str, Dict[str, float]] = {name: {} for name in variant_names}

    for train_variant in variant_names:
        for test_variant in variant_names:
            fold_scores: list[float] = []
            label_score_lists: Dict[int, list[float]] = {
                label_idx: [] for label_idx in range(semantic_labels.shape[1])
            }

            for train_idx, test_idx in fold_indices:
                score, label_scores = _train_and_evaluate_multilabel_transfer_once(
                    features_by_variant[train_variant][train_idx],
                    semantic_labels[train_idx],
                    features_by_variant[test_variant][test_idx],
                    semantic_labels[test_idx],
                    random_state=random_state,
                )
                if score > 0:
                    fold_scores.append(score)
                for label_idx, label_score in label_scores.items():
                    label_score_lists[label_idx].append(label_score)

            transfer_matrix[train_variant][test_variant] = float(np.mean(fold_scores)) if fold_scores else 0.0

    in_variant_accuracies = {variant: transfer_matrix[variant][variant] for variant in variant_names}
    cross_variant_scores = [
        transfer_matrix[train_variant][test_variant]
        for train_variant in variant_names
        for test_variant in variant_names
        if train_variant != test_variant
    ]
    cross_variant_accuracy = float(np.mean(cross_variant_scores)) if cross_variant_scores else 0.0
    in_variant_mean = float(np.mean(list(in_variant_accuracies.values()))) if in_variant_accuracies else 0.0

    return {
        "transfer_matrix": transfer_matrix,
        "in_variant_accuracies": in_variant_accuracies,
        "cross_variant_accuracy": cross_variant_accuracy,
        "transfer_gap": in_variant_mean - cross_variant_accuracy,
    }


def _compute_cross_variant_semantic_transfer_mean(
    features_by_variant: Dict[str, np.ndarray],
    semantic_labels: np.ndarray,
    variant_order: Sequence[str],
    fold_indices: Sequence[tuple[np.ndarray, np.ndarray]],
    random_state: int = 42,
) -> float:
    """Compute only the cross-variant semantic transfer mean used by uncertainty runs."""
    variant_names = [variant for variant in variant_order if variant in features_by_variant]
    cross_variant_scores: list[float] = []

    for train_variant in variant_names:
        for test_variant in variant_names:
            if train_variant == test_variant:
                continue

            fold_scores: list[float] = []
            for train_idx, test_idx in fold_indices:
                score, _label_scores = _train_and_evaluate_multilabel_transfer_once(
                    features_by_variant[train_variant][train_idx],
                    semantic_labels[train_idx],
                    features_by_variant[test_variant][test_idx],
                    semantic_labels[test_idx],
                    random_state=random_state,
                )
                if score > 0:
                    fold_scores.append(score)

            cross_variant_scores.append(float(np.mean(fold_scores)) if fold_scores else 0.0)

    return float(np.mean(cross_variant_scores)) if cross_variant_scores else 0.0


def _build_grouped_bootstrap_split_indices(
    sampled_case_indices: np.ndarray,
    semantic_labels: np.ndarray,
    n_splits: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build folds on unique sampled cases, then expand back to duplicated draws."""
    unique_case_indices = np.unique(sampled_case_indices)
    unique_labels = semantic_labels[unique_case_indices]
    unique_fold_indices = _build_matched_split_indices(
        unique_labels,
        n_splits=min(int(n_splits), int(unique_case_indices.size)),
        random_state=random_state,
    )

    grouped_folds: list[tuple[np.ndarray, np.ndarray]] = []
    for unique_train_idx, unique_test_idx in unique_fold_indices:
        train_cases = unique_case_indices[unique_train_idx]
        test_cases = unique_case_indices[unique_test_idx]
        train_mask = np.isin(sampled_case_indices, train_cases)
        test_mask = np.isin(sampled_case_indices, test_cases)
        grouped_folds.append((np.flatnonzero(train_mask), np.flatnonzero(test_mask)))
    return grouped_folds


def _manifest_variant_from_metric_path(metric_path: Path) -> str:
    relative_parts = metric_path.relative_to(OUTPUTS_ROOT).parts
    if len(relative_parts) >= 3:
        return str(relative_parts[2])
    return DEFAULT_MANIFEST_VARIANT


def _load_aligned_semantic_targets(
    analysis_name: str,
    manifest_variant: str,
    case_ids: Sequence[str],
) -> tuple[np.ndarray | None, np.ndarray | None, Path | None, Dict[str, Any] | None, str | None]:
    """Load semantic targets and align them to the saved bundle case order."""
    manifest_path = get_acquisition_manifest_path(
        dataset_name=analysis_name,
        manifest_kind="sampled",
        manifest_variant=manifest_variant,
    )
    if not manifest_path.exists():
        return None, None, None, None, "missing_semantic_manifest"

    semantic_labels_full, semantic_paths, semantic_metadata = _compute_semantic_task_labels(
        manifest_path,
        mode="auto",
        filter_dataset=analysis_name,
        cache_dir=get_cache_root(analysis_name, manifest_variant),
    )
    if semantic_labels_full.ndim != 2 or semantic_labels_full.shape[1] == 0:
        return None, None, manifest_path, semantic_metadata, "no_informative_semantic_labels"

    path_to_index = {str(path): idx for idx, path in enumerate(semantic_paths)}
    valid_positions: list[int] = []
    aligned_labels: list[np.ndarray] = []
    for case_position, case_id in enumerate(case_ids):
        label_index = path_to_index.get(str(case_id))
        if label_index is None:
            continue
        valid_positions.append(case_position)
        aligned_labels.append(semantic_labels_full[label_index])

    if not aligned_labels:
        return None, None, manifest_path, semantic_metadata, "no_aligned_semantic_labels_for_bundle_case_ids"
    if len(aligned_labels) < 10:
        return None, None, manifest_path, semantic_metadata, "insufficient_aligned_semantic_cases"

    return (
        np.stack(aligned_labels),
        np.asarray(valid_positions, dtype=np.int64),
        manifest_path,
        semantic_metadata,
        None,
    )


def _bootstrap_semantic_transfer(
    features_by_variant: Dict[str, np.ndarray],
    semantic_labels: np.ndarray,
    variant_order: Sequence[str],
    bootstrap_resamples: int,
    seed: int,
    n_cv_splits: int,
) -> Dict[str, Any]:
    """Estimate matched semantic-transfer uncertainty with grouped case bootstrap."""
    fold_indices = _build_matched_split_indices(
        semantic_labels,
        n_splits=n_cv_splits,
        random_state=seed,
    )
    point_mean = _compute_cross_variant_semantic_transfer_mean(
        features_by_variant,
        semantic_labels,
        variant_order,
        fold_indices,
        random_state=seed,
    )

    if bootstrap_resamples <= 0 or semantic_labels.shape[0] <= 1:
        return {
            "mean": point_mean,
            "bootstrap_std": None,
            "ci_lower": None,
            "ci_upper": None,
            "ci_status": "not_requested_set_bootstrap_resamples_positive",
            "valid_resamples": 0,
        }

    rng = np.random.default_rng(int(seed))
    bootstrap_values: list[float] = []
    for resample_index in range(int(bootstrap_resamples)):
        sampled_case_indices = rng.integers(0, semantic_labels.shape[0], size=semantic_labels.shape[0])
        if np.unique(sampled_case_indices).size < 2:
            continue

        resampled_features = {
            variant: features[sampled_case_indices]
            for variant, features in features_by_variant.items()
        }
        resampled_labels = semantic_labels[sampled_case_indices]
        try:
            resampled_fold_indices = _build_grouped_bootstrap_split_indices(
                sampled_case_indices,
                semantic_labels,
                n_splits=n_cv_splits,
                random_state=seed + resample_index + 1,
            )
            resampled_cross_variant_accuracy = _compute_cross_variant_semantic_transfer_mean(
                resampled_features,
                resampled_labels,
                variant_order,
                resampled_fold_indices,
                random_state=seed,
            )
        except ValueError:
            continue
        bootstrap_values.append(float(resampled_cross_variant_accuracy))

    if not bootstrap_values:
        return {
            "mean": point_mean,
            "bootstrap_std": None,
            "ci_lower": None,
            "ci_upper": None,
            "ci_status": "bootstrap_failed_no_valid_resamples",
            "valid_resamples": 0,
        }

    bootstrap_array = np.asarray(bootstrap_values, dtype=np.float64)
    return {
        "mean": point_mean,
        "bootstrap_std": float(np.std(bootstrap_array, ddof=0)),
        "ci_lower": float(np.percentile(bootstrap_array, 2.5)),
        "ci_upper": float(np.percentile(bootstrap_array, 97.5)),
        "ci_status": "ok",
        "valid_resamples": int(bootstrap_array.size),
    }


def _load_bundle(bundle_path: Path) -> tuple[Dict[str, np.ndarray], list[str], str, list[str], Dict[str, str]]:
    with np.load(bundle_path, allow_pickle=True) as bundle:
        feature_keys = sorted(key for key in bundle.files if key.startswith("features__"))
        features_by_variant = {
            key.replace("features__", "", 1): np.asarray(bundle[key], dtype=np.float32)
            for key in feature_keys
        }
        if "variant_names" in bundle:
            variant_order = [_scalar_text(value) for value in np.asarray(bundle["variant_names"]).tolist()]
        elif "spacing_names" in bundle:
            variant_order = [_scalar_text(value) for value in np.asarray(bundle["spacing_names"]).tolist()]
        else:
            variant_order = sorted(features_by_variant)
        if "reference_variant_name" in bundle:
            reference_variant = _scalar_text(bundle["reference_variant_name"])
        elif "reference_spacing_name" in bundle:
            reference_variant = _scalar_text(bundle["reference_spacing_name"])
        else:
            reference_variant = variant_order[0]
        if "case_ids" in bundle:
            case_ids = [_scalar_text(value) for value in np.asarray(bundle["case_ids"]).tolist()]
        else:
            case_ids = []
        if "variant_kinds" in bundle:
            kind_values = [_scalar_text(value) for value in np.asarray(bundle["variant_kinds"]).tolist()]
            variant_kinds = dict(zip(variant_order, kind_values))
        else:
            variant_kinds = {variant: "spacing" for variant in variant_order}
    return features_by_variant, variant_order, reference_variant, case_ids, variant_kinds


def _discover_metric_jsons(
    analysis_names: Sequence[str] | None,
    feature_types: Sequence[str] | None,
    checkpoints: Sequence[str] | None,
) -> list[Path]:
    metric_paths = sorted(OUTPUTS_ROOT.glob("*/*/results/*/*/perturbation_robustness.json"))
    if analysis_names:
        analysis_set = set(analysis_names)
        metric_paths = [path for path in metric_paths if path.relative_to(OUTPUTS_ROOT).parts[0] in analysis_set]
    if checkpoints:
        checkpoint_set = set(checkpoints)
        metric_paths = [path for path in metric_paths if path.parent.parent.name in checkpoint_set]
    if feature_types:
        feature_set = set(feature_types)
        metric_paths = [path for path in metric_paths if path.parent.name in feature_set]
    return metric_paths


def _analyze_metric_path(
    metric_path: Path,
    bootstrap_resamples: int,
    seed: int,
    cka_method: str,
    semantic_cv_splits: int,
) -> Dict[str, Any]:
    relative_parts = metric_path.relative_to(OUTPUTS_ROOT).parts
    analysis_name = relative_parts[0]
    checkpoint_name = metric_path.parent.parent.name
    feature_type = metric_path.parent.name
    payload = _load_json(metric_path)
    bundle_name = payload.get("controlled_embedding_bundle", "controlled_embedding_bundle.npz")
    bundle_path = metric_path.parent / str(bundle_name)

    row: Dict[str, Any] = {
        "analysis_name": analysis_name,
        "checkpoint_name": checkpoint_name,
        "feature_type": feature_type,
        "metric_json": str(metric_path),
        "bundle_path": str(bundle_path),
        "cka_method": cka_method,
        "bootstrap_resamples": int(bootstrap_resamples),
    }
    if not bundle_path.exists():
        row.update(
            {
                "status": "skipped",
                "skip_reason": "missing_controlled_embedding_bundle",
                "n_cases": None,
                "reference_variant": payload.get("reference_variant"),
            }
        )
        return row

    features_by_variant, variant_order, reference_variant, case_ids, variant_kinds = _load_bundle(bundle_path)
    if reference_variant not in features_by_variant:
        row.update({"status": "skipped", "skip_reason": "missing_reference_variant"})
        return row
    spacing_variant_order = [variant for variant in variant_order if variant_kinds.get(variant) == "spacing"]
    extra_acquisition_variant_order = [
        variant for variant in variant_order
        if variant_kinds.get(variant) == "extra_acquisition"
    ]

    reference_features = features_by_variant[reference_variant]
    distance_vectors = []
    for target_variant in variant_order:
        if target_variant == reference_variant or target_variant not in features_by_variant:
            continue
        target_features = features_by_variant[target_variant]
        if target_features.shape != reference_features.shape:
            continue
        distance_vectors.append(_rowwise_cosine_distance(reference_features, target_features))

    if not distance_vectors:
        row.update({"status": "skipped", "skip_reason": "missing_non_reference_variants"})
        return row

    compact_drift_per_case = np.mean(np.stack(distance_vectors, axis=0), axis=0)
    if bootstrap_resamples > 0:
        cka_summary = _bootstrap_compact_cka(
            features_by_variant,
            variant_order=variant_order,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
            method=cka_method,
        )
        cka_ci_status = "ok"
    else:
        cka_summary = {
            "mean": _compact_cka(features_by_variant, variant_order, method=cka_method),
            "ci_lower": None,
            "ci_upper": None,
        }
        cka_ci_status = "not_requested_set_bootstrap_resamples_positive"

    semantic_transfer_payload = payload.get("matched_semantic_transfer") or {}
    semantic_transfer_mean = None
    semantic_transfer_std = None
    semantic_transfer_ci_lower = None
    semantic_transfer_ci_upper = None
    semantic_transfer_ci_status = "missing_matched_semantic_transfer_artifact"
    semantic_manifest_path = None
    semantic_transfer_valid_resamples = 0
    n_semantic_cases = None
    n_semantic_labels = None
    semantic_skip_reason = None

    if isinstance(semantic_transfer_payload, dict) and semantic_transfer_payload:
        semantic_transfer_mean = float(semantic_transfer_payload.get("cross_variant_accuracy", 0.0))
        manifest_variant = _manifest_variant_from_metric_path(metric_path)
        aligned_labels, valid_indices, manifest_path, semantic_metadata, semantic_error = _load_aligned_semantic_targets(
            analysis_name,
            manifest_variant,
            case_ids,
        )
        semantic_manifest_path = str(manifest_path) if manifest_path is not None else None
        if semantic_error is None and aligned_labels is not None and valid_indices is not None:
            filtered_features_by_variant = {
                variant: features[valid_indices]
                for variant, features in features_by_variant.items()
            }
            semantic_summary = _bootstrap_semantic_transfer(
                filtered_features_by_variant,
                aligned_labels,
                variant_order,
                bootstrap_resamples=bootstrap_resamples,
                seed=seed,
                n_cv_splits=semantic_cv_splits,
            )
            semantic_transfer_mean = semantic_summary.get("mean")
            semantic_transfer_std = semantic_summary.get("bootstrap_std")
            semantic_transfer_ci_lower = semantic_summary.get("ci_lower")
            semantic_transfer_ci_upper = semantic_summary.get("ci_upper")
            semantic_transfer_ci_status = str(semantic_summary.get("ci_status"))
            semantic_transfer_valid_resamples = int(semantic_summary.get("valid_resamples", 0))
            n_semantic_cases = int(aligned_labels.shape[0])
            n_semantic_labels = int(aligned_labels.shape[1])
        else:
            semantic_skip_reason = semantic_error
            semantic_transfer_ci_status = semantic_error or semantic_transfer_ci_status
    row.update(
        {
            "status": "ok",
            "skip_reason": None,
            "n_cases": int(reference_features.shape[0]),
            "reference_variant": reference_variant,
            "variant_order": list(variant_order),
            "variant_kinds": dict(variant_kinds),
            "spacing_variant_order": spacing_variant_order,
            "extra_acquisition_variant_order": extra_acquisition_variant_order,
            "compact_drift_mean": float(np.mean(compact_drift_per_case)),
            "compact_drift_std": float(np.std(compact_drift_per_case, ddof=0)),
            "compact_cka_mean": cka_summary["mean"],
            "compact_cka_ci_lower": cka_summary["ci_lower"],
            "compact_cka_ci_upper": cka_summary["ci_upper"],
            "compact_cka_ci_status": cka_ci_status,
            "compact_semantic_transfer_mean": semantic_transfer_mean,
            "compact_semantic_transfer_bootstrap_std": semantic_transfer_std,
            "compact_semantic_transfer_ci_lower": semantic_transfer_ci_lower,
            "compact_semantic_transfer_ci_upper": semantic_transfer_ci_upper,
            "compact_semantic_transfer_ci_status": semantic_transfer_ci_status,
            "n_semantic_cases": n_semantic_cases,
            "n_semantic_labels": n_semantic_labels,
            "semantic_manifest_path": semantic_manifest_path,
            "semantic_transfer_bootstrap_method": "grouped_case_bootstrap_shared_matched_cv",
            "semantic_transfer_bootstrap_valid_resamples": semantic_transfer_valid_resamples,
            "semantic_transfer_uncertainty_status": semantic_transfer_ci_status,
        }
    )
    if semantic_skip_reason is not None:
        row["semantic_transfer_skip_reason"] = semantic_skip_reason
    return row


def _analyze_metric_path_task(task: tuple[Path, int, int, str, int]) -> Dict[str, Any]:
    metric_path, bootstrap_resamples, seed, cka_method, semantic_cv_splits = task
    return _analyze_metric_path(
        metric_path,
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
        cka_method=cka_method,
        semantic_cv_splits=semantic_cv_splits,
    )


def _analyze_metric_paths(
    metric_paths: Sequence[Path],
    *,
    bootstrap_resamples: int,
    seed: int,
    cka_method: str,
    semantic_cv_splits: int,
    workers: int,
    chunksize: int,
) -> list[Dict[str, Any]]:
    if workers <= 1 or len(metric_paths) <= 1:
        return [
            _analyze_metric_path(
                path,
                bootstrap_resamples=bootstrap_resamples,
                seed=seed,
                cka_method=cka_method,
                semantic_cv_splits=semantic_cv_splits,
            )
            for path in tqdm(metric_paths, desc="Acquisition Robustness controlled uncertainty")
        ]

    tasks = [
        (path, int(bootstrap_resamples), int(seed), cka_method, int(semantic_cv_splits))
        for path in metric_paths
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_analyze_metric_path_task, task): index
            for index, task in enumerate(tasks)
        }
        results: list[Dict[str, Any] | None] = [None] * len(tasks)
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Acquisition Robustness controlled uncertainty ({workers} workers)",
        ):
            results[futures[future]] = future.result()
        return [result for result in results if result is not None]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Acquisition Robustness controlled uncertainty from existing controlled embedding bundles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m reform3d.modules.acquisition_robustness.controlled_uncertainty_analysis --analysis-names abdomenatlas kits23 --feature-types cls avg_pool --bootstrap-resamples 1000 --workers 8\n"
            "  python -m reform3d.modules.acquisition_robustness.controlled_uncertainty_analysis --analysis-names imagecas --output-json /tmp/acquisition_controlled_uncertainty_imagecas.json\n"
        ),
    )
    parser.add_argument("--analysis-names", nargs="+", default=None)
    parser.add_argument("--feature-types", nargs="+", default=None)
    parser.add_argument("--checkpoints", nargs="+", default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cka-method",
        choices=("auto", "feature", "gram"),
        default="auto",
        help="Linear CKA backend. 'auto' chooses the exact feature-space or Gram-space formula by matrix size; 'gram' uses the controlled-evaluation Gram-space formulation.",
    )
    parser.add_argument(
        "--semantic-cv-splits",
        type=int,
        default=5,
        help="Number of shared case-disjoint folds used when recomputing matched semantic transfer.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of CPU worker processes. Use 1 for serial execution; use 0 for all available CPUs.",
    )
    parser.add_argument("--chunksize", type=int, default=1, help="Metric files assigned per worker task chunk.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args()

    if args.chunksize < 1:
        parser.error("--chunksize must be >= 1")
    if args.workers < 0:
        parser.error("--workers must be >= 0")
    workers = os.cpu_count() or 1 if args.workers == 0 else args.workers

    metric_paths = _discover_metric_jsons(args.analysis_names, args.feature_types, args.checkpoints)
    rows = _analyze_metric_paths(
        metric_paths,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
        cka_method=args.cka_method,
        semantic_cv_splits=args.semantic_cv_splits,
        workers=workers,
        chunksize=args.chunksize,
    )
    payload = {
        "analysis_scope": "acquisition_controlled_perturbation_fixed_checkpoint_case_uncertainty",
        "uncertainty_unit": "case_distribution_for_drift; case_bootstrap_for_cka; grouped_case_bootstrap_shared_matched_cv_for_semantic_transfer",
        "cka_method": args.cka_method,
        "semantic_cv_splits": int(args.semantic_cv_splits),
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "training_run_variance": "not_estimated_from_single_checkpoint_artifacts",
        "n_metric_files": len(metric_paths),
        "n_ok": sum(1 for row in rows if row.get("status") == "ok"),
        "n_skipped": sum(1 for row in rows if row.get("status") != "ok"),
        "rows": rows,
    }
    _write_json(args.output_json, payload)
    _write_csv(args.output_csv, rows)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
