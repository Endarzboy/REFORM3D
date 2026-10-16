#!/usr/bin/env python
"""Post-hoc uncertainty analysis for Acquisition Robustness native-spacing metrics.

This helper consumes cached observational feature bundles and existing
native-spacing result artifacts. It does not rerun feature extraction. It
reports stratified-bin bootstrap confidence intervals for the compact
native-spacing acquisition metrics currently shown in the benchmark tables:

- observational CKA: ``representation_geometry.json -> cka_stats.mean_cross_cka``
- observational semantic transfer: ``semantic_transfer.json -> cross_bin_accuracy``

Native bins contain different cases. Their CKA is an observational comparison,
not matched-case robustness under controlled spacing changes.

Examples:
    python -m reform3d.modules.acquisition_robustness.native_uncertainty_analysis \
        --analysis-names abdomenatlas kits23 imagecas totalsegmentermri jhu_stroke \
        --feature-types cls avg_pool multilayer \
        --bootstrap-resamples 1000 \
        --workers 8

    python -m reform3d.modules.acquisition_robustness.native_uncertainty_analysis \
        --analysis-names abdomenatlas \
        --feature-types avg_pool \
        --checkpoints med3dino_sa_96 \
        --output-json /tmp/acquisition_native_uncertainty_abdomenatlas.json
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is present in the project env.
    def tqdm(iterable, **_: Any):  # type: ignore[no-redef]
        return iterable

try:
    from .config import (
        CHECKPOINTS,
        DEFAULT_MANIFEST_VARIANT,
        ACQUISITION_OUTPUTS,
        get_cache_root,
        get_checkpoint_feature_dir,
        get_output_paths,
        get_acquisition_manifest_path,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    from config import (
        CHECKPOINTS,
        DEFAULT_MANIFEST_VARIANT,
        ACQUISITION_OUTPUTS,
        get_cache_root,
        get_checkpoint_feature_dir,
        get_output_paths,
        get_acquisition_manifest_path,
    )


def _compute_semantic_task_labels(*args: Any, **kwargs: Any) -> tuple[np.ndarray, list[str], Dict[str, Any]]:
    try:
        from .semantic_labels import compute_semantic_task_labels
    except ImportError:  # pragma: no cover - supports direct script execution.
        from semantic_labels import compute_semantic_task_labels
    return compute_semantic_task_labels(*args, **kwargs)


ACQUISITION_ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = ACQUISITION_OUTPUTS
DEFAULT_OUTPUT_DIR = OUTPUTS_ROOT / "statistical_rigor"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_DIR / "acquisition_native_uncertainty.json"
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "acquisition_native_uncertainty.csv"
DEFAULT_NATIVE_CKA_SAMPLES_PER_BIN = 500


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
        "manifest_variant",
        "status",
        "n_cases",
        "n_bins",
        "n_semantic_cases",
        "n_semantic_labels",
        "observational_cka_mean",
        "observational_cka_ci_lower",
        "observational_cka_ci_upper",
        "observational_cka_ci_status",
        "observational_cka_bootstrap_method",
        "observational_cka_bootstrap_valid_resamples",
        "observational_cka_n_samples_per_bin_cap",
        "observational_semantic_transfer_mean",
        "observational_semantic_transfer_bootstrap_std",
        "observational_semantic_transfer_ci_lower",
        "observational_semantic_transfer_ci_upper",
        "observational_semantic_transfer_ci_status",
        "observational_semantic_transfer_bootstrap_method",
        "observational_semantic_transfer_bootstrap_valid_resamples",
        "bootstrap_resamples",
        "seed",
        "cka_method",
        "semantic_min_class_count",
        "skip_reason",
        "semantic_manifest_path",
        "representation_geometry_path",
        "semantic_transfer_path",
        "feature_cache_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_crop_size(checkpoint_name: str) -> int:
    checkpoint = CHECKPOINTS.get(checkpoint_name)
    if checkpoint is not None:
        return int(checkpoint.crop_size)
    for token in checkpoint_name.split("_"):
        if token.startswith("c") and token[1:].isdigit():
            return int(token[1:])
    raise ValueError(f"Unable to infer crop size from checkpoint: {checkpoint_name}")


def _manifest_variant_from_metric_path(metric_path: Path) -> str:
    relative_parts = metric_path.relative_to(OUTPUTS_ROOT).parts
    if len(relative_parts) >= 6:
        return str(Path(*relative_parts[1:-4]))
    return DEFAULT_MANIFEST_VARIANT


def _discover_metric_jsons(
    analysis_names: Sequence[str] | None,
    feature_types: Sequence[str] | None,
    checkpoints: Sequence[str] | None,
) -> list[Path]:
    analysis_filter = set(analysis_names or [])
    feature_filter = set(feature_types or [])
    checkpoint_filter = set(checkpoints or [])
    metric_paths: list[Path] = []
    for metric_path in sorted(OUTPUTS_ROOT.glob("**/results/*/*/representation_geometry.json")):
        relative_parts = metric_path.relative_to(OUTPUTS_ROOT).parts
        if len(relative_parts) < 6:
            continue
        analysis_name = str(relative_parts[0])
        checkpoint_name = metric_path.parent.parent.name
        feature_type = metric_path.parent.name
        if analysis_filter and analysis_name not in analysis_filter:
            continue
        if feature_filter and feature_type not in feature_filter:
            continue
        if checkpoint_filter and checkpoint_name not in checkpoint_filter:
            continue
        metric_paths.append(metric_path)
    return metric_paths


def _expected_feature_cache_path(
    analysis_name: str,
    manifest_variant: str,
    checkpoint_name: str,
    feature_type: str,
) -> Path:
    manifest_path = get_acquisition_manifest_path(
        dataset_name=analysis_name,
        manifest_kind="sampled",
        manifest_variant=manifest_variant,
    )
    crop_size = _resolve_crop_size(checkpoint_name)
    manifest_hash = hashlib.md5(manifest_path.read_bytes()).hexdigest()[:8]
    features_root = get_output_paths(analysis_name, manifest_variant)["features"]
    feature_dir = get_checkpoint_feature_dir(features_root, checkpoint_name, feature_type)
    return feature_dir / f"features_c{crop_size}_{manifest_hash}.npz"


def _resolve_feature_cache_path(
    analysis_name: str,
    manifest_variant: str,
    checkpoint_name: str,
    feature_type: str,
) -> tuple[Path | None, str | None]:
    try:
        expected_path = _expected_feature_cache_path(
            analysis_name,
            manifest_variant,
            checkpoint_name,
            feature_type,
        )
    except FileNotFoundError:
        return None, "missing_manifest_for_feature_cache_lookup"
    except ValueError as exc:
        return None, str(exc)

    if expected_path.exists():
        return expected_path, None

    feature_dir = expected_path.parent
    crop_size = _resolve_crop_size(checkpoint_name)
    candidate_paths = sorted(feature_dir.glob(f"features_c{crop_size}_*.npz"))
    if len(candidate_paths) == 1:
        return candidate_paths[0], None
    if len(candidate_paths) > 1:
        return None, "ambiguous_feature_cache_candidates"
    return None, "missing_feature_cache"


def _linear_cka(source_features: np.ndarray, target_features: np.ndarray, method: str = "gram") -> float:
    try:
        from .representation_geometry_analysis import centered_linear_cka
    except ImportError:  # pragma: no cover - direct script execution
        from representation_geometry_analysis import centered_linear_cka
    return centered_linear_cka(source_features, target_features, method=method)


def _compute_observational_mean_cross_cka(
    features: np.ndarray,
    bin_labels: np.ndarray,
    *,
    sample_count_per_bin: int,
    rng: np.random.Generator,
    cka_method: str,
) -> float | None:
    unique_bins = sorted(int(bin_id) for bin_id in np.unique(bin_labels))
    if len(unique_bins) < 2:
        return None

    sampled_by_bin: dict[int, np.ndarray] = {}
    for bin_id in unique_bins:
        bin_indices = np.flatnonzero(bin_labels == bin_id)
        if bin_indices.size < 2:
            continue
        n_samples = min(int(sample_count_per_bin), int(bin_indices.size))
        sampled_indices = rng.choice(bin_indices, size=n_samples, replace=True)
        sampled_by_bin[bin_id] = np.asarray(features[sampled_indices], dtype=np.float64)

    cross_values: list[float] = []
    for source_index, source_bin in enumerate(unique_bins):
        source_features = sampled_by_bin.get(source_bin)
        if source_features is None:
            continue
        for target_bin in unique_bins[source_index + 1 :]:
            target_features = sampled_by_bin.get(target_bin)
            if target_features is None:
                continue
            pair_n = min(len(source_features), len(target_features))
            if pair_n < 2:
                continue
            cross_values.append(
                _linear_cka(
                    source_features[:pair_n],
                    target_features[:pair_n],
                    method=cka_method,
                )
            )
    if not cross_values:
        return None
    return float(np.mean(cross_values))


def _bootstrap_observational_cka(
    features: np.ndarray,
    bin_labels: np.ndarray,
    *,
    point_mean: float | None,
    bootstrap_resamples: int,
    seed: int,
    cka_method: str,
    sample_count_per_bin: int,
) -> Dict[str, Any]:
    if point_mean is None:
        return {
            "mean": None,
            "ci_lower": None,
            "ci_upper": None,
            "ci_status": "missing_point_estimate",
            "valid_resamples": 0,
        }
    if bootstrap_resamples <= 0:
        return {
            "mean": point_mean,
            "ci_lower": None,
            "ci_upper": None,
            "ci_status": "not_requested_set_bootstrap_resamples_positive",
            "valid_resamples": 0,
        }

    rng = np.random.default_rng(int(seed))
    bootstrap_values: list[float] = []
    for _ in range(int(bootstrap_resamples)):
        value = _compute_observational_mean_cross_cka(
            features,
            bin_labels,
            sample_count_per_bin=sample_count_per_bin,
            rng=rng,
            cka_method=cka_method,
        )
        if value is not None:
            bootstrap_values.append(float(value))

    if not bootstrap_values:
        return {
            "mean": point_mean,
            "ci_lower": None,
            "ci_upper": None,
            "ci_status": "bootstrap_failed_no_valid_resamples",
            "valid_resamples": 0,
        }
    return {
        "mean": point_mean,
        "ci_lower": float(np.percentile(bootstrap_values, 2.5)),
        "ci_upper": float(np.percentile(bootstrap_values, 97.5)),
        "ci_status": "ok",
        "valid_resamples": len(bootstrap_values),
    }


@lru_cache(maxsize=None)
def _load_semantic_targets_for_dataset(
    analysis_name: str,
    manifest_variant: str,
) -> tuple[np.ndarray | None, Dict[str, int] | None, Path | None, Dict[str, Any] | None, str | None]:
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
    return semantic_labels_full, path_to_index, manifest_path, semantic_metadata, None


def _load_aligned_semantic_targets(
    analysis_name: str,
    manifest_variant: str,
    file_paths: Sequence[str],
) -> tuple[np.ndarray | None, np.ndarray | None, Path | None, Dict[str, Any] | None, str | None]:
    semantic_labels_full, path_to_index, manifest_path, semantic_metadata, error = _load_semantic_targets_for_dataset(
        analysis_name,
        manifest_variant,
    )
    if error is not None or semantic_labels_full is None or path_to_index is None:
        return None, None, manifest_path, semantic_metadata, error

    valid_positions: list[int] = []
    aligned_labels: list[np.ndarray] = []
    for case_position, file_path in enumerate(file_paths):
        label_index = path_to_index.get(str(file_path))
        if label_index is None:
            continue
        valid_positions.append(case_position)
        aligned_labels.append(semantic_labels_full[label_index])

    if not aligned_labels:
        return None, None, manifest_path, semantic_metadata, "no_aligned_semantic_labels_for_feature_cache_file_paths"
    if len(aligned_labels) < 10:
        return None, None, manifest_path, semantic_metadata, "insufficient_aligned_semantic_cases"

    return (
        np.stack(aligned_labels),
        np.asarray(valid_positions, dtype=np.int64),
        manifest_path,
        semantic_metadata,
        None,
    )


def _train_and_evaluate_multilabel_transfer(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    *,
    random_state: int,
    min_class_count: int,
) -> float:
    if train_labels.ndim != 2 or test_labels.ndim != 2:
        raise ValueError("Multi-label transfer expects label matrices [N, K]")
    if len(train_features) < 10 or len(test_features) < 10 or train_labels.shape[1] == 0:
        return 0.0

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)

    valid_scores: list[float] = []
    for label_index in range(train_labels.shape[1]):
        y_train = train_labels[:, label_index]
        y_test = test_labels[:, label_index]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        pos_train = int(y_train.sum())
        neg_train = int(len(y_train) - pos_train)
        pos_test = int(y_test.sum())
        neg_test = int(len(y_test) - pos_test)
        if min(pos_train, neg_train, pos_test, neg_test) < min_class_count:
            continue

        classifier = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state,
            n_jobs=1,
        )
        try:
            classifier.fit(train_scaled, y_train)
            predictions = classifier.predict(test_scaled)
            valid_scores.append(float(balanced_accuracy_score(y_test, predictions)))
        except Exception:
            continue

    if not valid_scores:
        return 0.0
    return float(np.mean(valid_scores))


def _compute_cross_bin_semantic_transfer_mean(
    features: np.ndarray,
    bin_labels: np.ndarray,
    semantic_labels: np.ndarray,
    *,
    random_state: int,
    min_class_count: int,
) -> float | None:
    unique_bins = sorted(int(bin_id) for bin_id in np.unique(bin_labels))
    if len(unique_bins) < 2:
        return None

    cross_bin_values: list[float] = []
    for train_bin in unique_bins:
        train_mask = bin_labels == train_bin
        train_features = features[train_mask]
        train_task_labels = semantic_labels[train_mask]
        for test_bin in unique_bins:
            if train_bin == test_bin:
                continue
            test_mask = bin_labels == test_bin
            test_features = features[test_mask]
            test_task_labels = semantic_labels[test_mask]
            cross_bin_values.append(
                _train_and_evaluate_multilabel_transfer(
                    train_features,
                    train_task_labels,
                    test_features,
                    test_task_labels,
                    random_state=random_state,
                    min_class_count=min_class_count,
                )
            )
    if not cross_bin_values:
        return None
    return float(np.mean(cross_bin_values))


def _bootstrap_cross_bin_semantic_transfer(
    features: np.ndarray,
    bin_labels: np.ndarray,
    semantic_labels: np.ndarray,
    *,
    point_mean: float | None,
    bootstrap_resamples: int,
    seed: int,
    min_class_count: int,
) -> Dict[str, Any]:
    if point_mean is None:
        return {
            "mean": None,
            "bootstrap_std": None,
            "ci_lower": None,
            "ci_upper": None,
            "ci_status": "missing_point_estimate",
            "valid_resamples": 0,
        }
    if bootstrap_resamples <= 0:
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
    unique_bins = sorted(int(bin_id) for bin_id in np.unique(bin_labels))
    bin_indices = {
        bin_id: np.flatnonzero(bin_labels == bin_id)
        for bin_id in unique_bins
    }

    for resample_index in range(int(bootstrap_resamples)):
        sampled_index_parts = []
        for bin_id in unique_bins:
            indices = bin_indices[bin_id]
            if indices.size == 0:
                continue
            sampled_index_parts.append(rng.choice(indices, size=indices.size, replace=True))
        if len(sampled_index_parts) < 2:
            continue
        sampled_indices = np.concatenate(sampled_index_parts)
        value = _compute_cross_bin_semantic_transfer_mean(
            features[sampled_indices],
            bin_labels[sampled_indices],
            semantic_labels[sampled_indices],
            random_state=seed + resample_index + 1,
            min_class_count=min_class_count,
        )
        if value is not None:
            bootstrap_values.append(float(value))

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


def _analyze_metric_path(
    metric_path: Path,
    *,
    bootstrap_resamples: int,
    seed: int,
    cka_method: str,
    semantic_min_class_count: int,
) -> Dict[str, Any]:
    relative_parts = metric_path.relative_to(OUTPUTS_ROOT).parts
    analysis_name = str(relative_parts[0])
    checkpoint_name = metric_path.parent.parent.name
    feature_type = metric_path.parent.name
    manifest_variant = _manifest_variant_from_metric_path(metric_path)
    semantic_transfer_path = metric_path.with_name("semantic_transfer.json")

    row: Dict[str, Any] = {
        "analysis_name": analysis_name,
        "checkpoint_name": checkpoint_name,
        "feature_type": feature_type,
        "manifest_variant": manifest_variant,
        "status": "skipped",
        "skip_reason": None,
        "bootstrap_resamples": int(bootstrap_resamples),
        "seed": int(seed),
        "cka_method": cka_method,
        "semantic_min_class_count": int(semantic_min_class_count),
        "representation_geometry_path": str(metric_path),
        "semantic_transfer_path": str(semantic_transfer_path) if semantic_transfer_path.exists() else None,
        "feature_cache_path": None,
        "semantic_manifest_path": None,
    }

    feature_cache_path, feature_cache_error = _resolve_feature_cache_path(
        analysis_name,
        manifest_variant,
        checkpoint_name,
        feature_type,
    )
    if feature_cache_error is not None or feature_cache_path is None:
        row["skip_reason"] = feature_cache_error
        return row

    geometry_payload = _load_json(metric_path)
    with np.load(feature_cache_path, allow_pickle=False) as cache_payload:
        features = np.asarray(cache_payload["features"], dtype=np.float32)
        bin_labels = np.asarray(cache_payload["bin_labels"], dtype=np.int64)
        file_paths = np.asarray(cache_payload["file_paths"]).astype(str)

    point_cka_mean = _float_or_none(geometry_payload.get("cka_stats", {}).get("mean_cross_cka"))
    sample_count_per_bin = _int_or_none(
        geometry_payload.get("observational_distances", {}).get("summary", {}).get("n_samples_per_bin_cap")
    ) or DEFAULT_NATIVE_CKA_SAMPLES_PER_BIN
    cka_summary = _bootstrap_observational_cka(
        features,
        bin_labels,
        point_mean=point_cka_mean,
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
        cka_method=cka_method,
        sample_count_per_bin=sample_count_per_bin,
    )

    point_semantic_transfer_mean = None
    semantic_transfer_summary: Dict[str, Any] = {
        "mean": None,
        "bootstrap_std": None,
        "ci_lower": None,
        "ci_upper": None,
        "ci_status": "missing_semantic_transfer_artifact",
        "valid_resamples": 0,
    }
    semantic_manifest_path: Path | None = None
    n_semantic_cases = None
    n_semantic_labels = None

    if semantic_transfer_path.exists():
        semantic_transfer_payload = _load_json(semantic_transfer_path)
        point_semantic_transfer_mean = _float_or_none(semantic_transfer_payload.get("cross_bin_accuracy"))
        aligned_labels, valid_positions, semantic_manifest_path, _, semantic_error = _load_aligned_semantic_targets(
            analysis_name,
            manifest_variant,
            file_paths.tolist(),
        )
        row["semantic_manifest_path"] = str(semantic_manifest_path) if semantic_manifest_path is not None else None
        if semantic_error is None and aligned_labels is not None and valid_positions is not None:
            filtered_features = features[valid_positions]
            filtered_bin_labels = bin_labels[valid_positions]
            semantic_transfer_summary = _bootstrap_cross_bin_semantic_transfer(
                filtered_features,
                filtered_bin_labels,
                aligned_labels,
                point_mean=point_semantic_transfer_mean,
                bootstrap_resamples=bootstrap_resamples,
                seed=seed,
                min_class_count=semantic_min_class_count,
            )
            n_semantic_cases = int(aligned_labels.shape[0])
            n_semantic_labels = int(aligned_labels.shape[1])
        else:
            semantic_transfer_summary["ci_status"] = semantic_error

    row.update(
        {
            "status": "ok",
            "skip_reason": None,
            "n_cases": int(features.shape[0]),
            "n_bins": int(np.unique(bin_labels).size),
            "n_semantic_cases": n_semantic_cases,
            "n_semantic_labels": n_semantic_labels,
            "observational_cka_mean": cka_summary["mean"],
            "observational_cka_ci_lower": cka_summary["ci_lower"],
            "observational_cka_ci_upper": cka_summary["ci_upper"],
            "observational_cka_ci_status": cka_summary["ci_status"],
            "observational_cka_bootstrap_method": "stratified_bin_case_bootstrap_linear_cka",
            "observational_cka_bootstrap_valid_resamples": int(cka_summary.get("valid_resamples", 0)),
            "observational_cka_n_samples_per_bin_cap": int(sample_count_per_bin),
            "observational_semantic_transfer_mean": semantic_transfer_summary["mean"],
            "observational_semantic_transfer_bootstrap_std": semantic_transfer_summary["bootstrap_std"],
            "observational_semantic_transfer_ci_lower": semantic_transfer_summary["ci_lower"],
            "observational_semantic_transfer_ci_upper": semantic_transfer_summary["ci_upper"],
            "observational_semantic_transfer_ci_status": semantic_transfer_summary["ci_status"],
            "observational_semantic_transfer_bootstrap_method": "stratified_bin_case_bootstrap_multilabel_transfer",
            "observational_semantic_transfer_bootstrap_valid_resamples": int(semantic_transfer_summary.get("valid_resamples", 0)),
            "feature_cache_path": str(feature_cache_path),
        }
    )
    return row


def _analyze_metric_path_task(task: tuple[Path, int, int, str, int]) -> Dict[str, Any]:
    metric_path, bootstrap_resamples, seed, cka_method, semantic_min_class_count = task
    return _analyze_metric_path(
        metric_path,
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
        cka_method=cka_method,
        semantic_min_class_count=semantic_min_class_count,
    )


def _analyze_metric_paths(
    metric_paths: Sequence[Path],
    *,
    bootstrap_resamples: int,
    seed: int,
    cka_method: str,
    semantic_min_class_count: int,
    workers: int,
) -> list[Dict[str, Any]]:
    if workers <= 1 or len(metric_paths) <= 1:
        return [
            _analyze_metric_path(
                path,
                bootstrap_resamples=bootstrap_resamples,
                seed=seed,
                cka_method=cka_method,
                semantic_min_class_count=semantic_min_class_count,
            )
            for path in tqdm(metric_paths, desc="Acquisition Robustness native uncertainty")
        ]

    tasks = [
        (path, int(bootstrap_resamples), int(seed), cka_method, int(semantic_min_class_count))
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
            desc=f"Acquisition Robustness native uncertainty ({workers} workers)",
        ):
            results[futures[future]] = future.result()
        return [result for result in results if result is not None]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Acquisition Robustness native-spacing uncertainty from cached observational features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m reform3d.modules.acquisition_robustness.native_uncertainty_analysis --analysis-names abdomenatlas kits23 imagecas totalsegmentermri jhu_stroke --feature-types cls avg_pool multilayer --bootstrap-resamples 1000 --workers 8\n"
            "  python -m reform3d.modules.acquisition_robustness.native_uncertainty_analysis --analysis-names abdomenatlas --feature-types avg_pool --checkpoints med3dino_sa_96 --output-json /tmp/acquisition_native_uncertainty_abdomenatlas.json\n"
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
        default="gram",
        help="Linear CKA backend. 'gram' uses the observational Gram-space formulation.",
    )
    parser.add_argument(
        "--semantic-min-class-count",
        type=int,
        default=5,
        help="Minimum positive and negative examples required per label in each train/test bin, matching the native semantic transfer analysis.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of CPU worker processes. Use 1 for serial execution; use 0 for all available CPUs.",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args()

    if args.workers < 0:
        parser.error("--workers must be >= 0")
    if args.semantic_min_class_count < 1:
        parser.error("--semantic-min-class-count must be >= 1")
    workers = (os.cpu_count() or 1) if args.workers == 0 else args.workers

    metric_paths = _discover_metric_jsons(args.analysis_names, args.feature_types, args.checkpoints)
    rows = _analyze_metric_paths(
        metric_paths,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
        cka_method=args.cka_method,
        semantic_min_class_count=args.semantic_min_class_count,
        workers=workers,
    )
    payload = {
        "analysis_scope": "acquisition_native_spacing_case_bootstrap_uncertainty",
        "uncertainty_unit": "stratified_bin_case_bootstrap_for_observational_cka_and_semantic_transfer",
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "seed": int(args.seed),
        "cka_method": args.cka_method,
        "semantic_min_class_count": int(args.semantic_min_class_count),
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
