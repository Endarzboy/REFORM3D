"""Train readout probes for Acquisition Robustness embeddings.

The module fits lightweight models on frozen features to estimate anisotropy
readout, semantic label predictability, and cross-bin semantic transfer. It is
called by the evaluation pipeline after feature bundles and semantic labels are
available.

Inputs:
    feature bundles grouped by spacing bin.
    semantic labels produced by ``semantic_labels.py``.
    checkpoint and feature-type identifiers for result organization.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm

try:
    from .config import (
        CHECKPOINTS,
        EVALUATION_SEEDS,
        DEFAULT_BINNING_SCHEME,
        get_bin_name_map,
        get_checkpoint_feature_dir,
        get_dataset_name_from_manifest_path,
        get_manifest_variant_from_manifest_path,
        get_output_paths,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from config import (
        CHECKPOINTS,
        EVALUATION_SEEDS,
        DEFAULT_BINNING_SCHEME,
        get_bin_name_map,
        get_checkpoint_feature_dir,
        get_dataset_name_from_manifest_path,
        get_manifest_variant_from_manifest_path,
        get_output_paths,
    )

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


RATIO_STRATIFICATION_MAX_BINS = 5
SPACING_REGRESSION_TARGET_NAME = "anisotropy_ratio"
SPACING_REGRESSION_TARGET_TRANSFORM = "log"
SPACING_REGRESSION_TARGET_METRIC_SPACE = "log_anisotropy_ratio"


def _resolve_results_dir(
    output_dir: Optional[Path],
    analysis_name: str = "default",
    manifest_variant: str = "original_bins",
) -> Path:
    if output_dir is not None:
        return output_dir
    return get_output_paths(analysis_name, manifest_variant)["results"]


def _resolve_checkpoint_results_dir(
    output_dir: Optional[Path],
    checkpoint_name: str,
    feature_type: str,
    analysis_name: str = "default",
    manifest_variant: str = "original_bins",
) -> Path:
    base_dir = _resolve_results_dir(output_dir, analysis_name, manifest_variant)
    checkpoint_dir = get_checkpoint_feature_dir(base_dir, checkpoint_name, feature_type)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def _convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {_convert_numpy_types(k): _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(v) for v in obj]
    elif isinstance(obj, tuple):
        return [_convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


@dataclass
class SpacingReadoutResults:
    """Results from the spacing-stratified readout analysis."""
    checkpoint_name: str
    feature_type: str

    # Per-bin metrics
    bin_accuracies: Dict[int, float]
    bin_balanced_accuracies: Dict[int, float]
    bin_sample_counts: Dict[int, int]

    # Aggregated metrics
    overall_accuracy: float
    overall_balanced_accuracy: float
    accuracy_std_across_bins: float
    max_bin_gap: float  # Max difference between any two bins

    # Cross-validation details
    cv_scores_per_bin: Optional[Dict[int, List[float]]] = None

    # Balanced-bin sensitivity output (natural estimate remains top-level)
    balanced_bin_sensitivity: Optional[Dict[str, Any]] = None


@dataclass
class SpacingRegressionResults:
    """
    Results from spacing regression probe.

    Predicts continuous anisotropy ratio from embeddings.
    Interpretation:
    - Low R² / High MSE = spacing-invariant representation (GOOD)
    - High R² / Low MSE = representation encodes spacing artifacts (BAD)
    """
    checkpoint_name: str
    feature_type: str

    # Overall regression metrics
    r2_score: float
    mse: float
    mae: float

    # Per-bin regression (to see if certain regimes are more predictable)
    per_bin_r2: Dict[int, float]
    per_bin_mse: Dict[int, float]
    per_bin_mae: Dict[int, float]
    per_bin_sample_counts: Dict[int, int]

    # Cross-validation variance
    r2_std: float
    mse_std: float
    mae_std: float

    # CV policy metadata
    split_strategy: str = "ratio_quantile_stratified"
    target_name: str = SPACING_REGRESSION_TARGET_NAME
    target_transform: str = SPACING_REGRESSION_TARGET_TRANSFORM
    target_metric_space: str = SPACING_REGRESSION_TARGET_METRIC_SPACE


def _prepare_spacing_regression_targets(anisotropy_ratios: np.ndarray) -> np.ndarray:
    """Project raw anisotropy ratios into the regression target space used by representation geometry."""
    ratios = np.asarray(anisotropy_ratios, dtype=np.float64)
    if np.any(ratios <= 0):
        raise ValueError("Anisotropy ratios must be strictly positive for log regression targets")
    return np.log(ratios)


def serialize_spacing_regression_results(results: SpacingRegressionResults) -> Dict[str, Any]:
    """Serialize spacing regression results with explicit target-space metadata."""
    return _convert_numpy_types({
        "r2_score": results.r2_score,
        "r2_std": results.r2_std,
        "mse": results.mse,
        "mse_std": results.mse_std,
        "mae": results.mae,
        "mae_std": results.mae_std,
        "per_bin_r2": results.per_bin_r2,
        "per_bin_mse": results.per_bin_mse,
        "per_bin_mae": results.per_bin_mae,
        "per_bin_sample_counts": results.per_bin_sample_counts,
        "split_strategy": results.split_strategy,
        "target_name": results.target_name,
        "target_transform": results.target_transform,
        "target_metric_space": results.target_metric_space,
    })


@dataclass
class MultiSeedSpacingReadoutResults:
    """
    Aggregated spacing-stratified readout results across multiple random seeds.

    Provides mean ± std for all metrics to assess reproducibility.
    """
    checkpoint_name: str
    feature_type: str
    seeds: List[int]

    # Aggregated overall metrics (mean ± std)
    overall_accuracy_mean: float
    overall_accuracy_std: float
    overall_balanced_accuracy_mean: float
    overall_balanced_accuracy_std: float

    # Per-bin metrics (mean ± std)
    bin_accuracies_mean: Dict[int, float]
    bin_accuracies_std: Dict[int, float]
    bin_sample_counts: Dict[int, int]

    # Spacing regression (if computed)
    spacing_r2_mean: Optional[float] = None
    spacing_r2_std: Optional[float] = None
    spacing_mse_mean: Optional[float] = None
    spacing_mse_std: Optional[float] = None

    # Per-seed raw results for transparency
    per_seed_results: Optional[Dict[int, SpacingReadoutResults]] = None
    per_seed_spacing: Optional[Dict[int, SpacingRegressionResults]] = None


def _compute_ratio_quantile_bins(
    ratio_values: np.ndarray,
    max_bins: int = RATIO_STRATIFICATION_MAX_BINS,
) -> Optional[np.ndarray]:
    """Discretize continuous anisotropy ratios into support-aware quantile bins."""
    ratios = np.asarray(ratio_values, dtype=np.float64)
    if len(ratios) < 4 or np.allclose(ratios, ratios[0]):
        return None

    max_candidate_bins = min(max_bins, len(ratios) // 2)
    for n_bins in range(max_candidate_bins, 1, -1):
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(ratios, quantiles))
        if len(edges) < 3:
            continue
        strata = np.digitize(ratios, edges[1:-1], right=False).astype(np.int32)
        _, counts = np.unique(strata, return_counts=True)
        if counts.min() >= 2:
            return strata
    return None


def _resolve_stratified_cv(
    n_splits: int,
    random_state: int,
    label_values: Optional[np.ndarray] = None,
    ratio_values: Optional[np.ndarray] = None,
) -> Tuple[Optional[List[Tuple[np.ndarray, np.ndarray]]], int, str]:
    """Build the strongest support-preserving stratified CV available."""
    attempts: List[Tuple[str, Optional[np.ndarray]]] = []

    ratio_bins = None
    if ratio_values is not None:
        ratio_bins = _compute_ratio_quantile_bins(ratio_values)

    if label_values is not None and ratio_bins is not None:
        combined = np.asarray(
            [f"{int(label)}__rq{int(ratio_bin)}" for label, ratio_bin in zip(label_values, ratio_bins)],
            dtype=str,
        )
        attempts.append(("label_plus_ratio_quantile", combined))

    if label_values is not None:
        attempts.append(("label_only", np.asarray(label_values, dtype=str)))

    if ratio_bins is not None:
        attempts.append(("ratio_quantile", np.asarray(ratio_bins, dtype=str)))

    for strategy, strata in attempts:
        if strata is None:
            continue
        _, counts = np.unique(strata, return_counts=True)
        effective_splits = min(n_splits, int(counts.min()))
        if effective_splits >= 2:
            splitter = StratifiedKFold(
                n_splits=effective_splits,
                shuffle=True,
                random_state=random_state,
            )
            splits = list(splitter.split(np.zeros(len(strata)), strata))
            return splits, effective_splits, strategy

    return None, 0, "unavailable"


def _select_balanced_bin_indices(
    bin_labels: np.ndarray,
    random_state: int = 42,
) -> Tuple[Optional[np.ndarray], Dict[int, int], int]:
    """Select an equal-count subset from each observed bin for sensitivity analysis."""
    unique_bins = np.unique(bin_labels)
    if len(unique_bins) < 2:
        return None, {}, 0

    bin_counts = {int(bin_id): int(np.sum(bin_labels == bin_id)) for bin_id in unique_bins}
    sample_count_per_bin = min(bin_counts.values()) if bin_counts else 0
    if sample_count_per_bin < 2:
        return None, bin_counts, sample_count_per_bin

    rng = np.random.default_rng(random_state)
    selected_indices = []
    for bin_id in unique_bins:
        bin_indices = np.where(bin_labels == bin_id)[0]
        if len(bin_indices) > sample_count_per_bin:
            chosen = rng.choice(bin_indices, size=sample_count_per_bin, replace=False)
        else:
            chosen = bin_indices
        selected_indices.append(chosen)

    merged = np.concatenate(selected_indices)
    rng.shuffle(merged)
    return merged.astype(np.int64), bin_counts, sample_count_per_bin


def _evaluate_regression_cv(
    features_scaled: np.ndarray,
    targets: np.ndarray,
    n_splits: int,
    random_state: int,
) -> Tuple[List[float], List[float], List[float], str]:
    """Evaluate regression with ratio-aware stratified folds when possible."""
    cv_splits, effective_splits, strategy = _resolve_stratified_cv(
        n_splits=n_splits,
        random_state=random_state,
        ratio_values=targets,
    )
    if cv_splits is None:
        effective_splits = min(n_splits, len(features_scaled))
        if effective_splits < 2:
            return [], [], [], "unavailable"
        splitter = KFold(n_splits=effective_splits, shuffle=True, random_state=random_state)
        split_iter = splitter.split(features_scaled)
        strategy = "kfold_fallback"
    else:
        split_iter = cv_splits

    reg = Ridge(alpha=1.0, random_state=random_state)
    r2_scores: List[float] = []
    mse_scores: List[float] = []
    mae_scores: List[float] = []

    for train_idx, test_idx in split_iter:
        X_train, X_test = features_scaled[train_idx], features_scaled[test_idx]
        y_train, y_test = targets[train_idx], targets[test_idx]
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        r2_scores.append(r2_score(y_test, y_pred))
        mse_scores.append(mean_squared_error(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))

    return r2_scores, mse_scores, mae_scores, strategy


def train_linear_probe(
    features: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    ratio_values: Optional[np.ndarray] = None,
) -> Tuple[float, float, List[float]]:
    """
    Train a linear classifier with cross-validation.

    Args:
        features: Feature matrix [N, D]
        labels: Class labels [N]
        n_splits: Number of CV folds
        random_state: Random seed

    Returns:
        Tuple of (mean_accuracy, mean_balanced_accuracy, cv_scores)
    """
    # Skip if too few samples or only one class
    unique_classes = np.unique(labels)
    if len(features) < n_splits or len(unique_classes) < 2:
        logger.warning(f"Insufficient data: {len(features)} samples, {len(unique_classes)} classes")
        return 0.0, 0.0, []

    cv_splits, effective_splits, strategy = _resolve_stratified_cv(
        n_splits=n_splits,
        random_state=random_state,
        label_values=labels,
        ratio_values=ratio_values,
    )
    if cv_splits is None:
        logger.warning("Not enough support for stratified CV")
        return 0.0, 0.0, []

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Train classifier
    clf = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        random_state=random_state,
        n_jobs=4,  # Limit parallelism to prevent resource leaks
    )

    # Regular accuracy
    cv_scores = cross_val_score(clf, features_scaled, labels, cv=cv_splits, scoring='accuracy')

    # Balanced accuracy
    cv_balanced = cross_val_score(clf, features_scaled, labels, cv=cv_splits, scoring='balanced_accuracy')

    logger.debug("Linear probe CV strategy: %s (%d folds)", strategy, effective_splits)
    return float(np.mean(cv_scores)), float(np.mean(cv_balanced)), cv_scores.tolist()


def spacing_regression_probe(
    features: np.ndarray,
    anisotropy_ratios: np.ndarray,
    bin_labels: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> SpacingRegressionResults:
    """
    Train a regression probe to predict log-anisotropy ratio from embeddings.

    This is a key metric for spacing robustness:
    - Low R² = features are spacing-invariant (GOOD)
    - High R² = features encode spacing information (BAD)

    Args:
        features: Feature matrix [N, D]
        anisotropy_ratios: Raw continuous anisotropy ratio targets [N]
        bin_labels: Bin assignments for per-bin analysis [N]
        n_splits: Number of CV folds
        random_state: Random seed for reproducibility

    Returns:
        SpacingRegressionResults with overall and per-bin metrics in log-target space
    """
    regression_targets = _prepare_spacing_regression_targets(anisotropy_ratios)

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    r2_scores, mse_scores, mae_scores, split_strategy = _evaluate_regression_cv(
        features_scaled=features_scaled,
        targets=regression_targets,
        n_splits=n_splits,
        random_state=random_state,
    )

    # Per-bin regression analysis
    unique_bins = np.unique(bin_labels)
    per_bin_r2 = {}
    per_bin_mse = {}
    per_bin_mae = {}
    per_bin_counts = {}

    for bin_id in unique_bins:
        bin_id = int(bin_id)
        mask = bin_labels == bin_id
        bin_features = features_scaled[mask]
        bin_targets = regression_targets[mask]

        per_bin_counts[bin_id] = int(mask.sum())

        if len(bin_features) < n_splits * 2:
            logger.warning(f"Bin {bin_id}: insufficient samples ({len(bin_features)}) for regression CV")
            per_bin_r2[bin_id] = 0.0
            per_bin_mse[bin_id] = 0.0
            per_bin_mae[bin_id] = 0.0
            continue

        bin_r2_scores, bin_mse_scores, bin_mae_scores, _ = _evaluate_regression_cv(
            features_scaled=bin_features,
            targets=bin_targets,
            n_splits=min(n_splits, len(bin_features) // 2),
            random_state=random_state,
        )
        if not bin_r2_scores:
            per_bin_r2[bin_id] = 0.0
            per_bin_mse[bin_id] = 0.0
            per_bin_mae[bin_id] = 0.0
            continue

        per_bin_r2[bin_id] = float(np.mean(bin_r2_scores))
        per_bin_mse[bin_id] = float(np.mean(bin_mse_scores))
        per_bin_mae[bin_id] = float(np.mean(bin_mae_scores))

    return SpacingRegressionResults(
        checkpoint_name="",  # Will be set by caller
        feature_type="",     # Will be set by caller
        r2_score=float(np.mean(r2_scores)),
        mse=float(np.mean(mse_scores)),
        mae=float(np.mean(mae_scores)),
        per_bin_r2=per_bin_r2,
        per_bin_mse=per_bin_mse,
        per_bin_mae=per_bin_mae,
        per_bin_sample_counts=per_bin_counts,
        r2_std=float(np.std(r2_scores)),
        mse_std=float(np.std(mse_scores)),
        mae_std=float(np.std(mae_scores)),
        split_strategy=split_strategy,
        target_name=SPACING_REGRESSION_TARGET_NAME,
        target_transform=SPACING_REGRESSION_TARGET_TRANSFORM,
        target_metric_space=SPACING_REGRESSION_TARGET_METRIC_SPACE,
    )


def compute_probing_per_bin(
    features: np.ndarray,
    bin_labels: np.ndarray,
    task_labels: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    ratio_values: Optional[np.ndarray] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Train linear probes separately for each spacing bin.

    Args:
        features: Feature matrix [N, D]
        bin_labels: Anisotropy bin assignments [N]
        task_labels: Task labels (e.g., dataset ID) [N]
        n_splits: CV folds
        random_state: Random seed for reproducibility

    Returns:
        Results per bin
    """
    unique_bins = np.unique(bin_labels)
    results = {}

    for bin_id in unique_bins:
        mask = bin_labels == bin_id
        bin_features = features[mask]
        bin_task_labels = task_labels[mask]

        logger.info(f"  Bin {bin_id}: {len(bin_features)} samples, {len(np.unique(bin_task_labels))} classes")

        acc, bal_acc, cv_scores = train_linear_probe(
            bin_features,
            bin_task_labels,
            n_splits=n_splits,
            random_state=random_state,
            ratio_values=ratio_values[mask] if ratio_values is not None else None,
        )

        results[int(bin_id)] = {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "cv_scores": cv_scores,
            "n_samples": len(bin_features),
            "n_classes": len(np.unique(bin_task_labels)),
        }

    return results


def run_spacing_readout_analysis(
    features: np.ndarray,
    bin_labels: np.ndarray,
    task_labels: np.ndarray,
    checkpoint_name: str,
    feature_type: str = "cls",
    n_cv_splits: int = 5,
    output_dir: Optional[Path] = None,
    anisotropy_ratios: Optional[np.ndarray] = None,
    random_state: int = 42,
    bin_names: Optional[Dict[int, str]] = None,
    binning_scheme: str = DEFAULT_BINNING_SCHEME,
    analysis_name: str = "default",
    compute_balanced_sensitivity: bool = True,
) -> Tuple[SpacingReadoutResults, Optional[SpacingRegressionResults]]:
    """
    Run spacing readout analysis.

    Task: Classify which dataset a volume comes from (proxy for general representation quality).

    Args:
        features: Feature matrix [N, D]
        bin_labels: Anisotropy bin for each sample [N]
        task_labels: Task labels (dataset IDs) [N]
        checkpoint_name: Name of checkpoint being evaluated
        feature_type: Type of features (cls, avg_pool, multilayer)
        n_cv_splits: Number of cross-validation folds
        output_dir: Directory to save results
        anisotropy_ratios: Continuous anisotropy ratios for regression probe [N]
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (SpacingReadoutResults, SpacingRegressionResults or None)
    """
    output_dir = _resolve_results_dir(output_dir, analysis_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running spacing readout analysis for {checkpoint_name} ({feature_type})")
    logger.info(f"Features shape: {features.shape}, Unique bins: {np.unique(bin_labels)}")
    logger.info(f"Task classes: {len(np.unique(task_labels))}")
    bin_names = bin_names or get_bin_name_map(binning_scheme)

    # Overall probing (all bins together)
    logger.info("Computing overall probing accuracy...")
    overall_acc, overall_bal_acc, _ = train_linear_probe(
        features,
        task_labels,
        n_splits=n_cv_splits,
        random_state=random_state,
        ratio_values=anisotropy_ratios,
    )

    # Per-bin probing
    logger.info("Computing per-bin probing accuracy...")
    bin_results = compute_probing_per_bin(
        features,
        bin_labels,
        task_labels,
        n_splits=n_cv_splits,
        random_state=random_state,
        ratio_values=anisotropy_ratios,
    )

    # Spacing regression probe (if anisotropy_ratios provided)
    spacing_reg_results = None
    if anisotropy_ratios is not None:
        logger.info("Computing spacing regression probe...")
        spacing_reg_results = spacing_regression_probe(
            features=features,
            anisotropy_ratios=anisotropy_ratios,
            bin_labels=bin_labels,
            n_splits=n_cv_splits,
            random_state=random_state,
        )
        spacing_reg_results.checkpoint_name = checkpoint_name
        spacing_reg_results.feature_type = feature_type

        # Log spacing regression results
        logger.info(f"Spacing Regression R²: {spacing_reg_results.r2_score:.4f} ± {spacing_reg_results.r2_std:.4f}")
        logger.info(f"  (Lower R² = better spacing invariance)")

    # Aggregate metrics
    bin_accuracies = {k: v["accuracy"] for k, v in bin_results.items()}
    bin_balanced_accuracies = {k: v["balanced_accuracy"] for k, v in bin_results.items()}
    bin_sample_counts = {k: v["n_samples"] for k, v in bin_results.items()}

    valid_accs = [a for a in bin_accuracies.values() if a > 0]
    accuracy_std = float(np.std(valid_accs)) if len(valid_accs) > 1 else 0.0
    max_gap = float(max(valid_accs) - min(valid_accs)) if len(valid_accs) > 1 else 0.0

    cv_scores_per_bin = {k: v["cv_scores"] for k, v in bin_results.items()}

    results = SpacingReadoutResults(
        checkpoint_name=checkpoint_name,
        feature_type=feature_type,
        bin_accuracies=bin_accuracies,
        bin_balanced_accuracies=bin_balanced_accuracies,
        bin_sample_counts=bin_sample_counts,
        overall_accuracy=overall_acc,
        overall_balanced_accuracy=overall_bal_acc,
        accuracy_std_across_bins=accuracy_std,
        max_bin_gap=max_gap,
        cv_scores_per_bin=cv_scores_per_bin,
    )

    balanced_spacing_readout = None
    if compute_balanced_sensitivity:
        balanced_idx, original_bin_counts, sample_count_per_bin = _select_balanced_bin_indices(
            bin_labels,
            random_state=random_state,
        )
        if balanced_idx is not None:
            balanced_features = features[balanced_idx]
            balanced_bins = bin_labels[balanced_idx]
            balanced_task_labels = task_labels[balanced_idx]
            balanced_acc, balanced_bal_acc, _ = train_linear_probe(
                balanced_features,
                balanced_task_labels,
                n_splits=n_cv_splits,
                random_state=random_state,
                ratio_values=anisotropy_ratios[balanced_idx] if anisotropy_ratios is not None else None,
            )
            balanced_bin_results = compute_probing_per_bin(
                balanced_features,
                balanced_bins,
                balanced_task_labels,
                n_splits=n_cv_splits,
                random_state=random_state,
                ratio_values=anisotropy_ratios[balanced_idx] if anisotropy_ratios is not None else None,
            )
            balanced_spacing_readout = {
                "sample_count_per_bin": int(sample_count_per_bin),
                "total_selected_samples": int(len(balanced_idx)),
                "original_bin_counts": original_bin_counts,
                "overall_accuracy": balanced_acc,
                "overall_balanced_accuracy": balanced_bal_acc,
                "bin_accuracies": {k: v["accuracy"] for k, v in balanced_bin_results.items()},
                "bin_balanced_accuracies": {k: v["balanced_accuracy"] for k, v in balanced_bin_results.items()},
                "bin_sample_counts": {k: v["n_samples"] for k, v in balanced_bin_results.items()},
            }
            if anisotropy_ratios is not None:
                balanced_spacing_reg = spacing_regression_probe(
                    features=balanced_features,
                    anisotropy_ratios=anisotropy_ratios[balanced_idx],
                    bin_labels=balanced_bins,
                    n_splits=n_cv_splits,
                    random_state=random_state,
                )
                balanced_spacing_readout["spacing_regression"] = serialize_spacing_regression_results(balanced_spacing_reg)
            results.balanced_bin_sensitivity = balanced_spacing_readout
    else:
        logger.info("Skipping balanced spacing-readout sensitivity")

    # Save results
    results_dict = _convert_numpy_types({
        "checkpoint": checkpoint_name,
        "feature_type": feature_type,
        "binning_scheme": binning_scheme,
        "overall_accuracy": overall_acc,
        "overall_balanced_accuracy": overall_bal_acc,
        "bin_accuracies": bin_accuracies,
        "bin_balanced_accuracies": bin_balanced_accuracies,
        "bin_sample_counts": bin_sample_counts,
        "accuracy_std_across_bins": accuracy_std,
        "max_bin_gap": max_gap,
        "bin_details": bin_results,
    })

    # Add spacing regression results if available
    if spacing_reg_results is not None:
        results_dict["spacing_regression"] = serialize_spacing_regression_results(spacing_reg_results)
    if balanced_spacing_readout is not None:
        results_dict["balanced_bin_sensitivity"] = _convert_numpy_types(balanced_spacing_readout)

    results_path = output_dir / f"{checkpoint_name}_{feature_type}_spacing_readout.json"
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    logger.info(f"Saved results to {results_path}")

    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Spacing Readout Summary: {checkpoint_name} ({feature_type})")
    logger.info(f"{'='*50}")
    logger.info(f"Overall accuracy: {overall_acc:.4f}")
    logger.info(f"Overall balanced accuracy: {overall_bal_acc:.4f}")
    logger.info(f"Per-bin accuracies:")

    for bin_id in sorted(bin_accuracies.keys()):
        bin_name = bin_names.get(bin_id, f"Bin {bin_id}")
        logger.info(f"  {bin_name}: {bin_accuracies[bin_id]:.4f} (n={bin_sample_counts[bin_id]})")
    logger.info(f"Accuracy std across bins: {accuracy_std:.4f}")
    logger.info(f"Max bin gap: {max_gap:.4f}")

    # Log spacing regression summary if available
    if spacing_reg_results is not None:
        logger.info(f"\nSpacing Regression Probe (lower R² = more spacing-invariant):")
        logger.info(
            "  Overall R²: %.4f ± %.4f (split=%s)",
            spacing_reg_results.r2_score,
            spacing_reg_results.r2_std,
            spacing_reg_results.split_strategy,
        )
        logger.info(f"  Per-bin R²:")
        for bin_id in sorted(spacing_reg_results.per_bin_r2.keys()):
            bin_name = bin_names.get(bin_id, f"Bin {bin_id}")
            r2 = spacing_reg_results.per_bin_r2[bin_id]
            n = spacing_reg_results.per_bin_sample_counts[bin_id]
            logger.info(f"    {bin_name}: {r2:.4f} (n={n})")
    if balanced_spacing_readout is not None:
        logger.info(
            "Balanced-bin sensitivity (n/bin=%d): overall acc=%.4f, overall bal acc=%.4f",
            balanced_spacing_readout["sample_count_per_bin"],
            balanced_spacing_readout["overall_accuracy"],
            balanced_spacing_readout["overall_balanced_accuracy"],
        )
        if "spacing_regression" in balanced_spacing_readout:
            balanced_reg = balanced_spacing_readout["spacing_regression"]
            logger.info(
                "  Balanced spacing regression: R²=%.4f ± %.4f (split=%s)",
                balanced_reg["r2_score"],
                balanced_reg["r2_std"],
                balanced_reg["split_strategy"],
            )

    return results, spacing_reg_results


def run_spacing_readout_analysis_multiseed(
    features: np.ndarray,
    bin_labels: np.ndarray,
    task_labels: np.ndarray,
    checkpoint_name: str,
    feature_type: str = "cls",
    n_cv_splits: int = 5,
    output_dir: Optional[Path] = None,
    anisotropy_ratios: Optional[np.ndarray] = None,
    seeds: Optional[List[int]] = None,
    bin_names: Optional[Dict[int, str]] = None,
    binning_scheme: str = DEFAULT_BINNING_SCHEME,
    analysis_name: str = "default",
) -> MultiSeedSpacingReadoutResults:
    """
    Run spacing readout analysis multiple times with different random seeds.

    Provides reproducibility assessment via mean ± std across seeds.

    Args:
        features: Feature matrix [N, D]
        bin_labels: Anisotropy bin for each sample [N]
        task_labels: Task labels (dataset IDs) [N]
        checkpoint_name: Name of checkpoint being evaluated
        feature_type: Type of features (cls, avg_pool, multilayer)
        n_cv_splits: Number of cross-validation folds
        output_dir: Directory to save results
        anisotropy_ratios: Continuous anisotropy ratios for regression probe [N]
        seeds: List of random seeds (default: EVALUATION_SEEDS)

    Returns:
        MultiSeedSpacingReadoutResults with aggregated metrics
    """
    if seeds is None:
        seeds = EVALUATION_SEEDS

    output_dir = _resolve_results_dir(output_dir, analysis_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running multi-seed spacing readout analysis for {checkpoint_name} ({feature_type})")
    logger.info(f"Seeds: {seeds}")
    bin_names = bin_names or get_bin_name_map(binning_scheme)

    per_seed_results = {}
    per_seed_spacing = {}

    # Collect metrics across seeds
    overall_accs = []
    overall_bal_accs = []
    bin_accs_per_seed = {bid: [] for bid in np.unique(bin_labels)}
    spacing_r2s = []
    spacing_mses = []

    for seed in seeds:
        logger.info(f"\n--- Seed {seed} ---")
        spacing_readout_result, spacing_result = run_spacing_readout_analysis(
            features=features,
            bin_labels=bin_labels,
            task_labels=task_labels,
            checkpoint_name=checkpoint_name,
            feature_type=feature_type,
            n_cv_splits=n_cv_splits,
            output_dir=output_dir / f"seed_{seed}",
            anisotropy_ratios=anisotropy_ratios,
            random_state=seed,
            bin_names=bin_names,
            binning_scheme=binning_scheme,
            analysis_name=analysis_name,
        )

        per_seed_results[seed] = spacing_readout_result
        overall_accs.append(spacing_readout_result.overall_accuracy)
        overall_bal_accs.append(spacing_readout_result.overall_balanced_accuracy)

        for bid, acc in spacing_readout_result.bin_accuracies.items():
            if bid in bin_accs_per_seed:
                bin_accs_per_seed[bid].append(acc)

        if spacing_result is not None:
            per_seed_spacing[seed] = spacing_result
            spacing_r2s.append(spacing_result.r2_score)
            spacing_mses.append(spacing_result.mse)

    # Aggregate
    bin_sample_counts = per_seed_results[seeds[0]].bin_sample_counts
    bin_accs_mean = {bid: float(np.mean(accs)) for bid, accs in bin_accs_per_seed.items()}
    bin_accs_std = {bid: float(np.std(accs)) for bid, accs in bin_accs_per_seed.items()}

    result = MultiSeedSpacingReadoutResults(
        checkpoint_name=checkpoint_name,
        feature_type=feature_type,
        seeds=seeds,
        overall_accuracy_mean=float(np.mean(overall_accs)),
        overall_accuracy_std=float(np.std(overall_accs)),
        overall_balanced_accuracy_mean=float(np.mean(overall_bal_accs)),
        overall_balanced_accuracy_std=float(np.std(overall_bal_accs)),
        bin_accuracies_mean=bin_accs_mean,
        bin_accuracies_std=bin_accs_std,
        bin_sample_counts=bin_sample_counts,
        spacing_r2_mean=float(np.mean(spacing_r2s)) if spacing_r2s else None,
        spacing_r2_std=float(np.std(spacing_r2s)) if spacing_r2s else None,
        spacing_mse_mean=float(np.mean(spacing_mses)) if spacing_mses else None,
        spacing_mse_std=float(np.std(spacing_mses)) if spacing_mses else None,
        per_seed_results=per_seed_results,
        per_seed_spacing=per_seed_spacing if per_seed_spacing else None,
    )

    # Save aggregated results
    results_dict = _convert_numpy_types({
        "checkpoint": checkpoint_name,
        "feature_type": feature_type,
        "binning_scheme": binning_scheme,
        "seeds": seeds,
        "overall_accuracy": {"mean": result.overall_accuracy_mean, "std": result.overall_accuracy_std},
        "overall_balanced_accuracy": {"mean": result.overall_balanced_accuracy_mean, "std": result.overall_balanced_accuracy_std},
        "bin_accuracies_mean": bin_accs_mean,
        "bin_accuracies_std": bin_accs_std,
        "bin_sample_counts": bin_sample_counts,
        "spacing_regression": {
            "r2": {"mean": result.spacing_r2_mean, "std": result.spacing_r2_std},
            "mse": {"mean": result.spacing_mse_mean, "std": result.spacing_mse_std},
        } if result.spacing_r2_mean is not None else None,
    })

    results_path = output_dir / f"{checkpoint_name}_{feature_type}_spacing_readout_multiseed.json"
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    logger.info(f"\nSaved multi-seed results to {results_path}")

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Multi-Seed Summary: {checkpoint_name} ({feature_type})")
    logger.info(f"{'='*60}")
    logger.info(f"Overall accuracy: {result.overall_accuracy_mean:.4f} ± {result.overall_accuracy_std:.4f}")
    logger.info(f"Overall balanced accuracy: {result.overall_balanced_accuracy_mean:.4f} ± {result.overall_balanced_accuracy_std:.4f}")

    logger.info(f"Per-bin accuracies (mean ± std):")
    for bid in sorted(bin_accs_mean.keys()):
        bin_name = bin_names.get(bid, f"Bin {bid}")
        logger.info(f"  {bin_name}: {bin_accs_mean[bid]:.4f} ± {bin_accs_std[bid]:.4f}")

    if result.spacing_r2_mean is not None:
        logger.info(f"\nSpacing Regression R²: {result.spacing_r2_mean:.4f} ± {result.spacing_r2_std:.4f}")

    return result


def compare_checkpoint_spacing_readouts(
    results_list: List[SpacingReadoutResults],
    output_dir: Optional[Path] = None,
    analysis_name: str = "default",
) -> Dict[str, Any]:
    """
    Compare spacing readout results across multiple checkpoints.

    Key question: Do spacing-aware models have smaller performance gaps across bins?

    Args:
        results_list: List of SpacingReadoutResults from different checkpoints
        output_dir: Directory to save comparison

    Returns:
        Comparison summary
    """
    output_dir = _resolve_results_dir(output_dir, analysis_name)

    comparison = {
        "checkpoints": [],
        "overall_accuracies": [],
        "bin_gaps": [],
        "accuracy_stds": [],
        "per_bin_comparison": {},
    }

    all_bins = set()
    for r in results_list:
        all_bins.update(r.bin_accuracies.keys())

    for bin_id in sorted(all_bins):
        comparison["per_bin_comparison"][int(bin_id)] = []

    for r in results_list:
        comparison["checkpoints"].append(r.checkpoint_name)
        comparison["overall_accuracies"].append(r.overall_accuracy)
        comparison["bin_gaps"].append(r.max_bin_gap)
        comparison["accuracy_stds"].append(r.accuracy_std_across_bins)

        for bin_id in sorted(all_bins):
            acc = r.bin_accuracies.get(bin_id, 0.0)
            comparison["per_bin_comparison"][int(bin_id)].append(acc)

    # Statistical comparison
    comparison["summary"] = {
        "best_overall": comparison["checkpoints"][np.argmax(comparison["overall_accuracies"])],
        "most_consistent": comparison["checkpoints"][np.argmin(comparison["bin_gaps"])],
        "spacing_aware_reduces_gap": None,  # To be computed
    }

    # Check if spacing-aware models have smaller gaps
    sa_models = [r for r in results_list if "_sa" in r.checkpoint_name]
    non_sa_models = [r for r in results_list if "_sa" not in r.checkpoint_name]

    if sa_models and non_sa_models:
        sa_mean_gap = np.mean([r.max_bin_gap for r in sa_models])
        non_sa_mean_gap = np.mean([r.max_bin_gap for r in non_sa_models])
        comparison["summary"]["spacing_aware_reduces_gap"] = sa_mean_gap < non_sa_mean_gap
        comparison["summary"]["sa_mean_gap"] = float(sa_mean_gap)
        comparison["summary"]["non_sa_mean_gap"] = float(non_sa_mean_gap)

    # Save comparison
    comparison = _convert_numpy_types(comparison)
    comparison_path = output_dir / "spacing_readout_checkpoint_comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"Saved comparison to {comparison_path}")

    return comparison


# =============================================================================
# CROSS-BIN TRANSFER EXPERIMENTS
# =============================================================================

@dataclass
class CrossBinTransferResults:
    """Results from cross-bin transfer experiments."""
    checkpoint_name: str
    feature_type: str
    task_type: str  # "domain" or "semantic"

    # Transfer matrix: transfer_matrix[train_bin][test_bin] = accuracy
    transfer_matrix: Dict[int, Dict[int, float]]

    # In-bin accuracy (diagonal of transfer matrix)
    in_bin_accuracies: Dict[int, float]

    # Cross-bin accuracy (off-diagonal average)
    cross_bin_accuracy: float

    # Transfer gap: in-bin - cross-bin
    transfer_gap: float

    # Per-bin sample counts
    bin_sample_counts: Dict[int, int]

    metric_name: str = "accuracy"

    # Metadata
    dataset: str = "unknown"
    n_classes: int = 0


def train_and_evaluate_transfer(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    random_state: int = 42,
) -> Tuple[float, float]:
    """
    Train on one set, evaluate on another (transfer evaluation).

    Args:
        train_features: Training features [N_train, D]
        train_labels: Training labels [N_train]
        test_features: Test features [N_test, D]
        test_labels: Test labels [N_test]
        random_state: Random seed

    Returns:
        Tuple of (accuracy, balanced_accuracy)
    """
    # Skip if insufficient data
    unique_train = np.unique(train_labels)
    unique_test = np.unique(test_labels)

    if len(train_features) < 10 or len(test_features) < 10:
        logger.warning("Insufficient samples for transfer evaluation")
        return 0.0, 0.0

    if len(unique_train) < 2:
        logger.warning("Insufficient classes in training set")
        return 0.0, 0.0

    # Check label overlap
    overlap = set(unique_train) & set(unique_test)
    if len(overlap) == 0:
        logger.warning("No overlapping classes between train and test")
        return 0.0, 0.0

    # Filter to overlapping classes only
    if len(overlap) < len(unique_train) or len(overlap) < len(unique_test):
        train_mask = np.isin(train_labels, list(overlap))
        test_mask = np.isin(test_labels, list(overlap))
        train_features = train_features[train_mask]
        train_labels = train_labels[train_mask]
        test_features = test_features[test_mask]
        test_labels = test_labels[test_mask]

    # Standardize features (fit on train, apply to test)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)

    # Train classifier
    clf = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        random_state=random_state,
        n_jobs=4,  # Limit parallelism to prevent resource leaks
    )

    try:
        clf.fit(train_scaled, train_labels)
        predictions = clf.predict(test_scaled)

        acc = accuracy_score(test_labels, predictions)
        bal_acc = balanced_accuracy_score(test_labels, predictions)

        return float(acc), float(bal_acc)
    except Exception as e:
        logger.warning(f"Transfer evaluation failed: {e}")
        return 0.0, 0.0


def run_cross_bin_transfer(
    features: np.ndarray,
    bin_labels: np.ndarray,
    task_labels: np.ndarray,
    checkpoint_name: str,
    feature_type: str = "cls",
    task_type: str = "domain",
    dataset: str = "unknown",
    output_dir: Optional[Path] = None,
    analysis_name: str = "default",
) -> CrossBinTransferResults:
    """
    Run cross-bin transfer experiments.

    For each pair of bins (train_bin, test_bin), train a probe on train_bin
    and evaluate on test_bin. This measures how well representations transfer
    across spacing regimes.

    Args:
        features: Feature matrix [N, D]
        bin_labels: Anisotropy bin for each sample [N]
        task_labels: Task labels (domain or semantic) [N]
        checkpoint_name: Name of checkpoint
        feature_type: Type of features
        task_type: "domain" or "semantic"
        dataset: Dataset name
        output_dir: Output directory for results

    Returns:
        CrossBinTransferResults dataclass
    """
    output_dir = _resolve_results_dir(output_dir, analysis_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    unique_bins = sorted(np.unique(bin_labels))
    n_classes = len(np.unique(task_labels))

    logger.info(f"Running cross-bin transfer for {checkpoint_name} ({task_type})")
    logger.info(f"Bins: {unique_bins}, Classes: {n_classes}")

    # Build transfer matrix
    transfer_matrix = {int(b): {} for b in unique_bins}
    bin_sample_counts = {}

    for train_bin in unique_bins:
        train_mask = bin_labels == train_bin
        train_features = features[train_mask]
        train_labels = task_labels[train_mask]
        bin_sample_counts[int(train_bin)] = int(train_mask.sum())

        for test_bin in unique_bins:
            test_mask = bin_labels == test_bin
            test_features = features[test_mask]
            test_labels = task_labels[test_mask]

            # Compute transfer accuracy
            acc, bal_acc = train_and_evaluate_transfer(
                train_features, train_labels,
                test_features, test_labels
            )

            transfer_matrix[int(train_bin)][int(test_bin)] = acc

            logger.info(f"  Train Bin {train_bin} -> Test Bin {test_bin}: {acc:.4f}")

    # Compute summary metrics
    in_bin_accs = {int(b): transfer_matrix[int(b)][int(b)] for b in unique_bins}

    # Cross-bin accuracy (off-diagonal average)
    cross_bin_vals = []
    for train_bin in unique_bins:
        for test_bin in unique_bins:
            if train_bin != test_bin:
                cross_bin_vals.append(transfer_matrix[int(train_bin)][int(test_bin)])

    cross_bin_acc = float(np.mean(cross_bin_vals)) if cross_bin_vals else 0.0
    in_bin_mean = float(np.mean(list(in_bin_accs.values())))
    transfer_gap = in_bin_mean - cross_bin_acc

    results = CrossBinTransferResults(
        checkpoint_name=checkpoint_name,
        feature_type=feature_type,
        task_type=task_type,
        metric_name="accuracy",
        transfer_matrix=transfer_matrix,
        in_bin_accuracies=in_bin_accs,
        cross_bin_accuracy=cross_bin_acc,
        transfer_gap=transfer_gap,
        bin_sample_counts=bin_sample_counts,
        dataset=dataset,
        n_classes=n_classes,
    )

    # Save results
    results_dict = _convert_numpy_types({
        "checkpoint": checkpoint_name,
        "feature_type": feature_type,
        "task_type": task_type,
        "metric_name": results.metric_name,
        "dataset": dataset,
        "n_classes": n_classes,
        "transfer_matrix": transfer_matrix,
        "in_bin_accuracies": in_bin_accs,
        "cross_bin_accuracy": cross_bin_acc,
        "transfer_gap": transfer_gap,
        "bin_sample_counts": bin_sample_counts,
    })

    results_path = output_dir / f"{checkpoint_name}_{feature_type}_{task_type}_transfer.json"
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    logger.info(f"Saved transfer results to {results_path}")

    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Cross-Bin Transfer Summary ({task_type})")
    logger.info(f"{'='*50}")
    logger.info(f"In-bin accuracy (mean): {in_bin_mean:.4f}")
    logger.info(f"Cross-bin accuracy (mean): {cross_bin_acc:.4f}")
    logger.info(f"Transfer gap: {transfer_gap:.4f}")

    return results


# =============================================================================
# SEMANTIC PROBING (Multi-Label Classification)
# =============================================================================

def train_multilabel_probe(
    features: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    ratio_values: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[int, float], List[float]]:
    """
    Train a multi-label classifier (one-vs-rest) with cross-validation.

    Args:
        features: Feature matrix [N, D]
        labels: Multi-label matrix [N, K] where K is number of labels
        n_splits: Number of CV folds
        random_state: Random seed

    Returns:
        Tuple of (mean_balanced_accuracy, per_label_balanced_accuracy, cv_scores)
    """
    n_samples, n_labels = labels.shape

    if n_labels == 0:
        logger.warning("No semantic labels available for probing")
        return 0.0, {}, []

    if n_samples < n_splits:
        logger.warning(f"Insufficient samples ({n_samples}) for CV")
        return 0.0, {}, []

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Train per-label classifiers
    per_label_accs = {}
    all_accs = []

    for label_idx in range(n_labels):
        label_col = labels[:, label_idx]

        # Skip if label is too rare or too common
        pos_count = label_col.sum()
        if pos_count < n_splits or pos_count > n_samples - n_splits:
            per_label_accs[label_idx] = 0.0
            continue

        clf = LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            random_state=random_state,
            n_jobs=4,  # Limit parallelism to prevent resource leaks
        )

        try:
            cv_splits, _, _ = _resolve_stratified_cv(
                n_splits=n_splits,
                random_state=random_state,
                label_values=label_col,
                ratio_values=ratio_values,
            )
            if cv_splits is None:
                per_label_accs[label_idx] = 0.0
                continue
            scores = cross_val_score(
                clf,
                features_scaled,
                label_col,
                cv=cv_splits,
                scoring='balanced_accuracy',
            )
            per_label_accs[label_idx] = float(np.mean(scores))
            all_accs.append(float(np.mean(scores)))
        except Exception as e:
            logger.warning(f"Label {label_idx} failed: {e}")
            per_label_accs[label_idx] = 0.0

    mean_acc = float(np.mean(all_accs)) if all_accs else 0.0
    return mean_acc, per_label_accs, all_accs


@dataclass
class SemanticProbingResults:
    """Results from semantic (multi-label) probing analysis."""
    checkpoint_name: str
    feature_type: str

    # Overall score (mean balanced accuracy across informative labels)
    mean_balanced_accuracy: float

    # Per-bin mean balanced accuracy
    bin_balanced_accuracies: Dict[int, float]

    # Per-label balanced accuracies
    per_label_balanced_accuracies: Dict[int, float]

    # Label names
    label_names: List[str]

    # Sample counts
    bin_sample_counts: Dict[int, int]

    # Balanced-bin sensitivity output (natural estimate remains top-level)
    balanced_bin_sensitivity: Optional[Dict[str, Any]] = None

    # Cross-bin transfer results (if computed)
    transfer_results: Optional[CrossBinTransferResults] = None


def run_semantic_probing(
    features: np.ndarray,
    bin_labels: np.ndarray,
    semantic_labels: np.ndarray,
    checkpoint_name: str,
    feature_type: str = "cls",
    label_names: Optional[List[str]] = None,
    anisotropy_ratios: Optional[np.ndarray] = None,
    n_cv_splits: int = 5,
    output_dir: Optional[Path] = None,
    analysis_name: str = "default",
    manifest_variant: str = "original_bins",
    random_state: int = 42,
    compute_balanced_sensitivity: bool = True,
) -> SemanticProbingResults:
    """
    Run semantic probing with multi-label classification.

    Args:
        features: Feature matrix [N, D]
        bin_labels: Anisotropy bin for each sample [N]
        semantic_labels: Multi-label matrix [N, K]
        checkpoint_name: Name of checkpoint
        feature_type: Type of features
        label_names: Names for each label column
        n_cv_splits: Number of CV folds
        output_dir: Output directory

    Returns:
        SemanticProbingResults dataclass
    """
    output_dir = _resolve_checkpoint_results_dir(
        output_dir,
        checkpoint_name,
        feature_type,
        analysis_name,
        manifest_variant,
    )

    logger.info(f"Running semantic probing for {checkpoint_name}")
    logger.info(f"Features: {features.shape}, Labels: {semantic_labels.shape}")

    if semantic_labels.shape[1] == 0:
        raise ValueError("Semantic probing requires at least one informative label")

    unique_bins = sorted(np.unique(bin_labels))
    n_labels = semantic_labels.shape[1]

    if label_names is None:
        label_names = [f"label_{i}" for i in range(n_labels)]

    # Overall probing
    logger.info("Computing overall semantic balanced accuracy...")
    mean_acc, per_label_accs, _ = train_multilabel_probe(
        features,
        semantic_labels,
        n_splits=n_cv_splits,
        ratio_values=anisotropy_ratios,
        random_state=random_state,
    )

    # Per-bin probing
    logger.info("Computing per-bin semantic balanced accuracy...")
    bin_accuracies = {}
    bin_sample_counts = {}

    for bin_id in unique_bins:
        mask = bin_labels == bin_id
        bin_features = features[mask]
        bin_semantic = semantic_labels[mask]
        bin_sample_counts[int(bin_id)] = int(mask.sum())

        acc, _, _ = train_multilabel_probe(
            bin_features,
            bin_semantic,
            n_splits=n_cv_splits,
            ratio_values=anisotropy_ratios[mask] if anisotropy_ratios is not None else None,
            random_state=random_state,
        )
        bin_accuracies[int(bin_id)] = acc
        logger.info(f"  Bin {bin_id}: {acc:.4f} (n={bin_sample_counts[int(bin_id)]})")

    results = SemanticProbingResults(
        checkpoint_name=checkpoint_name,
        feature_type=feature_type,
        mean_balanced_accuracy=mean_acc,
        bin_balanced_accuracies=bin_accuracies,
        per_label_balanced_accuracies=per_label_accs,
        label_names=label_names,
        bin_sample_counts=bin_sample_counts,
    )

    balanced_semantic = None
    if compute_balanced_sensitivity:
        balanced_idx, original_bin_counts, sample_count_per_bin = _select_balanced_bin_indices(
            bin_labels,
            random_state=random_state,
        )
        if balanced_idx is not None:
            balanced_features = features[balanced_idx]
            balanced_bins = bin_labels[balanced_idx]
            balanced_semantic_labels = semantic_labels[balanced_idx]
            balanced_ratios = anisotropy_ratios[balanced_idx] if anisotropy_ratios is not None else None
            balanced_mean_acc, balanced_per_label_accs, _ = train_multilabel_probe(
                balanced_features,
                balanced_semantic_labels,
                n_splits=n_cv_splits,
                ratio_values=balanced_ratios,
                random_state=random_state,
            )
            balanced_bin_accuracies = {}
            balanced_bin_counts = {}
            for bin_id in sorted(np.unique(balanced_bins)):
                mask = balanced_bins == bin_id
                acc, _, _ = train_multilabel_probe(
                    balanced_features[mask],
                    balanced_semantic_labels[mask],
                    n_splits=n_cv_splits,
                    ratio_values=balanced_ratios[mask] if balanced_ratios is not None else None,
                )
                balanced_bin_accuracies[int(bin_id)] = acc
                balanced_bin_counts[int(bin_id)] = int(mask.sum())
            balanced_semantic = {
                "sample_count_per_bin": int(sample_count_per_bin),
                "total_selected_samples": int(len(balanced_idx)),
                "original_bin_counts": original_bin_counts,
                "mean_balanced_accuracy": balanced_mean_acc,
                "bin_balanced_accuracies": balanced_bin_accuracies,
                "per_label_balanced_accuracies": balanced_per_label_accs,
                "bin_sample_counts": balanced_bin_counts,
            }
            results.balanced_bin_sensitivity = balanced_semantic
    else:
        logger.info("Skipping balanced semantic-probing sensitivity")

    # Save results
    results_dict = _convert_numpy_types({
        "checkpoint": checkpoint_name,
        "feature_type": feature_type,
        "metric_name": "balanced_accuracy",
        "mean_balanced_accuracy": mean_acc,
        "bin_balanced_accuracies": bin_accuracies,
        "per_label_balanced_accuracies": per_label_accs,
        "label_names": label_names,
        "bin_sample_counts": bin_sample_counts,
    })
    if balanced_semantic is not None:
        results_dict["balanced_bin_sensitivity"] = _convert_numpy_types(balanced_semantic)

    results_path = output_dir / "semantic_probing.json"
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    logger.info(f"Saved semantic probing results to {results_path}")
    logger.info("Semantic probing summary:")
    logger.info("  Natural bins: mean balanced accuracy=%.4f", mean_acc)
    if balanced_semantic is not None:
        logger.info(
            "  Balanced-bin sensitivity (n/bin=%d): mean balanced accuracy=%.4f",
            balanced_semantic["sample_count_per_bin"],
            balanced_semantic["mean_balanced_accuracy"],
        )

    return results


# =============================================================================
# COMBINED SPACING READOUT ANALYSIS (Domain + Semantic + Transfer)
# =============================================================================

@dataclass
class FullReadoutResults:
    """Complete spacing readout results with all evaluation modes."""
    checkpoint_name: str
    feature_type: str

    # Domain probing (existing functionality)
    domain_probing: SpacingReadoutResults

    # Semantic probing (multi-label organ classification)
    semantic_probing: Optional[SemanticProbingResults] = None

    # Cross-bin transfer - domain
    domain_transfer: Optional[CrossBinTransferResults] = None

    # Cross-bin transfer - semantic (same multi-label task as semantic probing)
    semantic_transfer: Optional[CrossBinTransferResults] = None


def train_and_evaluate_multilabel_transfer(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    random_state: int = 42,
    min_class_count: int = 5,
) -> Tuple[float, Dict[int, float]]:
    """Train one binary classifier per label and evaluate transfer with balanced accuracy."""
    if train_labels.ndim != 2 or test_labels.ndim != 2:
        raise ValueError("Multi-label transfer expects label matrices [N, K]")

    if len(train_features) < 10 or len(test_features) < 10 or train_labels.shape[1] == 0:
        return 0.0, {}

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)

    per_label_scores = {}
    valid_scores = []

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
            solver='lbfgs',
            random_state=random_state,
            n_jobs=4,
        )
        try:
            clf.fit(train_scaled, y_train)
            predictions = clf.predict(test_scaled)
            score = balanced_accuracy_score(y_test, predictions)
            per_label_scores[label_idx] = float(score)
            valid_scores.append(float(score))
        except Exception as e:
            logger.warning(f"Multi-label transfer failed for label {label_idx}: {e}")

    if not valid_scores:
        return 0.0, {}

    return float(np.mean(valid_scores)), per_label_scores


def run_multilabel_cross_bin_transfer(
    features: np.ndarray,
    bin_labels: np.ndarray,
    semantic_labels: np.ndarray,
    checkpoint_name: str,
    feature_type: str = "cls",
    dataset: str = "unknown",
    output_dir: Optional[Path] = None,
    analysis_name: str = "default",
    manifest_variant: str = "original_bins",
    random_state: int = 42,
) -> CrossBinTransferResults:
    """Run cross-bin transfer on the same multi-label semantic task used in probing."""
    output_dir = _resolve_checkpoint_results_dir(
        output_dir,
        checkpoint_name,
        feature_type,
        analysis_name,
        manifest_variant,
    )

    if semantic_labels.ndim != 2 or semantic_labels.shape[1] == 0:
        raise ValueError("Semantic transfer requires at least one informative semantic label")

    unique_bins = sorted(np.unique(bin_labels))
    transfer_matrix = {int(b): {} for b in unique_bins}
    bin_sample_counts = {}

    logger.info(f"Running cross-bin transfer for {checkpoint_name} (semantic multi-label)")
    logger.info(f"Bins: {unique_bins}, Labels: {semantic_labels.shape[1]}")

    for train_bin in unique_bins:
        train_mask = bin_labels == train_bin
        train_features = features[train_mask]
        train_task_labels = semantic_labels[train_mask]
        bin_sample_counts[int(train_bin)] = int(train_mask.sum())

        for test_bin in unique_bins:
            test_mask = bin_labels == test_bin
            test_features = features[test_mask]
            test_task_labels = semantic_labels[test_mask]

            score, _ = train_and_evaluate_multilabel_transfer(
                train_features,
                train_task_labels,
                test_features,
                test_task_labels,
                random_state=random_state,
            )
            transfer_matrix[int(train_bin)][int(test_bin)] = score
            logger.info(
                f"  Train Bin {train_bin} -> Test Bin {test_bin}: {score:.4f} "
                "(mean balanced accuracy)"
            )

    in_bin_accs = {int(b): transfer_matrix[int(b)][int(b)] for b in unique_bins}
    cross_bin_vals = []
    for train_bin in unique_bins:
        for test_bin in unique_bins:
            if train_bin != test_bin:
                cross_bin_vals.append(transfer_matrix[int(train_bin)][int(test_bin)])

    cross_bin_acc = float(np.mean(cross_bin_vals)) if cross_bin_vals else 0.0
    in_bin_mean = float(np.mean(list(in_bin_accs.values())))
    transfer_gap = in_bin_mean - cross_bin_acc

    results = CrossBinTransferResults(
        checkpoint_name=checkpoint_name,
        feature_type=feature_type,
        task_type="semantic",
        metric_name="mean_balanced_accuracy",
        transfer_matrix=transfer_matrix,
        in_bin_accuracies=in_bin_accs,
        cross_bin_accuracy=cross_bin_acc,
        transfer_gap=transfer_gap,
        bin_sample_counts=bin_sample_counts,
        dataset=dataset,
        n_classes=int(semantic_labels.shape[1]),
    )

    results_dict = _convert_numpy_types({
        "checkpoint": checkpoint_name,
        "feature_type": feature_type,
        "task_type": "semantic",
        "metric_name": results.metric_name,
        "dataset": dataset,
        "n_classes": int(semantic_labels.shape[1]),
        "transfer_matrix": transfer_matrix,
        "in_bin_accuracies": in_bin_accs,
        "cross_bin_accuracy": cross_bin_acc,
        "transfer_gap": transfer_gap,
        "bin_sample_counts": bin_sample_counts,
    })

    results_path = output_dir / "semantic_transfer.json"
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    logger.info(f"Saved transfer results to {results_path}")

    logger.info(f"\n{'='*50}")
    logger.info("Cross-Bin Transfer Summary (semantic multi-label)")
    logger.info(f"{'='*50}")
    logger.info(f"In-bin mean balanced accuracy: {in_bin_mean:.4f}")
    logger.info(f"Cross-bin mean balanced accuracy: {cross_bin_acc:.4f}")
    logger.info(f"Transfer gap: {transfer_gap:.4f}")

    return results


def run_full_readout_analysis(
    features: np.ndarray,
    bin_labels: np.ndarray,
    domain_labels: np.ndarray,
    semantic_labels: Optional[np.ndarray] = None,
    dominant_organ_labels: Optional[np.ndarray] = None,
    checkpoint_name: str = "unknown",
    feature_type: str = "cls",
    label_names: Optional[List[str]] = None,
    dataset: str = "unknown",
    run_transfer: bool = True,
    n_cv_splits: int = 5,
    output_dir: Optional[Path] = None,
    analysis_name: str = "default",
) -> FullReadoutResults:
    """
    Run complete spacing readout analysis with domain probing, semantic probing,
    and cross-bin transfer experiments.

    Args:
        features: Feature matrix [N, D]
        bin_labels: Anisotropy bin for each sample [N]
        domain_labels: Domain/dataset labels [N]
        semantic_labels: Multi-label organ presence matrix [N, K] (optional)
        dominant_organ_labels: Single-label dominant organ [N] (optional)
        checkpoint_name: Name of checkpoint
        feature_type: Type of features
        label_names: Names for semantic labels
        dataset: Dataset name for transfer experiments
        run_transfer: Whether to run cross-bin transfer experiments
        n_cv_splits: Number of CV folds
        output_dir: Output directory

    Returns:
        FullReadoutResults dataclass
    """
    output_dir = _resolve_results_dir(output_dir, analysis_name)

    # 1. Domain probing (existing functionality)
    logger.info("\n=== Domain Probing ===")
    domain_results, _ = run_spacing_readout_analysis(
        features=features,
        bin_labels=bin_labels,
        task_labels=domain_labels,
        checkpoint_name=checkpoint_name,
        feature_type=feature_type,
        n_cv_splits=n_cv_splits,
        output_dir=output_dir,
        analysis_name=analysis_name,
    )

    # 2. Semantic probing (if labels provided)
    semantic_results = None
    if semantic_labels is not None:
        logger.info("\n=== Semantic Probing ===")
        semantic_results = run_semantic_probing(
            features=features,
            bin_labels=bin_labels,
            semantic_labels=semantic_labels,
            checkpoint_name=checkpoint_name,
            feature_type=feature_type,
            label_names=label_names,
            n_cv_splits=n_cv_splits,
            output_dir=output_dir,
            analysis_name=analysis_name,
        )

    # 3. Cross-bin transfer - domain
    domain_transfer = None
    semantic_transfer = None

    if run_transfer:
        logger.info("\n=== Cross-Bin Transfer (Domain) ===")
        domain_transfer = run_cross_bin_transfer(
            features=features,
            bin_labels=bin_labels,
            task_labels=domain_labels,
            checkpoint_name=checkpoint_name,
            feature_type=feature_type,
            task_type="domain",
            dataset=dataset,
            output_dir=output_dir,
            analysis_name=analysis_name,
        )

        # 4. Cross-bin transfer - semantic (using dominant organ)
        if dominant_organ_labels is not None:
            logger.info("\n=== Cross-Bin Transfer (Semantic) ===")
            semantic_transfer = run_cross_bin_transfer(
                features=features,
                bin_labels=bin_labels,
                task_labels=dominant_organ_labels,
                checkpoint_name=checkpoint_name,
                feature_type=feature_type,
                task_type="semantic",
                dataset=dataset,
                output_dir=output_dir,
                analysis_name=analysis_name,
            )

    return FullReadoutResults(
        checkpoint_name=checkpoint_name,
        feature_type=feature_type,
        domain_probing=domain_results,
        semantic_probing=semantic_results,
        domain_transfer=domain_transfer,
        semantic_transfer=semantic_transfer,
    )


def main():
    """Main entry point for spacing readout analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Acquisition Robustness: Spacing Readout",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python semantic_analysis.py -m abdomenatlas/original_bins/manifest_sampled.json -a abdomenatlas -c med3dino_rel_96 -f cls\n"
            "  python semantic_analysis.py -m totalsegmentermri/original_bins/manifest_sampled.json -a totalsegmentermri -c med3dino_rel_96 -f avg_pool --cv-splits 3"
        ),
    )
    parser.add_argument(
        "-m", "--manifest",
        type=Path,
        default=None,
        help="Path to manifest JSON with volumes",
    )
    parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        choices=list(CHECKPOINTS.keys()),
        default="med3dino_rel_96",
        help="Checkpoint to evaluate",
    )
    parser.add_argument(
        "-f", "--feature-type",
        type=str,
        choices=["cls", "avg_pool", "multilayer"],
        default="cls",
        help="Feature type to extract",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of cross-validation splits",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for feature extraction",
    )
    parser.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="Run analysis on all available checkpoints",
    )
    parser.add_argument(
        "-a", "--analysis-name",
        type=str,
        default=None,
        help="Dataset/analysis namespace for default output paths",
    )

    args = parser.parse_args()

    # Import here to avoid circular imports
    try:
        from .data_loader import create_acquisition_dataloader
        from .feature_extractor import FeatureExtractor
        from .config import (
            get_acquisition_manifest_path,
            get_binning_scheme_from_manifest,
            get_bin_name_map,
            resolve_acquisition_manifest_path,
        )
    except ImportError:  # pragma: no cover - direct script execution fallback
        from data_loader import create_acquisition_dataloader
        from feature_extractor import FeatureExtractor
        from config import (
            get_acquisition_manifest_path,
            get_binning_scheme_from_manifest,
            get_bin_name_map,
            resolve_acquisition_manifest_path,
        )

    # Load manifest
    if args.manifest is None:
        manifest_path = get_acquisition_manifest_path("abdomenatlas", "sampled")
    else:
        manifest_arg = Path(args.manifest)
        if manifest_arg.is_absolute():
            manifest_path = manifest_arg
        elif len(manifest_arg.parts) == 1:
            manifest_path = get_acquisition_manifest_path("abdomenatlas", manifest_arg.stem)
        else:
            manifest_path = resolve_acquisition_manifest_path(manifest_arg)

    with open(manifest_path) as f:
        manifest_data = json.load(f)

    volumes = manifest_data["volumes"]
    binning_scheme = get_binning_scheme_from_manifest(manifest_data)
    bin_names = get_bin_name_map(binning_scheme)
    analysis_name = args.analysis_name or get_dataset_name_from_manifest_path(manifest_path)
    manifest_variant = get_manifest_variant_from_manifest_path(manifest_path)
    output_dir = get_output_paths(analysis_name, manifest_variant)["results"]
    bin_labels = np.array([v["anisotropy_bin"] for v in volumes])

    # Create task labels from dataset names
    unique_datasets = sorted(set(v["dataset"] for v in volumes))
    dataset_to_id = {d: i for i, d in enumerate(unique_datasets)}
    task_labels = np.array([dataset_to_id[v["dataset"]] for v in volumes])

    logger.info(f"Loaded {len(volumes)} volumes from {manifest_path}")
    logger.info(f"Task: Classify {len(unique_datasets)} datasets")
    logger.info(f"Binning scheme: {binning_scheme}")

    if args.all_checkpoints:
        # Run on all checkpoints
        all_results = []
        for ckpt_name in CHECKPOINTS.keys():
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing checkpoint: {ckpt_name}")
            logger.info(f"{'='*60}")

            # Get crop size from checkpoint config
            crop_size = CHECKPOINTS[ckpt_name].crop_size

            # Create dataloader and extractor
            dataloader = create_acquisition_dataloader(
                manifest_path,
                crop_size=crop_size,
                batch_size=args.batch_size,
            )
            extractor = FeatureExtractor(ckpt_name)

            # Extract features
            features_dict = extractor.extract_from_dataloader(dataloader, show_progress=True)
            features = features_dict[args.feature_type]

            # Run spacing readout
            results, _ = run_spacing_readout_analysis(
                features=features,
                bin_labels=bin_labels,
                task_labels=task_labels,
                checkpoint_name=ckpt_name,
                feature_type=args.feature_type,
                n_cv_splits=args.cv_splits,
                output_dir=output_dir,
                bin_names=bin_names,
                binning_scheme=binning_scheme,
                analysis_name=analysis_name,
            )
            all_results.append(results)

        # Compare checkpoints
        comparison = compare_checkpoint_spacing_readouts(all_results, output_dir=output_dir, analysis_name=analysis_name)
        logger.info(f"\n{'='*60}")
        logger.info("FINAL COMPARISON")
        logger.info(f"{'='*60}")
        logger.info(f"Best overall accuracy: {comparison['summary']['best_overall']}")
        logger.info(f"Most consistent across bins: {comparison['summary']['most_consistent']}")
        if comparison['summary']['spacing_aware_reduces_gap'] is not None:
            logger.info(f"SA reduces bin gap: {comparison['summary']['spacing_aware_reduces_gap']}")
            logger.info(f"  SA mean gap: {comparison['summary']['sa_mean_gap']:.4f}")
            logger.info(f"  Non-SA mean gap: {comparison['summary']['non_sa_mean_gap']:.4f}")

    else:
        # Single checkpoint
        crop_size = CHECKPOINTS[args.checkpoint].crop_size

        dataloader = create_acquisition_dataloader(
            manifest_path,
            crop_size=crop_size,
            batch_size=args.batch_size,
        )
        extractor = FeatureExtractor(args.checkpoint)

        # Extract features
        features_dict = extractor.extract_from_dataloader(dataloader, show_progress=True)
        features = features_dict[args.feature_type]

        # Run spacing readout
        results, _ = run_spacing_readout_analysis(
            features=features,
            bin_labels=bin_labels,
            task_labels=task_labels,
            checkpoint_name=args.checkpoint,
            feature_type=args.feature_type,
            n_cv_splits=args.cv_splits,
            output_dir=output_dir,
            bin_names=bin_names,
            binning_scheme=binning_scheme,
            analysis_name=analysis_name,
        )


if __name__ == "__main__":
    main()
