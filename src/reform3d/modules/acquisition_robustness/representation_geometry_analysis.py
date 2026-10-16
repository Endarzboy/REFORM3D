"""Compute native-spacing representation geometry metrics.

This module compares frozen embeddings across anisotropy bins using CKA,
inter-bin and intra-bin distances, silhouette scores, and optional low-dimensional
visualization coordinates. The evaluation pipeline writes one result payload per
checkpoint and feature family.

Inputs:
    features_by_bin: embedding matrices grouped by spacing bin.
    bin metadata: bin names, counts, and spacing ranges.
    output_dir: result directory for JSON and figure artifacts.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import json

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from tqdm import tqdm


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


try:
    from .config import (
        CHECKPOINTS,
        get_acquisition_manifest_path,
        DEFAULT_BINNING_SCHEME,
        get_binning_scheme_from_manifest,
        get_bin_name_map,
        get_output_paths,
        get_dataset_name_from_manifest_path,
        get_manifest_variant_from_manifest_path,
        get_checkpoint_feature_dir,
        get_summary_feature_dir,
        resolve_acquisition_manifest_path,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from config import (
        CHECKPOINTS,
        get_acquisition_manifest_path,
        DEFAULT_BINNING_SCHEME,
        get_binning_scheme_from_manifest,
        get_bin_name_map,
        get_output_paths,
        get_dataset_name_from_manifest_path,
        get_manifest_variant_from_manifest_path,
        get_checkpoint_feature_dir,
        get_summary_feature_dir,
        resolve_acquisition_manifest_path,
    )

logger = logging.getLogger(__name__)


OBSERVED_METRIC_RANDOM_SEED = 42
ENABLE_BALANCED_REPRESENTATION_GEOMETRY = os.environ.get(
    "MED3DINO_ENABLE_BALANCED_REPRESENTATION_GEOMETRY",
    "1",
) == "1"
ENABLE_BALANCED_TSNE = os.environ.get("MED3DINO_ENABLE_BALANCED_TSNE", "0") == "1" # Defaults to False since t-SNE


def _format_metric(value: Optional[float]) -> str:
    """Format optional diagnostics for logs."""
    return "skipped" if value is None else f"{float(value):.4f}"


def _select_balanced_bin_indices(
    bin_labels: np.ndarray,
    random_state: int = OBSERVED_METRIC_RANDOM_SEED,
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


def _resolve_geometry_dirs(
    output_dir: Optional[Path],
    figures_dir: Optional[Path],
    dataset_name: str,
    manifest_variant: str,
) -> Tuple[Path, Path]:
    output_paths = get_output_paths(dataset_name, manifest_variant)
    return output_dir or output_paths["results"], figures_dir or output_paths["figures"]


# =============================================================================
# CKA (Centered Kernel Alignment)
# =============================================================================

def centering_matrix(n: int) -> np.ndarray:
    """Create centering matrix H = I - (1/n) * 1 * 1^T"""
    return np.eye(n) - np.ones((n, n)) / n


def hsic(K: np.ndarray, L: np.ndarray) -> float:
    """
    Compute Hilbert-Schmidt Independence Criterion.

    HSIC(K, L) = (1/(n-1)^2) * tr(KHLH)
    """
    n = K.shape[0]
    H = centering_matrix(n)
    return np.trace(K @ H @ L @ H) / ((n - 1) ** 2)


def cka_linear(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute linear CKA between two feature matrices.

    CKA(X, Y) = HSIC(X @ X^T, Y @ Y^T) / sqrt(HSIC(X @ X^T, X @ X^T) * HSIC(Y @ Y^T, Y @ Y^T))

    Args:
        X: Feature matrix [N, D1]
        Y: Feature matrix [N, D2]

    Returns:
        CKA similarity in [0, 1]
    """
    # Compute Gram matrices
    K = X @ X.T
    L = Y @ Y.T

    # Compute HSIC values
    hsic_kl = hsic(K, L)
    hsic_kk = hsic(K, K)
    hsic_ll = hsic(L, L)

    # Avoid division by zero
    if hsic_kk < 1e-10 or hsic_ll < 1e-10:
        return 0.0

    return hsic_kl / np.sqrt(hsic_kk * hsic_ll)


def centered_linear_cka(X: np.ndarray, Y: np.ndarray, method: str = "auto") -> float:
    """Compute linear CKA efficiently for bootstrap samples with paired rows.

    The Gram and feature backends are algebraically equivalent to ``cka_linear``.
    They avoid explicitly multiplying centering matrices for each resample.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError("CKA requires two feature matrices with the same number of rows")
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("CKA features must be finite")
    if method not in {"auto", "gram", "feature"}:
        raise ValueError(f"Unsupported CKA method: {method}")
    n = X.shape[0]
    if n < 2 or not X.shape[1] or not Y.shape[1]:
        return 0.0
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    if method == "auto":
        method = "feature" if max(X.shape[1], Y.shape[1]) <= n else "gram"
    if method == "gram":
        K, L = X @ X.T, Y @ Y.T
        numerator = float(np.sum(K * L))
        source_scale, target_scale = np.linalg.norm(K, "fro"), np.linalg.norm(L, "fro")
    else:
        numerator = float(np.linalg.norm(X.T @ Y, "fro") ** 2)
        source_scale = np.linalg.norm(X.T @ X, "fro")
        target_scale = np.linalg.norm(Y.T @ Y, "fro")
    # Match the primary evaluator's HSIC self-similarity threshold.
    if (source_scale / (n - 1)) ** 2 < 1e-10 or (target_scale / (n - 1)) ** 2 < 1e-10:
        return 0.0
    return float(numerator / (source_scale * target_scale))


def cka_rbf(X: np.ndarray, Y: np.ndarray, sigma: float = None) -> float:
    """
    Compute RBF kernel CKA between two feature matrices.

    Args:
        X: Feature matrix [N, D1]
        Y: Feature matrix [N, D2]
        sigma: RBF bandwidth (default: median heuristic)

    Returns:
        CKA similarity in [0, 1]
    """
    from scipy.spatial.distance import pdist, squareform

    def rbf_kernel(Z, sigma):
        dists = squareform(pdist(Z, 'euclidean'))
        if sigma is None:
            sigma = np.median(dists[dists > 0])
        return np.exp(-dists ** 2 / (2 * sigma ** 2))

    K = rbf_kernel(X, sigma)
    L = rbf_kernel(Y, sigma)

    hsic_kl = hsic(K, L)
    hsic_kk = hsic(K, K)
    hsic_ll = hsic(L, L)

    if hsic_kk < 1e-10 or hsic_ll < 1e-10:
        return 0.0

    return hsic_kl / np.sqrt(hsic_kk * hsic_ll)


def compute_cka_matrix(
    features: np.ndarray,
    bin_labels: np.ndarray,
    kernel: str = "linear",
    n_samples_per_bin: int = 500,
) -> Tuple[np.ndarray, Dict]:
    """
    Compute CKA matrix between anisotropy bins.

    Args:
        features: Feature matrix [N, D]
        bin_labels: Bin assignment for each sample [N]
        kernel: "linear" or "rbf"
        n_samples_per_bin: Max samples to use per bin (for efficiency)

    Returns:
        cka_matrix: [n_bins, n_bins] CKA similarity matrix
        stats: Dictionary with additional statistics
    """
    unique_bins = np.unique(bin_labels)
    n_bins = len(unique_bins)
    cka_matrix = np.zeros((n_bins, n_bins))

    # Sample features per bin
    bin_features = {}
    for bin_id in unique_bins:
        mask = bin_labels == bin_id
        bin_feats = features[mask]

        # Subsample if needed
        if len(bin_feats) > n_samples_per_bin:
            idx = np.random.choice(len(bin_feats), n_samples_per_bin, replace=False)
            bin_feats = bin_feats[idx]

        bin_features[bin_id] = bin_feats

    # Compute CKA for all pairs
    cka_func = cka_linear if kernel == "linear" else cka_rbf

    for i, bin_i in enumerate(unique_bins):
        for j, bin_j in enumerate(unique_bins):
            if i <= j:
                # Need same number of samples for CKA
                n = min(len(bin_features[bin_i]), len(bin_features[bin_j]))
                X = bin_features[bin_i][:n]
                Y = bin_features[bin_j][:n]

                cka_val = cka_func(X, Y)
                cka_matrix[i, j] = cka_val
                cka_matrix[j, i] = cka_val

    # Compute statistics
    off_diag_mask = ~np.eye(n_bins, dtype=bool)
    stats = {
        "mean_self_cka": np.mean(np.diag(cka_matrix)),
        "mean_cross_cka": np.mean(cka_matrix[off_diag_mask]),
        "min_cross_cka": np.min(cka_matrix[off_diag_mask]),
        "max_cross_cka": np.max(cka_matrix[off_diag_mask]),
        "bin_ids": unique_bins.tolist(),
    }

    return cka_matrix, stats


def _sample_bin_features(
    features: np.ndarray,
    bin_labels: np.ndarray,
    n_samples_per_bin: int,
    random_state: int = OBSERVED_METRIC_RANDOM_SEED,
) -> Dict[int, np.ndarray]:
    """Subsample features per bin for stable pairwise observational metrics."""
    rng = np.random.default_rng(random_state)
    sampled = {}
    for bin_id in np.unique(bin_labels):
        mask = bin_labels == bin_id
        bin_feats = features[mask]
        if len(bin_feats) > n_samples_per_bin:
            idx = rng.choice(len(bin_feats), n_samples_per_bin, replace=False)
            bin_feats = bin_feats[idx]
        sampled[int(bin_id)] = bin_feats
    return sampled


def _median_heuristic_sigma(X: np.ndarray, Y: np.ndarray) -> float:
    """Estimate RBF bandwidth from pairwise distances in the combined sample."""
    from scipy.spatial.distance import pdist

    combined = np.vstack([X, Y])
    if len(combined) < 2:
        return 1.0
    distances = pdist(combined, metric="euclidean")
    positive = distances[distances > 0]
    if len(positive) == 0:
        return 1.0
    return float(np.median(positive))


def compute_mmd_rbf(X: np.ndarray, Y: np.ndarray, sigma: Optional[float] = None) -> float:
    """Compute unbiased RBF MMD^2 between two feature sets."""
    from scipy.spatial.distance import cdist

    if len(X) < 2 or len(Y) < 2:
        return 0.0

    sigma = sigma or _median_heuristic_sigma(X, Y)
    sigma = max(sigma, 1e-6)
    gamma = 1.0 / (2.0 * sigma * sigma)

    k_xx = np.exp(-gamma * cdist(X, X, metric="sqeuclidean"))
    k_yy = np.exp(-gamma * cdist(Y, Y, metric="sqeuclidean"))
    k_xy = np.exp(-gamma * cdist(X, Y, metric="sqeuclidean"))

    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)

    n_x = len(X)
    n_y = len(Y)
    xx = k_xx.sum() / (n_x * (n_x - 1))
    yy = k_yy.sum() / (n_y * (n_y - 1))
    xy = k_xy.mean()
    return float(xx + yy - 2.0 * xy)


def compute_sliced_wasserstein_distance(
    X: np.ndarray,
    Y: np.ndarray,
    n_projections: int = 64,
    random_state: int = OBSERVED_METRIC_RANDOM_SEED,
) -> float:
    """Approximate Wasserstein distance by averaging 1D projections."""
    rng = np.random.default_rng(random_state)
    dim = X.shape[1]
    projections = rng.normal(size=(n_projections, dim))
    projections /= np.linalg.norm(projections, axis=1, keepdims=True) + 1e-12

    distances = []
    for direction in projections:
        proj_x = np.sort(X @ direction)
        proj_y = np.sort(Y @ direction)
        n = min(len(proj_x), len(proj_y))
        if n == 0:
            continue
        distances.append(float(np.mean(np.abs(proj_x[:n] - proj_y[:n]))))

    if not distances:
        return 0.0
    return float(np.mean(distances))


def _bootstrap_confidence_interval(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    X: np.ndarray,
    Y: np.ndarray,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    random_state: int = OBSERVED_METRIC_RANDOM_SEED,
) -> Dict[str, float]:
    """Bootstrap confidence interval for a two-sample metric."""
    rng = np.random.default_rng(random_state)
    estimates = []
    for _ in range(n_bootstrap):
        x_idx = rng.choice(len(X), len(X), replace=True)
        y_idx = rng.choice(len(Y), len(Y), replace=True)
        estimates.append(metric_fn(X[x_idx], Y[y_idx]))

    lower = float(np.quantile(estimates, alpha / 2.0))
    upper = float(np.quantile(estimates, 1.0 - alpha / 2.0))
    return {
        "mean": float(np.mean(estimates)),
        "std": float(np.std(estimates)),
        "ci_lower": lower,
        "ci_upper": upper,
        "n_bootstrap": int(n_bootstrap),
    }


def _permutation_test(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    X: np.ndarray,
    Y: np.ndarray,
    observed_value: float,
    n_permutations: int = 200,
    random_state: int = OBSERVED_METRIC_RANDOM_SEED,
) -> Dict[str, float]:
    """Permutation test for a two-sample metric."""
    rng = np.random.default_rng(random_state)
    combined = np.vstack([X, Y])
    n_x = len(X)
    null_values = []

    for _ in range(n_permutations):
        permuted = rng.permutation(len(combined))
        perm_x = combined[permuted[:n_x]]
        perm_y = combined[permuted[n_x:]]
        null_values.append(metric_fn(perm_x, perm_y))

    null_values_arr = np.asarray(null_values)
    p_value = (1.0 + np.sum(null_values_arr >= observed_value)) / (1.0 + n_permutations)
    return {
        "p_value": float(p_value),
        "null_mean": float(np.mean(null_values_arr)),
        "null_std": float(np.std(null_values_arr)),
        "n_permutations": int(n_permutations),
    }


def compute_distribution_shift_metrics(
    features: np.ndarray,
    bin_labels: np.ndarray,
    n_samples_per_bin: int = 500,
    n_bootstrap: int = 200,
    n_permutations: int = 200,
    n_projections: int = 64,
    random_state: int = OBSERVED_METRIC_RANDOM_SEED,
) -> Dict[str, Any]:
    """Compute required observational set-level metrics between spacing bins."""
    unique_bins = np.unique(bin_labels)
    sampled_features = _sample_bin_features(features, bin_labels, n_samples_per_bin, random_state=random_state)
    pair_metrics: Dict[str, Dict[str, Any]] = {}
    mmd_values = []
    swd_values = []

    for i, bin_i in enumerate(unique_bins):
        for bin_j in unique_bins[i + 1:]:
            X = sampled_features[int(bin_i)]
            Y = sampled_features[int(bin_j)]
            pair_key = f"{int(bin_i)}_vs_{int(bin_j)}"
            sigma = _median_heuristic_sigma(X, Y)

            mmd_metric = lambda a, b, sigma=sigma: compute_mmd_rbf(a, b, sigma=sigma)
            swd_metric = lambda a, b, rs=random_state: compute_sliced_wasserstein_distance(
                a,
                b,
                n_projections=n_projections,
                random_state=rs,
            )

            observed_mmd = mmd_metric(X, Y)
            observed_swd = swd_metric(X, Y)

            pair_metrics[pair_key] = {
                "bins": [int(bin_i), int(bin_j)],
                "sample_counts": {int(bin_i): int(len(X)), int(bin_j): int(len(Y))},
                "mmd_rbf": {
                    "observed": observed_mmd,
                    "sigma": float(sigma),
                    "bootstrap": _bootstrap_confidence_interval(
                        mmd_metric,
                        X,
                        Y,
                        n_bootstrap=n_bootstrap,
                        random_state=random_state,
                    ),
                    "permutation": _permutation_test(
                        mmd_metric,
                        X,
                        Y,
                        observed_mmd,
                        n_permutations=n_permutations,
                        random_state=random_state,
                    ),
                },
                "sliced_wasserstein": {
                    "observed": observed_swd,
                    "n_projections": int(n_projections),
                    "bootstrap": _bootstrap_confidence_interval(
                        swd_metric,
                        X,
                        Y,
                        n_bootstrap=n_bootstrap,
                        random_state=random_state,
                    ),
                    "permutation": _permutation_test(
                        swd_metric,
                        X,
                        Y,
                        observed_swd,
                        n_permutations=n_permutations,
                        random_state=random_state,
                    ),
                },
            }
            mmd_values.append(observed_mmd)
            swd_values.append(observed_swd)

    return {
        "pairwise": pair_metrics,
        "summary": {
            "mean_mmd_rbf": float(np.mean(mmd_values)) if mmd_values else 0.0,
            "max_mmd_rbf": float(np.max(mmd_values)) if mmd_values else 0.0,
            "mean_sliced_wasserstein": float(np.mean(swd_values)) if swd_values else 0.0,
            "max_sliced_wasserstein": float(np.max(swd_values)) if swd_values else 0.0,
            "n_bin_pairs": int(len(pair_metrics)),
            "n_samples_per_bin_cap": int(n_samples_per_bin),
            "n_bootstrap": int(n_bootstrap),
            "n_permutations": int(n_permutations),
        },
    }


def skipped_distribution_shift_metrics(reason: str) -> Dict[str, Any]:
    """Return a result-shaped payload when distribution diagnostics are skipped."""
    return {
        "pairwise": {},
        "summary": {
            "mean_mmd_rbf": None,
            "max_mmd_rbf": None,
            "mean_sliced_wasserstein": None,
            "max_sliced_wasserstein": None,
            "n_bin_pairs": 0,
            "n_samples_per_bin_cap": None,
            "n_bootstrap": 0,
            "n_permutations": 0,
            "skipped": True,
            "skip_reason": reason,
        },
    }


# =============================================================================
# Embedding Geometry Analysis
# =============================================================================

def compute_embedding_geometry(
    features: np.ndarray,
    bin_labels: np.ndarray,
) -> Dict:
    """
    Compute embedding geometry metrics.

    Metrics:
    - Intra-bin distance: Average pairwise distance within each bin
    - Inter-bin distance: Average distance between bin centroids
    - Silhouette score: Clustering quality metric
    - Bin separability: Ratio of inter/intra distances

    Args:
        features: Feature matrix [N, D]
        bin_labels: Bin assignment for each sample [N]

    Returns:
        Dictionary with geometry metrics
    """
    from scipy.spatial.distance import pdist, cdist

    unique_bins = np.unique(bin_labels)
    n_bins = len(unique_bins)

    # Compute centroids
    centroids = {}
    for bin_id in unique_bins:
        mask = bin_labels == bin_id
        centroids[bin_id] = features[mask].mean(axis=0)

    # Intra-bin distances
    intra_distances = {}
    for bin_id in unique_bins:
        mask = bin_labels == bin_id
        bin_feats = features[mask]
        if len(bin_feats) > 1:
            # Sample for efficiency
            if len(bin_feats) > 1000:
                idx = np.random.choice(len(bin_feats), 1000, replace=False)
                bin_feats = bin_feats[idx]
            dists = pdist(bin_feats, metric='cosine')
            intra_distances[bin_id] = float(np.mean(dists))
        else:
            intra_distances[bin_id] = 0.0

    # Inter-bin distances (centroid-to-centroid)
    centroid_matrix = np.stack([centroids[b] for b in unique_bins])
    inter_distances = pdist(centroid_matrix, metric='cosine')

    # Silhouette score (sample for efficiency)
    n_samples = min(5000, len(features))
    idx = np.random.choice(len(features), n_samples, replace=False)
    try:
        silhouette = silhouette_score(
            features[idx],
            bin_labels[idx],
            metric='cosine',
            sample_size=min(2000, n_samples)
        )
    except Exception:
        silhouette = 0.0

    # Separability ratio
    mean_intra = np.mean(list(intra_distances.values()))
    mean_inter = np.mean(inter_distances)
    separability = mean_inter / (mean_intra + 1e-8)

    return {
        "intra_distances": intra_distances,
        "mean_intra_distance": float(mean_intra),
        "inter_distances": inter_distances.tolist(),
        "mean_inter_distance": float(mean_inter),
        "silhouette_score": float(silhouette),
        "separability_ratio": float(separability),
        "n_samples_per_bin": {
            int(b): int(np.sum(bin_labels == b)) for b in unique_bins
        },
    }


def skipped_embedding_geometry(features: np.ndarray, bin_labels: np.ndarray, reason: str) -> Dict[str, Any]:
    """Return a result-shaped payload when embedding geometry diagnostics are skipped."""
    return {
        "intra_distances": {},
        "mean_intra_distance": None,
        "inter_distances": [],
        "mean_inter_distance": None,
        "silhouette_score": None,
        "separability_ratio": None,
        "n_samples_per_bin": {
            int(b): int(np.sum(bin_labels == b)) for b in np.unique(bin_labels)
        },
        "skipped": True,
        "skip_reason": reason,
    }


# =============================================================================
# Visualization
# =============================================================================

def compute_tsne_embedding(
    features: np.ndarray,
    bin_labels: np.ndarray,
    n_samples: int = 2000,
    perplexity: int = 30,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute t-SNE embedding for visualization.

    Args:
        features: Feature matrix [N, D]
        bin_labels: Bin assignment for each sample [N]
        n_samples: Number of samples to include
        perplexity: t-SNE perplexity parameter
        random_state: Random seed

    Returns:
        tsne_coords: [n_samples, 2] t-SNE coordinates
        sampled_labels: [n_samples] bin labels for sampled points
    """
    # Stratified sampling
    unique_bins = np.unique(bin_labels)
    samples_per_bin = n_samples // len(unique_bins)

    sampled_idx = []
    for bin_id in unique_bins:
        mask = bin_labels == bin_id
        bin_idx = np.where(mask)[0]
        n = min(samples_per_bin, len(bin_idx))
        sampled_idx.extend(np.random.choice(bin_idx, n, replace=False))

    sampled_idx = np.array(sampled_idx)
    np.random.shuffle(sampled_idx)

    X = features[sampled_idx]
    y = bin_labels[sampled_idx]

    # Compute t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(X) - 1),  # perplexity must be < n_samples
        random_state=random_state,
        max_iter=1000,
    )
    coords = tsne.fit_transform(X)

    return coords, y


def save_tsne_plot(
    coords: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str = "t-SNE by Anisotropy Bin",
    bin_names: Optional[Dict[int, str]] = None,
):
    """Save t-SNE visualization to file."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))

    unique_bins = np.unique(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_bins)))
    bin_names = bin_names or get_bin_name_map(DEFAULT_BINNING_SCHEME)

    for i, bin_id in enumerate(unique_bins):
        mask = labels == bin_id
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[colors[i]],
            label=f"Bin {bin_id}: {bin_names.get(bin_id, 'unknown')}",
            alpha=0.6,
            s=20,
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved t-SNE plot to {output_path}")


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

@dataclass
class RepresentationGeometryResults:
    """Results from representation geometry analysis."""
    checkpoint_name: str
    feature_type: str
    cka_matrix: np.ndarray
    cka_stats: Dict
    observational_distances: Dict[str, Any]
    geometry_metrics: Dict
    balanced_bin_sensitivity: Optional[Dict[str, Any]] = None
    tsne_coords: Optional[np.ndarray] = None
    tsne_labels: Optional[np.ndarray] = None


def run_representation_geometry_analysis(
    features: np.ndarray,
    bin_labels: np.ndarray,
    checkpoint_name: str,
    feature_type: str = "cls",
    compute_tsne: bool = True,
    output_dir: Optional[Path] = None,
    figures_dir: Optional[Path] = None,
    bin_names: Optional[Dict[int, str]] = None,
    binning_scheme: str = "original",
    dataset_name: str = "default",
    manifest_variant: str = "original_bins",
    compute_distribution_metrics: bool = True,
    compute_embedding_geometry_metrics: bool = True,
    compute_balanced_sensitivity: bool = True,
) -> RepresentationGeometryResults:
    """
    Run complete representation geometry analysis.

    Args:
        features: Feature matrix [N, D]
        bin_labels: Bin assignment for each sample [N]
        checkpoint_name: Name of the checkpoint
        feature_type: Type of features ("cls", "avg_pool", "multilayer")
        compute_tsne: Whether to compute t-SNE embedding
        output_dir: Directory to save results
        figures_dir: Directory to save figures

    Returns:
        RepresentationGeometryResults object
    """
    output_dir, figures_dir = _resolve_geometry_dirs(output_dir, figures_dir, dataset_name, manifest_variant)
    checkpoint_results_dir = get_checkpoint_feature_dir(output_dir, checkpoint_name, feature_type)
    checkpoint_figures_dir = get_checkpoint_feature_dir(figures_dir, checkpoint_name, feature_type)
    checkpoint_results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running representation geometry analysis for {checkpoint_name} ({feature_type})")
    logger.info(f"Features shape: {features.shape}, Unique bins: {np.unique(bin_labels)}")

    # 1. CKA Analysis
    logger.info("Computing CKA matrix...")
    cka_matrix, cka_stats = compute_cka_matrix(features, bin_labels, kernel="linear")

    if compute_distribution_metrics:
        logger.info("Computing observational MMD and Sliced Wasserstein metrics...")
        observational_distances = compute_distribution_shift_metrics(features, bin_labels)
    else:
        logger.info("Skipping observational MMD/SWD diagnostics")
        observational_distances = skipped_distribution_shift_metrics("fast_seeded_evaluation")

    if compute_embedding_geometry_metrics:
        logger.info("Computing embedding geometry...")
        geometry = compute_embedding_geometry(features, bin_labels)
    else:
        logger.info("Skipping embedding geometry diagnostics")
        geometry = skipped_embedding_geometry(features, bin_labels, "fast_seeded_evaluation")

    # 4. t-SNE (optional)
    tsne_coords, tsne_labels = None, None
    if compute_tsne:
        logger.info("Computing t-SNE embedding...")
        tsne_coords, tsne_labels = compute_tsne_embedding(features, bin_labels)

        # Save visualization
        save_tsne_plot(
            tsne_coords,
            tsne_labels,
            checkpoint_figures_dir / "tsne.png",
            title=f"t-SNE: {checkpoint_name} ({feature_type})",
            bin_names=bin_names,
        )

    results = RepresentationGeometryResults(
        checkpoint_name=checkpoint_name,
        feature_type=feature_type,
        cka_matrix=cka_matrix,
        cka_stats=cka_stats,
        observational_distances=observational_distances,
        geometry_metrics=geometry,
        tsne_coords=tsne_coords,
        tsne_labels=tsne_labels,
    )

    balanced_sensitivity = None
    if compute_balanced_sensitivity and ENABLE_BALANCED_REPRESENTATION_GEOMETRY:
        balanced_idx, original_bin_counts, sample_count_per_bin = _select_balanced_bin_indices(bin_labels)
        if balanced_idx is not None:
            balanced_features = features[balanced_idx]
            balanced_bins = bin_labels[balanced_idx]
            balanced_cka_matrix, balanced_cka_stats = compute_cka_matrix(
                balanced_features,
                balanced_bins,
                kernel="linear",
                n_samples_per_bin=sample_count_per_bin,
            )
            balanced_observational_distances = compute_distribution_shift_metrics(
                balanced_features,
                balanced_bins,
                n_samples_per_bin=sample_count_per_bin,
            )
            balanced_geometry = compute_embedding_geometry(balanced_features, balanced_bins)
            balanced_sensitivity = {
                "sample_count_per_bin": int(sample_count_per_bin),
                "total_selected_samples": int(len(balanced_idx)),
                "original_bin_counts": original_bin_counts,
                "cka_matrix": balanced_cka_matrix.tolist(),
                "cka_stats": balanced_cka_stats,
                "observational_distances": balanced_observational_distances,
                "geometry": balanced_geometry,
            }
            if compute_tsne and ENABLE_BALANCED_TSNE:
                balanced_tsne_coords, balanced_tsne_labels = compute_tsne_embedding(
                    balanced_features,
                    balanced_bins,
                    n_samples=min(2000, len(balanced_features)),
                    random_state=OBSERVED_METRIC_RANDOM_SEED,
                )
                save_tsne_plot(
                    balanced_tsne_coords,
                    balanced_tsne_labels,
                    checkpoint_figures_dir / "tsne_balanced.png",
                    title=f"t-SNE Balanced: {checkpoint_name} ({feature_type})",
                    bin_names=bin_names,
                )
            elif compute_tsne:
                logger.info("Skipping balanced t-SNE for %s (%s)", checkpoint_name, feature_type)
            results.balanced_bin_sensitivity = balanced_sensitivity
    else:
        logger.info(
            "Skipping balanced representation-geometry sensitivity for %s (%s)",
            checkpoint_name,
            feature_type,
        )

    # Save results
    results_dict = _convert_numpy_types({
        "checkpoint": checkpoint_name,
        "feature_type": feature_type,
        "binning_scheme": binning_scheme,
        "cka_matrix": cka_matrix.tolist(),
        "cka_stats": cka_stats,
        "observational_distances": observational_distances,
        "geometry": geometry,
    })
    if balanced_sensitivity is not None:
        results_dict["balanced_bin_sensitivity"] = _convert_numpy_types(balanced_sensitivity)

    results_path = checkpoint_results_dir / "representation_geometry.json"
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    logger.info(f"Saved results to {results_path}")

    logger.info("Representation geometry summary:")
    logger.info(
        "  Natural bins: cross-CKA=%.4f, mean MMD=%s, mean SWD=%s, silhouette=%s, separability=%s",
        cka_stats["mean_cross_cka"],
        _format_metric(observational_distances["summary"]["mean_mmd_rbf"]),
        _format_metric(observational_distances["summary"]["mean_sliced_wasserstein"]),
        _format_metric(geometry["silhouette_score"]),
        _format_metric(geometry["separability_ratio"]),
    )
    if balanced_sensitivity is not None:
        logger.info(
            "  Balanced-bin sensitivity (n/bin=%d): cross-CKA=%.4f, mean MMD=%.4f, mean SWD=%.4f, silhouette=%.4f, separability=%.4f",
            balanced_sensitivity["sample_count_per_bin"],
            balanced_sensitivity["cka_stats"]["mean_cross_cka"],
            balanced_sensitivity["observational_distances"]["summary"]["mean_mmd_rbf"],
            balanced_sensitivity["observational_distances"]["summary"]["mean_sliced_wasserstein"],
            balanced_sensitivity["geometry"]["silhouette_score"],
            balanced_sensitivity["geometry"]["separability_ratio"],
        )

    return results


def analyze_all_checkpoints(
    manifest_path: Path,
    checkpoints: Optional[List[str]] = None,
    feature_types: List[str] = ["cls"],
    batch_size: int = 8,
    output_dir: Optional[Path] = None,
    dataset_name: Optional[str] = None,
    manifest_variant: Optional[str] = None,
) -> Dict[str, Dict[str, RepresentationGeometryResults]]:
    """
    Run representation geometry analysis for all checkpoints.

    Args:
        manifest_path: Path to sampled manifest JSON
        checkpoints: List of checkpoint names (default: all)
        feature_types: Feature types to analyze
        batch_size: Batch size for feature extraction
        output_dir: Directory to save results

    Returns:
        Nested dict: {checkpoint: {feature_type: RepresentationGeometryResults}}
    """
    try:
        from .feature_extractor import FeatureExtractor, ExtractionConfig
        from .data_loader import create_acquisition_dataloader
    except ImportError:  # pragma: no cover - direct script execution fallback
        from feature_extractor import FeatureExtractor, ExtractionConfig
        from data_loader import create_acquisition_dataloader

    dataset_name = dataset_name or get_dataset_name_from_manifest_path(manifest_path)
    manifest_variant = manifest_variant or get_manifest_variant_from_manifest_path(manifest_path)
    output_paths = get_output_paths(dataset_name, manifest_variant)
    output_dir = output_dir or output_paths["results"]
    figures_dir = output_paths["figures"]
    checkpoints = checkpoints or list(CHECKPOINTS.keys())

    # Load manifest for bin labels
    with open(manifest_path) as f:
        manifest = json.load(f)
    volumes = manifest["volumes"]
    bin_labels = np.array([v["anisotropy_bin"] for v in volumes])
    binning_scheme = get_binning_scheme_from_manifest(manifest)
    bin_names = get_bin_name_map(binning_scheme)

    all_results = {}

    for ckpt_name in checkpoints:
        logger.info(f"\n{'='*60}\nAnalyzing {ckpt_name}\n{'='*60}")

        try:
            # Create dataloader
            crop_size = CHECKPOINTS[ckpt_name].crop_size
            dataloader = create_acquisition_dataloader(
                manifest_path=manifest_path,
                crop_size=crop_size,
                batch_size=batch_size,
                num_workers=4,
            )

            # Extract features
            config = ExtractionConfig(
                extract_cls="cls" in feature_types,
                extract_avg_pool="avg_pool" in feature_types,
                extract_multilayer="multilayer" in feature_types,
            )
            extractor = FeatureExtractor(ckpt_name, config=config)
            features_dict = extractor.extract_from_dataloader(dataloader)

            # Analyze each feature type
            ckpt_results = {}
            for feat_type in feature_types:
                if feat_type in features_dict:
                    results = run_representation_geometry_analysis(
                        features=features_dict[feat_type],
                        bin_labels=bin_labels,
                        checkpoint_name=ckpt_name,
                        feature_type=feat_type,
                        compute_tsne=True,
                        output_dir=output_dir,
                        figures_dir=figures_dir,
                        bin_names=bin_names,
                        binning_scheme=binning_scheme,
                        dataset_name=dataset_name,
                        manifest_variant=manifest_variant,
                    )
                    ckpt_results[feat_type] = results

            all_results[ckpt_name] = ckpt_results

        except Exception as e:
            logger.error(f"Failed to analyze {ckpt_name}: {e}")
            continue

    # Generate summary report
    generate_summary_report(all_results, output_dir)

    return all_results


def generate_summary_report(
    results: Dict[str, Dict[str, RepresentationGeometryResults]],
    output_dir: Path,
):
    """Generate summary report comparing all checkpoints."""

    summary = {
        "checkpoints": {},
        "comparison": {},
    }

    for ckpt_name, feat_results in results.items():
        for feat_type, result in feat_results.items():
            key = f"{ckpt_name}_{feat_type}"
            summary["checkpoints"][key] = {
                "mean_cross_cka": result.cka_stats["mean_cross_cka"],
                "mean_mmd_rbf": result.observational_distances["summary"]["mean_mmd_rbf"],
                "mean_sliced_wasserstein": result.observational_distances["summary"]["mean_sliced_wasserstein"],
                "silhouette_score": result.geometry_metrics["silhouette_score"],
                "separability_ratio": result.geometry_metrics["separability_ratio"],
            }

    # Rank by spacing invariance (higher cross-CKA = more invariant)
    if summary["checkpoints"]:
        ranked = sorted(
            summary["checkpoints"].items(),
            key=lambda x: x[1]["mean_cross_cka"],
            reverse=True
        )
        summary["ranking_by_invariance"] = [
            {"model": k, **v} for k, v in ranked
        ]

    # Save summary
    summary_feature_type = next(iter(next(iter(results.values())).keys()), "cls") if results else "cls"
    summary_dir = get_summary_feature_dir(output_dir, summary_feature_type)
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "checkpoint_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("REPRESENTATION GEOMETRY SUMMARY: Spacing Invariance Ranking")
    print("="*60)
    print(f"{'Model':<30} {'Cross-CKA':>12} {'MMD':>10} {'SWD':>10} {'Silhouette':>12}")
    print("-"*60)

    for item in summary.get("ranking_by_invariance", []):
        print(
            f"{item['model']:<30} {item['mean_cross_cka']:>12.4f} "
            f"{item['mean_mmd_rbf']:>10.4f} {item['mean_sliced_wasserstein']:>10.4f} "
            f"{item['silhouette_score']:>12.4f}"
        )

    logger.info(f"Summary saved to {summary_path}")


def compare_checkpoint_geometry(
    results_list: List[RepresentationGeometryResults],
    output_dir: Optional[Path] = None,
    dataset_name: str = "default",
    manifest_variant: str = "original_bins",
) -> Dict[str, Any]:
    """
    Compare representation geometry results across multiple checkpoints.

    Key question: Which model produces most spacing-invariant representations?

    Args:
        results_list: List of RepresentationGeometryResults from different checkpoints
        output_dir: Directory to save comparison

    Returns:
        Comparison summary dict
    """
    output_dir, _ = _resolve_geometry_dirs(output_dir, None, dataset_name, manifest_variant)
    feature_type = results_list[0].feature_type if results_list else "cls"
    summary_dir = get_summary_feature_dir(output_dir, feature_type)
    summary_dir.mkdir(parents=True, exist_ok=True)

    comparison = {
        "checkpoints": [],
        "mean_cross_cka": [],
        "mean_mmd_rbf": [],
        "mean_sliced_wasserstein": [],
        "silhouette_scores": [],
        "separability_ratios": [],
    }

    for r in results_list:
        comparison["checkpoints"].append(r.checkpoint_name)
        comparison["mean_cross_cka"].append(r.cka_stats["mean_cross_cka"])
        comparison["mean_mmd_rbf"].append(r.observational_distances["summary"]["mean_mmd_rbf"])
        comparison["mean_sliced_wasserstein"].append(r.observational_distances["summary"]["mean_sliced_wasserstein"])
        comparison["silhouette_scores"].append(r.geometry_metrics["silhouette_score"])
        comparison["separability_ratios"].append(r.geometry_metrics["separability_ratio"])

    # Statistical comparison
    comparison["summary"] = {
        "most_invariant": comparison["checkpoints"][np.argmax(comparison["mean_cross_cka"])],
        "least_invariant": comparison["checkpoints"][np.argmin(comparison["mean_cross_cka"])],
        "best_separability": comparison["checkpoints"][np.argmax(comparison["separability_ratios"])],
    }

    # Check if spacing-aware models are more invariant
    sa_models = [(i, r) for i, r in enumerate(results_list) if "_sa" in r.checkpoint_name]
    non_sa_models = [(i, r) for i, r in enumerate(results_list) if "_sa" not in r.checkpoint_name]

    if sa_models and non_sa_models:
        sa_mean_cka = np.mean([comparison["mean_cross_cka"][i] for i, _ in sa_models])
        non_sa_mean_cka = np.mean([comparison["mean_cross_cka"][i] for i, _ in non_sa_models])
        comparison["summary"]["sa_more_invariant"] = sa_mean_cka > non_sa_mean_cka
        comparison["summary"]["sa_mean_cka"] = float(sa_mean_cka)
        comparison["summary"]["non_sa_mean_cka"] = float(non_sa_mean_cka)

    # Save comparison
    comparison = _convert_numpy_types(comparison)
    comparison_path = summary_dir / "representation_geometry_comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"Saved comparison to {comparison_path}")

    # Print summary
    logger.info(f"\nMost invariant: {comparison['summary']['most_invariant']}")
    logger.info(f"Best separability: {comparison['summary']['best_separability']}")
    if "sa_more_invariant" in comparison["summary"]:
        logger.info(f"SA more invariant: {comparison['summary']['sa_more_invariant']}")
        logger.info(f"  SA mean CKA: {comparison['summary']['sa_mean_cka']:.4f}")
        logger.info(f"  Non-SA mean CKA: {comparison['summary']['non_sa_mean_cka']:.4f}")

    return comparison


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Acquisition Robustness: Representation Geometry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python representation_geometry_analysis.py -m abdomenatlas/original_bins/manifest_sampled.json -c med3dino_rel_96 med3dino_sa_96 -f cls -a abdomenatlas\n"
            "  python representation_geometry_analysis.py -m totalsegmentermri/original_bins/manifest_sampled.json -c med3dino_rel_96 -f avg_pool multilayer -a totalsegmentermri"
        ),
    )
    parser.add_argument(
        "--manifest", "-m",
        type=str,
        default="abdomenatlas/original_bins/manifest_sampled.json",
        help="Manifest path relative to the acquisition manifest directory"
    )
    parser.add_argument(
        "--checkpoints", "-c",
        nargs="+",
        default=None,
        help="Checkpoints to analyze (default: all)"
    )
    parser.add_argument(
        "--feature-types", "-f",
        nargs="+",
        default=["cls"],
        choices=["cls", "avg_pool", "multilayer"],
        help="Feature types to analyze"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=8,
        help="Batch size for feature extraction"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory"
    )
    parser.add_argument(
        "-a", "--analysis-name",
        type=str,
        default=None,
        help="Dataset/analysis namespace for default output paths"
    )

    args = parser.parse_args()

    manifest_arg = Path(args.manifest)
    if manifest_arg.is_absolute():
        manifest_path = manifest_arg
    elif len(manifest_arg.parts) == 1:
        manifest_path = get_acquisition_manifest_path("abdomenatlas", manifest_kind=manifest_arg.stem)
    else:
        manifest_path = resolve_acquisition_manifest_path(manifest_arg)
    dataset_name = args.analysis_name or get_dataset_name_from_manifest_path(manifest_path)
    manifest_variant = get_manifest_variant_from_manifest_path(manifest_path)
    output_dir = Path(args.output_dir) if args.output_dir else get_output_paths(dataset_name, manifest_variant)["results"]

    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    results = analyze_all_checkpoints(
        manifest_path=manifest_path,
        checkpoints=args.checkpoints,
        feature_types=args.feature_types,
        batch_size=args.batch_size,
        output_dir=output_dir,
        dataset_name=dataset_name,
        manifest_variant=manifest_variant,
    )

    print(f"\nAnalyzed {len(results)} checkpoints")
