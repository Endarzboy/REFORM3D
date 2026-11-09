"""Linear CKA for matched feature matrices."""

import numpy as np


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
