"""Reference-patch cosine similarity on a native 3D feature grid."""
from __future__ import annotations

from numbers import Integral

import numpy as np


def _triplet(value, name):
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain three integers") from exc
    if len(values) != 3 or any(
        isinstance(item, (bool, np.bool_)) or not isinstance(item, Integral)
        for item in values
    ):
        raise ValueError(f"{name} must contain three integers")
    return tuple(int(item) for item in values)


def reference_similarity(tokens, grid_shape, reference_patch):
    """Compare every patch feature with one reference patch using cosine similarity.

    ``tokens`` has shape ``[patches, channels]`` in C-order ``grid_shape`` order.
    ``reference_patch`` gives its three zero-based native grid indices. Features
    must be finite, real, and nonzero; neither inputs nor feature channels are
    centered, thresholded, or modified. Return a float32 3D array in [-1, 1].
    """
    grid_shape = _triplet(grid_shape, "grid_shape")
    if any(size <= 0 for size in grid_shape):
        raise ValueError("grid_shape dimensions must be positive")
    reference_patch = _triplet(reference_patch, "reference_patch")
    if any(index < 0 or index >= size for index, size in zip(reference_patch, grid_shape)):
        raise ValueError("reference_patch indices must be inside grid_shape")

    try:
        features = np.asarray(tokens)
        if features.dtype.kind not in "iuf":
            raise ValueError
        with np.errstate(over="ignore", invalid="ignore"):
            features = features.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("tokens must contain real numeric features") from exc
    if features.ndim != 2 or not all(features.shape):
        raise ValueError("tokens must have nonempty shape [patches, channels]")
    if features.shape[0] != grid_shape[0] * grid_shape[1] * grid_shape[2]:
        raise ValueError("tokens patch count must match grid_shape")
    if not np.isfinite(features).all():
        raise ValueError("tokens must contain finite features")

    # Scaling before the norm avoids overflow or underflow for finite vectors.
    scales = np.max(np.abs(features), axis=1, keepdims=True)
    if (scales == 0).any():
        raise ValueError("Every patch feature must have a positive norm")
    unit_features = features / scales
    unit_features /= np.linalg.norm(unit_features, axis=1, keepdims=True)
    reference_index = np.ravel_multi_index(reference_patch, grid_shape, order="C")
    similarity = np.clip(unit_features @ unit_features[reference_index], -1., 1.)
    similarity[reference_index] = 1.
    return similarity.reshape(grid_shape, order="C").astype(np.float32)
