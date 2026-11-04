"""Strict loading for sample-identified organ feature bundles."""
from __future__ import annotations


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


__all__ = ["load_organ_feature_bundle"]
