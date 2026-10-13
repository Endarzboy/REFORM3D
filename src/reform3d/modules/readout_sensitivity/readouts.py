"""Shared readout primitives for the readout-sensitivity analysis.

Two readouts are supported everywhere:

* ``logistic`` with a configurable inverse-regularization ``C``. With ``C=1.0``
  this is numerically identical to the production readouts, because
  scikit-learn's ``LogisticRegression`` defaults to ``C=1.0``; passing it
  explicitly changes nothing.
* ``nearest_centroid`` — an optimization-free readout that assigns each query to
  the nearest class centroid in the standardized feature space. It reuses the
  production ``_predict_nearest_centroid`` helper for both the binary
  (per-semantic-label) and multiclass (anatomical) cases.

The standardization, eligibility checks, folds, and aggregation are held fixed
across readouts; only the classifier (or its ``C``) changes.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

# Reuse the exact production nearest-centroid implementation so the anatomical
# multiclass centroid remains bit-identical to the stored baseline and the
# binary semantic centroid shares one definition.
from reform3d.modules.anatomical_generalization.few_shot_transfer_analysis import (
    _predict_nearest_centroid,
)

LOGISTIC = "logistic"
NEAREST_CENTROID = "nearest_centroid"
READOUTS = (LOGISTIC, NEAREST_CENTROID)


def _fit_predict_binary(
    train_scaled: np.ndarray,
    train_labels: np.ndarray,
    test_scaled: np.ndarray,
    *,
    readout: str,
    logistic_c: float,
    max_iter: int,
    random_state: int,
) -> np.ndarray | None:
    """Return predictions for one binary label, or ``None`` on classifier failure."""
    if readout == LOGISTIC:
        classifier = LogisticRegression(
            max_iter=max_iter,
            solver="lbfgs",
            random_state=random_state,
            n_jobs=1,
            C=float(logistic_c),
        )
        try:
            classifier.fit(train_scaled, train_labels)
        except Exception:
            return None
        return classifier.predict(test_scaled)
    if readout == NEAREST_CENTROID:
        return _predict_nearest_centroid(train_scaled, train_labels, test_scaled)
    raise ValueError(f"Unsupported readout: {readout}")


def transfer_once(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    *,
    readout: str,
    logistic_c: float,
    random_state: int,
    max_iter: int,
    min_class_count: int = 2,
) -> Tuple[float, Dict[int, float]]:
    """Train one readout per semantic label on one variant, evaluate on another.

    Mirrors the production ``_train_and_evaluate_multilabel_transfer_once``
    exactly for ``readout="logistic", logistic_c=1.0``: same training-fold-only
    ``StandardScaler``, same per-label eligibility gates, same balanced-accuracy
    average over eligible labels. Only the per-label classifier changes.
    """
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

        predictions = _fit_predict_binary(
            train_scaled,
            y_train,
            test_scaled,
            readout=readout,
            logistic_c=logistic_c,
            max_iter=max_iter,
            random_state=random_state,
        )
        if predictions is None:
            continue
        score = float(balanced_accuracy_score(y_test, predictions))
        per_label_scores[label_idx] = score
        valid_scores.append(score)

    if not valid_scores:
        return 0.0, {}
    return float(np.mean(valid_scores)), per_label_scores


def multiclass_readout_balanced_accuracy(
    support_scaled: np.ndarray,
    support_labels: np.ndarray,
    query_scaled: np.ndarray,
    query_labels: np.ndarray,
    *,
    readout: str,
    logistic_c: float,
    random_state: int,
    max_iter: int,
) -> float:
    """Balanced accuracy of one multiclass readout on a fixed support/query split.

    With ``readout="logistic", logistic_c=1.0, max_iter=2000`` this reproduces the
    production anatomical logistic probe; with ``readout="nearest_centroid"`` it
    reproduces the production centroid baseline.
    """
    predictions = multiclass_readout_predictions(
        support_scaled,
        support_labels,
        query_scaled,
        readout=readout,
        logistic_c=logistic_c,
        random_state=random_state,
        max_iter=max_iter,
    )
    return float(balanced_accuracy_score(query_labels, predictions))


def multiclass_readout_predictions(
    support_scaled: np.ndarray,
    support_labels: np.ndarray,
    query_scaled: np.ndarray,
    *,
    readout: str,
    logistic_c: float,
    random_state: int,
    max_iter: int,
) -> np.ndarray:
    """Predict one fixed multiclass support/query split with the selected readout."""
    if readout == LOGISTIC:
        classifier = LogisticRegression(
            max_iter=max_iter,
            solver="lbfgs",
            random_state=random_state,
            C=float(logistic_c),
        )
        classifier.fit(support_scaled, support_labels)
        predictions = classifier.predict(query_scaled)
    elif readout == NEAREST_CENTROID:
        predictions = _predict_nearest_centroid(support_scaled, support_labels, query_scaled)
    else:
        raise ValueError(f"Unsupported readout: {readout}")
    return np.asarray(predictions)


def readout_variants(logistic_c_values: Sequence[float]) -> list[Dict[str, Any]]:
    """Enumerate the (readout, C) variants: logistic per C, plus one centroid."""
    variants: list[Dict[str, Any]] = [
        {"readout": LOGISTIC, "logistic_c": float(c)} for c in logistic_c_values
    ]
    variants.append({"readout": NEAREST_CENTROID, "logistic_c": None})
    return variants
