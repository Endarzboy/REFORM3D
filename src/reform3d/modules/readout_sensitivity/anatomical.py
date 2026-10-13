"""Low-shot anatomical-transfer readout-sensitivity for one cell.

Reuses the production case-disjoint support/query construction so every readout
sees identical support and query sample IDs; only the classifier (or its ``C``)
changes. For each target-organ fold and seed it records the balanced accuracy of
logistic ``C in {0.1, 1, 10}`` and the nearest-centroid baseline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from reform3d.modules.anatomical_generalization.data_loader import (
    get_eligible_holdout_organs,
    load_anatomical_manifest,
)
from reform3d.modules.anatomical_generalization.few_shot_transfer_analysis import (
    _build_case_disjoint_support_and_query_sets,
    _eligible_records,
    _group_by_organ_and_modality,
)

from . import config
from .readouts import LOGISTIC, NEAREST_CENTROID, multiclass_readout_predictions


def _readout_key(readout: str, logistic_c: float | None) -> str:
    if readout == LOGISTIC:
        return f"logistic_C{logistic_c:g}"
    return "nearest_centroid"


def load_anatomical_features(npz_path: Path) -> Dict[str, np.ndarray]:
    """Build sample_id -> feature row from a frozen Cross-Modal embedding bundle.

    The frozen NPZ is the authoritative feature source. Unlike the strict pipeline
    loader, this does not require the manifest to be a subset of the bundle: any
    manifest sample lacking a feature is dropped downstream by the production
    ``_eligible_records`` filter (exactly as the production few-shot loop does).
    All bundle features must be finite and sample ids unique.
    """
    with np.load(npz_path, allow_pickle=True) as payload:
        if "features" not in payload.files:
            raise ValueError(f"Embedding bundle {npz_path} must contain a 'features' array")
        features = np.asarray(payload["features"], dtype=np.float32)
        if "sample_ids" not in payload.files:
            raise ValueError(f"Embedding bundle {npz_path} must contain 'sample_ids'")
        sample_ids = [str(item) for item in np.asarray(payload["sample_ids"]).reshape(-1).tolist()]
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError(f"Embedding bundle {npz_path} has invalid feature shape {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError(f"Embedding bundle {npz_path} contains non-finite features")
    if len(sample_ids) != features.shape[0]:
        raise ValueError(f"Embedding bundle {npz_path} sample_ids length mismatch")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"Embedding bundle {npz_path} contains duplicate sample_ids")
    return {sample_id: features[index] for index, sample_id in enumerate(sample_ids)}


def evaluate_anatomical_cell(
    dataset: str,
    checkpoint: str,
    feature_type: str,
    seed: int,
    logistic_c_values: Sequence[float] = config.LOGISTIC_C_VALUES,
) -> Dict[str, Any]:
    """Evaluate all readout variants for one anatomical (dataset, ckpt, feat, seed) cell."""
    manifest_path = config.anatomical_manifest_path(dataset)
    embeddings_path = config.anatomical_embeddings_path(dataset, checkpoint, feature_type, seed)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Anatomical manifest not found: {manifest_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Anatomical embeddings not found: {embeddings_path}")

    samples = load_anatomical_manifest(manifest_path)
    required_modalities = config.ANATOMICAL_REQUIRED_MODALITIES
    supported_organs = get_eligible_holdout_organs(
        samples,
        required_modalities=required_modalities,
        min_samples_per_modality=config.ANATOMICAL_MIN_SAMPLES_PER_MODALITY,
    )
    features_by_id = load_anatomical_features(embeddings_path)
    n_manifest_without_features = sum(
        1 for sample in samples if str(sample["sample_id"]) not in features_by_id
    )

    records = _eligible_records(samples, features_by_id, supported_organs, required_modalities)
    grouped_records = _group_by_organ_and_modality(records, required_modalities)
    label_to_index = {organ: index for index, organ in enumerate(sorted(supported_organs))}

    rng = np.random.default_rng(int(seed))
    support_records, query_records, balanced_query_count = _build_case_disjoint_support_and_query_sets(
        grouped_records,
        supported_organs,
        required_modalities,
        support_per_modality=config.ANATOMICAL_SUPPORT_PER_MODALITY,
        query_per_modality=config.ANATOMICAL_QUERY_PER_MODALITY,
        rng=rng,
    )
    if support_records is None or query_records is None:
        raise ValueError(f"Insufficient case-disjoint support/query data for {dataset}/seed_{seed}")

    support_features = np.stack(
        [features_by_id[record["sample_id"]] for record in support_records], axis=0
    )
    query_features = np.stack(
        [features_by_id[record["sample_id"]] for record in query_records], axis=0
    )
    support_labels = np.asarray(
        [label_to_index[str(record["primary_organ"])] for record in support_records],
        dtype=np.int64,
    )
    query_labels = np.asarray(
        [label_to_index[str(record["primary_organ"])] for record in query_records],
        dtype=np.int64,
    )
    if not (np.isfinite(support_features).all() and np.isfinite(query_features).all()):
        raise ValueError(f"Non-finite anatomical features for {dataset}")

    scaler = StandardScaler()
    support_scaled = scaler.fit_transform(support_features)
    query_scaled = scaler.transform(query_features)

    predictions_by_readout: Dict[str, np.ndarray] = {}
    for logistic_c in logistic_c_values:
        key = _readout_key(LOGISTIC, logistic_c)
        predictions_by_readout[key] = multiclass_readout_predictions(
            support_scaled,
            support_labels,
            query_scaled,
            readout=LOGISTIC,
            logistic_c=float(logistic_c),
            random_state=int(seed),
            max_iter=config.ANATOMICAL_LOGISTIC_MAX_ITER,
        )
    predictions_by_readout[_readout_key(NEAREST_CENTROID, None)] = multiclass_readout_predictions(
        support_scaled,
        support_labels,
        query_scaled,
        readout=NEAREST_CENTROID,
        logistic_c=1.0,
        random_state=int(seed),
        max_iter=config.ANATOMICAL_LOGISTIC_MAX_ITER,
    )
    balanced_accuracy_by_readout = {
        key: float(balanced_accuracy_score(query_labels, predictions))
        for key, predictions in predictions_by_readout.items()
    }

    organ_results: Dict[str, Any] = {}
    query_organs = np.asarray([str(record["primary_organ"]) for record in query_records])
    for organ in sorted(supported_organs):
        organ_mask = query_organs == organ
        if not np.any(organ_mask):
            organ_results[organ] = {"status": "skipped", "reason": "no_target_queries"}
            continue
        organ_results[organ] = {
            "status": "ok",
            "target_organ": organ,
            "n_target_query_samples": int(organ_mask.sum()),
            "target_query_recall_by_readout": {
                key: float(np.mean(predictions[organ_mask] == query_labels[organ_mask]))
                for key, predictions in predictions_by_readout.items()
            },
        }
    for key, balanced_accuracy in balanced_accuracy_by_readout.items():
        recalls = [
            payload["target_query_recall_by_readout"][key]
            for payload in organ_results.values()
            if payload.get("status") == "ok"
        ]
        reconstructed = float(np.mean(recalls)) if recalls else 0.0
        if abs(reconstructed - balanced_accuracy) > 1e-12:
            raise RuntimeError(
                f"Anatomical BA reconstruction failed for {dataset}/{key}: "
                f"balanced_accuracy={balanced_accuracy:.12f}, "
                f"mean_target_recall={reconstructed:.12f}"
            )

    evaluated = [organ for organ, payload in organ_results.items() if payload.get("status") == "ok"]
    return {
        "schema_version": 2,
        "dataset": dataset,
        "checkpoint": checkpoint,
        "feature_type": feature_type,
        "seed": int(seed),
        "manifest_path": str(manifest_path),
        "embeddings_npz": str(embeddings_path),
        "required_modalities": list(required_modalities),
        "support_per_modality": int(config.ANATOMICAL_SUPPORT_PER_MODALITY),
        "query_per_modality": config.ANATOMICAL_QUERY_PER_MODALITY,
        "min_samples_per_modality": int(config.ANATOMICAL_MIN_SAMPLES_PER_MODALITY),
        "supported_organs": sorted(supported_organs),
        "n_supported_organs": len(list(supported_organs)),
        "n_evaluated_organs": len(evaluated),
        "n_manifest_samples": len(samples),
        "n_feature_samples": len(features_by_id),
        "n_manifest_samples_without_features": int(n_manifest_without_features),
        "query_per_modality_actual": int(balanced_query_count),
        "n_support_samples": int(len(support_records)),
        "n_query_samples": int(len(query_records)),
        "support_sample_ids": [str(record["sample_id"]) for record in support_records],
        "query_sample_ids": [str(record["sample_id"]) for record in query_records],
        "balanced_accuracy_by_readout": balanced_accuracy_by_readout,
        "per_organ": organ_results,
    }


def production_anatomical_baseline(
    dataset: str,
    checkpoint: str,
    feature_type: str,
    seed: int,
) -> Dict[str, Any]:
    """Freshly recompute the production logistic (C=1) and centroid macro means.

    Independent code path (``evaluate_low_shot_target_transfer``) used only to
    gate the readout-sensitivity results. Anchored to current production because
    the stored JSON artifacts are stale for seeds 123/456 (see module docs).
    """
    from reform3d.modules.anatomical_generalization.few_shot_transfer_analysis import (
        evaluate_low_shot_target_transfer,
    )

    manifest_path = config.anatomical_manifest_path(dataset)
    embeddings_path = config.anatomical_embeddings_path(dataset, checkpoint, feature_type, seed)
    samples = load_anatomical_manifest(manifest_path)
    required_modalities = config.ANATOMICAL_REQUIRED_MODALITIES
    supported_organs = get_eligible_holdout_organs(
        samples,
        required_modalities=required_modalities,
        min_samples_per_modality=config.ANATOMICAL_MIN_SAMPLES_PER_MODALITY,
    )
    features_by_id = load_anatomical_features(embeddings_path)
    result = evaluate_low_shot_target_transfer(
        features_by_id,
        samples,
        supported_organs=supported_organs,
        required_modalities=required_modalities,
        support_per_modality=config.ANATOMICAL_SUPPORT_PER_MODALITY,
        query_per_modality=config.ANATOMICAL_QUERY_PER_MODALITY,
        seeds=(int(seed),),
        feature_projection=None,
    )
    macro = result["macro_mean"]
    target_query_by_organ = {
        organ: {
            "logistic_C1": float(payload["target_query_accuracy"]),
            "nearest_centroid": float(payload["nearest_centroid_target_query_accuracy"]),
        }
        for organ, payload in result["per_organ"].items()
        if payload.get("status") == "ok"
    }
    return {
        "logistic_C1": float(macro["balanced_accuracy"]),
        "nearest_centroid": float(macro["nearest_centroid_balanced_accuracy"]),
        "n_evaluated_organs": int(result["n_evaluated_organs"]),
        "target_query_recall_by_organ": target_query_by_organ,
    }
