"""Fixed-budget few-shot transfer analysis for Anatomical Generalization.

This module implements the Surface B evaluation layer: for each low-shot target
organ, it samples a balanced, case-disjoint support/query split across organs and
modalities, trains a lightweight linear probe, and measures how well the target
organ is recognized on query examples whose source cases were not used for
adaptation.

The protocol is intentionally simple and deterministic:

    1. keep only samples with valid features and supported organs
    2. sample a fixed support budget per organ and modality
    3. build a balanced query pool from the remaining records
    4. train one multiclass linear probe on the support pool
    5. report target-organ query accuracy, in-distribution reference accuracy,
       generalization gap, and transfer efficiency

This gives Anatomical Generalization a task-facing transfer surface without introducing a new
segmentation trainer inside the package.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from .feature_projection import (
    FeatureProjectionSettings,
    apply_feature_projection,
    fit_feature_projection,
)


PROJECTED_METRIC_KEYS = (
    "projected_balanced_accuracy",
    "projected_target_query_accuracy",
    "projected_reference_organs_accuracy",
    "projected_target_vs_reference_gap",
    "projected_transfer_efficiency",
    "projected_centroid_balanced_accuracy",
    "projected_centroid_target_query_accuracy",
    "projected_adaptation_gain_balanced_accuracy",
    "projected_adaptation_gain_target_query_accuracy",
    "projected_relative_recovery_balanced_accuracy",
    "projected_relative_recovery_target_query_accuracy",
)


def _eligible_records(
    samples: Sequence[Dict[str, Any]],
    features_by_id: Dict[str, np.ndarray],
    supported_organs: Sequence[str],
    required_modalities: Sequence[str],
) -> list[Dict[str, Any]]:
    supported = set(supported_organs)
    required = set(required_modalities)
    return [
        sample
        for sample in samples
        if sample.get("sample_id") in features_by_id
        and sample.get("primary_organ") in supported
        and sample.get("modality") in required
    ]


def _group_by_organ_and_modality(
    records: Sequence[Dict[str, Any]],
    required_modalities: Sequence[str],
) -> Dict[str, Dict[str, list[Dict[str, Any]]]]:
    grouped: Dict[str, Dict[str, list[Dict[str, Any]]]] = defaultdict(
        lambda: {str(modality): [] for modality in required_modalities}
    )
    for record in records:
        grouped[str(record["primary_organ"])][str(record["modality"])].append(record)
    return grouped


def _case_id(record: Dict[str, Any]) -> str:
    return str(record.get("source_case_id") or record.get("patient_id") or record.get("sample_id"))


def _sample_case_representatives(
    records: Sequence[Dict[str, Any]],
    case_count: int,
    rng: np.random.Generator,
    candidate_case_ids: Sequence[str] | None = None,
) -> tuple[list[Dict[str, Any]], set[str]]:
    grouped_by_case: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        case_id = _case_id(record)
        grouped_by_case[case_id].append(record)

    available_case_ids = sorted(set(candidate_case_ids) if candidate_case_ids is not None else set(grouped_by_case))
    case_ids = [case_id for case_id in available_case_ids if case_id in grouped_by_case]
    if len(case_ids) < int(case_count):
        return [], set()
    indices = rng.choice(len(case_ids), size=int(case_count), replace=False)
    selected_case_ids = [case_ids[int(index)] for index in indices]

    selected_records = []
    for case_id in selected_case_ids:
        case_records = grouped_by_case[case_id]
        selected_records.append(case_records[int(rng.integers(len(case_records)))])
    return selected_records, set(selected_case_ids)


def _build_case_disjoint_support_and_query_sets(
    grouped_records: Dict[str, Dict[str, list[Dict[str, Any]]]],
    supported_organs: Sequence[str],
    required_modalities: Sequence[str],
    support_per_modality: int,
    query_per_modality: int | None,
    rng: np.random.Generator,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], int] | tuple[None, None, None]:
    support_records: list[Dict[str, Any]] = []
    support_case_ids: set[str] = set()
    for modality in required_modalities:
        common_case_ids: set[str] | None = None
        for organ in supported_organs:
            modality_records = list(grouped_records.get(organ, {}).get(modality, []))
            organ_case_ids = {_case_id(record) for record in modality_records}
            common_case_ids = organ_case_ids if common_case_ids is None else common_case_ids.intersection(organ_case_ids)

        shared_case_ids = sorted(common_case_ids or set())
        if len(shared_case_ids) >= int(support_per_modality):
            selected_indices = rng.choice(len(shared_case_ids), size=int(support_per_modality), replace=False)
            selected_shared_case_ids = [shared_case_ids[int(index)] for index in selected_indices]
            for organ in supported_organs:
                modality_records = list(grouped_records[organ][modality])
                support_subset, selected_support_case_ids = _sample_case_representatives(
                    modality_records,
                    support_per_modality,
                    rng,
                    candidate_case_ids=selected_shared_case_ids,
                )
                if len(support_subset) != int(support_per_modality):
                    return None, None, None
                support_case_ids.update(selected_support_case_ids)
                support_records.extend(support_subset)
            continue

        for organ in supported_organs:
            modality_records = list(grouped_records[organ][modality])
            support_subset, selected_support_case_ids = _sample_case_representatives(modality_records, support_per_modality, rng)
            if len(support_subset) != int(support_per_modality):
                return None, None, None
            support_case_ids.update(selected_support_case_ids)
            support_records.extend(support_subset)

    query_candidates_by_stratum: dict[tuple[str, str], list[Dict[str, Any]]] = {}
    min_remaining_cases = None
    for organ in supported_organs:
        for modality in required_modalities:
            modality_records = list(grouped_records[organ][modality])
            query_candidates = [record for record in modality_records if _case_id(record) not in support_case_ids]
            unique_query_case_ids = {_case_id(record) for record in query_candidates}
            if not unique_query_case_ids:
                return None, None, None
            min_remaining_cases = (
                len(unique_query_case_ids)
                if min_remaining_cases is None
                else min(min_remaining_cases, len(unique_query_case_ids))
            )
            query_candidates_by_stratum[(str(organ), str(modality))] = query_candidates

    if min_remaining_cases is None or min_remaining_cases <= 0:
        return None, None, None

    balanced_query_count = (
        min_remaining_cases if query_per_modality is None else min(min_remaining_cases, int(query_per_modality))
    )
    if balanced_query_count <= 0:
        return None, None, None

    query_records: list[Dict[str, Any]] = []
    for organ in supported_organs:
        for modality in required_modalities:
            query_subset, selected_query_case_ids = _sample_case_representatives(
                query_candidates_by_stratum[(str(organ), str(modality))],
                balanced_query_count,
                rng,
            )
            if len(query_subset) != int(balanced_query_count) or support_case_ids.intersection(selected_query_case_ids):
                return None, None, None
            query_records.extend(query_subset)

    return support_records, query_records, balanced_query_count


def _organ_accuracy(
    records: Sequence[Dict[str, Any]],
    predictions: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    grouped_true: Dict[str, list[int]] = defaultdict(list)
    grouped_pred: Dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        organ = str(record["primary_organ"])
        grouped_true[organ].append(int(labels[index]))
        grouped_pred[organ].append(int(predictions[index]))
    return {
        organ: float(accuracy_score(grouped_true[organ], grouped_pred[organ]))
        for organ in sorted(grouped_true)
    }


def _accuracy_by_modality(
    records: Sequence[Dict[str, Any]],
    predictions: np.ndarray,
    labels: np.ndarray,
    target_organ: str,
) -> Dict[str, float]:
    grouped_true: Dict[str, list[int]] = defaultdict(list)
    grouped_pred: Dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if str(record["primary_organ"]) != target_organ:
            continue
        modality = str(record["modality"])
        grouped_true[modality].append(int(labels[index]))
        grouped_pred[modality].append(int(predictions[index]))
    return {
        modality: float(accuracy_score(grouped_true[modality], grouped_pred[modality]))
        for modality in sorted(grouped_true)
    }


def _predict_nearest_centroid(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    query_features: np.ndarray,
) -> np.ndarray:
    labels = np.asarray(sorted(set(int(label) for label in support_labels)), dtype=np.int64)
    centroids = np.stack(
        [support_features[support_labels == label].mean(axis=0) for label in labels],
        axis=0,
    )
    squared_distances = np.sum((query_features[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return labels[np.argmin(squared_distances, axis=1)]


def _relative_recovery(adapted_score: float | None, baseline_score: float | None) -> float | None:
    if adapted_score is None or baseline_score is None:
        return None
    denominator = 1.0 - float(baseline_score)
    if denominator <= 1e-12:
        return None
    return float((float(adapted_score) - float(baseline_score)) / denominator)


def _summarize_predictions(
    *,
    records: Sequence[Dict[str, Any]],
    predictions: np.ndarray,
    labels: np.ndarray,
    target_organ: str,
) -> Dict[str, Any]:
    balanced_accuracy = float(balanced_accuracy_score(labels, predictions))
    organ_accuracy = _organ_accuracy(records, predictions, labels)
    target_query_accuracy = organ_accuracy.get(target_organ)
    reference_scores = [score for candidate, score in organ_accuracy.items() if candidate != target_organ]
    reference_organs_accuracy = _mean(reference_scores)
    target_vs_reference_gap = (
        None
        if target_query_accuracy is None or reference_organs_accuracy is None
        else float(reference_organs_accuracy - target_query_accuracy)
    )
    transfer_efficiency = (
        None
        if target_query_accuracy is None or reference_organs_accuracy in (None, 0.0)
        else float(target_query_accuracy / reference_organs_accuracy)
    )
    return {
        "balanced_accuracy": balanced_accuracy,
        "target_query_accuracy": target_query_accuracy,
        "reference_organs_accuracy": reference_organs_accuracy,
        "target_vs_reference_gap": target_vs_reference_gap,
        "transfer_efficiency": transfer_efficiency,
        "per_organ_query_accuracy": organ_accuracy,
        "target_query_accuracy_by_modality": _accuracy_by_modality(
            records,
            predictions,
            labels,
            target_organ=target_organ,
        ),
    }


def _mean(values: Iterable[float]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return float(np.mean(numeric))


def _std(values: Iterable[float]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if len(numeric) < 2:
        return None
    return float(np.std(numeric, ddof=1))


def evaluate_low_shot_target_transfer(
    features_by_id: Dict[str, np.ndarray],
    samples: Sequence[Dict[str, Any]],
    supported_organs: Sequence[str],
    required_modalities: Sequence[str] = ("ct", "mr"),
    support_per_modality: int = 2,
    query_per_modality: int | None = None,
    seeds: Sequence[int] = (42, 123, 456),
    feature_projection: FeatureProjectionSettings | None = None,
) -> Dict[str, Any]:
    """Run the Surface B fixed-budget low-shot target-organ transfer evaluation.

    Each fold names one target organ. The target organ is included in the
    support set with the same fixed budget as every other organ, and evaluation
    is performed on query examples from source cases that are disjoint from all
    support cases.
    """
    records = _eligible_records(samples, features_by_id, supported_organs, required_modalities)
    grouped_records = _group_by_organ_and_modality(records, required_modalities)
    surface_scope = "within_modality" if len(required_modalities) == 1 else "cross_modality"
    transfer_protocol = "fixed_budget_low_shot_target_transfer_case_disjoint"

    label_to_index = {organ: index for index, organ in enumerate(sorted(supported_organs))}
    organ_payloads: Dict[str, Dict[str, Any]] = {}
    macro_balanced_accuracies: list[float] = []
    macro_target_query_scores: list[float] = []
    macro_reference_scores: list[float] = []
    macro_gaps: list[float] = []
    macro_efficiencies: list[float] = []
    macro_centroid_balanced_accuracies: list[float] = []
    macro_centroid_target_query_scores: list[float] = []
    macro_adaptation_gains: list[float] = []
    macro_target_query_adaptation_gains: list[float] = []
    macro_relative_recoveries: list[float] = []
    projected_macro_values: dict[str, list[float]] = {key: [] for key in PROJECTED_METRIC_KEYS}

    organ_iterator = tqdm(sorted(supported_organs), desc="Anatomical few-shot organs", unit="organ", dynamic_ncols=True, leave=False)
    for organ in organ_iterator:
        seed_results: list[Dict[str, Any]] = []
        seed_iterator = tqdm(seeds, desc=f"Few-shot seeds {organ}", unit="seed", dynamic_ncols=True, leave=False)
        for seed in seed_iterator:
            rng = np.random.default_rng(int(seed))
            support_records, query_records, balanced_query_count = _build_case_disjoint_support_and_query_sets(
                grouped_records,
                supported_organs,
                required_modalities,
                support_per_modality=support_per_modality,
                query_per_modality=query_per_modality,
                rng=rng,
            )
            if support_records is None or query_records is None:
                continue

            support_features = np.stack([features_by_id[record["sample_id"]] for record in support_records], axis=0)
            query_features = np.stack([features_by_id[record["sample_id"]] for record in query_records], axis=0)
            support_labels = np.asarray([label_to_index[str(record["primary_organ"])] for record in support_records], dtype=np.int64)
            query_labels = np.asarray([label_to_index[str(record["primary_organ"])] for record in query_records], dtype=np.int64)

            scaler = StandardScaler()
            support_scaled = scaler.fit_transform(support_features)
            query_scaled = scaler.transform(query_features)

            classifier = LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                random_state=int(seed),
            )
            classifier.fit(support_scaled, support_labels)
            predictions = classifier.predict(query_scaled)
            probe_summary = _summarize_predictions(
                records=query_records,
                predictions=predictions,
                labels=query_labels,
                target_organ=organ,
            )
            centroid_predictions = _predict_nearest_centroid(support_scaled, support_labels, query_scaled)
            centroid_summary = _summarize_predictions(
                records=query_records,
                predictions=centroid_predictions,
                labels=query_labels,
                target_organ=organ,
            )
            target_query_records = [record for record in query_records if str(record["primary_organ"]) == organ]
            adaptation_gain_balanced_accuracy = float(
                probe_summary["balanced_accuracy"] - centroid_summary["balanced_accuracy"]
            )
            adaptation_gain_target_query_accuracy = (
                None
                if probe_summary["target_query_accuracy"] is None or centroid_summary["target_query_accuracy"] is None
                else float(probe_summary["target_query_accuracy"] - centroid_summary["target_query_accuracy"])
            )
            projected_payload: Dict[str, Any] = {}
            if feature_projection is not None and feature_projection.enabled:
                projection_fit = fit_feature_projection(
                    feature_projection,
                    support_records=support_records,
                    support_features=support_scaled,
                    scaler=scaler,
                )
                projected_support = apply_feature_projection(support_scaled, projection_fit)
                projected_query = apply_feature_projection(query_scaled, projection_fit)

                projected_classifier = LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=int(seed),
                )
                projected_classifier.fit(projected_support, support_labels)
                projected_predictions = projected_classifier.predict(projected_query)
                projected_probe_summary = _summarize_predictions(
                    records=query_records,
                    predictions=projected_predictions,
                    labels=query_labels,
                    target_organ=organ,
                )
                projected_centroid_predictions = _predict_nearest_centroid(
                    projected_support,
                    support_labels,
                    projected_query,
                )
                projected_centroid_summary = _summarize_predictions(
                    records=query_records,
                    predictions=projected_centroid_predictions,
                    labels=query_labels,
                    target_organ=organ,
                )
                projected_adaptation_gain_balanced_accuracy = float(
                    projected_probe_summary["balanced_accuracy"] - projected_centroid_summary["balanced_accuracy"]
                )
                projected_adaptation_gain_target_query_accuracy = (
                    None
                    if projected_probe_summary["target_query_accuracy"] is None
                    or projected_centroid_summary["target_query_accuracy"] is None
                    else float(
                        projected_probe_summary["target_query_accuracy"]
                        - projected_centroid_summary["target_query_accuracy"]
                    )
                )
                projected_payload = {
                    "projected_balanced_accuracy": projected_probe_summary["balanced_accuracy"],
                    "projected_target_query_accuracy": projected_probe_summary["target_query_accuracy"],
                    "projected_reference_organs_accuracy": projected_probe_summary["reference_organs_accuracy"],
                    "projected_target_vs_reference_gap": projected_probe_summary["target_vs_reference_gap"],
                    "projected_transfer_efficiency": projected_probe_summary["transfer_efficiency"],
                    "projected_per_organ_query_accuracy": projected_probe_summary["per_organ_query_accuracy"],
                    "projected_target_query_accuracy_by_modality": projected_probe_summary[
                        "target_query_accuracy_by_modality"
                    ],
                    "projected_centroid_balanced_accuracy": projected_centroid_summary["balanced_accuracy"],
                    "projected_centroid_target_query_accuracy": projected_centroid_summary["target_query_accuracy"],
                    "projected_centroid_reference_organs_accuracy": projected_centroid_summary[
                        "reference_organs_accuracy"
                    ],
                    "projected_centroid_target_vs_reference_gap": projected_centroid_summary[
                        "target_vs_reference_gap"
                    ],
                    "projected_centroid_transfer_efficiency": projected_centroid_summary["transfer_efficiency"],
                    "projected_centroid_per_organ_query_accuracy": projected_centroid_summary[
                        "per_organ_query_accuracy"
                    ],
                    "projected_centroid_target_query_accuracy_by_modality": projected_centroid_summary[
                        "target_query_accuracy_by_modality"
                    ],
                    "projected_adaptation_gain_balanced_accuracy": projected_adaptation_gain_balanced_accuracy,
                    "projected_adaptation_gain_target_query_accuracy": projected_adaptation_gain_target_query_accuracy,
                    "projected_relative_recovery_balanced_accuracy": _relative_recovery(
                        projected_probe_summary["balanced_accuracy"],
                        projected_centroid_summary["balanced_accuracy"],
                    ),
                    "projected_relative_recovery_target_query_accuracy": _relative_recovery(
                        projected_probe_summary["target_query_accuracy"],
                        projected_centroid_summary["target_query_accuracy"],
                    ),
                    "projection_diagnostics": dict(projection_fit.diagnostics),
                }

            seed_results.append(
                {
                    "seed": int(seed),
                    "support_per_modality": int(support_per_modality),
                    "query_per_modality": int(balanced_query_count),
                    "n_support_samples": int(len(support_records)),
                    "n_query_samples": int(len(query_records)),
                    "target_query_samples": int(len(target_query_records)),
                    "reference_query_samples": int(len(query_records) - len(target_query_records)),
                    "balanced_accuracy": probe_summary["balanced_accuracy"],
                    "target_query_accuracy": probe_summary["target_query_accuracy"],
                    "reference_organs_accuracy": probe_summary["reference_organs_accuracy"],
                    "target_vs_reference_gap": probe_summary["target_vs_reference_gap"],
                    "transfer_efficiency": probe_summary["transfer_efficiency"],
                    "per_organ_query_accuracy": probe_summary["per_organ_query_accuracy"],
                    "target_query_accuracy_by_modality": probe_summary["target_query_accuracy_by_modality"],
                    "nearest_centroid_balanced_accuracy": centroid_summary["balanced_accuracy"],
                    "nearest_centroid_target_query_accuracy": centroid_summary["target_query_accuracy"],
                    "nearest_centroid_reference_organs_accuracy": centroid_summary["reference_organs_accuracy"],
                    "nearest_centroid_target_vs_reference_gap": centroid_summary["target_vs_reference_gap"],
                    "nearest_centroid_transfer_efficiency": centroid_summary["transfer_efficiency"],
                    "nearest_centroid_per_organ_query_accuracy": centroid_summary["per_organ_query_accuracy"],
                    "nearest_centroid_target_query_accuracy_by_modality": centroid_summary[
                        "target_query_accuracy_by_modality"
                    ],
                    "adaptation_gain_balanced_accuracy": adaptation_gain_balanced_accuracy,
                    "adaptation_gain_target_query_accuracy": adaptation_gain_target_query_accuracy,
                    "relative_recovery_balanced_accuracy": _relative_recovery(
                        probe_summary["balanced_accuracy"],
                        centroid_summary["balanced_accuracy"],
                    ),
                    "relative_recovery_target_query_accuracy": _relative_recovery(
                        probe_summary["target_query_accuracy"],
                        centroid_summary["target_query_accuracy"],
                    ),
                    **projected_payload,
                }
            )

        if not seed_results:
            organ_payloads[organ] = {
                "status": "skipped",
                "reason": "insufficient_support_for_few_shot_surface",
                "fold_context": {
                    "low_shot_target_organ": organ,
                    "other_organs_in_multiclass_adaptation": [
                        candidate for candidate in sorted(supported_organs) if candidate != organ
                    ],
                    "fold_policy": "one_target_organ_per_low_shot_fold",
                    "surface_scope": surface_scope,
                    "support_query_split_policy": "case_disjoint_support_query",
                },
            }
            continue

        balanced_scores = [result["balanced_accuracy"] for result in seed_results if result.get("balanced_accuracy") is not None]
        target_query_scores = [
            result["target_query_accuracy"] for result in seed_results if result.get("target_query_accuracy") is not None
        ]
        reference_scores = [
            result["reference_organs_accuracy"]
            for result in seed_results
            if result.get("reference_organs_accuracy") is not None
        ]
        gaps = [
            result["target_vs_reference_gap"]
            for result in seed_results
            if result.get("target_vs_reference_gap") is not None
        ]
        efficiencies = [result["transfer_efficiency"] for result in seed_results if result.get("transfer_efficiency") is not None]
        centroid_balanced_scores = [
            result["nearest_centroid_balanced_accuracy"]
            for result in seed_results
            if result.get("nearest_centroid_balanced_accuracy") is not None
        ]
        centroid_target_query_scores = [
            result["nearest_centroid_target_query_accuracy"]
            for result in seed_results
            if result.get("nearest_centroid_target_query_accuracy") is not None
        ]
        adaptation_gains = [
            result["adaptation_gain_balanced_accuracy"]
            for result in seed_results
            if result.get("adaptation_gain_balanced_accuracy") is not None
        ]
        target_query_adaptation_gains = [
            result["adaptation_gain_target_query_accuracy"]
            for result in seed_results
            if result.get("adaptation_gain_target_query_accuracy") is not None
        ]
        relative_recoveries = [
            result["relative_recovery_balanced_accuracy"]
            for result in seed_results
            if result.get("relative_recovery_balanced_accuracy") is not None
        ]
        projected_scores = {
            key: [result[key] for result in seed_results if result.get(key) is not None]
            for key in PROJECTED_METRIC_KEYS
        }

        organ_summary = {
            "status": "ok",
            "fold_context": {
                "low_shot_target_organ": organ,
                "other_organs_in_multiclass_adaptation": [candidate for candidate in sorted(supported_organs) if candidate != organ],
                "fold_policy": "one_target_organ_per_low_shot_fold",
                "surface_scope": surface_scope,
                "support_query_split_policy": "case_disjoint_support_query",
                "surface_b_protocol": transfer_protocol,
            },
            "seed_results": seed_results,
            "balanced_accuracy": _mean(balanced_scores),
            "balanced_accuracy_std": _std(balanced_scores),
            "target_query_accuracy": _mean(target_query_scores),
            "target_query_accuracy_std": _std(target_query_scores),
            "reference_organs_accuracy": _mean(reference_scores),
            "target_vs_reference_gap": _mean(gaps),
            "transfer_efficiency": _mean(efficiencies),
            "nearest_centroid_balanced_accuracy": _mean(centroid_balanced_scores),
            "nearest_centroid_balanced_accuracy_std": _std(centroid_balanced_scores),
            "nearest_centroid_target_query_accuracy": _mean(centroid_target_query_scores),
            "nearest_centroid_target_query_accuracy_std": _std(centroid_target_query_scores),
            "adaptation_gain_balanced_accuracy": _mean(adaptation_gains),
            "adaptation_gain_balanced_accuracy_std": _std(adaptation_gains),
            "adaptation_gain_target_query_accuracy": _mean(target_query_adaptation_gains),
            "adaptation_gain_target_query_accuracy_std": _std(target_query_adaptation_gains),
            "relative_recovery_balanced_accuracy": _mean(relative_recoveries),
            "relative_recovery_balanced_accuracy_std": _std(relative_recoveries),
        }
        if feature_projection is not None and feature_projection.enabled:
            organ_summary.update(
                {
                    key: _mean(values)
                    for key, values in projected_scores.items()
                }
            )
            organ_summary.update(
                {
                    f"{key}_std": _std(values)
                    for key, values in projected_scores.items()
                }
            )
        organ_payloads[organ] = organ_summary

        if organ_summary["balanced_accuracy"] is not None:
            macro_balanced_accuracies.append(float(organ_summary["balanced_accuracy"]))
        if organ_summary["target_query_accuracy"] is not None:
            macro_target_query_scores.append(float(organ_summary["target_query_accuracy"]))
        if organ_summary["reference_organs_accuracy"] is not None:
            macro_reference_scores.append(float(organ_summary["reference_organs_accuracy"]))
        if organ_summary["target_vs_reference_gap"] is not None:
            macro_gaps.append(float(organ_summary["target_vs_reference_gap"]))
        if organ_summary["transfer_efficiency"] is not None:
            macro_efficiencies.append(float(organ_summary["transfer_efficiency"]))
        if organ_summary["nearest_centroid_balanced_accuracy"] is not None:
            macro_centroid_balanced_accuracies.append(float(organ_summary["nearest_centroid_balanced_accuracy"]))
        if organ_summary["nearest_centroid_target_query_accuracy"] is not None:
            macro_centroid_target_query_scores.append(float(organ_summary["nearest_centroid_target_query_accuracy"]))
        if organ_summary["adaptation_gain_balanced_accuracy"] is not None:
            macro_adaptation_gains.append(float(organ_summary["adaptation_gain_balanced_accuracy"]))
        if organ_summary["adaptation_gain_target_query_accuracy"] is not None:
            macro_target_query_adaptation_gains.append(float(organ_summary["adaptation_gain_target_query_accuracy"]))
        if organ_summary["relative_recovery_balanced_accuracy"] is not None:
            macro_relative_recoveries.append(float(organ_summary["relative_recovery_balanced_accuracy"]))
        if feature_projection is not None and feature_projection.enabled:
            for key in PROJECTED_METRIC_KEYS:
                if organ_summary.get(key) is not None:
                    projected_macro_values[key].append(float(organ_summary[key]))

    evaluated_organs = [organ for organ, payload in organ_payloads.items() if payload.get("status") == "ok"]
    macro_seed_results = []
    for seed in seeds:
        seed_payloads = [
            result
            for payload in organ_payloads.values()
            if payload.get("status") == "ok"
            for result in payload.get("seed_results", [])
            if int(result.get("seed")) == int(seed)
        ]
        if not seed_payloads:
            continue
        macro_seed_results.append(
            {
                "seed": int(seed),
                "balanced_accuracy": _mean(result.get("balanced_accuracy") for result in seed_payloads),
                "target_query_accuracy": _mean(result.get("target_query_accuracy") for result in seed_payloads),
                "nearest_centroid_balanced_accuracy": _mean(
                    result.get("nearest_centroid_balanced_accuracy") for result in seed_payloads
                ),
                "nearest_centroid_target_query_accuracy": _mean(
                    result.get("nearest_centroid_target_query_accuracy") for result in seed_payloads
                ),
                "adaptation_gain_balanced_accuracy": _mean(
                    result.get("adaptation_gain_balanced_accuracy") for result in seed_payloads
                ),
                "adaptation_gain_target_query_accuracy": _mean(
                    result.get("adaptation_gain_target_query_accuracy") for result in seed_payloads
                ),
                "relative_recovery_balanced_accuracy": _mean(
                    result.get("relative_recovery_balanced_accuracy") for result in seed_payloads
                ),
                "n_evaluated_organs": len(seed_payloads),
            }
        )
        if feature_projection is not None and feature_projection.enabled:
            macro_seed_results[-1].update(
                {
                    key: _mean(result.get(key) for result in seed_payloads)
                    for key in PROJECTED_METRIC_KEYS
                }
            )

    macro_mean = {
        "balanced_accuracy": _mean(macro_balanced_accuracies),
        "target_query_accuracy": _mean(macro_target_query_scores),
        "reference_organs_accuracy": _mean(macro_reference_scores),
        "target_vs_reference_gap": _mean(macro_gaps),
        "transfer_efficiency": _mean(macro_efficiencies),
        "nearest_centroid_balanced_accuracy": _mean(macro_centroid_balanced_accuracies),
        "nearest_centroid_target_query_accuracy": _mean(macro_centroid_target_query_scores),
        "adaptation_gain_balanced_accuracy": _mean(macro_adaptation_gains),
        "adaptation_gain_target_query_accuracy": _mean(macro_target_query_adaptation_gains),
        "relative_recovery_balanced_accuracy": _mean(macro_relative_recoveries),
    }
    macro_mean_std = {
        "balanced_accuracy": _std(result.get("balanced_accuracy") for result in macro_seed_results),
        "target_query_accuracy": _std(result.get("target_query_accuracy") for result in macro_seed_results),
        "nearest_centroid_balanced_accuracy": _std(
            result.get("nearest_centroid_balanced_accuracy") for result in macro_seed_results
        ),
        "nearest_centroid_target_query_accuracy": _std(
            result.get("nearest_centroid_target_query_accuracy") for result in macro_seed_results
        ),
        "adaptation_gain_balanced_accuracy": _std(
            result.get("adaptation_gain_balanced_accuracy") for result in macro_seed_results
        ),
        "adaptation_gain_target_query_accuracy": _std(
            result.get("adaptation_gain_target_query_accuracy") for result in macro_seed_results
        ),
        "relative_recovery_balanced_accuracy": _std(
            result.get("relative_recovery_balanced_accuracy") for result in macro_seed_results
        ),
    }
    if feature_projection is not None and feature_projection.enabled:
        macro_mean.update({key: _mean(values) for key, values in projected_macro_values.items()})
        macro_mean_std.update(
            {
                key: _std(result.get(key) for result in macro_seed_results)
                for key in PROJECTED_METRIC_KEYS
            }
        )

    return {
        "status": "ok" if evaluated_organs else "skipped",
        "surface_name": "surface_b_few_shot_transfer_evidence",
        "surface_scope": surface_scope,
        "target_definition": "low_shot_target_organ",
        "target_definition_note": (
            "The target organ is represented in the fixed support set; metrics are evaluated on query examples "
            "from source cases disjoint from all support cases."
        ),
        "support_query_split_policy": "case_disjoint_support_query",
        "transfer_protocol": transfer_protocol,
        "required_modalities": list(required_modalities),
        "support_per_modality": int(support_per_modality),
        "query_per_modality": query_per_modality,
        "evaluation_seeds": [int(seed) for seed in seeds],
        "n_supported_organs": len(list(supported_organs)),
        "n_evaluated_organs": len(evaluated_organs),
        "feature_projection": feature_projection.to_payload()
        if feature_projection is not None and feature_projection.enabled
        else None,
        "macro_mean": macro_mean,
        "macro_mean_std": macro_mean_std,
        "macro_seed_results": macro_seed_results,
        "per_organ": organ_payloads,
    }


def _max_feasible_support_per_modality(
    grouped_records: Dict[str, Dict[str, list[Dict[str, Any]]]],
    supported_organs: Sequence[str],
    required_modalities: Sequence[str],
) -> int:
    """Return the largest support budget that still leaves at least one query case."""
    minima: list[int] = []
    for organ in supported_organs:
        for modality in required_modalities:
            case_ids = {
                _case_id(record)
                for record in grouped_records.get(str(organ), {}).get(str(modality), [])
            }
            minima.append(max(len(case_ids) - 1, 0))
    return min(minima) if minima else 0


def evaluate_support_size_sweep(
    features_by_id: Dict[str, np.ndarray],
    samples: Sequence[Dict[str, Any]],
    supported_organs: Sequence[str],
    required_modalities: Sequence[str] = ("ct", "mr"),
    support_sizes: Sequence[int] = (1, 2, 4, 8),
    query_per_modality: int | None = None,
    seeds: Sequence[int] = (42, 123, 456),
    feature_projection: FeatureProjectionSettings | None = None,
) -> Dict[str, Any]:
    """Evaluate low-shot transfer across support budgets with availability caps."""
    records = _eligible_records(samples, features_by_id, supported_organs, required_modalities)
    grouped_records = _group_by_organ_and_modality(records, required_modalities)
    max_support = _max_feasible_support_per_modality(grouped_records, supported_organs, required_modalities)
    rows: list[Dict[str, Any]] = []
    seen_effective: set[int] = set()

    for requested_support in [int(value) for value in support_sizes if int(value) > 0]:
        effective_support = min(requested_support, max_support)
        if effective_support <= 0:
            rows.append(
                {
                    "requested_support_per_modality": requested_support,
                    "effective_support_per_modality": None,
                    "status": "skipped",
                    "reason": "insufficient_case_disjoint_support_and_query_cases",
                }
            )
            continue
        if effective_support in seen_effective:
            rows.append(
                {
                    "requested_support_per_modality": requested_support,
                    "effective_support_per_modality": effective_support,
                    "status": "aliased_to_prior_budget",
                    "reason": "requested_support_exceeded_available_case_disjoint_budget",
                }
            )
            continue
        seen_effective.add(effective_support)
        result = evaluate_low_shot_target_transfer(
            features_by_id,
            samples,
            supported_organs=supported_organs,
            required_modalities=required_modalities,
            support_per_modality=effective_support,
            query_per_modality=query_per_modality,
            seeds=seeds,
            feature_projection=feature_projection,
        )
        macro = result.get("macro_mean", {})
        macro_std = result.get("macro_mean_std", {})
        row = {
            "requested_support_per_modality": requested_support,
            "effective_support_per_modality": effective_support,
            "status": result.get("status"),
            "n_supported_organs": result.get("n_supported_organs"),
            "n_evaluated_organs": result.get("n_evaluated_organs"),
            "balanced_accuracy": macro.get("balanced_accuracy"),
            "balanced_accuracy_seed_std": macro_std.get("balanced_accuracy"),
            "nearest_centroid_balanced_accuracy": macro.get("nearest_centroid_balanced_accuracy"),
            "adaptation_gain_balanced_accuracy": macro.get("adaptation_gain_balanced_accuracy"),
            "adaptation_gain_balanced_accuracy_seed_std": macro_std.get("adaptation_gain_balanced_accuracy"),
            "query_per_modality": result.get("query_per_modality"),
            "evaluation_seeds": result.get("evaluation_seeds"),
            "full_result": result,
        }
        if feature_projection is not None and feature_projection.enabled:
            row.update(
                {
                    "projected_balanced_accuracy": macro.get("projected_balanced_accuracy"),
                    "projected_balanced_accuracy_seed_std": macro_std.get("projected_balanced_accuracy"),
                    "projected_centroid_balanced_accuracy": macro.get("projected_centroid_balanced_accuracy"),
                    "projected_adaptation_gain_balanced_accuracy": macro.get(
                        "projected_adaptation_gain_balanced_accuracy"
                    ),
                    "projected_adaptation_gain_balanced_accuracy_seed_std": macro_std.get(
                        "projected_adaptation_gain_balanced_accuracy"
                    ),
                }
            )
        rows.append(row)

    return {
        "artifact": "support_size_sweep",
        "support_sizes_requested": [int(value) for value in support_sizes],
        "max_feasible_support_per_modality": int(max_support),
        "required_modalities": list(required_modalities),
        "support_query_split_policy": "case_disjoint_support_query",
        "class_balance_policy": "fixed support and balanced query count per organ/modality stratum; balanced accuracy is the headline metric",
        "feature_projection": feature_projection.to_payload()
        if feature_projection is not None and feature_projection.enabled
        else None,
        "rows": rows,
    }


def support_size_sweep_csv_rows(sweep: Dict[str, Any]) -> list[Dict[str, Any]]:
    rows = []
    for row in sweep.get("rows", []):
        rows.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"full_result", "evaluation_seeds"}
            }
        )
    return rows
