"""Deterministic sampling plans for Cross-Modal protocol comparisons."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from reform3d.modules.anatomical_generalization.few_shot_transfer_analysis import (
    _build_case_disjoint_support_and_query_sets,
    _eligible_records,
    _group_by_organ_and_modality,
    _predict_nearest_centroid,
    _relative_recovery,
)

from .retrieval_metrics import (
    _average_precision,
    _balanced_top_chance,
    _l2_normalize,
    _top_metric_key,
)


def _case_id(record: Mapping[str, Any]) -> str:
    return str(
        record.get("source_case_id")
        or record.get("patient_id")
        or record.get("sample_id")
    )


def validate_sample_records(samples: Sequence[Mapping[str, Any]]) -> None:
    sample_ids = [str(sample["sample_id"]) for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Sample records contain duplicate sample IDs")
    for sample in samples:
        if not sample.get("primary_organ"):
            raise ValueError(f"Sample {sample.get('sample_id')} has no primary organ")
        if not sample.get("modality"):
            raise ValueError(f"Sample {sample.get('sample_id')} has no modality")


def _sample_ids(
    records: Sequence[Mapping[str, Any]],
    sample_count: int,
    rng: np.random.Generator,
) -> list[str]:
    if sample_count >= len(records):
        return [str(record["sample_id"]) for record in records]
    indices = rng.choice(len(records), size=sample_count, replace=False)
    return [str(records[int(index)]["sample_id"]) for index in indices]


def _build_direction_plan(
    samples: Sequence[Mapping[str, Any]],
    *,
    query_modality: str,
    target_modality: str,
    seed: int,
    supported_organs: Sequence[str] | None = None,
) -> dict[str, Any]:
    supported = set(supported_organs or [])
    queries_by_organ: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    targets_by_organ: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        organ = str(sample["primary_organ"])
        if supported and organ not in supported:
            continue
        modality = str(sample["modality"])
        if modality == query_modality:
            queries_by_organ[organ].append(sample)
        if modality == target_modality:
            targets_by_organ[organ].append(sample)

    common_organs = sorted(set(queries_by_organ) & set(targets_by_organ))
    if len(common_organs) < 2:
        raise ValueError(
            f"Need at least two shared organs for {query_modality}->{target_modality}"
        )
    queries_per_organ = min(len(queries_by_organ[organ]) for organ in common_organs)
    targets_per_organ = min(len(targets_by_organ[organ]) for organ in common_organs)
    if queries_per_organ <= 0 or targets_per_organ <= 0:
        raise ValueError(
            f"Unable to form balanced retrieval plan for {query_modality}->{target_modality}"
        )

    rng = np.random.default_rng(int(seed))
    query_ids = [
        sample_id
        for organ in common_organs
        for sample_id in _sample_ids(
            queries_by_organ[organ],
            queries_per_organ,
            rng,
        )
    ]
    target_ids = [
        sample_id
        for organ in common_organs
        for sample_id in _sample_ids(
            targets_by_organ[organ],
            targets_per_organ,
            rng,
        )
    ]
    return {
        "query_modality": query_modality,
        "target_modality": target_modality,
        "sampling_seed": int(seed),
        "organs": common_organs,
        "queries_per_organ": int(queries_per_organ),
        "targets_per_organ": int(targets_per_organ),
        "query_sample_ids": query_ids,
        "target_sample_ids": target_ids,
    }


def build_retrieval_plan(
    samples: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    supported_organs: Sequence[str] | None = None,
) -> dict[str, Any]:
    validate_sample_records(samples)
    plan = {
        "schema_version": 1,
        "evaluation_seed": int(seed),
        "directions": {
            "ct_to_mr": _build_direction_plan(
                samples,
                query_modality="ct",
                target_modality="mr",
                seed=int(seed),
                supported_organs=supported_organs,
            ),
            "mr_to_ct": _build_direction_plan(
                samples,
                query_modality="mr",
                target_modality="ct",
                seed=int(seed) + 1,
                supported_organs=supported_organs,
            ),
        },
    }
    validate_retrieval_plan(plan, samples)
    return plan


def validate_retrieval_plan(
    plan: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
) -> None:
    records_by_id = {str(sample["sample_id"]): sample for sample in samples}
    for direction_name, direction in plan.get("directions", {}).items():
        query_ids = [str(value) for value in direction.get("query_sample_ids", [])]
        target_ids = [str(value) for value in direction.get("target_sample_ids", [])]
        if not query_ids or not target_ids:
            raise ValueError(f"Retrieval plan {direction_name} is empty")
        if len(query_ids) != len(set(query_ids)):
            raise ValueError(f"Retrieval plan {direction_name} has duplicate queries")
        if len(target_ids) != len(set(target_ids)):
            raise ValueError(f"Retrieval plan {direction_name} has duplicate targets")
        missing = sorted((set(query_ids) | set(target_ids)) - set(records_by_id))
        if missing:
            raise ValueError(
                f"Retrieval plan {direction_name} references missing samples: {missing[:3]}"
            )

        query_modality = str(direction["query_modality"])
        target_modality = str(direction["target_modality"])
        organs = [str(value) for value in direction["organs"]]
        query_counts: dict[str, int] = defaultdict(int)
        target_counts: dict[str, int] = defaultdict(int)
        for sample_id in query_ids:
            record = records_by_id[sample_id]
            if str(record["modality"]) != query_modality:
                raise ValueError(f"Query modality mismatch for {sample_id}")
            query_counts[str(record["primary_organ"])] += 1
        for sample_id in target_ids:
            record = records_by_id[sample_id]
            if str(record["modality"]) != target_modality:
                raise ValueError(f"Target modality mismatch for {sample_id}")
            target_counts[str(record["primary_organ"])] += 1
        expected_query_count = int(direction["queries_per_organ"])
        expected_target_count = int(direction["targets_per_organ"])
        if any(query_counts.get(organ, 0) != expected_query_count for organ in organs):
            raise ValueError(f"Unbalanced query plan for {direction_name}")
        if any(target_counts.get(organ, 0) != expected_target_count for organ in organs):
            raise ValueError(f"Unbalanced target plan for {direction_name}")


def build_anatomical_plan(
    samples: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    supported_organs: Sequence[str],
    required_modalities: Sequence[str] = ("ct", "mr"),
    support_per_modality: int = 2,
    query_per_modality: int | None = None,
) -> dict[str, Any]:
    validate_sample_records(samples)
    available = {str(sample["sample_id"]): np.empty(1, dtype=np.float32) for sample in samples}
    records = _eligible_records(
        list(samples),
        available,
        supported_organs,
        required_modalities,
    )
    grouped = _group_by_organ_and_modality(records, required_modalities)
    support, query, balanced_query_count = _build_case_disjoint_support_and_query_sets(
        grouped,
        supported_organs,
        required_modalities,
        support_per_modality=int(support_per_modality),
        query_per_modality=query_per_modality,
        rng=np.random.default_rng(int(seed)),
    )
    if support is None or query is None or balanced_query_count is None:
        raise ValueError(f"Unable to build anatomical plan for seed {seed}")

    plan = {
        "schema_version": 1,
        "evaluation_seed": int(seed),
        "supported_organs": sorted(str(value) for value in supported_organs),
        "required_modalities": [str(value) for value in required_modalities],
        "support_per_modality": int(support_per_modality),
        "query_per_modality": int(balanced_query_count),
        "support_sample_ids": [str(record["sample_id"]) for record in support],
        "query_sample_ids": [str(record["sample_id"]) for record in query],
    }
    validate_anatomical_plan(plan, samples)
    return plan


def validate_anatomical_plan(
    plan: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
) -> None:
    records_by_id = {str(sample["sample_id"]): sample for sample in samples}
    support_ids = [str(value) for value in plan.get("support_sample_ids", [])]
    query_ids = [str(value) for value in plan.get("query_sample_ids", [])]
    if not support_ids or not query_ids:
        raise ValueError("Anatomical plan has an empty support or query set")
    if len(support_ids) != len(set(support_ids)):
        raise ValueError("Anatomical support plan has duplicate sample IDs")
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Anatomical query plan has duplicate sample IDs")
    missing = sorted((set(support_ids) | set(query_ids)) - set(records_by_id))
    if missing:
        raise ValueError(f"Anatomical plan references missing samples: {missing[:3]}")

    support_cases = {_case_id(records_by_id[sample_id]) for sample_id in support_ids}
    query_cases = {_case_id(records_by_id[sample_id]) for sample_id in query_ids}
    overlap = sorted(support_cases & query_cases)
    if overlap:
        raise ValueError(f"Support/query source-case overlap: {overlap[:3]}")

    organs = [str(value) for value in plan["supported_organs"]]
    modalities = [str(value) for value in plan["required_modalities"]]
    support_count = int(plan["support_per_modality"])
    query_count = int(plan["query_per_modality"])
    support_strata: dict[tuple[str, str], int] = defaultdict(int)
    query_strata: dict[tuple[str, str], int] = defaultdict(int)
    for sample_id in support_ids:
        record = records_by_id[sample_id]
        support_strata[
            (str(record["primary_organ"]), str(record["modality"]))
        ] += 1
    for sample_id in query_ids:
        record = records_by_id[sample_id]
        query_strata[
            (str(record["primary_organ"]), str(record["modality"]))
        ] += 1
    for organ in organs:
        for modality in modalities:
            if support_strata[(organ, modality)] != support_count:
                raise ValueError(f"Unbalanced support stratum: {organ}/{modality}")
            if query_strata[(organ, modality)] != query_count:
                raise ValueError(f"Unbalanced query stratum: {organ}/{modality}")


def _evaluate_retrieval_direction(
    features_by_id: Mapping[str, np.ndarray],
    records_by_id: Mapping[str, Mapping[str, Any]],
    direction: Mapping[str, Any],
    top_ks: Sequence[int],
) -> dict[str, Any]:
    query_ids = [str(value) for value in direction["query_sample_ids"]]
    target_ids = [str(value) for value in direction["target_sample_ids"]]
    missing = sorted((set(query_ids) | set(target_ids)) - set(features_by_id))
    if missing:
        raise ValueError(f"Feature bundle is missing planned retrieval IDs: {missing[:3]}")

    query_features = _l2_normalize(
        np.stack([features_by_id[sample_id] for sample_id in query_ids], axis=0)
    )
    target_features = _l2_normalize(
        np.stack([features_by_id[sample_id] for sample_id in target_ids], axis=0)
    )
    similarity = query_features @ target_features.T
    target_organs = np.asarray(
        [str(records_by_id[sample_id]["primary_organ"]) for sample_id in target_ids],
        dtype=object,
    )
    metric_ks = tuple(sorted(set(int(value) for value in top_ks if int(value) > 0)))
    query_rows: list[dict[str, Any]] = []
    per_organ: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"map": [], **{_top_metric_key(value): [] for value in metric_ks}}
    )
    for index, sample_id in enumerate(query_ids):
        query_record = records_by_id[sample_id]
        organ = str(query_record["primary_organ"])
        ranking = np.argsort(-similarity[index])
        relevance = target_organs[ranking] == organ
        row = {
            "sample_id": sample_id,
            "source_case_id": _case_id(query_record),
            "organ": organ,
            "query_modality": str(direction["query_modality"]),
            "target_modality": str(direction["target_modality"]),
            "map": _average_precision(relevance),
            "n_positive_targets": int(np.count_nonzero(relevance)),
            "n_targets": int(len(target_ids)),
        }
        row["average_precision"] = row["map"]
        per_organ[organ]["map"].append(float(row["map"]))
        for value in metric_ks:
            key = _top_metric_key(value)
            row[key] = float(np.any(relevance[:value]))
            per_organ[organ][key].append(float(row[key]))
        query_rows.append(row)

    overall = {
        key: float(np.mean([row[key] for row in query_rows]))
        for key in ("map", *(_top_metric_key(value) for value in metric_ks))
    }
    organs = [str(value) for value in direction["organs"]]
    return {
        "status": "ok",
        "evaluation": "balanced_cross_modal_organ_retrieval",
        "query_modality": str(direction["query_modality"]),
        "target_modality": str(direction["target_modality"]),
        "n_queries": len(query_ids),
        "n_targets": len(target_ids),
        "n_organs": len(organs),
        "queries_per_organ": int(direction["queries_per_organ"]),
        "targets_per_organ": int(direction["targets_per_organ"]),
        "overall": overall,
        "organ_balanced_chance": {
            _top_metric_key(value): _balanced_top_chance(
                len(organs),
                int(direction["targets_per_organ"]),
                value,
            )
            for value in metric_ks
        },
        "per_organ": {
            organ: {
                "n_queries": len(per_organ[organ]["map"]),
                **{
                    key: float(np.mean(values))
                    for key, values in per_organ[organ].items()
                },
            }
            for organ in sorted(per_organ)
        },
        "query_results": query_rows,
        "query_sample_ids": query_ids,
        "target_sample_ids": target_ids,
    }


def evaluate_retrieval_plan(
    features_by_id: Mapping[str, np.ndarray],
    samples: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    top_ks: Sequence[int] = (1, 5),
) -> dict[str, Any]:
    validate_retrieval_plan(plan, samples)
    records_by_id = {str(sample["sample_id"]): sample for sample in samples}
    ct_to_mr = _evaluate_retrieval_direction(
        features_by_id,
        records_by_id,
        plan["directions"]["ct_to_mr"],
        top_ks,
    )
    mr_to_ct = _evaluate_retrieval_direction(
        features_by_id,
        records_by_id,
        plan["directions"]["mr_to_ct"],
        top_ks,
    )
    shared_keys = sorted(set(ct_to_mr["overall"]) & set(mr_to_ct["overall"]))
    return {
        "status": "ok",
        "evaluation_seed": int(plan["evaluation_seed"]),
        "sampling_plan_schema_version": int(plan["schema_version"]),
        "ct_to_mr": ct_to_mr,
        "mr_to_ct": mr_to_ct,
        "bidirectional_mean": {
            key: float(
                (ct_to_mr["overall"][key] + mr_to_ct["overall"][key]) / 2.0
            )
            for key in shared_keys
        },
    }


def _organ_accuracies(
    records: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    values: dict[str, list[int]] = defaultdict(list)
    predictions_by_organ: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        organ = str(record["primary_organ"])
        values[organ].append(int(labels[index]))
        predictions_by_organ[organ].append(int(predictions[index]))
    return {
        organ: float(accuracy_score(values[organ], predictions_by_organ[organ]))
        for organ in sorted(values)
    }


def _modality_accuracy_for_organ(
    records: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    predictions: np.ndarray,
    organ: str,
) -> dict[str, float]:
    indices_by_modality: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if str(record["primary_organ"]) == organ:
            indices_by_modality[str(record["modality"])].append(index)
    return {
        modality: float(
            accuracy_score(
                labels[np.asarray(indices, dtype=np.int64)],
                predictions[np.asarray(indices, dtype=np.int64)],
            )
        )
        for modality, indices in sorted(indices_by_modality.items())
    }


def _mean(values: Sequence[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return float(np.mean(numeric)) if numeric else None


def _std(values: Sequence[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return float(np.std(numeric, ddof=1)) if len(numeric) >= 2 else None


def _seed_prediction_summary(
    *,
    records: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    predictions: np.ndarray,
    target_organ: str,
) -> dict[str, Any]:
    per_organ = _organ_accuracies(records, labels, predictions)
    target = per_organ[target_organ]
    reference = _mean(
        [score for organ, score in per_organ.items() if organ != target_organ]
    )
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "target_query_accuracy": target,
        "reference_organs_accuracy": reference,
        "target_vs_reference_gap": (
            None if reference is None else float(reference - target)
        ),
        "transfer_efficiency": (
            None if reference in (None, 0.0) else float(target / reference)
        ),
        "per_organ_query_accuracy": per_organ,
        "target_query_accuracy_by_modality": _modality_accuracy_for_organ(
            records,
            labels,
            predictions,
            target_organ,
        ),
    }


def evaluate_anatomical_plans(
    features_by_id: Mapping[str, np.ndarray],
    samples: Sequence[Mapping[str, Any]],
    plans: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not plans:
        raise ValueError("At least one anatomical sampling plan is required")
    records_by_id = {str(sample["sample_id"]): sample for sample in samples}
    supported_organs = [str(value) for value in plans[0]["supported_organs"]]
    required_modalities = [str(value) for value in plans[0]["required_modalities"]]
    label_to_index = {
        organ: index for index, organ in enumerate(sorted(supported_organs))
    }
    per_organ_seed_results: dict[str, list[dict[str, Any]]] = {
        organ: [] for organ in sorted(supported_organs)
    }

    for plan in plans:
        validate_anatomical_plan(plan, samples)
        if [str(value) for value in plan["supported_organs"]] != supported_organs:
            raise ValueError("Anatomical plans use different organ sets")
        support_ids = [str(value) for value in plan["support_sample_ids"]]
        query_ids = [str(value) for value in plan["query_sample_ids"]]
        missing = sorted((set(support_ids) | set(query_ids)) - set(features_by_id))
        if missing:
            raise ValueError(f"Feature bundle is missing anatomical IDs: {missing[:3]}")
        support_records = [records_by_id[sample_id] for sample_id in support_ids]
        query_records = [records_by_id[sample_id] for sample_id in query_ids]
        support_features = np.stack(
            [features_by_id[sample_id] for sample_id in support_ids],
            axis=0,
        )
        query_features = np.stack(
            [features_by_id[sample_id] for sample_id in query_ids],
            axis=0,
        )
        support_labels = np.asarray(
            [label_to_index[str(record["primary_organ"])] for record in support_records],
            dtype=np.int64,
        )
        query_labels = np.asarray(
            [label_to_index[str(record["primary_organ"])] for record in query_records],
            dtype=np.int64,
        )

        scaler = StandardScaler()
        support_scaled = scaler.fit_transform(support_features)
        query_scaled = scaler.transform(query_features)
        seed = int(plan["evaluation_seed"])
        classifier = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=seed,
        )
        classifier.fit(support_scaled, support_labels)
        probe_predictions = classifier.predict(query_scaled)
        centroid_predictions = _predict_nearest_centroid(
            support_scaled,
            support_labels,
            query_scaled,
        )

        for organ in sorted(supported_organs):
            probe = _seed_prediction_summary(
                records=query_records,
                labels=query_labels,
                predictions=probe_predictions,
                target_organ=organ,
            )
            centroid = _seed_prediction_summary(
                records=query_records,
                labels=query_labels,
                predictions=centroid_predictions,
                target_organ=organ,
            )
            target_count = sum(
                str(record["primary_organ"]) == organ for record in query_records
            )
            per_organ_seed_results[organ].append(
                {
                    "seed": seed,
                    "support_per_modality": int(plan["support_per_modality"]),
                    "query_per_modality": int(plan["query_per_modality"]),
                    "n_support_samples": len(support_records),
                    "n_query_samples": len(query_records),
                    "target_query_samples": int(target_count),
                    "reference_query_samples": int(len(query_records) - target_count),
                    **probe,
                    "nearest_centroid_balanced_accuracy": centroid[
                        "balanced_accuracy"
                    ],
                    "nearest_centroid_target_query_accuracy": centroid[
                        "target_query_accuracy"
                    ],
                    "nearest_centroid_reference_organs_accuracy": centroid[
                        "reference_organs_accuracy"
                    ],
                    "nearest_centroid_target_vs_reference_gap": centroid[
                        "target_vs_reference_gap"
                    ],
                    "nearest_centroid_transfer_efficiency": centroid[
                        "transfer_efficiency"
                    ],
                    "nearest_centroid_per_organ_query_accuracy": centroid[
                        "per_organ_query_accuracy"
                    ],
                    "nearest_centroid_target_query_accuracy_by_modality": centroid[
                        "target_query_accuracy_by_modality"
                    ],
                    "adaptation_gain_balanced_accuracy": float(
                        probe["balanced_accuracy"] - centroid["balanced_accuracy"]
                    ),
                    "adaptation_gain_target_query_accuracy": float(
                        probe["target_query_accuracy"]
                        - centroid["target_query_accuracy"]
                    ),
                    "relative_recovery_balanced_accuracy": _relative_recovery(
                        probe["balanced_accuracy"],
                        centroid["balanced_accuracy"],
                    ),
                    "relative_recovery_target_query_accuracy": _relative_recovery(
                        probe["target_query_accuracy"],
                        centroid["target_query_accuracy"],
                    ),
                }
            )

    per_organ: dict[str, dict[str, Any]] = {}
    metric_keys = (
        "balanced_accuracy",
        "target_query_accuracy",
        "reference_organs_accuracy",
        "target_vs_reference_gap",
        "transfer_efficiency",
        "nearest_centroid_balanced_accuracy",
        "nearest_centroid_target_query_accuracy",
        "adaptation_gain_balanced_accuracy",
        "adaptation_gain_target_query_accuracy",
        "relative_recovery_balanced_accuracy",
        "relative_recovery_target_query_accuracy",
    )
    for organ, seed_results in sorted(per_organ_seed_results.items()):
        payload: dict[str, Any] = {
            "status": "ok",
            "fold_context": {
                "low_shot_target_organ": organ,
                "other_organs_in_multiclass_adaptation": [
                    candidate for candidate in sorted(supported_organs)
                    if candidate != organ
                ],
                "fold_policy": "one_target_organ_per_low_shot_fold",
                "surface_scope": "cross_modality",
                "support_query_split_policy": "case_disjoint_support_query",
                "surface_b_protocol": (
                    "fixed_budget_low_shot_target_transfer_case_disjoint"
                ),
            },
            "seed_results": seed_results,
        }
        for key in metric_keys:
            values = [result.get(key) for result in seed_results]
            payload[key] = _mean(values)
            if key in {
                "balanced_accuracy",
                "target_query_accuracy",
                "nearest_centroid_balanced_accuracy",
                "nearest_centroid_target_query_accuracy",
                "adaptation_gain_balanced_accuracy",
                "adaptation_gain_target_query_accuracy",
                "relative_recovery_balanced_accuracy",
            }:
                payload[f"{key}_std"] = _std(values)
        per_organ[organ] = payload

    macro_seed_results: list[dict[str, Any]] = []
    for plan in plans:
        seed = int(plan["evaluation_seed"])
        rows = [
            result
            for organ_payload in per_organ.values()
            for result in organ_payload["seed_results"]
            if int(result["seed"]) == seed
        ]
        macro_seed_results.append(
            {
                "seed": seed,
                "balanced_accuracy": _mean(
                    [row["balanced_accuracy"] for row in rows]
                ),
                "target_query_accuracy": _mean(
                    [row["target_query_accuracy"] for row in rows]
                ),
                "nearest_centroid_balanced_accuracy": _mean(
                    [row["nearest_centroid_balanced_accuracy"] for row in rows]
                ),
                "nearest_centroid_target_query_accuracy": _mean(
                    [row["nearest_centroid_target_query_accuracy"] for row in rows]
                ),
                "adaptation_gain_balanced_accuracy": _mean(
                    [row["adaptation_gain_balanced_accuracy"] for row in rows]
                ),
                "adaptation_gain_target_query_accuracy": _mean(
                    [row["adaptation_gain_target_query_accuracy"] for row in rows]
                ),
                "relative_recovery_balanced_accuracy": _mean(
                    [row["relative_recovery_balanced_accuracy"] for row in rows]
                ),
                "n_evaluated_organs": len(per_organ),
            }
        )

    macro_keys = (
        "balanced_accuracy",
        "target_query_accuracy",
        "nearest_centroid_balanced_accuracy",
        "nearest_centroid_target_query_accuracy",
        "adaptation_gain_balanced_accuracy",
        "adaptation_gain_target_query_accuracy",
        "relative_recovery_balanced_accuracy",
    )
    return {
        "status": "ok",
        "surface_name": "surface_b_few_shot_transfer_evidence",
        "surface_scope": "cross_modality",
        "target_definition": "low_shot_target_organ",
        "target_definition_note": (
            "The target organ is represented in the fixed support set; metrics are "
            "evaluated on query examples from source cases disjoint from all support cases."
        ),
        "support_query_split_policy": "case_disjoint_support_query",
        "transfer_protocol": "fixed_budget_low_shot_target_transfer_case_disjoint",
        "required_modalities": required_modalities,
        "support_per_modality": int(plans[0]["support_per_modality"]),
        "query_per_modality": int(plans[0]["query_per_modality"]),
        "evaluation_seeds": [int(plan["evaluation_seed"]) for plan in plans],
        "n_supported_organs": len(supported_organs),
        "n_evaluated_organs": len(per_organ),
        "feature_projection": None,
        "macro_mean": {
            key: _mean([payload[key] for payload in per_organ.values()])
            for key in macro_keys
        },
        "macro_mean_std": {
            key: _std([row[key] for row in macro_seed_results])
            for key in macro_keys
        },
        "macro_seed_results": macro_seed_results,
        "per_organ": per_organ,
    }
