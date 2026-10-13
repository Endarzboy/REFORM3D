"""Retrieval metrics for Cross-Modal organ embeddings.

The functions compute balanced CT-to-MR and MR-to-CT retrieval over retained
organs, including top-k accuracy, mean average precision, chance baselines, and
query-bootstrap confidence intervals.

Inputs:
    features_by_id: mapping from manifest sample IDs to embedding vectors.
    samples: normalized Cross-Modal manifest records.
    supported_organs: organs retained for balanced retrieval.
"""

from collections import defaultdict
from math import comb
from typing import Any, Dict, Iterable, Sequence

import numpy as np
from tqdm import tqdm


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return features / norms


def _sample_records(records: Sequence[Dict[str, Any]], sample_count: int, rng: np.random.Generator) -> list[Dict[str, Any]]:
    entries = list(records)
    if sample_count >= len(entries):
        return entries
    indices = rng.choice(len(entries), size=sample_count, replace=False)
    return [entries[index] for index in indices]


def _balanced_top_chance(n_organs: int, samples_per_organ: int, k: int) -> float | None:
    total_targets = n_organs * samples_per_organ
    if n_organs <= 1 or samples_per_organ <= 0 or total_targets <= 0:
        return None
    k = min(k, total_targets)
    if k <= 0:
        return None
    if total_targets - samples_per_organ < k:
        return 1.0
    missed = comb(total_targets - samples_per_organ, k) / comb(total_targets, k)
    return float(1.0 - missed)


def _average_precision(relevance: np.ndarray) -> float:
    positive_ranks = np.flatnonzero(relevance) + 1
    if positive_ranks.size == 0:
        return 0.0

    precision_at_hits = np.arange(1, positive_ranks.size + 1, dtype=np.float64) / positive_ranks
    return float(np.mean(precision_at_hits))


def _top_metric_key(k: int) -> str:
    return f"top@{k}"


def _bootstrap_query_metric_ci(
    query_results: Sequence[Dict[str, Any]],
    metric_keys: Sequence[str],
    bootstrap_resamples: int,
    seed: int,
) -> Dict[str, Any]:
    if bootstrap_resamples <= 0 or not query_results:
        return {}

    rng = np.random.default_rng(int(seed))
    metrics_by_key = {
        key: np.asarray([float(row[key]) for row in query_results if row.get(key) is not None], dtype=np.float64)
        for key in metric_keys
    }
    ci_payload: Dict[str, Any] = {}
    for key, values in metrics_by_key.items():
        if values.size == 0:
            continue
        bootstrap_means = np.empty(int(bootstrap_resamples), dtype=np.float64)
        for resample_index in tqdm(range(int(bootstrap_resamples)), desc=f"Retrieval bootstrap {key}", unit="resample", dynamic_ncols=True, leave=False):
            indices = rng.integers(0, values.size, size=values.size)
            bootstrap_means[resample_index] = float(np.mean(values[indices]))
        ci_payload[key] = {
            "mean": float(np.mean(values)),
            "ci_lower": float(np.percentile(bootstrap_means, 2.5)),
            "ci_upper": float(np.percentile(bootstrap_means, 97.5)),
            "n_queries": int(values.size),
            "bootstrap_resamples": int(bootstrap_resamples),
            "bootstrap_unit": "query",
        }
    return ci_payload


def _bootstrap_bidirectional_metric_ci(
    ct_to_mr_query_results: Sequence[Dict[str, Any]],
    mr_to_ct_query_results: Sequence[Dict[str, Any]],
    metric_keys: Sequence[str],
    bootstrap_resamples: int,
    seed: int,
) -> Dict[str, Any]:
    if bootstrap_resamples <= 0 or not ct_to_mr_query_results or not mr_to_ct_query_results:
        return {}

    rng = np.random.default_rng(int(seed))
    ci_payload: Dict[str, Any] = {}
    for key in metric_keys:
        ct_values = np.asarray(
            [float(row[key]) for row in ct_to_mr_query_results if row.get(key) is not None],
            dtype=np.float64,
        )
        mr_values = np.asarray(
            [float(row[key]) for row in mr_to_ct_query_results if row.get(key) is not None],
            dtype=np.float64,
        )
        if ct_values.size == 0 or mr_values.size == 0:
            continue
        bootstrap_means = np.empty(int(bootstrap_resamples), dtype=np.float64)
        for resample_index in tqdm(range(int(bootstrap_resamples)), desc=f"Bidirectional retrieval bootstrap {key}", unit="resample", dynamic_ncols=True, leave=False):
            ct_indices = rng.integers(0, ct_values.size, size=ct_values.size)
            mr_indices = rng.integers(0, mr_values.size, size=mr_values.size)
            bootstrap_means[resample_index] = float(
                (np.mean(ct_values[ct_indices]) + np.mean(mr_values[mr_indices])) / 2.0
            )
        ci_payload[key] = {
            "mean": float((np.mean(ct_values) + np.mean(mr_values)) / 2.0),
            "ci_lower": float(np.percentile(bootstrap_means, 2.5)),
            "ci_upper": float(np.percentile(bootstrap_means, 97.5)),
            "n_ct_to_mr_queries": int(ct_values.size),
            "n_mr_to_ct_queries": int(mr_values.size),
            "bootstrap_resamples": int(bootstrap_resamples),
            "bootstrap_unit": "direction_balanced_query",
        }
    return ci_payload


def compute_cross_modal_retrieval(
    features_by_id: Dict[str, np.ndarray],
    samples: Sequence[Dict[str, Any]],
    supported_organs: Sequence[str] | None = None,
    query_modality: str = "ct",
    target_modality: str = "mr",
    top_ks: Sequence[int] = (1, 5),
    max_queries_per_organ: int | None = None,
    max_targets_per_organ: int | None = None,
    seed: int = 42,
    return_query_results: bool = False,
    bootstrap_resamples: int = 0,
) -> Dict[str, Any]:
    supported_organ_set = set(supported_organs or [])
    query_records = [
        sample for sample in samples
        if sample.get("modality") == query_modality
        and sample.get("primary_organ")
        and sample.get("sample_id") in features_by_id
        and (not supported_organ_set or sample.get("primary_organ") in supported_organ_set)
    ]
    target_records = [
        sample for sample in samples
        if sample.get("modality") == target_modality
        and sample.get("primary_organ")
        and sample.get("sample_id") in features_by_id
        and (not supported_organ_set or sample.get("primary_organ") in supported_organ_set)
    ]

    if not query_records or not target_records:
        return {
            "status": "skipped",
            "reason": "insufficient_primary_organ_records_for_retrieval",
            "n_queries": len(query_records),
            "n_targets": len(target_records),
        }

    query_by_organ = defaultdict(list)
    target_by_organ = defaultdict(list)
    for sample in query_records:
        query_by_organ[sample["primary_organ"]].append(sample)
    for sample in target_records:
        target_by_organ[sample["primary_organ"]].append(sample)

    common_organs = sorted(set(query_by_organ) & set(target_by_organ))
    if len(common_organs) < 2:
        return {
            "status": "skipped",
            "reason": "need_at_least_two_organs_for_balanced_retrieval",
            "n_queries": len(query_records),
            "n_targets": len(target_records),
            "n_common_organs": len(common_organs),
        }

    queries_per_organ = min(len(query_by_organ[organ]) for organ in common_organs)
    targets_per_organ = min(len(target_by_organ[organ]) for organ in common_organs)
    if max_queries_per_organ is not None:
        queries_per_organ = min(queries_per_organ, max_queries_per_organ)
    if max_targets_per_organ is not None:
        targets_per_organ = min(targets_per_organ, max_targets_per_organ)
    if queries_per_organ <= 0 or targets_per_organ <= 0:
        return {
            "status": "skipped",
            "reason": "balanced_candidate_pool_could_not_be_formed",
            "n_common_organs": len(common_organs),
        }

    rng = np.random.default_rng(seed)
    balanced_queries = [
        query
        for organ in common_organs
        for query in _sample_records(query_by_organ[organ], queries_per_organ, rng)
    ]
    balanced_targets = [
        target
        for organ in common_organs
        for target in _sample_records(target_by_organ[organ], targets_per_organ, rng)
    ]

    query_features = _l2_normalize(np.stack([features_by_id[sample["sample_id"]] for sample in balanced_queries], axis=0))
    target_features = _l2_normalize(np.stack([features_by_id[sample["sample_id"]] for sample in balanced_targets], axis=0))
    similarity = query_features @ target_features.T

    top_ks = tuple(sorted(set(int(k) for k in top_ks if int(k) > 0)))
    top_counts = {k: 0 for k in top_ks}
    per_organ = defaultdict(lambda: {_top_metric_key(k): 0 for k in top_ks} | {"map_total": 0.0, "n_queries": 0})

    target_organs = np.asarray([sample["primary_organ"] for sample in balanced_targets], dtype=object)
    average_precision_total = 0.0
    query_results: list[Dict[str, Any]] = []
    retrieval_iterator = tqdm(
        balanced_queries,
        desc=f"Cross-Modal retrieval {query_modality}->{target_modality}",
        unit="query",
        dynamic_ncols=True,
        leave=False,
    )
    for query_index, query_record in enumerate(retrieval_iterator):
        organ = query_record["primary_organ"]
        ranking = np.argsort(-similarity[query_index])
        per_organ[organ]["n_queries"] += 1
        ranked_organs = target_organs[ranking]
        relevance = ranked_organs == organ
        average_precision = _average_precision(relevance)
        average_precision_total += average_precision
        per_organ[organ]["map_total"] += average_precision
        query_result = {
            "sample_id": str(query_record["sample_id"]),
            "organ": str(organ),
            "query_modality": str(query_modality),
            "target_modality": str(target_modality),
            "average_precision": average_precision,
            "map": average_precision,
            "n_positive_targets": int(np.sum(relevance)),
            "n_targets": int(len(balanced_targets)),
        }
        for k in top_ks:
            top = int(np.any(relevance[:k]))
            top_counts[k] += top
            per_organ[organ][_top_metric_key(k)] += top
            query_result[_top_metric_key(k)] = float(top)
        query_results.append(query_result)

    n_queries = len(balanced_queries)
    metric_keys = ["map", *[_top_metric_key(k) for k in top_ks]]
    result = {
        "status": "ok",
        "evaluation": "balanced_cross_modal_organ_retrieval",
        "query_modality": query_modality,
        "target_modality": target_modality,
        "n_queries": n_queries,
        "n_targets": len(balanced_targets),
        "n_organs": len(common_organs),
        "queries_per_organ": queries_per_organ,
        "targets_per_organ": targets_per_organ,
        "overall": {
            **{_top_metric_key(k): float(top_counts[k] / n_queries) for k in top_ks},
            "map": float(average_precision_total / n_queries),
        },
        "organ_balanced_chance": {
            _top_metric_key(k): _balanced_top_chance(len(common_organs), targets_per_organ, k)
            for k in top_ks
        },
        "per_organ": {
            organ: {
                "n_queries": values["n_queries"],
                **{
                    key: float(values[key] / values["n_queries"])
                    for key in values
                    if key.startswith("top@") and values["n_queries"] > 0
                },
                "map": float(values["map_total"] / values["n_queries"]) if values["n_queries"] > 0 else None,
            }
            for organ, values in sorted(per_organ.items())
        },
    }
    if bootstrap_resamples > 0:
        result["overall_bootstrap_95ci"] = _bootstrap_query_metric_ci(
            query_results,
            metric_keys=metric_keys,
            bootstrap_resamples=int(bootstrap_resamples),
            seed=int(seed),
        )
    if return_query_results:
        result["query_results"] = query_results
    return result


def compute_bidirectional_cross_modal_retrieval(
    features_by_id: Dict[str, np.ndarray],
    samples: Sequence[Dict[str, Any]],
    supported_organs: Sequence[str] | None = None,
    top_ks: Sequence[int] = (1, 5),
    max_queries_per_organ: int | None = None,
    max_targets_per_organ: int | None = None,
    seed: int = 42,
    return_query_results: bool = False,
    bootstrap_resamples: int = 0,
) -> Dict[str, Any]:
    include_query_results = bool(return_query_results or bootstrap_resamples > 0)
    ct_to_mr = compute_cross_modal_retrieval(
        features_by_id,
        samples,
        supported_organs=supported_organs,
        query_modality="ct",
        target_modality="mr",
        top_ks=top_ks,
        max_queries_per_organ=max_queries_per_organ,
        max_targets_per_organ=max_targets_per_organ,
        seed=seed,
        return_query_results=include_query_results,
        bootstrap_resamples=bootstrap_resamples,
    )
    mr_to_ct = compute_cross_modal_retrieval(
        features_by_id,
        samples,
        supported_organs=supported_organs,
        query_modality="mr",
        target_modality="ct",
        top_ks=top_ks,
        max_queries_per_organ=max_queries_per_organ,
        max_targets_per_organ=max_targets_per_organ,
        seed=seed + 1,
        return_query_results=include_query_results,
        bootstrap_resamples=bootstrap_resamples,
    )

    summary = None
    if ct_to_mr.get("status") == "ok" and mr_to_ct.get("status") == "ok":
        shared_metric_keys = sorted(set(ct_to_mr["overall"]) & set(mr_to_ct["overall"]))
        summary = {
            key: float((ct_to_mr["overall"][key] + mr_to_ct["overall"][key]) / 2.0)
            for key in shared_metric_keys
        }

    bidirectional_bootstrap = None
    if summary is not None and bootstrap_resamples > 0:
        bidirectional_bootstrap = _bootstrap_bidirectional_metric_ci(
            ct_to_mr.get("query_results", []),
            mr_to_ct.get("query_results", []),
            metric_keys=sorted(summary),
            bootstrap_resamples=int(bootstrap_resamples),
            seed=int(seed) + 1009,
        )

    if not return_query_results:
        ct_to_mr.pop("query_results", None)
        mr_to_ct.pop("query_results", None)

    result = {
        "status": "ok" if summary is not None else "partial",
        "ct_to_mr": ct_to_mr,
        "mr_to_ct": mr_to_ct,
        "bidirectional_mean": summary,
    }
    if bidirectional_bootstrap:
        result["bidirectional_bootstrap_95ci"] = bidirectional_bootstrap
    return result
