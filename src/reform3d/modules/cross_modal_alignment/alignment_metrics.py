"""Metric helpers for Cross-Modal Alignment.

The functions in this module operate on normalized organ embeddings and
manifest samples to compute geometry and alignment diagnostics such as CKA,
LISI, silhouette scores, anatomy-over-modality margins, and centroid summaries.

Inputs:
    features: two-dimensional embedding arrays or sample-id keyed feature maps.
    samples: normalized Cross-Modal manifest records.
    supported_organs: organs retained by the cohort builder.
"""

from collections import defaultdict
from typing import Any, Dict, Sequence

import numpy as np
from tqdm import tqdm

from .sampling import (eligible_samples, require_features, margin_selection,
                       lisi_selection, validate_evaluation_plan)

try:
    from sklearn.metrics import silhouette_score
except ImportError:  # pragma: no cover
    silhouette_score = None


def _as_2d_array(features: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape {array.shape}")
    return array


def _l2_normalize_vector(vector: np.ndarray | Sequence[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"Expected a 1D feature vector, got shape {array.shape}")
    norm = np.linalg.norm(array)
    if norm <= 0:
        return array
    return array / norm


def _summary_stats(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "n_samples": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "n_samples": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _entropy_probabilities(squared_distances: np.ndarray, beta: float) -> tuple[np.ndarray, float]:
    logits = -squared_distances * beta
    logits -= np.max(logits)
    weights = np.exp(logits)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        probs = np.full(squared_distances.shape, 1.0 / max(1, squared_distances.size), dtype=np.float64)
    else:
        probs = weights / weight_sum
    entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, None))))
    return probs, entropy


def _probabilities_with_perplexity(
    squared_distances: np.ndarray,
    perplexity: float,
    tolerance: float = 1e-5,
    max_iter: int = 64,
) -> np.ndarray:
    if squared_distances.size == 0:
        return np.empty(0, dtype=np.float64)

    target_entropy = float(np.log(max(1.0, perplexity)))
    beta = 1.0
    beta_min = None
    beta_max = None
    probs = np.full(squared_distances.shape, 1.0 / squared_distances.size, dtype=np.float64)

    for _ in range(max_iter):
        probs, entropy = _entropy_probabilities(squared_distances, beta)
        entropy_error = entropy - target_entropy
        if abs(entropy_error) <= tolerance:
            break

        if entropy_error > 0:
            beta_min = beta
            beta = 2.0 * beta if beta_max is None else 0.5 * (beta + beta_max)
        else:
            beta_max = beta
            beta = 0.5 * beta if beta_min is None else 0.5 * (beta + beta_min)

    return probs


def _inverse_simpson(label_values: Sequence[str], probabilities: np.ndarray) -> float:
    label_mass: Dict[str, float] = defaultdict(float)
    for label, probability in zip(label_values, probabilities):
        label_mass[str(label)] += float(probability)

    squared_mass = sum(mass * mass for mass in label_mass.values())
    if squared_mass <= 0:
        return 0.0
    return float(1.0 / squared_mass)


def linear_cka(features_a: np.ndarray | Sequence[Sequence[float]], features_b: np.ndarray | Sequence[Sequence[float]]) -> float:
    x = _as_2d_array(features_a)
    y = _as_2d_array(features_b)
    if x.shape[0] != y.shape[0]:
        raise ValueError("Linear CKA requires the same number of samples in both matrices")

    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)

    xxty = x.T @ y
    numerator = np.linalg.norm(xxty, ord="fro") ** 2
    denominator = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _safe_silhouette(features: np.ndarray, labels: Sequence[str]) -> Dict[str, Any]:
    if silhouette_score is None:
        return {"status": "unavailable", "score": None, "reason": "scikit-learn not installed"}

    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        return {"status": "skipped", "score": None, "reason": "need at least two label classes"}
    if len(unique_labels) >= len(labels):
        return {"status": "skipped", "score": None, "reason": "need repeated labels for silhouette"}

    try:
        score = silhouette_score(features, labels)
    except Exception as exc:  # pragma: no cover
        return {"status": "error", "score": None, "reason": str(exc)}

    return {"status": "ok", "score": float(score), "reason": None}


def compute_global_alignment_summary(
    features: np.ndarray | Sequence[Sequence[float]],
    organ_labels: Sequence[str],
    modality_labels: Sequence[str],
) -> Dict[str, Any]:
    feature_matrix = _as_2d_array(features)
    organ_result = _safe_silhouette(feature_matrix, organ_labels)
    modality_result = _safe_silhouette(feature_matrix, modality_labels)

    organ_score = organ_result.get("score")
    modality_score = modality_result.get("score")
    if organ_score is None or modality_score is None:
        dominance_ratio = None
        dominant_axis = None
    else:
        dominance_ratio = None if abs(modality_score) < 1e-12 else float(organ_score / modality_score)
        dominant_axis = "organ" if organ_score > modality_score else "modality"

    return {
        "n_samples": int(feature_matrix.shape[0]),
        "organ_silhouette": organ_result,
        "modality_silhouette": modality_result,
        "organ_over_modality_ratio": dominance_ratio,
        "dominant_axis": dominant_axis,
    }


def compute_cross_modal_cka_by_organ(
    features_by_id: Dict[str, np.ndarray],
    cohorts: Dict[str, Any],
    max_samples_per_modality: int | None = None,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    supported_organs = cohorts.get("supported_organs", {})
    required_modalities = cohorts.get("required_modalities", ["ct", "mr"])
    if len(required_modalities) != 2:
        raise ValueError("Cross-modal CKA helper expects exactly two required modalities")

    modality_a, modality_b = required_modalities
    for organ, cohort in tqdm(supported_organs.items(), desc="Cross-Modal CKA organs", unit="organ", dynamic_ncols=True, leave=False):
        sample_ids_a = [sample_id for sample_id in cohort["sample_ids"][modality_a] if sample_id in features_by_id]
        sample_ids_b = [sample_id for sample_id in cohort["sample_ids"][modality_b] if sample_id in features_by_id]
        shared_count = min(len(sample_ids_a), len(sample_ids_b))
        if max_samples_per_modality is not None:
            shared_count = min(shared_count, max_samples_per_modality)

        if shared_count < 2:
            results[organ] = {
                "status": "skipped",
                "reason": "insufficient_features_for_balanced_cka",
                modality_a: len(sample_ids_a),
                modality_b: len(sample_ids_b),
            }
            continue

        features_a = np.stack([features_by_id[sample_id] for sample_id in sample_ids_a[:shared_count]], axis=0)
        features_b = np.stack([features_by_id[sample_id] for sample_id in sample_ids_b[:shared_count]], axis=0)
        results[organ] = {
            "status": "ok",
            modality_a: int(features_a.shape[0]),
            modality_b: int(features_b.shape[0]),
            "cross_modal_cka": linear_cka(features_a, features_b),
        }

    return results


def compute_anatomy_over_modality_margin(
    features_by_id: Dict[str, np.ndarray],
    samples: Sequence[Dict[str, Any]],
    supported_organs: Sequence[str] | None = None,
    max_samples_per_pool: int | None = None,
    bootstrap_resamples: int = 1000,
    seed: int = 42,
    evaluation_plan: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if evaluation_plan is not None:
        validate_evaluation_plan(evaluation_plan, samples, seed=seed, supported_organs=supported_organs,
                                 max_samples_per_pool=max_samples_per_pool, bootstrap_resamples=bootstrap_resamples)
        supported_organs = evaluation_plan["supported_organs"]
    records = eligible_samples(samples, supported_organs)
    require_features(features_by_id, records)
    if not records:
        return {
            "status": "skipped",
            "reason": "no_samples_with_primary_organs_and_features",
        }

    normalized_features = {
        sample_id: _l2_normalize_vector(feature)
        for sample_id, feature in features_by_id.items()
    }

    selection = (evaluation_plan["margin"] if evaluation_plan is not None
                 else margin_selection(records, max_samples_per_pool, seed))
    by_id = {record["sample_id"]: record for record in records}
    query_results: list[Dict[str, Any]] = []
    per_organ_margins: Dict[str, list[float]] = defaultdict(list)
    skipped_queries = selection["n_skipped_queries"]
    for query in tqdm(selection["queries"], desc="Cross-Modal anatomy-margin queries", unit="query", dynamic_ncols=True, leave=False):
        sample_id = query["sample_id"]
        sample = by_id[sample_id]
        organ, modality = sample["primary_organ"], sample["modality"]
        selected_positive_ids = query["positive_ids"]
        selected_negative_ids = query["negative_ids"]
        sample_count = len(selected_positive_ids)

        query_feature = normalized_features[sample_id]
        positive_features = np.stack([normalized_features[candidate_id] for candidate_id in selected_positive_ids], axis=0)
        negative_features = np.stack([normalized_features[candidate_id] for candidate_id in selected_negative_ids], axis=0)

        positive_distance = float(np.mean(1.0 - positive_features @ query_feature))
        negative_distance = float(np.mean(1.0 - negative_features @ query_feature))
        margin = negative_distance - positive_distance

        query_results.append(
            {
                "sample_id": sample_id,
                "organ": organ,
                "modality": modality,
                "n_positive": sample_count,
                "n_negative": sample_count,
                "same_organ_opposite_modality_distance": positive_distance,
                "different_organ_same_modality_distance": negative_distance,
                "margin": margin,
            }
        )
        per_organ_margins[organ].append(margin)

    if not query_results:
        return {
            "status": "skipped",
            "reason": "no_queries_satisfied_metric_assumptions",
            "n_skipped_queries": skipped_queries,
        }

    margins = np.asarray([result["margin"] for result in query_results], dtype=np.float64)
    bootstrap_ci = None
    if bootstrap_resamples > 0 and len(margins) > 1:
        bootstrap_seed = evaluation_plan["bootstrap"]["margin_seed"] if evaluation_plan is not None else seed + 1
        bootstrap_rng = np.random.Generator(np.random.PCG64(bootstrap_seed))
        bootstrap_means = np.empty(bootstrap_resamples, dtype=np.float64)
        for index in tqdm(range(bootstrap_resamples), desc="Cross-Modal margin bootstrap", unit="resample", dynamic_ncols=True, leave=False):
            sample_indices = bootstrap_rng.integers(0, len(margins), size=len(margins))
            bootstrap_means[index] = float(np.mean(margins[sample_indices]))
        bootstrap_ci = {
            "lower": float(np.quantile(bootstrap_means, 0.025)),
            "upper": float(np.quantile(bootstrap_means, 0.975)),
            "n_resamples": int(bootstrap_resamples),
        }

    return {
        "status": "ok",
        "distance": "cosine_distance_on_l2_normalized_features",
        "n_queries": len(query_results),
        "n_skipped_queries": skipped_queries,
        "n_organs": len(per_organ_margins),
        "overall": {
            "mean_margin": float(np.mean(margins)),
            "median_margin": float(np.median(margins)),
            "fraction_positive_margin": float(np.mean(margins > 0)),
            "bootstrap_95ci": bootstrap_ci,
        },
        "per_organ": {
            organ: {
                "n_queries": len(organ_margins),
                "mean_margin": float(np.mean(organ_margins)),
                "median_margin": float(np.median(organ_margins)),
                "fraction_positive_margin": float(np.mean(np.asarray(organ_margins) > 0)),
            }
            for organ, organ_margins in sorted(per_organ_margins.items())
        },
    }


def compute_balanced_lisi(
    features_by_id: Dict[str, np.ndarray],
    samples: Sequence[Dict[str, Any]],
    supported_organs: Sequence[str] | None = None,
    required_modalities: Sequence[str] = ("ct", "mr"),
    max_samples_per_group: int | None = None,
    perplexity: float = 30.0,
    k_neighbors: int | None = None,
    seed: int = 42,
    evaluation_plan: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if evaluation_plan is not None:
        validate_evaluation_plan(evaluation_plan, samples, seed=seed, supported_organs=supported_organs,
                                 required_modalities=required_modalities,
                                 max_lisi_samples_per_group=max_samples_per_group,
                                 lisi_perplexity=perplexity, lisi_k_neighbors=k_neighbors)
        supported_organs = evaluation_plan["supported_organs"]
    required_modalities = tuple(sorted(set(required_modalities)))
    records = [row for row in eligible_samples(samples, supported_organs)
               if row["modality"] in required_modalities]
    require_features(features_by_id, records)
    selection = (evaluation_plan["lisi"] if evaluation_plan is not None
                 else lisi_selection(records, required_modalities, max_samples_per_group, seed))
    common_organs = selection["organs"]
    samples_per_group = selection["samples_per_group"]
    if len(common_organs) < 2:
        return {"status": "skipped", "reason": "need_at_least_two_organs_with_full_modality_support_for_lisi",
                "n_organs": len(common_organs)}
    by_id = {record["sample_id"]: record for record in records}
    balanced_records = [by_id[sample_id] for sample_id in selection["sample_ids"]]
    if len(balanced_records) < 3:
        return {
            "status": "skipped",
            "reason": "need_at_least_three_samples_for_lisi",
            "n_samples": len(balanced_records),
        }

    feature_matrix = np.stack(
        [_l2_normalize_vector(features_by_id[record["sample_id"]]) for record in balanced_records],
        axis=0,
    )
    organ_labels = np.asarray([record["primary_organ"] for record in balanced_records], dtype=object)
    modality_labels = np.asarray([record["modality"] for record in balanced_records], dtype=object)

    n_samples = int(feature_matrix.shape[0])
    effective_perplexity = float(min(max(1.0, perplexity), max(1, n_samples - 1)))
    default_neighbors = max(3, int(np.ceil(3.0 * effective_perplexity)))
    requested_neighbors = default_neighbors if k_neighbors is None else int(k_neighbors)
    effective_neighbors = min(max(1, requested_neighbors), n_samples - 1)
    effective_perplexity = float(min(effective_perplexity, max(1, effective_neighbors)))

    sample_ids = np.asarray([row["sample_id"] for row in balanced_records])

    modality_ilisi_values: list[float] = []
    organ_clisi_values: list[float] = []
    per_organ: Dict[str, Dict[str, list[float]]] = defaultdict(lambda: {"modality_ilisi": [], "organ_clisi": []})
    per_modality: Dict[str, Dict[str, list[float]]] = defaultdict(lambda: {"modality_ilisi": [], "organ_clisi": []})

    lisi_iterator = tqdm(balanced_records, desc="Cross-Modal balanced LISI", unit="sample", dynamic_ncols=True, leave=False)
    for sample_index, record in enumerate(lisi_iterator):
        # Exclude this sample explicitly: duplicate vectors need not rank self first.
        squared_distances = np.sum((feature_matrix - feature_matrix[sample_index]) ** 2, axis=1)
        candidate_indices = np.flatnonzero(np.arange(n_samples) != sample_index)
        order = np.lexsort((sample_ids[candidate_indices], squared_distances[candidate_indices]))
        neighbors = candidate_indices[order[:effective_neighbors]]
        probabilities = _probabilities_with_perplexity(
            squared_distances[neighbors], perplexity=effective_perplexity,
        )
        neighbor_organs = organ_labels[neighbors]
        neighbor_modalities = modality_labels[neighbors]

        modality_ilisi = _inverse_simpson(neighbor_modalities, probabilities)
        organ_clisi = _inverse_simpson(neighbor_organs, probabilities)
        modality_ilisi_values.append(modality_ilisi)
        organ_clisi_values.append(organ_clisi)
        per_organ[record["primary_organ"]]["modality_ilisi"].append(modality_ilisi)
        per_organ[record["primary_organ"]]["organ_clisi"].append(organ_clisi)
        per_modality[record["modality"]]["modality_ilisi"].append(modality_ilisi)
        per_modality[record["modality"]]["organ_clisi"].append(organ_clisi)

    return {
        "status": "ok",
        "distance": "euclidean_distance_on_l2_normalized_features",
        "required_modalities": list(required_modalities),
        "n_samples": n_samples,
        "n_organs": len(common_organs),
        "samples_per_group": int(samples_per_group),
        "perplexity": effective_perplexity,
        "k_neighbors": int(effective_neighbors),
        "overall": {
            "modality_ilisi": {
                **_summary_stats(modality_ilisi_values),
                "interpretation": "higher_is_better",
                "theoretical_min": 1.0,
                "theoretical_max": float(len(required_modalities)),
            },
            "organ_clisi": {
                **_summary_stats(organ_clisi_values),
                "interpretation": "lower_is_better",
                "theoretical_min": 1.0,
                "theoretical_max": float(len(common_organs)),
            },
        },
        "per_organ": {
            organ: {
                "modality_ilisi": _summary_stats(values["modality_ilisi"]),
                "organ_clisi": _summary_stats(values["organ_clisi"]),
            }
            for organ, values in sorted(per_organ.items())
        },
        "per_modality": {
            modality: {
                "modality_ilisi": _summary_stats(values["modality_ilisi"]),
                "organ_clisi": _summary_stats(values["organ_clisi"]),
            }
            for modality, values in sorted(per_modality.items())
        },
    }
