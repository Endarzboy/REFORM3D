"""Feature-independent, reproducible cohorts for cross-modal evaluation."""

from collections import defaultdict

import numpy as np

from reform3d.core.reproducibility import canonical_records

from .correspondence import CorrespondenceConfig, build_population_cohorts


def canonical_samples(samples):
    return canonical_records(samples, "samples")


def sample_identity(samples):
    return [
        {"sample_id": row["sample_id"], "patient_id": row.get("patient_id"),
         "modality": row.get("modality"), "primary_organ": row.get("primary_organ"),
         "organs": sorted(row.get("organs") or [])}
        for row in canonical_samples(samples)
    ]


def eligible_samples(samples, supported_organs=None):
    organs = None if supported_organs is None else set(supported_organs)
    return [row for row in canonical_samples(samples)
            if row.get("primary_organ") and (organs is None or row["primary_organ"] in organs)]


def require_features(features, samples):
    missing = sorted(row["sample_id"] for row in samples if row["sample_id"] not in features)
    if missing:
        raise ValueError(f"Missing features for {len(missing)} planned samples: {missing[:5]}")
    shapes = set()
    for row in samples:
        vector = np.asarray(features[row["sample_id"]])
        if (vector.ndim != 1 or vector.size == 0 or vector.dtype.kind not in "fiu"
                or not np.isfinite(vector).all()):
            raise ValueError(f"Invalid features for sample {row['sample_id']}")
        shapes.add(vector.shape)
    if len(shapes) > 1:
        raise ValueError("Cross-modal feature dimensions differ between samples")


def _sample(ids, count, rng):
    ids = sorted(ids)
    if count < len(ids):
        ids = [ids[index] for index in rng.choice(len(ids), size=count, replace=False)]
    return sorted(ids)


def margin_selection(samples, max_samples_per_pool, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    queries = []
    for row in samples:
        positive = [other["sample_id"] for other in samples
                    if other["primary_organ"] == row["primary_organ"] and other["modality"] != row["modality"]]
        negative = [other["sample_id"] for other in samples
                    if other["primary_organ"] != row["primary_organ"] and other["modality"] == row["modality"]]
        count = min(len(positive), len(negative))
        if max_samples_per_pool is not None:
            count = min(count, max_samples_per_pool)
        if count > 0:
            queries.append({"sample_id": row["sample_id"],
                            "positive_ids": _sample(positive, count, rng),
                            "negative_ids": _sample(negative, count, rng)})
    return {"queries": queries, "n_skipped_queries": len(samples) - len(queries)}


def retrieval_selection(samples, query_modality, target_modality, max_queries_per_organ, max_targets_per_organ, seed):
    queries, targets = defaultdict(list), defaultdict(list)
    for row in samples:
        if row["modality"] == query_modality:
            queries[row["primary_organ"]].append(row["sample_id"])
        if row["modality"] == target_modality:
            targets[row["primary_organ"]].append(row["sample_id"])
    organs = sorted(queries.keys() & targets.keys())
    nq = min((len(queries[organ]) for organ in organs), default=0)
    nt = min((len(targets[organ]) for organ in organs), default=0)
    if max_queries_per_organ is not None:
        nq = min(nq, max_queries_per_organ)
    if max_targets_per_organ is not None:
        nt = min(nt, max_targets_per_organ)
    rng = np.random.Generator(np.random.PCG64(seed))
    return {"organs": organs, "queries_per_organ": nq, "targets_per_organ": nt,
            "query_ids": sorted(sid for organ in organs for sid in _sample(queries[organ], nq, rng)),
            "target_ids": sorted(sid for organ in organs for sid in _sample(targets[organ], nt, rng))}


def lisi_selection(samples, required_modalities, max_samples_per_group, seed):
    groups = defaultdict(lambda: defaultdict(list))
    for row in samples:
        groups[row["primary_organ"]][row["modality"]].append(row["sample_id"])
    organs = sorted(organ for organ, groups_by_modality in groups.items()
                    if all(groups_by_modality.get(modality) for modality in required_modalities))
    count = min((len(groups[organ][modality]) for organ in organs for modality in required_modalities), default=0)
    if max_samples_per_group is not None:
        count = min(count, max_samples_per_group)
    rng = np.random.Generator(np.random.PCG64(seed))
    return {"organs": organs, "samples_per_group": count,
            "sample_ids": sorted(sid for organ in organs for modality in required_modalities
                                 for sid in _sample(groups[organ][modality], count, rng))}


def build_evaluation_plan(samples, *, seed=42, required_modalities=("ct", "mr"),
                          min_samples_per_modality=5, max_samples_per_pool=None,
                          max_queries_per_organ=None, max_targets_per_organ=None,
                          bootstrap_resamples=1000, max_lisi_samples_per_group=None,
                          lisi_perplexity=30.0, lisi_k_neighbors=None, top_ks=(1, 5)):
    """Choose every metric cohort before loading encoder features."""
    samples = canonical_samples(samples)
    modalities = sorted(set(str(value).lower() for value in required_modalities))
    if len(modalities) < 2:
        raise ValueError("Cross-modal evaluation requires at least two modalities")
    if seed < 0 or min_samples_per_modality < 1 or bootstrap_resamples < 0:
        raise ValueError("Invalid seed, cohort support, or bootstrap count")
    for cap in (max_samples_per_pool, max_queries_per_organ, max_targets_per_organ,
                max_lisi_samples_per_group, lisi_k_neighbors):
        if cap is not None and cap < 1:
            raise ValueError("Cross-modal sampling caps must be positive")
    if not np.isfinite(lisi_perplexity) or lisi_perplexity < 1:
        raise ValueError("LISI perplexity must be finite and at least one")
    cohorts = build_population_cohorts(samples, CorrespondenceConfig(tuple(modalities), min_samples_per_modality))
    organs = sorted(cohorts["supported_organs"])
    eligible = eligible_samples(samples, organs)
    protocol = {"required_modalities": modalities, "min_samples_per_modality": min_samples_per_modality,
                "max_samples_per_pool": max_samples_per_pool, "max_queries_per_organ": max_queries_per_organ,
                "max_targets_per_organ": max_targets_per_organ, "bootstrap_resamples": bootstrap_resamples,
                "max_lisi_samples_per_group": max_lisi_samples_per_group, "lisi_perplexity": lisi_perplexity,
                "lisi_k_neighbors": lisi_k_neighbors, "top_ks": sorted(set(top_ks))}
    return {"schema_version": 1, "module": "cross_modal", "seed": int(seed), "protocol": protocol,
            "samples": sample_identity(samples), "supported_organs": organs,
            "margin": margin_selection(eligible, max_samples_per_pool, seed),
            "retrieval": {
                "ct_to_mr": retrieval_selection(eligible, "ct", "mr", max_queries_per_organ, max_targets_per_organ, seed),
                "mr_to_ct": retrieval_selection(eligible, "mr", "ct", max_queries_per_organ, max_targets_per_organ, seed + 1)},
            "lisi": lisi_selection(eligible, modalities, max_lisi_samples_per_group, seed),
            "bootstrap": {"generator": "PCG64", "query_order": "sample_id", "margin_seed": seed + 1,
                          "ct_to_mr_seed": seed, "mr_to_ct_seed": seed + 1, "bidirectional_seed": seed + 1009,
                          "unit": "query", "bidirectional_unit": "direction_balanced_query"},
            "tie_breaking": "sample_id_ascending; LISI excludes self by sample ID"}


def validate_evaluation_plan(plan, samples, *, seed, supported_organs=None, **protocol):
    """Reject changed populations or selections instead of silently resampling."""
    if plan.get("schema_version") != 1 or plan.get("module") != "cross_modal":
        raise ValueError("Unsupported cross-modal evaluation plan")
    if plan.get("seed") != seed or plan.get("samples") != sample_identity(samples):
        raise ValueError("Evaluation plan seed or sample identities do not match")
    if supported_organs is not None and sorted(supported_organs) != plan.get("supported_organs"):
        raise ValueError("Evaluation plan supported organs do not match")
    for key, value in protocol.items():
        if key in ("required_modalities", "top_ks"):
            value = sorted(set(value))
        if plan.get("protocol", {}).get(key) != value:
            raise ValueError(f"Evaluation plan protocol differs: {key}")
    try:
        expected = build_evaluation_plan(samples, seed=seed, **plan["protocol"])
    except (KeyError, TypeError) as exc:
        raise ValueError("Invalid cross-modal evaluation plan protocol") from exc
    if plan != expected:
        raise ValueError("Cross-modal evaluation plan selections do not match its protocol")
    return plan
