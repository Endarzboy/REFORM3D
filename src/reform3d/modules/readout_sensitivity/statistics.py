"""Paired statistics for the readout-sensitivity analysis.

Two resampling designs are supported:

* Semantic transfer uses a hierarchical bootstrap over seeds and matched directed
  spacing-transfer units (6 directed transfers per seed).
* Low-shot transfer uses a crossed bootstrap over seeds and target organs.

All paired comparisons (REL-ISO, SA-ISO, SA-REL) are reported with an unadjusted
95% interval and a Bonferroni-adjusted interval across the family of three. An
interval containing zero is never interpreted as evidence of equivalence.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from . import config

CHECKPOINT_ORDER = ("med3dino_iso_96", "med3dino_rel_96", "med3dino_sa_96")
_LABEL = config.CHECKPOINT_LABELS


def _percentile_bounds(alpha: float) -> Tuple[float, float]:
    return 100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)


def _summary_from_bootstrap(
    point: float,
    boot: np.ndarray,
    *,
    family_size: int,
    confidence_level: float,
) -> Dict[str, Any]:
    alpha = 1.0 - confidence_level
    alpha_adj = alpha / float(family_size)
    lo, hi = _percentile_bounds(alpha)
    lo_adj, hi_adj = _percentile_bounds(alpha_adj)
    ci = [float(np.percentile(boot, lo)), float(np.percentile(boot, hi))]
    ci_adj = [float(np.percentile(boot, lo_adj)), float(np.percentile(boot, hi_adj))]
    return {
        "effect": float(point),
        "ci_95": ci,
        "ci_bonferroni": ci_adj,
        "bonferroni_family_size": int(family_size),
        "confidence_level": float(confidence_level),
        "excludes_zero_95": bool(ci[0] > 0.0 or ci[1] < 0.0),
        "excludes_zero_bonferroni": bool(ci_adj[0] > 0.0 or ci_adj[1] < 0.0),
        "direction": "positive" if point > 0 else ("negative" if point < 0 else "zero"),
        "n_bootstrap": int(boot.size),
    }


def hierarchical_paired_bootstrap(
    diffs_by_seed: Mapping[int, np.ndarray],
    *,
    resamples: int,
    seed: int,
    family_size: int = config.BONFERRONI_FAMILY_SIZE,
    confidence_level: float = config.CONFIDENCE_LEVEL,
) -> Dict[str, Any]:
    """Hierarchical bootstrap: resample seeds, then matched units within each seed.

    ``diffs_by_seed`` maps each evaluation seed to a 1-D array of matched paired
    differences (one per directed transfer). All seeds must carry the same number
    of units for the point estimate to weight seeds equally.
    """
    seeds = sorted(diffs_by_seed)
    matrix = np.stack([np.asarray(diffs_by_seed[s], dtype=np.float64) for s in seeds], axis=0)
    n_seeds, n_units = matrix.shape
    point = float(matrix.mean())

    rng = np.random.default_rng(int(seed))
    seed_idx = rng.integers(0, n_seeds, size=(int(resamples), n_seeds))
    unit_idx = rng.integers(0, n_units, size=(int(resamples), n_seeds, n_units))
    picked = matrix[seed_idx[:, :, None], unit_idx]  # [R, n_seeds, n_units]
    boot = picked.reshape(int(resamples), -1).mean(axis=1)
    summary = _summary_from_bootstrap(
        point, boot, family_size=family_size, confidence_level=confidence_level
    )
    summary.update({"design": "hierarchical_seed_then_matched_transfer", "n_seeds": n_seeds, "n_units_per_seed": n_units})
    return summary


def crossed_paired_bootstrap(
    diff_matrix: np.ndarray,
    *,
    resamples: int,
    seed: int,
    family_size: int = config.BONFERRONI_FAMILY_SIZE,
    confidence_level: float = config.CONFIDENCE_LEVEL,
) -> Dict[str, Any]:
    """Crossed bootstrap over seeds (rows) and target organs (columns).

    ``diff_matrix`` is a complete [n_seeds, n_organs] matrix of matched paired
    differences. Seeds and organs are resampled independently and the statistic
    is averaged over the crossed product.
    """
    matrix = np.asarray(diff_matrix, dtype=np.float64)
    n_seeds, n_organs = matrix.shape
    point = float(matrix.mean())

    rng = np.random.default_rng(int(seed))
    seed_idx = rng.integers(0, n_seeds, size=(int(resamples), n_seeds))
    organ_idx = rng.integers(0, n_organs, size=(int(resamples), n_organs))
    picked = matrix[seed_idx[:, :, None], organ_idx[:, None, :]]  # [R, n_seeds, n_organs]
    boot = picked.reshape(int(resamples), -1).mean(axis=1)
    summary = _summary_from_bootstrap(
        point, boot, family_size=family_size, confidence_level=confidence_level
    )
    summary.update({"design": "crossed_seed_by_organ", "n_seeds": n_seeds, "n_organs": n_organs})
    return summary


def checkpoint_ordering(mean_by_checkpoint: Mapping[str, float]) -> Dict[str, Any]:
    """Rank checkpoints (descending headline) and render an ISO/REL/SA ordering string."""
    present = [c for c in CHECKPOINT_ORDER if c in mean_by_checkpoint and mean_by_checkpoint[c] is not None]
    ranked = sorted(present, key=lambda c: mean_by_checkpoint[c], reverse=True)
    return {
        "ranking": [_LABEL[c] for c in ranked],
        "ordering_string": ">".join(_LABEL[c] for c in ranked),
        "iso_rel_sa_means": {_LABEL[c]: float(mean_by_checkpoint[c]) for c in present},
    }


def mean_std(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray([float(v) for v in values], dtype=np.float64)
    return {
        "mean": float(array.mean()) if array.size else None,
        "std": float(array.std(ddof=1)) if array.size >= 2 else None,
        "n": int(array.size),
        "per_seed": [float(v) for v in array],
    }


def paired_comparison_pairs() -> List[Tuple[str, str, str]]:
    """Return (label, high_checkpoint, low_checkpoint) for each paired comparison."""
    key = {"ISO": "med3dino_iso_96", "REL": "med3dino_rel_96", "SA": "med3dino_sa_96"}
    pairs = []
    for name in config.PAIRED_COMPARISONS:
        high, low = name.split("-")
        pairs.append((name, key[high], key[low]))
    return pairs


def ordering_agreement(
    ordering_by_readout: Mapping[str, Sequence[str]],
    reference_readout: str,
    comparison_readout: str,
) -> bool:
    """Whether two readouts produce the identical ISO/REL/SA ranking."""
    return list(ordering_by_readout.get(reference_readout, [])) == list(
        ordering_by_readout.get(comparison_readout, [])
    )


def direction_agreement(effects_by_readout: Mapping[str, float]) -> Dict[str, Any]:
    """Whether the SA-REL effect keeps the same sign across all readouts."""
    signs = {
        readout: (1 if effect > 0 else (-1 if effect < 0 else 0))
        for readout, effect in effects_by_readout.items()
    }
    unique_signs = set(signs.values())
    return {
        "signs": signs,
        "all_agree": len(unique_signs) == 1,
        "sign": next(iter(unique_signs)) if len(unique_signs) == 1 else None,
    }
