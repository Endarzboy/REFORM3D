"""Cross-Modal Alignment retrieval diagnostics."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Sequence


DEFAULT_CONTEXT_ABLATION_SWEEP = (
    {"name": "baseline_full_bbox_context", "bbox_margin": 5, "outside_organ_mask": "none"},
    {"name": "organ_only_bbox0", "bbox_margin": 0, "outside_organ_mask": "zero"},
)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def context_ablation_tag(bbox_margin: int, outside_organ_mask: str, default_margin: int = 5) -> str | None:
    mode = str(outside_organ_mask or "none").strip().lower()
    if int(bbox_margin) == int(default_margin) and mode == "none":
        return None
    return f"ctx_bbox{int(bbox_margin)}_mask{mode}"


def build_retrieval_pool_diagnostics(
    samples: Sequence[Dict[str, Any]],
    *,
    required_modalities: Sequence[str] = ("ct", "mr"),
    supported_organs: Sequence[str] | None = None,
    bbox_margin: int = 5,
    outside_organ_mask: str = "none",
) -> Dict[str, Any]:
    required = [str(modality).lower() for modality in required_modalities]
    supported = set(supported_organs or [])
    by_organ: dict[str, Counter] = defaultdict(Counter)
    for sample in samples:
        organ = sample.get("primary_organ")
        modality = str(sample.get("modality") or "unknown").lower()
        if not organ or modality not in required:
            continue
        if supported and organ not in supported:
            continue
        by_organ[str(organ)][modality] += 1

    organ_rows = []
    for organ, counts in sorted(by_organ.items()):
        complete = all(counts.get(modality, 0) > 0 for modality in required)
        organ_rows.append(
            {
                "organ": organ,
                "complete_required_modalities": complete,
                **{f"{modality}_samples": int(counts.get(modality, 0)) for modality in required},
                "min_required_modality_samples": min(int(counts.get(modality, 0)) for modality in required),
            }
        )

    complete_organs = [row for row in organ_rows if row["complete_required_modalities"]]
    return {
        "artifact": "retrieval_pool_diagnostics",
        "required_modalities": required,
        "n_organs_with_required_modalities": len(complete_organs),
        "organ_pool_counts": organ_rows,
        "balanced_pool_policy": {
            "query_pool": "sample min available records per organ for query modality",
            "target_pool": "sample min available records per organ for target modality",
            "chance_control": "organ_balanced_chance is computed from n_organs and targets_per_organ",
        },
        "context_controls": {
            "organ_crop_source": "segmentation-mask bounding box",
            "bbox_margin": int(bbox_margin),
            "outside_organ_mask": str(outside_organ_mask),
            "context_ablation_tag": context_ablation_tag(bbox_margin, outside_organ_mask),
        },
    }


def _metric_value(payload: Dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def build_context_ablation_summary(
    baseline_metrics: Dict[str, Any] | None,
    ablation_metrics: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    baseline = baseline_metrics or {}
    baseline_mean = _metric_value(baseline, "balanced_cross_modal_retrieval", "bidirectional_mean") or {}
    rows = []
    for metrics in ablation_metrics:
        retrieval = _metric_value(metrics, "balanced_cross_modal_retrieval", "bidirectional_mean") or {}
        tag = metrics.get("context_ablation_tag") or metrics.get("checkpoint_name")
        rows.append(
            {
                "context_ablation_tag": tag,
                "map": retrieval.get("map"),
                "top@1": retrieval.get("top@1"),
                "top@5": retrieval.get("top@5"),
                "delta_map_vs_baseline": (
                    None
                    if retrieval.get("map") is None or baseline_mean.get("map") is None
                    else float(retrieval["map"] - baseline_mean["map"])
                ),
                "delta_top@1_vs_baseline": (
                    None
                    if retrieval.get("top@1") is None or baseline_mean.get("top@1") is None
                    else float(retrieval["top@1"] - baseline_mean["top@1"])
                ),
            }
        )
    return {
        "artifact": "context_ablation_retrieval",
        "baseline": {
            "map": baseline_mean.get("map"),
            "top@1": baseline_mean.get("top@1"),
            "top@5": baseline_mean.get("top@5"),
        },
        "ablations": rows,
        "interpretation": "Large positive/negative deltas indicate retrieval may be sensitive to FOV/context outside the target organ.",
    }


def write_retrieval_pool_diagnostics(
    output_path: Path,
    samples: Sequence[Dict[str, Any]],
    *,
    required_modalities: Sequence[str],
    supported_organs: Sequence[str] | None,
    bbox_margin: int,
    outside_organ_mask: str,
) -> Dict[str, Any]:
    payload = build_retrieval_pool_diagnostics(
        samples,
        required_modalities=required_modalities,
        supported_organs=supported_organs,
        bbox_margin=bbox_margin,
        outside_organ_mask=outside_organ_mask,
    )
    _write_json(output_path, payload)
    return payload


def build_ct_pet_corpus_constraint(
    *,
    dataset_name: str,
    configured_sources: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "artifact": "ct_pet_alignment",
        "dataset": dataset_name,
        "status": "corpus_constraint",
        "reason": "No organ-level paired CT-PET manifest builder is currently available with shared CT/PET organ labels.",
        "configured_sources": configured_sources,
        "required_to_enable": [
            "paired or cohort-level CT and PET image records",
            "organ-level masks or labels shared across modalities",
            "case-disjoint IDs for leakage audit",
        ],
    }


def write_ct_pet_corpus_constraint(
    output_path: Path,
    *,
    dataset_name: str,
    configured_sources: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    payload = build_ct_pet_corpus_constraint(dataset_name=dataset_name, configured_sources=configured_sources)
    _write_json(output_path, payload)
    return payload


__all__ = [
    "DEFAULT_CONTEXT_ABLATION_SWEEP",
    "build_context_ablation_summary",
    "build_ct_pet_corpus_constraint",
    "build_retrieval_pool_diagnostics",
    "context_ablation_tag",
    "write_ct_pet_corpus_constraint",
    "write_retrieval_pool_diagnostics",
]
