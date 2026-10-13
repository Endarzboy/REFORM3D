"""Build organ/modality support cohorts for Cross-Modal Alignment.

The cohort builder groups manifest samples by organ and modality, retains organs
with enough CT and MR support, and records the dropped organs with reasons.

Inputs:
    samples: normalized Cross-Modal manifest records.
    CorrespondenceConfig: required modalities and minimum per-modality support.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from .config import DEFAULT_REQUIRED_MODALITIES


@dataclass(frozen=True)
class CorrespondenceConfig:
    required_modalities: Tuple[str, ...] = DEFAULT_REQUIRED_MODALITIES
    min_samples_per_modality: int = 5


def build_population_cohorts(
    samples: Sequence[Dict[str, Any]],
    config: CorrespondenceConfig | None = None,
) -> Dict[str, Any]:
    config = config or CorrespondenceConfig()
    required_modalities = tuple(modality.lower() for modality in config.required_modalities)

    organ_to_modality_samples: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for sample in samples:
        for organ in sample.get("organs") or []:
            organ_bucket = organ_to_modality_samples.setdefault(organ, {})
            organ_bucket.setdefault(sample["modality"], []).append(sample)

    supported_organs: Dict[str, Any] = {}
    dropped_organs: Dict[str, Any] = {}
    for organ, modality_map in sorted(organ_to_modality_samples.items()):
        counts = {modality: len(modality_map.get(modality, [])) for modality in required_modalities}
        missing_or_underpowered = [
            modality
            for modality, count in counts.items()
            if count < config.min_samples_per_modality
        ]

        if missing_or_underpowered:
            dropped_organs[organ] = {
                "counts": counts,
                "reason": "insufficient_cross_modal_support",
                "underpowered_modalities": missing_or_underpowered,
            }
            continue

        supported_organs[organ] = {
            "organ": organ,
            "counts": counts,
            "sample_ids": {
                modality: [sample["sample_id"] for sample in modality_map.get(modality, [])]
                for modality in required_modalities
            },
        }

    return {
        "required_modalities": list(required_modalities),
        "min_samples_per_modality": config.min_samples_per_modality,
        "supported_organs": supported_organs,
        "dropped_organs": dropped_organs,
        "n_supported_organs": len(supported_organs),
        "n_dropped_organs": len(dropped_organs),
    }
