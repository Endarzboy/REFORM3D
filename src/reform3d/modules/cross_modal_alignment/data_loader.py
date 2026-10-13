"""Load, normalize, validate, and summarize Cross-Modal Alignment manifests.

This module standardizes organ-level samples into the minimal cohort format used
by the Cross-Modal Anatomical Alignment evaluation pipeline.

Usage example:
    from reform3d.modules.cross_modal_alignment.data_loader import (
        load_cross_modal_manifest,
        validate_cross_modal_manifest,
    )

    samples = load_cross_modal_manifest("amos22_ct_mr/core/manifest_sampled.json")
    report = validate_cross_modal_manifest(samples)
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .config import DEFAULT_REQUIRED_MODALITIES
from .path_resolution import normalize_totalsegmenter_ct_record


ORGAN_KEY_CANDIDATES: Tuple[str, ...] = (
    "organs",
    "organ_labels",
    "shared_organs",
    "anatomy_labels",
)


def _normalize_modality(value: Any) -> str:
    if value is None:
        return "unknown"
    return str(value).strip().lower()


def _normalize_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = value.strip().lower()
        return [normalized] if normalized else []
    if isinstance(value, Sequence):
        items: List[str] = []
        for item in value:
            if item is None:
                continue
            normalized = str(item).strip().lower()
            if normalized:
                items.append(normalized)
        return sorted(set(items))
    return []


def _extract_organs(sample: Dict[str, Any]) -> List[str]:
    for key in ORGAN_KEY_CANDIDATES:
        if key in sample:
            organs = _normalize_string_list(sample.get(key))
            if organs:
                return organs

    primary_organ = sample.get("primary_organ") or sample.get("organ")
    return _normalize_string_list(primary_organ)


def _extract_primary_organ(sample: Dict[str, Any], organs: List[str]) -> str | None:
    explicit = sample.get("primary_organ")
    if explicit is not None:
        normalized = _normalize_string_list(explicit)
        return normalized[0] if normalized else None
    if len(organs) == 1:
        return organs[0]
    return None


def load_cross_modal_manifest(manifest_path: str | Path) -> List[Dict[str, Any]]:
    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        raw_samples = payload.get("samples") or payload.get("volumes") or []
    else:
        raw_samples = payload

    normalized_samples: List[Dict[str, Any]] = []
    for index, sample in enumerate(raw_samples):
        if not isinstance(sample, dict):
            continue

        sample = normalize_totalsegmenter_ct_record(sample)
        sample_id = sample.get("sample_id") or sample.get("id") or sample.get("file_path") or f"sample_{index:06d}"
        organs = _extract_organs(sample)
        normalized_samples.append(
            {
                "sample_id": str(sample_id),
                "file_path": sample.get("file_path"),
                "dataset": sample.get("dataset"),
                "modality": _normalize_modality(sample.get("modality")),
                "organs": organs,
                "primary_organ": _extract_primary_organ(sample, organs),
                "patient_id": sample.get("patient_id") or sample.get("subject_id"),
                "study_id": sample.get("study_id") or sample.get("exam_id"),
                "split": sample.get("split"),
                "raw_sample": sample,
            }
        )

    return normalized_samples


def validate_cross_modal_manifest(
    samples: Sequence[Dict[str, Any]],
    required_modalities: Sequence[str] = DEFAULT_REQUIRED_MODALITIES,
) -> Dict[str, Any]:
    required = tuple(str(modality).lower() for modality in required_modalities)
    modality_counter = Counter(sample.get("modality", "unknown") for sample in samples)
    missing_modalities = [modality for modality in required if modality_counter.get(modality, 0) == 0]

    missing_organs = [sample["sample_id"] for sample in samples if not sample.get("organs")]
    missing_paths = [sample["sample_id"] for sample in samples if not sample.get("file_path")]

    return {
        "n_samples": len(samples),
        "required_modalities": list(required),
        "available_modalities": dict(sorted(modality_counter.items())),
        "missing_modalities": missing_modalities,
        "samples_missing_organs": missing_organs,
        "samples_missing_file_path": missing_paths,
        "is_valid": not missing_modalities and not missing_organs,
    }


def summarize_cross_modal_manifest(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    modality_counter = Counter(sample.get("modality", "unknown") for sample in samples)
    split_counter = Counter(sample.get("split") or "unspecified" for sample in samples)
    primary_organ_counter = Counter(sample.get("primary_organ") or "unassigned" for sample in samples)
    organ_counter = Counter()
    multi_organ_count = 0

    for sample in samples:
        organs = sample.get("organs") or []
        if len(organs) > 1:
            multi_organ_count += 1
        organ_counter.update(organs)

    return {
        "n_samples": len(samples),
        "modalities": dict(sorted(modality_counter.items())),
        "splits": dict(sorted(split_counter.items())),
        "primary_organs": dict(sorted(primary_organ_counter.items())),
        "organs": dict(sorted(organ_counter.items())),
        "n_multi_organ_samples": multi_organ_count,
    }
