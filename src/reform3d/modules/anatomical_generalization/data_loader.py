"""Load and validate Anatomical Generalization manifests.

This module is the schema-normalization layer for Anatomical Generalization. It
accepts Cross-Modal-style organ manifests, resolves multiple organ-key aliases,
and converts them into the compact sample contract used by the evaluator.

Required inputs:
        - an Anatomical or Cross-Modal-style manifest JSON with a top-level list of samples
            or a dictionary containing ``samples`` or ``volumes``.
        - per-sample organ information through one of the supported organ keys.
        - per-sample modality labels such as ``ct`` and ``mr``.

Examples:
        >>> samples = load_anatomical_manifest("totalsegmenter_ct_mr_anchor/core/manifest_sampled.json")
        >>> validation = validate_anatomical_manifest(samples, required_modalities=("ct", "mr"), min_holdout_organs=2, min_samples_per_modality=5)
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .config import DEFAULT_MIN_HOLDOUT_ORGANS, DEFAULT_REQUIRED_MODALITIES


ORGAN_KEY_CANDIDATES: Tuple[str, ...] = (
    "organs",
    "organ_labels",
    "shared_organs",
    "anatomy_labels",
)

PRETRAINING_EXPOSURE_ALIASES = {
    "abdomenatlas": "pretraining_seen_source",
    "abdomenct1k": "pretraining_seen_source",
    "kits23": "pretraining_seen_source",
    "kits": "pretraining_seen_source",
    "totalsegmenter_ct": "pretraining_seen_source",
    "totalsegmenterct": "pretraining_seen_source",
    "totalsegmenter_mri": "pretraining_unseen_source",
    "totalsegmentermri": "pretraining_unseen_source",
    "mmwhs_ct_mr": "pretraining_unseen_source",
    "mmwhs": "pretraining_unseen_source",
    "amos22_ct_mr": "pretraining_unseen_source",
    "amos_ct_mr": "pretraining_unseen_source",
    "amos22": "pretraining_unseen_source",
    "chaos_ct_mr": "pretraining_unseen_source",
    "word": "pretraining_unseen_source",
}

ORGAN_FAMILY_ALIASES = {
    "aorta": "vascular",
    "ascending_aorta": "vascular",
    "inferior_vena_cava": "vascular",
    "pulmonary_artery": "vascular",
    "left_atrium": "cardiac",
    "left_ventricle": "cardiac",
    "myocardium": "cardiac",
    "right_atrium": "cardiac",
    "right_ventricle": "cardiac",
    "colon": "gastrointestinal",
    "duodenum": "gastrointestinal",
    "esophagus": "gastrointestinal",
    "small_bowel": "gastrointestinal",
    "stomach": "gastrointestinal",
    "adrenal_gland_left": "endocrine",
    "adrenal_gland_right": "endocrine",
    "gallbladder": "hepatobiliary",
    "liver": "hepatobiliary",
    "pancreas": "hepatobiliary",
    "kidney_left": "urinary",
    "kidney_right": "urinary",
    "urinary_bladder": "urinary",
    "spleen": "lymphatic",
}


def _normalize_modality(value: Any) -> str:
    if value is None:
        return "unknown"
    return str(value).strip().lower()


def _normalize_dataset_name(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized or None


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


def _infer_organ_family(organ_name: str | None) -> str:
    if organ_name is None:
        return "unknown"
    return ORGAN_FAMILY_ALIASES.get(str(organ_name).strip().lower(), "other")


def _extract_image_path(sample: Dict[str, Any]) -> Any:
    return sample.get("image_path") or sample.get("file_path")


def _extract_mask_path(sample: Dict[str, Any]) -> Any:
    return sample.get("mask_path") or sample.get("label_path")


def _extract_mask_metadata(sample: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "mask_path": _extract_mask_path(sample),
        "label_path": sample.get("label_path"),
        "label_voxels": sample.get("label_voxels"),
        "shape": sample.get("shape"),
        "spacing": sample.get("spacing"),
    }


def _infer_pretraining_exposure(sample: Dict[str, Any]) -> str:
    candidates = [
        sample.get("source_dataset"),
        sample.get("dataset"),
        sample.get("image_path"),
        sample.get("file_path"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        normalized = str(candidate).strip().lower()
        for alias, exposure in PRETRAINING_EXPOSURE_ALIASES.items():
            if alias in normalized:
                return exposure
    return "pretraining_exposure_unknown"


def load_anatomical_manifest(manifest_path: str | Path) -> List[Dict[str, Any]]:
    """Load one Anatomical Generalization manifest and normalize samples into the evaluator contract.

    Required parameter:
        manifest_path: path to a manifest JSON file.

    Returned sample fields include:
        - ``sample_id``
        - ``file_path``
        - ``modality``
        - ``organs``
        - ``primary_organ``
        - ``source_case_id``

    Example:
        >>> samples = load_anatomical_manifest("manifest_sampled.json")
    """
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

        sample_id = sample.get("sample_id") or sample.get("id") or sample.get("file_path") or f"sample_{index:06d}"
        organs = _extract_organs(sample)
        primary_organ = _extract_primary_organ(sample, organs)
        source_dataset = _normalize_dataset_name(sample.get("source_dataset") or sample.get("dataset"))
        image_path = _extract_image_path(sample)
        mask_path = _extract_mask_path(sample)
        normalized_samples.append(
            {
                "sample_id": str(sample_id),
                "patient_id": sample.get("patient_id") or sample.get("subject_id"),
                "file_path": sample.get("file_path"),
                "image_path": image_path,
                "dataset": sample.get("dataset"),
                "source_dataset": source_dataset,
                "modality": _normalize_modality(sample.get("modality")),
                "organs": organs,
                "primary_organ": primary_organ,
                "organ_name": primary_organ,
                "organ_family": _infer_organ_family(primary_organ),
                "organ_status": sample.get("organ_status") or "holdout_candidate",
                "pretraining_exposure": sample.get("pretraining_exposure") or _infer_pretraining_exposure(sample),
                "evidence_type": sample.get("evidence_type"),
                "mask_path": mask_path,
                "mask_metadata": sample.get("mask_metadata") or _extract_mask_metadata(sample),
                "study_id": sample.get("study_id") or sample.get("exam_id"),
                "source_case_id": sample.get("source_case_id") or sample.get("patient_id"),
                "split": sample.get("split"),
                "holdout_candidate": bool(primary_organ),
                "raw_sample": sample,
            }
        )

    return normalized_samples


def compute_organ_modality_support(
    samples: Sequence[Dict[str, Any]],
    required_modalities: Sequence[str] = DEFAULT_REQUIRED_MODALITIES,
) -> Dict[str, Dict[str, int]]:
    """Count how many samples each organ has in each required modality.

    Required parameters:
        samples: normalized Anatomical Generalization samples.
        required_modalities: modalities that must be counted, usually ``ct`` and
            ``mr``.
    """
    required = tuple(str(modality).lower() for modality in required_modalities)
    support: Dict[str, Dict[str, int]] = defaultdict(lambda: {modality: 0 for modality in required})
    for sample in samples:
        organ = sample.get("organ_name") or sample.get("primary_organ")
        modality = sample.get("modality")
        if not organ or modality not in required:
            continue
        support[str(organ)][str(modality)] += 1
    return {organ: dict(counts) for organ, counts in sorted(support.items())}


def _source_case_id(sample: Dict[str, Any]) -> str:
    return str(sample.get("source_case_id") or sample.get("patient_id") or sample.get("sample_id"))


def compute_organ_modality_case_support(
    samples: Sequence[Dict[str, Any]],
    required_modalities: Sequence[str] = DEFAULT_REQUIRED_MODALITIES,
) -> Dict[str, Dict[str, int]]:
    """Count unique source cases each organ has in each required modality."""
    required = tuple(str(modality).lower() for modality in required_modalities)
    support: Dict[str, Dict[str, set[str]]] = defaultdict(lambda: {modality: set() for modality in required})
    for sample in samples:
        organ = sample.get("organ_name") or sample.get("primary_organ")
        modality = sample.get("modality")
        if not organ or modality not in required:
            continue
        support[str(organ)][str(modality)].add(_source_case_id(sample))
    return {
        organ: {modality: len(case_ids) for modality, case_ids in counts.items()}
        for organ, counts in sorted(support.items())
    }


def get_eligible_holdout_organs(
    samples: Sequence[Dict[str, Any]],
    required_modalities: Sequence[str] = DEFAULT_REQUIRED_MODALITIES,
    min_samples_per_modality: int = 1,
) -> List[str]:
    """Return organs that satisfy the Anatomical Generalization per-fold target criterion.

    Required parameters:
        samples: normalized Anatomical Generalization samples.
        required_modalities: modalities required by the Anatomical surface.
        min_samples_per_modality: minimum unique source-case support required
            per organ in every required modality.

    The returned organs are eligible to serve as the low-shot target organ in a
    target-organ fold. This is a per-organ, per-fold notion of eligibility rather
    than a statement that all returned organs are held out at the same time.
    """
    support = compute_organ_modality_case_support(samples, required_modalities)
    eligible = []
    for organ, modality_counts in support.items():
        if all(modality_counts.get(modality, 0) >= min_samples_per_modality for modality in required_modalities):
            eligible.append(organ)
    return sorted(eligible)


def validate_anatomical_manifest(
    samples: Sequence[Dict[str, Any]],
    required_modalities: Sequence[str] = DEFAULT_REQUIRED_MODALITIES,
    min_holdout_organs: int = DEFAULT_MIN_HOLDOUT_ORGANS,
    min_samples_per_modality: int = 1,
) -> Dict[str, Any]:
    """Validate whether a normalized manifest is usable for Anatomical Generalization.

    Required parameters:
        samples: normalized Anatomical Generalization samples.
        required_modalities: modalities that must exist in the manifest.
        min_holdout_organs: minimum number of organs eligible for target-organ
            analysis.
        min_samples_per_modality: minimum unique source-case support required
            per eligible organ and modality.

    Example:
        >>> validate_anatomical_manifest(samples, ("ct", "mr"), 2, 5)["is_valid"]
        True
    """
    required = tuple(str(modality).lower() for modality in required_modalities)
    modality_counter = Counter(sample.get("modality", "unknown") for sample in samples)
    missing_modalities = [modality for modality in required if modality_counter.get(modality, 0) == 0]
    missing_organs = [sample["sample_id"] for sample in samples if not sample.get("organs")]
    missing_paths = [sample["sample_id"] for sample in samples if not sample.get("image_path")]
    support = compute_organ_modality_support(samples, required)
    case_support = compute_organ_modality_case_support(samples, required)
    eligible_holdout_organs = get_eligible_holdout_organs(samples, required, min_samples_per_modality)

    return {
        "n_samples": len(samples),
        "required_modalities": list(required),
        "available_modalities": dict(sorted(modality_counter.items())),
        "missing_modalities": missing_modalities,
        "samples_missing_organs": missing_organs,
        "samples_missing_file_path": missing_paths,
        "organ_support": support,
        "organ_case_support": case_support,
        "eligible_holdout_organs": eligible_holdout_organs,
        "eligible_target_organs": eligible_holdout_organs,
        "eligible_holdout_organs_note": (
            "Eligible target organs are organs that individually satisfy the per-fold support rule: "
            "each organ can be evaluated in a separate low-shot target-organ fold."
        ),
        "min_samples_per_modality": int(min_samples_per_modality),
        "min_holdout_organs": int(min_holdout_organs),
        "is_valid": (
            not missing_modalities
            and not missing_organs
            and len(eligible_holdout_organs) >= int(min_holdout_organs)
        ),
    }


def summarize_anatomical_manifest(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute a lightweight manifest summary for result metadata."""
    modality_counter = Counter(sample.get("modality", "unknown") for sample in samples)
    split_counter = Counter(sample.get("split") or "unspecified" for sample in samples)
    primary_organ_counter = Counter(sample.get("organ_name") or sample.get("primary_organ") or "unassigned" for sample in samples)
    source_case_counter = Counter(sample.get("source_case_id") or "unknown" for sample in samples)
    exposure_counter = Counter(sample.get("pretraining_exposure") or "unknown" for sample in samples)
    dataset_counter = Counter(sample.get("source_dataset") or sample.get("dataset") or "unknown" for sample in samples)

    return {
        "n_samples": len(samples),
        "modalities": dict(sorted(modality_counter.items())),
        "splits": dict(sorted(split_counter.items())),
        "primary_organs": dict(sorted(primary_organ_counter.items())),
        "source_datasets": dict(sorted(dataset_counter.items())),
        "pretraining_exposure": dict(sorted(exposure_counter.items())),
        "n_unique_source_cases": len(source_case_counter),
    }
