#!/usr/bin/env python

"""Build Cross-Modal Alignment manifests for supported CT/MR datasets.

The command scans one dataset surface, converts image/mask records into
organ-level samples, and writes full, sampled, and metadata manifests under the
Cross-Modal manifest root.

Inputs:
    --dataset: ``amos22_ct_mr``, ``mmwhs_ct_mr``, ``totalsegmenter_ct_mr_anchor``,
        ``chaos_ct_mr``, or ``all``.
    --min-voxels: minimum organ-mask support for retaining an organ sample.
    --min-samples-per-modality: minimum CT and MR support for an organ.

Example:
    python -m reform3d.modules.cross_modal_alignment.build_manifests \
        --dataset amos22_ct_mr
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm

from reform3d.modules.acquisition_robustness.config import (
    get_acquisition_manifest_path,
    get_output_paths as get_acquisition_output_paths,
)

from .config import get_cross_modal_manifest_dir, normalize_cross_modal_dataset_name
from .path_resolution import normalize_totalsegmenter_ct_record


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


MANIFEST_VARIANT = "core"
COHORT_EVIDENCE_TYPE = "cohort_level_same_organ_ct_mr"
PAIRED_EVIDENCE_TYPE = "paired_same_case_ct_mr"
CHAOS_EVIDENCE_TYPE = "single_organ_liver_ct_mr"

TOTALSEGMENTER_DATASET_NAME = "totalsegmenter_ct_mr_anchor"
MMWHS_DATASET_NAME = "mmwhs_ct_mr"
AMOS22_DATASET_NAME = "amos22_ct_mr"
CHAOS_DATASET_NAME = "chaos_ct_mr"
CROSS_MODAL_DATASETS = (
    TOTALSEGMENTER_DATASET_NAME,
    MMWHS_DATASET_NAME,
    AMOS22_DATASET_NAME,
    CHAOS_DATASET_NAME,
)

AMOS22_ROOT = Path.cwd() / "datasets" / "amos22"
MMWHS_ROOT = Path.cwd() / "datasets" / "mmwhs"
CHAOS_ROOT = Path.cwd() / "datasets" / "chaos"

TOTALSEGMENTER_SHARED_ORGANS = [
    "spleen",
    "kidney_left",
    "kidney_right",
    "gallbladder",
    "liver",
    "stomach",
    "pancreas",
    "small_bowel",
    "colon",
    "urinary_bladder",
    "aorta",
    "inferior_vena_cava",
]

MMWHS_LABELS: Dict[int, str] = {
    205: "myocardium",
    420: "left_atrium",
    500: "left_ventricle",
    550: "right_atrium",
    600: "right_ventricle",
    820: "ascending_aorta",
    850: "pulmonary_artery",
}

AMOS22_LABELS: Dict[int, str] = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "esophagus",
    6: "liver",
    7: "stomach",
    8: "aorta",
    9: "inferior_vena_cava",
    10: "pancreas",
    11: "adrenal_gland_right",
    12: "adrenal_gland_left",
    13: "duodenum",
    14: "urinary_bladder",
    15: "prostate_uterus",
}
AMOS22_EXCLUDED_ORGANS = {"urinary_bladder", "prostate_uterus"}
AMOS22_RAW_LABEL_OVERRIDES = {8: "arota", 9: "postcava", 15: "prostate/uterus"}
AMOS22_DEFAULT_SPLITS = ("training", "validation")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_manifest_trio(dataset_name: str, manifest_variant: str, trio: Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]) -> Path:
    manifest_dir = get_cross_modal_manifest_dir(dataset_name, manifest_variant)
    full_manifest, sampled_manifest, manifest_meta = trio
    _write_json(manifest_dir / "manifest_full.json", full_manifest)
    _write_json(manifest_dir / "manifest_sampled.json", sampled_manifest)
    _write_json(manifest_dir / "manifest_meta.json", manifest_meta)
    return manifest_dir


def _load_json(path: Path) -> Dict[str, Any] | List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _nifti_shape_spacing(path: Path) -> Tuple[List[int], List[float]]:
    image = nib.load(str(path))
    return [int(value) for value in image.shape[:3]], [float(value) for value in image.header.get_zooms()[:3]]


def _sitk_shape_spacing(path: Path) -> Tuple[List[int], List[float]]:
    image = _load_sitk_image(path)
    return list(reversed(image.GetSize()[:3])), [float(value) for value in image.GetSpacing()[:3]]


def _load_sitk_image(path: Path) -> sitk.Image:
    if path.is_dir():
        reader = sitk.ImageSeriesReader()
        series = reader.GetGDCMSeriesFileNames(str(path))
        if not series:
            raise ValueError(f"No DICOM series found under {path}")
        reader.SetFileNames(series)
        return reader.Execute()
    return sitk.ReadImage(str(path))


def _count_nifti_label(label_path: Path, label_value: int | None = None) -> int:
    data = np.asanyarray(nib.load(str(label_path)).dataobj)
    if label_value is None:
        return int(np.count_nonzero(data > 0))
    return int(np.count_nonzero(data == label_value))


def _count_sitk_label(label_path: Path, label_value: int | None = None) -> int:
    data = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path)))
    if label_value is None:
        return int(np.count_nonzero(data > 0))
    return int(np.count_nonzero(data == label_value))


def _sitk_label_counts(label_path: Path, labels: Dict[int, str], min_voxels: int) -> Dict[str, int]:
    data = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path)))
    counts: Dict[str, int] = {}
    for label_value, organ_name in labels.items():
        voxel_count = int(np.count_nonzero(data == label_value))
        counts[organ_name] = voxel_count if voxel_count >= min_voxels else 0
    return counts


def _count_png_stack(mask_dir: Path) -> int:
    count = 0
    for png_path in sorted(mask_dir.glob("*.png")):
        count += int(np.count_nonzero(np.asarray(Image.open(png_path).convert("L")) > 0))
    return count


def _manifest_payload(
    *,
    dataset_name: str,
    description: str,
    evidence_type: str,
    retained_organs: Sequence[str],
    samples: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    full = {
        "version": "1.0",
        "description": f"Cross-Modal full manifest: {description}",
        "reform3d_module": "cross_modal_alignment",
        "dataset": dataset_name,
        "manifest_variant": MANIFEST_VARIANT,
        "evidence_type": evidence_type,
        "retained_organs": list(retained_organs),
        "samples": list(samples),
    }
    sampled = {
        **full,
        "description": f"Cross-Modal sampled manifest: {description}",
        "sampling": {"applied": False, "strategy": "none", "reason": "manifest_sampled_equals_manifest_full"},
    }
    return full, sampled


def _sample_counts_by_modality(samples: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for sample in samples:
        counts[str(sample["modality"])] += 1
    return dict(sorted(counts.items()))


def _support_by_organ(ct_support: Dict[str, int], mr_support: Dict[str, int], organs: Iterable[str]) -> Dict[str, Dict[str, int]]:
    return {
        organ: {"ct_cases": int(ct_support.get(organ, 0)), "mr_cases": int(mr_support.get(organ, 0))}
        for organ in organs
    }


def _acquisition_manifest_path(dataset_name: str, manifest_kind: str = "full") -> Path:
    return get_acquisition_manifest_path(dataset_name, manifest_kind, "original_bins")


def _acquisition_semantic_cache_path(dataset_name: str) -> Path | None:
    if dataset_name != "totalsegmentermri":
        return None
    features_root = get_acquisition_output_paths(dataset_name, "original_bins")["features"]
    cache_path = features_root / "semantic_labels_totalsegmentermri_shared_structure_presence_totalsegmentermri.json"
    return cache_path if cache_path.exists() else None


def _load_presence_from_cache(cache_path: Path) -> Dict[str, Dict[str, int]]:
    payload = _load_json(cache_path)
    if not isinstance(payload, dict):
        return {}
    return {
        str(file_path): {str(organ): int(count) for organ, count in (entry.get("organ_volumes") or {}).items()}
        for file_path, entry in payload.items()
        if isinstance(entry, dict)
    }


def _extract_totalsegmenter_counts(label_dir: Path, min_voxels: int) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for organ in TOTALSEGMENTER_SHARED_ORGANS:
        mask_path = label_dir / f"{organ}.nii.gz"
        if not mask_path.exists():
            counts[organ] = 0
            continue
        try:
            voxel_count = _count_nifti_label(mask_path)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", mask_path, exc)
            voxel_count = 0
        counts[organ] = voxel_count if voxel_count >= min_voxels else 0
    return counts


def _load_acquisition_source_volumes(dataset_name: str) -> List[Dict[str, Any]]:
    manifest = _load_json(_acquisition_manifest_path(dataset_name, "full"))
    if isinstance(manifest, dict):
        volumes = list(manifest.get("volumes") or manifest.get("samples") or [])
    else:
        volumes = list(manifest)
    if dataset_name == "totalsegmenter_ct":
        return [normalize_totalsegmenter_ct_record(volume) for volume in volumes if isinstance(volume, dict)]
    return volumes


def _build_totalsegmenter_case_records(dataset_name: str, modality: str, min_voxels: int) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    volumes = _load_acquisition_source_volumes(dataset_name)
    cached_counts: Dict[str, Dict[str, int]] = {}
    cache_path = _acquisition_semantic_cache_path(dataset_name)
    if cache_path is not None:
        logger.info("Loading cached TotalSegmenter structure counts from %s", cache_path)
        cached_counts = _load_presence_from_cache(cache_path)

    case_records: List[Dict[str, Any]] = []
    support_counts = {organ: 0 for organ in TOTALSEGMENTER_SHARED_ORGANS}
    for volume in tqdm(volumes, desc=f"Scanning {dataset_name}", unit="case"):
        image_path = Path(volume["file_path"])
        label_dir = Path(volume["label_path"])
        organ_counts = cached_counts.get(str(image_path)) or _extract_totalsegmenter_counts(label_dir, min_voxels)
        present_organs = [organ for organ in TOTALSEGMENTER_SHARED_ORGANS if int(organ_counts.get(organ, 0)) > 0]
        for organ in present_organs:
            support_counts[organ] += 1
        case_id = image_path.parent.name
        case_records.append(
            {
                "dataset": dataset_name,
                "modality": modality,
                "case_id": case_id,
                "patient_id": case_id,
                "file_path": str(image_path),
                "label_path": str(label_dir),
                "shape": volume.get("shape"),
                "spacing": volume.get("spacing"),
                "present_organs": present_organs,
                "organ_counts": {organ: int(organ_counts.get(organ, 0)) for organ in TOTALSEGMENTER_SHARED_ORGANS},
            }
        )
    return case_records, support_counts


def _samples_from_binary_organ_cases(
    dataset_name: str,
    evidence_type: str,
    cases: Sequence[Dict[str, Any]],
    retained_organs: Iterable[str],
    *,
    source_dataset: str | None = None,
) -> List[Dict[str, Any]]:
    retained = set(retained_organs)
    samples: List[Dict[str, Any]] = []
    for case in cases:
        for organ in case.get("present_organs", []):
            if organ not in retained:
                continue
            mask_path = case.get("mask_path") or str(Path(case["label_path"]) / f"{organ}.nii.gz")
            sample = {
                "sample_id": f"{case['modality']}:{case['case_id']}:{organ}",
                "file_path": case["file_path"],
                "mask_path": mask_path,
                "label_path": case.get("label_path"),
                "dataset": dataset_name,
                "source_dataset": source_dataset or case.get("dataset", dataset_name),
                "modality": case["modality"],
                "primary_organ": organ,
                "organs": [organ],
                "patient_id": case["patient_id"],
                "source_case_id": case["case_id"],
                "evidence_type": evidence_type,
                "shape": case.get("shape"),
                "spacing": case.get("spacing"),
                "label_voxels": int(case["organ_counts"].get(organ, 0)),
            }
            if case.get("mask_loader"):
                sample["mask_loader"] = case["mask_loader"]
            samples.append(sample)
    return sorted(samples, key=lambda sample: (sample["primary_organ"], sample["modality"], sample["source_case_id"]))


def build_totalsegmenter_manifest(min_voxels: int, min_samples_per_modality: int) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    ct_cases, ct_support = _build_totalsegmenter_case_records("totalsegmenter_ct", "ct", min_voxels)
    mr_cases, mr_support = _build_totalsegmenter_case_records("totalsegmentermri", "mr", min_voxels)
    retained_organs = [
        organ for organ in TOTALSEGMENTER_SHARED_ORGANS
        if ct_support.get(organ, 0) >= min_samples_per_modality and mr_support.get(organ, 0) >= min_samples_per_modality
    ]
    samples = _samples_from_binary_organ_cases(
        TOTALSEGMENTER_DATASET_NAME,
        COHORT_EVIDENCE_TYPE,
        [*ct_cases, *mr_cases],
        retained_organs,
        source_dataset="totalsegmenter_ct_mr",
    )
    full, sampled = _manifest_payload(
        dataset_name=TOTALSEGMENTER_DATASET_NAME,
        description="TotalSegmenter CT + TotalSegmenter-MR anchor surface",
        evidence_type=COHORT_EVIDENCE_TYPE,
        retained_organs=retained_organs,
        samples=samples,
    )
    meta = {
        "dataset": TOTALSEGMENTER_DATASET_NAME,
        "reform3d_module": "cross_modal_alignment",
        "manifest_variant": MANIFEST_VARIANT,
        "source_manifests": {
            "ct": str(_acquisition_manifest_path("totalsegmenter_ct", "full")),
            "mr": str(_acquisition_manifest_path("totalsegmentermri", "full")),
        },
        "source_case_counts": {"ct": len(ct_cases), "mr": len(mr_cases)},
        "retained_organs": retained_organs,
        "excluded_organs": {
            organ: {
                **_support_by_organ(ct_support, mr_support, [organ])[organ],
                "reason": "insufficient_cross_modal_case_support",
            }
            for organ in TOTALSEGMENTER_SHARED_ORGANS
            if organ not in retained_organs
        },
        "shared_organ_label_set": TOTALSEGMENTER_SHARED_ORGANS,
        "support_by_organ": _support_by_organ(ct_support, mr_support, retained_organs),
        "total_samples_full": len(samples),
        "total_samples_sampled": len(samples),
        "sample_counts_by_modality": _sample_counts_by_modality(samples),
        "evidence_type": COHORT_EVIDENCE_TYPE,
        "min_voxels": int(min_voxels),
        "min_samples_per_modality": int(min_samples_per_modality),
        "sampling": sampled["sampling"],
    }
    return full, sampled, meta


def _resolve_mmwhs_image_path(image_entry: Path) -> Path:
    if image_entry.is_file():
        return image_entry
    if image_entry.is_dir():
        files = sorted(path for path in image_entry.iterdir() if path.is_file())
        if len(files) != 1:
            raise ValueError(f"Expected exactly one image file under {image_entry}, found {len(files)}")
        return files[0]
    raise FileNotFoundError(f"Missing MMWHS image entry: {image_entry}")


def _build_mmwhs_case_records(dataset_root: Path, modality: str, min_voxels: int) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    subset_dir = dataset_root / f"{modality}_train"
    support_counts = {organ: 0 for organ in MMWHS_LABELS.values()}
    case_records: List[Dict[str, Any]] = []
    for label_path in tqdm(sorted(subset_dir.glob(f"{modality}_train_*_label.nii")), desc=f"Scanning MMWHS {modality}", unit="case"):
        case_id = label_path.stem.replace("_label", "")
        image_path = _resolve_mmwhs_image_path(subset_dir / f"{case_id}_image.nii")
        shape, spacing = _sitk_shape_spacing(image_path)
        organ_counts = _sitk_label_counts(label_path, MMWHS_LABELS, min_voxels)
        present_organs = [organ for organ, count in organ_counts.items() if count > 0]
        for organ in present_organs:
            support_counts[organ] += 1
        case_records.append(
            {
                "dataset": MMWHS_DATASET_NAME,
                "modality": modality,
                "case_id": case_id,
                "patient_id": case_id.split("_")[-1],
                "file_path": str(image_path),
                "label_path": str(label_path),
                "shape": shape,
                "spacing": spacing,
                "organ_counts": organ_counts,
            }
        )
    return case_records, support_counts


def _samples_from_multiclass_cases(
    *,
    dataset_name: str,
    evidence_type: str,
    cases: Sequence[Dict[str, Any]],
    retained_organs: Sequence[str],
    label_map: Dict[int, str],
    source_dataset: str,
) -> List[Dict[str, Any]]:
    label_value_by_organ = {organ: label_value for label_value, organ in label_map.items()}
    samples: List[Dict[str, Any]] = []
    for case in cases:
        for organ in retained_organs:
            voxel_count = int(case["organ_counts"].get(organ, 0))
            if voxel_count <= 0:
                continue
            samples.append(
                {
                    "sample_id": f"{case['modality']}:{case['case_id']}:{organ}",
                    "file_path": case["file_path"],
                    "mask_path": case["label_path"],
                    "label_path": case["label_path"],
                    "mask_label_value": int(label_value_by_organ[organ]),
                    "dataset": dataset_name,
                    "source_dataset": source_dataset,
                    "modality": case["modality"],
                    "primary_organ": organ,
                    "organs": [organ],
                    "patient_id": case["patient_id"],
                    "source_case_id": case["case_id"],
                    "split": case.get("split"),
                    "evidence_type": evidence_type,
                    "shape": case["shape"],
                    "spacing": case["spacing"],
                    "label_voxels": voxel_count,
                }
            )
    return sorted(samples, key=lambda sample: (sample["primary_organ"], sample["modality"], sample["source_case_id"]))


def build_mmwhs_manifest(dataset_root: Path, min_voxels: int, min_samples_per_modality: int) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    ct_cases, ct_support = _build_mmwhs_case_records(dataset_root, "ct", min_voxels)
    mr_cases, mr_support = _build_mmwhs_case_records(dataset_root, "mr", min_voxels)
    retained_organs = [
        organ for organ in MMWHS_LABELS.values()
        if ct_support.get(organ, 0) >= min_samples_per_modality and mr_support.get(organ, 0) >= min_samples_per_modality
    ]
    samples = _samples_from_multiclass_cases(
        dataset_name=MMWHS_DATASET_NAME,
        evidence_type=PAIRED_EVIDENCE_TYPE,
        cases=[*ct_cases, *mr_cases],
        retained_organs=retained_organs,
        label_map=MMWHS_LABELS,
        source_dataset="mmwhs",
    )
    full, sampled = _manifest_payload(
        dataset_name=MMWHS_DATASET_NAME,
        description="MMWHS CT/MR cardiac validation surface",
        evidence_type=PAIRED_EVIDENCE_TYPE,
        retained_organs=retained_organs,
        samples=samples,
    )
    meta = {
        "dataset": MMWHS_DATASET_NAME,
        "reform3d_module": "cross_modal_alignment",
        "manifest_variant": MANIFEST_VARIANT,
        "source_root": str(dataset_root),
        "source_case_counts": {"ct": len(ct_cases), "mr": len(mr_cases)},
        "retained_organs": retained_organs,
        "excluded_organs": {
            organ: {
                **_support_by_organ(ct_support, mr_support, [organ])[organ],
                "reason": "insufficient_cross_modal_case_support",
            }
            for organ in MMWHS_LABELS.values()
            if organ not in retained_organs
        },
        "organ_label_map": {str(label_value): organ for label_value, organ in MMWHS_LABELS.items()},
        "support_by_organ": _support_by_organ(ct_support, mr_support, retained_organs),
        "total_samples_full": len(samples),
        "total_samples_sampled": len(samples),
        "sample_counts_by_modality": _sample_counts_by_modality(samples),
        "evidence_type": PAIRED_EVIDENCE_TYPE,
        "min_voxels": int(min_voxels),
        "min_samples_per_modality": int(min_samples_per_modality),
    }
    return full, sampled, meta


def _load_amos22_dataset_json(dataset_root: Path) -> Dict[str, Any]:
    metadata_path = dataset_root / "dataset.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing AMOS22 dataset.json: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _amos22_case_id(image_entry: str) -> str:
    return Path(image_entry).name.removesuffix(".nii.gz")


def _amos22_case_number(case_id: str) -> int:
    try:
        return int(case_id.split("_")[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Unable to parse AMOS22 case id: {case_id}") from exc


def _amos22_modality(case_id: str) -> str:
    return "ct" if _amos22_case_number(case_id) < 500 else "mr"


def _iter_amos22_entries(metadata: Dict[str, Any], splits: Iterable[str]) -> List[Tuple[str, Dict[str, str]]]:
    entries: List[Tuple[str, Dict[str, str]]] = []
    for split in splits:
        for entry in metadata.get(split, []):
            if isinstance(entry, dict) and "image" in entry and "label" in entry:
                entries.append((split, entry))
    return entries


def _build_amos22_case_records(
    dataset_root: Path,
    splits: Iterable[str],
    min_voxels: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    metadata = _load_amos22_dataset_json(dataset_root)
    support_counts = {organ: {"ct": 0, "mr": 0} for organ in AMOS22_LABELS.values()}
    case_records: List[Dict[str, Any]] = []
    for split, entry in tqdm(_iter_amos22_entries(metadata, splits), desc="Scanning AMOS22", unit="case"):
        case_id = _amos22_case_id(entry["image"])
        modality = _amos22_modality(case_id)
        image_path = (dataset_root / entry["image"]).resolve()
        label_path = (dataset_root / entry["label"]).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Missing AMOS22 image: {image_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Missing AMOS22 label: {label_path}")
        shape, spacing = _nifti_shape_spacing(image_path)
        organ_counts = {}
        for label_value, organ in AMOS22_LABELS.items():
            count = _count_nifti_label(label_path, label_value)
            organ_counts[organ] = count if count >= min_voxels else 0
        for organ, count in organ_counts.items():
            if count > 0:
                support_counts[organ][modality] += 1
        case_records.append(
            {
                "dataset": AMOS22_DATASET_NAME,
                "source_dataset": "amos22",
                "modality": modality,
                "case_id": case_id,
                "patient_id": case_id,
                "split": split,
                "file_path": str(image_path),
                "label_path": str(label_path),
                "shape": shape,
                "spacing": spacing,
                "organ_counts": organ_counts,
            }
        )
    return case_records, support_counts


def build_amos22_manifest(
    dataset_root: Path,
    splits: Iterable[str],
    min_voxels: int,
    min_samples_per_modality: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    selected_splits = list(splits)
    cases, support = _build_amos22_case_records(dataset_root, selected_splits, min_voxels)
    retained_organs = [
        organ for organ in AMOS22_LABELS.values()
        if organ not in AMOS22_EXCLUDED_ORGANS
        and support.get(organ, {}).get("ct", 0) >= min_samples_per_modality
        and support.get(organ, {}).get("mr", 0) >= min_samples_per_modality
    ]
    samples = _samples_from_multiclass_cases(
        dataset_name=AMOS22_DATASET_NAME,
        evidence_type=COHORT_EVIDENCE_TYPE,
        cases=cases,
        retained_organs=retained_organs,
        label_map=AMOS22_LABELS,
        source_dataset="amos22",
    )
    full, sampled = _manifest_payload(
        dataset_name=AMOS22_DATASET_NAME,
        description="AMOS22 cohort-level CT/MR abdominal-organ validation surface",
        evidence_type=COHORT_EVIDENCE_TYPE,
        retained_organs=retained_organs,
        samples=samples,
    )
    source_case_counts = defaultdict(int)
    split_counts = defaultdict(int)
    for case in cases:
        source_case_counts[case["modality"]] += 1
        split_counts[f"{case['split']}_{case['modality']}"] += 1
    meta = {
        "dataset": AMOS22_DATASET_NAME,
        "reform3d_module": "cross_modal_alignment",
        "manifest_variant": MANIFEST_VARIANT,
        "source_root": str(dataset_root),
        "source_case_counts": dict(sorted(source_case_counts.items())),
        "source_case_counts_by_split": dict(sorted(split_counts.items())),
        "retained_organs": retained_organs,
        "excluded_organs": {
            organ: {
                "ct_cases": int(support.get(organ, {}).get("ct", 0)),
                "mr_cases": int(support.get(organ, {}).get("mr", 0)),
                "reason": "excluded_by_protocol_low_mr_support" if organ in AMOS22_EXCLUDED_ORGANS else "insufficient_cross_modal_case_support",
            }
            for organ in AMOS22_LABELS.values()
            if organ not in retained_organs
        },
        "organ_label_map": {str(label_value): organ for label_value, organ in AMOS22_LABELS.items()},
        "raw_organ_label_map": {
            str(label_value): AMOS22_RAW_LABEL_OVERRIDES.get(label_value, organ)
            for label_value, organ in AMOS22_LABELS.items()
        },
        "support_by_organ": {
            organ: {
                "ct_cases": int(support.get(organ, {}).get("ct", 0)),
                "mr_cases": int(support.get(organ, {}).get("mr", 0)),
            }
            for organ in retained_organs
        },
        "total_samples_full": len(samples),
        "total_samples_sampled": len(samples),
        "sample_counts_by_modality": _sample_counts_by_modality(samples),
        "evidence_type": COHORT_EVIDENCE_TYPE,
        "min_voxels": int(min_voxels),
        "min_samples_per_modality": int(min_samples_per_modality),
        "selected_splits": selected_splits,
        "modality_rule": "AMOS22 case ids below 500 are CT; ids 500 and above are MRI.",
    }
    return full, sampled, meta


def _is_dicom_like_file(path: Path) -> bool:
    if not path.is_file():
        return False
    suffix = path.suffix.lower()
    return suffix in {"", ".dcm", ".ima"}


def _has_dicom_files(path: Path) -> bool:
    try:
        return any(_is_dicom_like_file(child) for child in path.iterdir())
    except OSError:
        return False


def _chaos_relative_parts(path: Path, root: Path | None = None) -> List[str]:
    if root is not None:
        try:
            return [part.lower() for part in path.resolve().relative_to(root.resolve()).parts]
        except ValueError:
            pass
    return [part.lower() for part in path.parts]


def _infer_chaos_modality(path: Path, root: Path | None = None) -> str | None:
    parts = _chaos_relative_parts(path, root)
    if any(part in {"mr", "mri"} or part.startswith("mr_") for part in parts):
        return "mr"
    if any(part == "ct" or part.startswith("ct_") for part in parts):
        return "ct"
    if any(part in {"t1dual", "t2spir"} for part in parts):
        return "mr"
    return None


def _is_chaos_ground_path(path: Path, root: Path | None = None) -> bool:
    return "ground" in _chaos_relative_parts(path, root)


def _infer_chaos_patient_id(path: Path, root: Path) -> str:
    try:
        parts = path.resolve().relative_to(root.resolve()).parts
    except ValueError:
        parts = path.parts
    for part in parts:
        token = part.strip()
        if token.isdigit():
            return token
        if token.lower().startswith("chaos_"):
            return token
    return path.parent.name


def _chaos_case_id(path: Path, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        relative = path
    return "__".join(part for part in relative.parts if part not in {".", ""})


def _find_chaos_ground_dir(series_dir: Path, root: Path) -> Path | None:
    for parent in [series_dir.parent, *series_dir.parents]:
        if root not in parent.parents and parent != root:
            continue
        direct = parent / "Ground"
        if direct.is_dir() and any(direct.glob("*.png")):
            return direct
        try:
            relative_depth = len(parent.resolve().relative_to(root.resolve()).parts)
        except ValueError:
            break
        if parent == root or relative_depth <= 3:
            break
    return None


def _is_maskish_nifti(path: Path) -> bool:
    text = path.as_posix().lower()
    return any(token in text for token in ("mask", "label", "seg", "ground", "gt"))


def _nifti_files(root: Path) -> List[Path]:
    return sorted([*root.rglob("*.nii"), *root.rglob("*.nii.gz")])


def _find_chaos_nifti_mask(image_path: Path, root: Path) -> Path | None:
    image_stem = image_path.name.replace(".nii.gz", "").replace(".nii", "")
    candidate_roots = [image_path.parent, image_path.parent.parent if image_path.parent.parent.exists() else image_path.parent]
    for candidate_root in candidate_roots:
        masks = [
            path for path in _nifti_files(candidate_root)
            if path != image_path and _is_maskish_nifti(path)
        ]
        matching = [path for path in masks if image_stem.lower() in path.name.lower()]
        if matching:
            return sorted(matching)[0]
        if masks:
            return sorted(masks)[0]
    masks = [path for path in _nifti_files(root) if _is_maskish_nifti(path) and _infer_chaos_patient_id(path, root) == _infer_chaos_patient_id(image_path, root)]
    return sorted(masks)[0] if masks else None


def _discover_chaos_case_records(dataset_root: Path, min_voxels: int) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    case_records: List[Dict[str, Any]] = []
    support_counts = {"ct": 0, "mr": 0}
    seen_sources: set[str] = set()

    dicom_dirs = [
        path for path in dataset_root.rglob("*")
        if path.is_dir()
        and not _is_chaos_ground_path(path, dataset_root)
        and _has_dicom_files(path)
    ]
    for series_dir in tqdm(dicom_dirs, desc="Scanning CHAOS DICOM series", unit="series"):
        modality = _infer_chaos_modality(series_dir, dataset_root)
        if modality not in {"ct", "mr"}:
            continue
        ground_dir = _find_chaos_ground_dir(series_dir, dataset_root)
        if ground_dir is None:
            continue
        voxel_count = _count_png_stack(ground_dir)
        if voxel_count < min_voxels:
            continue
        try:
            shape, spacing = _sitk_shape_spacing(series_dir)
        except Exception as exc:
            logger.warning("Skipping CHAOS series %s: %s", series_dir, exc)
            continue
        patient_id = _infer_chaos_patient_id(series_dir, dataset_root)
        source_case_id = _chaos_case_id(series_dir, dataset_root)
        if source_case_id in seen_sources:
            continue
        seen_sources.add(source_case_id)
        support_counts[modality] += 1
        case_records.append(
            {
                "dataset": CHAOS_DATASET_NAME,
                "source_dataset": "chaos",
                "modality": modality,
                "case_id": source_case_id,
                "patient_id": patient_id,
                "file_path": str(series_dir),
                "label_path": str(ground_dir),
                "mask_path": str(ground_dir),
                "mask_loader": "chaos_png_stack",
                "shape": shape,
                "spacing": spacing,
                "present_organs": ["liver"],
                "organ_counts": {"liver": int(voxel_count)},
            }
        )

    for image_path in tqdm([path for path in _nifti_files(dataset_root) if not _is_maskish_nifti(path)], desc="Scanning CHAOS NIfTI images", unit="case"):
        modality = _infer_chaos_modality(image_path, dataset_root)
        if modality not in {"ct", "mr"}:
            continue
        mask_path = _find_chaos_nifti_mask(image_path, dataset_root)
        if mask_path is None:
            continue
        source_case_id = _chaos_case_id(image_path, dataset_root)
        if source_case_id in seen_sources:
            continue
        voxel_count = _count_nifti_label(mask_path)
        if voxel_count < min_voxels:
            continue
        shape, spacing = _nifti_shape_spacing(image_path)
        seen_sources.add(source_case_id)
        support_counts[modality] += 1
        case_records.append(
            {
                "dataset": CHAOS_DATASET_NAME,
                "source_dataset": "chaos",
                "modality": modality,
                "case_id": source_case_id,
                "patient_id": _infer_chaos_patient_id(image_path, dataset_root),
                "file_path": str(image_path),
                "label_path": str(mask_path),
                "mask_path": str(mask_path),
                "shape": shape,
                "spacing": spacing,
                "present_organs": ["liver"],
                "organ_counts": {"liver": int(voxel_count)},
            }
        )

    return case_records, support_counts


def build_chaos_manifest(dataset_root: Path, min_voxels: int, min_samples_per_modality: int) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    cases, support = _discover_chaos_case_records(dataset_root, min_voxels)
    retained_organs = ["liver"] if support.get("ct", 0) >= min_samples_per_modality and support.get("mr", 0) >= min_samples_per_modality else []
    samples = _samples_from_binary_organ_cases(
        CHAOS_DATASET_NAME,
        CHAOS_EVIDENCE_TYPE,
        cases,
        retained_organs,
        source_dataset="chaos",
    )
    full, sampled = _manifest_payload(
        dataset_name=CHAOS_DATASET_NAME,
        description="CHAOS liver-only CT/MR supporting surface",
        evidence_type=CHAOS_EVIDENCE_TYPE,
        retained_organs=retained_organs,
        samples=samples,
    )
    meta = {
        "dataset": CHAOS_DATASET_NAME,
        "reform3d_module": "cross_modal_alignment",
        "manifest_variant": MANIFEST_VARIANT,
        "source_root": str(dataset_root),
        "source_case_counts": {"ct": int(support.get("ct", 0)), "mr": int(support.get("mr", 0))},
        "retained_organs": retained_organs,
        "excluded_organs": {
            "liver": {
                "ct_cases": int(support.get("ct", 0)),
                "mr_cases": int(support.get("mr", 0)),
                "reason": "insufficient_cross_modal_case_support",
            }
        } if not retained_organs else {},
        "support_by_organ": {
            "liver": {"ct_cases": int(support.get("ct", 0)), "mr_cases": int(support.get("mr", 0))}
        },
        "total_samples_full": len(samples),
        "total_samples_sampled": len(samples),
        "sample_counts_by_modality": _sample_counts_by_modality(samples),
        "evidence_type": CHAOS_EVIDENCE_TYPE,
        "min_voxels": int(min_voxels),
        "min_samples_per_modality": int(min_samples_per_modality),
        "interpretation_scope": "CHAOS is retained as a liver-only supporting surface, not as a broad multi-organ cohort.",
        "notes": [
            "The builder supports common CHAOS DICOM image directories with PNG Ground masks and preprocessed NIfTI image/mask pairs.",
            "PNG mask stacks are marked with mask_loader=chaos_png_stack for the Cross-Modal feature extractor.",
        ],
    }
    return full, sampled, meta


def build_dataset_manifest(
    dataset_name: str,
    *,
    manifest_variant: str,
    min_voxels: int,
    min_samples_per_modality: int,
    dataset_root: Path | None,
    amos_splits: Sequence[str],
) -> Path:
    dataset_name = normalize_cross_modal_dataset_name(dataset_name)
    if manifest_variant != MANIFEST_VARIANT:
        raise ValueError(f"Cross-Modal manifest builders currently materialize only the {MANIFEST_VARIANT!r} variant")

    if dataset_name == TOTALSEGMENTER_DATASET_NAME:
        trio = build_totalsegmenter_manifest(min_voxels, min_samples_per_modality)
    elif dataset_name == MMWHS_DATASET_NAME:
        trio = build_mmwhs_manifest(dataset_root or MMWHS_ROOT, min_voxels, min_samples_per_modality)
    elif dataset_name == AMOS22_DATASET_NAME:
        trio = build_amos22_manifest(dataset_root or AMOS22_ROOT, amos_splits, min_voxels, min_samples_per_modality)
    elif dataset_name == CHAOS_DATASET_NAME:
        trio = build_chaos_manifest(dataset_root or CHAOS_ROOT, min_voxels, min_samples_per_modality)
    else:
        raise ValueError(f"Unknown Cross-Modal dataset surface: {dataset_name}")

    manifest_dir = _write_manifest_trio(dataset_name, manifest_variant, trio)
    full_manifest = trio[0]
    logger.info("Wrote %s manifests to %s", dataset_name, manifest_dir)
    logger.info("Retained organs: %s", ", ".join(full_manifest["retained_organs"]) or "none")
    logger.info("Total organ-level samples: %d", len(full_manifest["samples"]))
    return manifest_dir


def _resolve_dataset_root_override(dataset_name: str, args: argparse.Namespace) -> Path | None:
    if args.dataset_root:
        return Path(args.dataset_root).resolve()
    if dataset_name == MMWHS_DATASET_NAME and args.mmwhs_root:
        return Path(args.mmwhs_root).resolve()
    if dataset_name == AMOS22_DATASET_NAME and args.amos22_root:
        return Path(args.amos22_root).resolve()
    if dataset_name == CHAOS_DATASET_NAME and args.chaos_root:
        return Path(args.chaos_root).resolve()
    return None


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build Cross-Modal Alignment manifests for supported CT/MR datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m reform3d.modules.cross_modal_alignment.build_manifests --dataset amos22_ct_mr\n"
            "  python -m reform3d.modules.cross_modal_alignment.build_manifests --dataset chaos_ct_mr\n"
            "  python -m reform3d.modules.cross_modal_alignment.build_manifests --dataset all --skip-missing"
        ),
    )
    parser.add_argument("--dataset", default=AMOS22_DATASET_NAME, choices=[*CROSS_MODAL_DATASETS, "all"])
    parser.add_argument("--manifest-variant", default=MANIFEST_VARIANT)
    parser.add_argument("--min-voxels", type=int, default=100)
    parser.add_argument("--min-samples-per-modality", type=int, default=5)
    parser.add_argument("--dataset-root", default=None, help="Root override for the selected dataset")
    parser.add_argument("--amos22-root", default=str(AMOS22_ROOT))
    parser.add_argument("--mmwhs-root", default=str(MMWHS_ROOT))
    parser.add_argument("--chaos-root", default=str(CHAOS_ROOT))
    parser.add_argument("--amos-splits", nargs="+", default=list(AMOS22_DEFAULT_SPLITS), choices=("training", "validation"))
    parser.add_argument("--skip-missing", action="store_true", help="Continue when a selected source root or manifest is missing")
    args = parser.parse_args(argv)

    selected = list(CROSS_MODAL_DATASETS) if args.dataset == "all" else [normalize_cross_modal_dataset_name(args.dataset)]
    for dataset_name in selected:
        try:
            build_dataset_manifest(
                dataset_name,
                manifest_variant=args.manifest_variant,
                min_voxels=args.min_voxels,
                min_samples_per_modality=args.min_samples_per_modality,
                dataset_root=_resolve_dataset_root_override(dataset_name, args),
                amos_splits=args.amos_splits,
            )
        except (FileNotFoundError, ValueError) as exc:
            if not args.skip_missing:
                raise
            logger.warning("Skipping %s: %s", dataset_name, exc)


if __name__ == "__main__":
    main()
