#!/usr/bin/env python

"""Extract organ-aware Cross-Modal Alignment embeddings for one feature family.

This entry point reads an organ-level Cross-Modal Alignment manifest and writes
the embedding bundle consumed by the Cross-Modal metric pipeline. A reusable
crop-cache stage lets checkpoints with the same input crop size share the
expensive CPU-side preprocessing work. The extractor can:

- build an organ crop cache once per crop size
- reuse cached organ crops across multiple checkpoints with the same crop size
- write the final NPZ bundle expected by the Cross-Modal metric pipeline

Without a crop cache, it loads each source case once, applies the same
percentile normalization and spacing-aware axis permutation used by the
validated Acquisition Robustness preprocessing path, crops around the organ
mask bounding box, resizes to the checkpoint crop size, and writes an NPZ bundle
with this contract:

- features: [n_samples, dim]
- sample_ids: [n_samples]

The output NPZ can be passed directly to the Cross-Modal evaluation pipeline.

    Usage examples:
    python -m reform3d.modules.cross_modal_alignment.feature_extractor \
        -m amos22_ct_mr/core/manifest_sampled.json \
        -c med3dino_rel_96

    python -m reform3d.modules.cross_modal_alignment.feature_extractor \
        -m amos22_ct_mr/core/manifest_sampled.json \
        -c med3dino_rel_96 \
        --max-samples 32 \
        --batch-size 4

    python -m reform3d.modules.cross_modal_alignment.feature_extractor \
        -m amos22_ct_mr/core/manifest_sampled.json \
        --crop-size 96 \
        --build-crop-cache-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from reform3d.core.checkpoints import resolve_checkpoint_config
from reform3d.core.paths import get_reform3d_cache_path_candidates
from reform3d.modules.acquisition_robustness.feature_extractor import ExtractionConfig, FeatureExtractor

from reform3d.encoders.med3dino.spacing_aware_transforms import CropForegroundSwapSliceDimsV2  # noqa: E402
from .config import (
    DEFAULT_FEATURE_TYPE,
    FEATURE_TYPES,
    get_checkpoint_feature_npz_path,
    get_crop_cache_dir,
    get_dataset_name_from_manifest_path,
    get_manifest_variant_from_manifest_path,
    get_output_paths,
    normalize_feature_type,
)
from .path_resolution import normalize_totalsegmenter_ct_record


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


PREPROCESS_PERCENTILE_LOWER = 0.05
PREPROCESS_PERCENTILE_UPPER = 99.95
PREPROCESS_OUTPUT_MIN = -1.0
PREPROCESS_OUTPUT_MAX = 1.0
DEFAULT_BBOX_MARGIN = 5
OUTSIDE_ORGAN_MASK_MODES = ("none", "zero")
PNG_MASK_STACK_LOADERS = frozenset({"png_mask_stack", "chaos_png_stack"})

CROP_CACHE_SAMPLE_SIGNATURE_VERSION = "normalized_sample_stat_v1"
CROP_CACHE_INDEX_NAME = "crop_cache_index.json"
CROP_CACHE_CASES_DIRNAME = "cases"
CROP_CACHE_LOCK_NAME = ".crop_cache.lock"


@dataclass(frozen=True)
class OrganSample:
    """Normalized organ-level Cross-Modal sample record."""

    sample_id: str
    file_path: Path
    mask_path: Path
    modality: str
    primary_organ: str
    source_case_id: str
    spacing: Tuple[float, float, float] | None
    mask_label_value: int | None = None
    mask_label_range: Tuple[int, int] | None = None
    mask_loader: str | None = None


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _normalize_modality(value: Any) -> str:
    return str(value or "unknown").strip().lower()


def _normalize_organ(sample: Dict[str, Any]) -> str:
    primary = sample.get("primary_organ") or sample.get("organ")
    if primary:
        return str(primary).strip().lower()
    organs = sample.get("organs") or []
    if isinstance(organs, Sequence) and organs:
        return str(organs[0]).strip().lower()
    raise ValueError(f"Sample is missing primary organ: {sample}")


def _normalize_spacing(value: Any) -> Tuple[float, float, float] | None:
    if not isinstance(value, Sequence) or len(value) < 3:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None


def _normalize_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_int_range(value: Any) -> Tuple[int, int] | None:
    if not isinstance(value, Sequence) or len(value) < 2:
        return None
    lower = _normalize_int(value[0])
    upper = _normalize_int(value[1])
    if lower is None or upper is None:
        return None
    return (min(lower, upper), max(lower, upper))


def _load_manifest_samples(manifest_path: Path, max_samples: int | None = None) -> List[OrganSample]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_samples = payload.get("samples") if isinstance(payload, dict) else payload
    if raw_samples is None:
        raw_samples = payload.get("volumes", []) if isinstance(payload, dict) else []

    samples: List[OrganSample] = []
    for index, raw_sample in enumerate(raw_samples):
        if not isinstance(raw_sample, dict):
            continue
        raw_sample = normalize_totalsegmenter_ct_record(raw_sample)
        sample_id = str(raw_sample.get("sample_id") or raw_sample.get("id") or f"sample_{index:06d}")
        file_path = Path(raw_sample.get("file_path") or "").resolve()
        mask_path = Path(raw_sample.get("mask_path") or "").resolve()
        if not file_path:
            raise ValueError(f"Sample {sample_id} is missing file_path")
        if not mask_path:
            raise ValueError(f"Sample {sample_id} is missing mask_path")

        samples.append(
            OrganSample(
                sample_id=sample_id,
                file_path=file_path,
                mask_path=mask_path,
                modality=_normalize_modality(raw_sample.get("modality")),
                primary_organ=_normalize_organ(raw_sample),
                source_case_id=str(raw_sample.get("source_case_id") or raw_sample.get("patient_id") or sample_id),
                spacing=_normalize_spacing(raw_sample.get("spacing")),
                mask_label_value=_normalize_int(raw_sample.get("mask_label_value")),
                mask_label_range=_normalize_int_range(raw_sample.get("mask_label_range")),
                mask_loader=(str(raw_sample.get("mask_loader")).strip().lower() if raw_sample.get("mask_loader") else None),
            )
        )
        if max_samples is not None and len(samples) >= max_samples:
            break

    return samples


def _replace_nonfinite(array: np.ndarray) -> np.ndarray:
    if np.isfinite(array).all():
        return array
    finite = array[np.isfinite(array)]
    fill_value = float(finite.mean()) if finite.size else 0.0
    return np.nan_to_num(array, nan=fill_value, posinf=fill_value, neginf=fill_value)


def _percentile_scale(image: np.ndarray) -> np.ndarray:
    lower = float(np.percentile(image, PREPROCESS_PERCENTILE_LOWER))
    upper = float(np.percentile(image, PREPROCESS_PERCENTILE_UPPER))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return np.zeros_like(image, dtype=np.float32)

    clipped = np.clip(image, lower, upper)
    scaled = (clipped - lower) / (upper - lower)
    scaled = scaled * (PREPROCESS_OUTPUT_MAX - PREPROCESS_OUTPUT_MIN) + PREPROCESS_OUTPUT_MIN
    return scaled.astype(np.float32, copy=False)


def _path_uses_sitk(path: Path) -> bool:
    name = path.name.lower()
    return path.is_dir() or name.endswith(".nrrd") or name.endswith(".seg.nrrd") or name.endswith(".mha") or name.endswith(".mhd")


def _load_dicom_series(path: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    file_names = list(reader.GetGDCMSeriesFileNames(str(path)))
    if not file_names:
        raise ValueError(f"No DICOM slices found under {path}")
    reader.SetFileNames(file_names)
    return reader.Execute()


def _load_image_array_and_spacing(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    if _path_uses_sitk(path):
        image_obj = _load_dicom_series(path) if path.is_dir() else sitk.ReadImage(str(path))
        image = sitk.GetArrayFromImage(image_obj)
        if image.ndim == 4:
            if 1 in image.shape:
                image = np.squeeze(image)
            else:
                raise ValueError(f"Only single-channel images are supported, got shape {image.shape} for {path}")
        if image.ndim != 3:
            raise ValueError(f"Expected 3D image, got shape {image.shape} for {path}")
        image = np.transpose(image, (1, 2, 0))
        spacing = tuple(float(x) for x in image_obj.GetSpacing()[:3])
        return np.asarray(image, dtype=np.float32), spacing

    image_obj = nib.load(str(path))
    image = np.asarray(image_obj.dataobj, dtype=np.float32)
    if image.ndim == 4:
        if 1 in image.shape:
            image = np.squeeze(image)
        else:
            raise ValueError(f"Only single-channel images are supported, got shape {image.shape} for {path}")
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape {image.shape} for {path}")
    spacing = tuple(float(x) for x in image_obj.header.get_zooms()[:3])
    return image, spacing


def _load_case_image(image_path: Path, spacing_hint: Tuple[float, float, float] | None) -> Tuple[np.ndarray, List[int]]:
    image, detected_spacing = _load_image_array_and_spacing(image_path)
    spacing = spacing_hint or detected_spacing
    perm, _ = CropForegroundSwapSliceDimsV2.get_permutation_and_metadata(spacing)

    image = _replace_nonfinite(image)
    image = _percentile_scale(image)
    image = np.transpose(image[None, ...], axes=perm)
    return image.astype(np.float32, copy=False), perm


def _load_binary_mask_volume(mask_path: Path) -> np.ndarray:
    if _path_uses_sitk(mask_path):
        mask_obj = _load_dicom_series(mask_path) if mask_path.is_dir() else sitk.ReadImage(str(mask_path))
        mask = sitk.GetArrayFromImage(mask_obj)
        if mask.ndim == 4:
            if 1 in mask.shape:
                mask = np.squeeze(mask)
            else:
                raise ValueError(f"Expected 3D mask, got shape {mask.shape} for {mask_path}")
        if mask.ndim != 3:
            raise ValueError(f"Expected 3D mask, got shape {mask.shape} for {mask_path}")
        return np.transpose(mask, (1, 2, 0))

    mask_obj = nib.load(str(mask_path))
    mask = np.asarray(mask_obj.dataobj)
    if mask.ndim == 4:
        if 1 in mask.shape:
            mask = np.squeeze(mask)
        else:
            raise ValueError(f"Expected 3D mask, got shape {mask.shape} for {mask_path}")
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {mask.shape} for {mask_path}")
    return mask


def _load_png_mask_stack(mask_dir: Path) -> np.ndarray:
    png_paths = sorted(path for path in mask_dir.iterdir() if path.suffix.lower() == ".png")
    if not png_paths:
        raise ValueError(f"No PNG masks found under {mask_dir}")
    slices = [np.asarray(Image.open(path).convert("L")) for path in png_paths]
    return np.transpose(np.stack(slices, axis=0), (1, 2, 0))


def _uses_png_mask_stack(mask_loader: str | None) -> bool:
    if not mask_loader:
        return False
    return str(mask_loader).strip().lower() in PNG_MASK_STACK_LOADERS


def _binarize_mask(mask: np.ndarray, sample: OrganSample) -> np.ndarray:
    if sample.mask_label_value is not None:
        return (mask == sample.mask_label_value).astype(np.uint8, copy=False)
    if sample.mask_label_range is not None:
        lower, upper = sample.mask_label_range
        return ((mask >= lower) & (mask <= upper)).astype(np.uint8, copy=False)
    return (mask > 0).astype(np.uint8, copy=False)


def _load_mask(sample: OrganSample, perm: Sequence[int]) -> np.ndarray:
    if _uses_png_mask_stack(sample.mask_loader):
        mask = _load_png_mask_stack(sample.mask_path)
    else:
        mask = _load_binary_mask_volume(sample.mask_path)
    mask = _binarize_mask(mask, sample)
    return np.transpose(mask[None, ...], axes=perm)[0]


def _get_bounding_box(mask: np.ndarray, margin: int) -> Tuple[slice, slice, slice] | None:
    coords = np.where(mask > 0)
    if coords[0].size == 0:
        return None

    slices: List[slice] = []
    for axis, axis_coords in enumerate(coords):
        start = max(0, int(axis_coords.min()) - margin)
        stop = min(mask.shape[axis], int(axis_coords.max()) + margin + 1)
        slices.append(slice(start, stop))
    return tuple(slices)  # type: ignore[return-value]


def _normalize_outside_organ_mask_mode(outside_organ_mask: str) -> str:
    normalized = str(outside_organ_mask).strip().lower()
    if normalized not in OUTSIDE_ORGAN_MASK_MODES:
        raise ValueError(
            f"Unknown outside-organ mask mode: {outside_organ_mask}. "
            f"Expected one of {OUTSIDE_ORGAN_MASK_MODES}."
        )
    return normalized


def _context_ablation_tag(bbox_margin: int, outside_organ_mask: str) -> str | None:
    normalized_mask = _normalize_outside_organ_mask_mode(outside_organ_mask)
    if int(bbox_margin) == DEFAULT_BBOX_MARGIN and normalized_mask == "none":
        return None
    return f"ctx_bbox{int(bbox_margin)}_mask{normalized_mask}"


def _resize_crop(
    image: np.ndarray,
    bbox: Tuple[slice, slice, slice],
    crop_size: int,
    mask: np.ndarray | None = None,
    outside_organ_mask: str = "none",
) -> torch.Tensor:
    outside_organ_mask = _normalize_outside_organ_mask_mode(outside_organ_mask)
    cropped = image[(slice(None),) + bbox]
    if outside_organ_mask == "zero":
        if mask is None:
            raise ValueError("mask_required_for_outside_organ_mask")
        cropped_mask = (mask[bbox] > 0).astype(cropped.dtype, copy=False)
        cropped = cropped * cropped_mask[None, ...]
    tensor = torch.from_numpy(cropped.copy()).unsqueeze(0)
    resized = F.interpolate(
        tensor.float(),
        size=(crop_size, crop_size, crop_size),
        mode="trilinear",
        align_corners=False,
    )
    return resized.squeeze(0)


def _group_samples_by_case(samples: Sequence[OrganSample]) -> Dict[Tuple[str, str], List[OrganSample]]:
    grouped: Dict[Tuple[str, str], List[OrganSample]] = defaultdict(list)
    for sample in samples:
        case_key = (sample.source_case_id, str(sample.file_path))
        grouped[case_key].append(sample)
    return grouped


def _flush_batch(
    extractor: Any,
    pending_tensors: List[torch.Tensor],
    pending_ids: List[str],
    feature_type: str,
    feature_rows: List[np.ndarray],
    feature_ids: List[str],
) -> None:
    if not pending_tensors:
        return
    batch = torch.stack(pending_tensors, dim=0)
    batch_features = extractor.extract(batch, normalize=True)[feature_type].cpu().numpy()
    feature_rows.extend(batch_features)
    feature_ids.extend(pending_ids)
    pending_tensors.clear()
    pending_ids.clear()


def _default_output_npz(
    manifest_path: Path,
    checkpoint_name: str,
    feature_type: str,
    context_ablation_tag: str | None = None,
) -> Path:
    analysis_name = get_dataset_name_from_manifest_path(manifest_path)
    manifest_variant = get_manifest_variant_from_manifest_path(manifest_path)
    base_dir = get_output_paths(analysis_name, manifest_variant)["features"]
    output_path = get_checkpoint_feature_npz_path(base_dir, checkpoint_name, feature_type)
    if context_ablation_tag:
        output_path = output_path.parent / context_ablation_tag / output_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _build_extraction_config(feature_type: str, batch_size: int, device: str) -> ExtractionConfig:
    feature_type = normalize_feature_type(feature_type)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    return ExtractionConfig(
        extract_cls=(feature_type == "cls"),
        extract_avg_pool=(feature_type == "avg_pool"),
        extract_multilayer=(feature_type == "multilayer"),
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )


def _resolve_crop_cache_dir(
    manifest_path: Path,
    crop_size: int,
    explicit_cache_dir: str | None,
    context_ablation_tag: str | None = None,
) -> Path:
    if explicit_cache_dir:
        return Path(explicit_cache_dir).resolve()
    analysis_name = get_dataset_name_from_manifest_path(manifest_path)
    manifest_variant = get_manifest_variant_from_manifest_path(manifest_path)
    cache_dir = get_crop_cache_dir(analysis_name, manifest_variant, crop_size)
    if context_ablation_tag:
        return cache_dir.parent / f"{cache_dir.name}__{context_ablation_tag}"
    return cache_dir


def _crop_cache_index_path(cache_dir: Path) -> Path:
    return cache_dir / CROP_CACHE_INDEX_NAME


def _crop_cache_cases_dir(cache_dir: Path) -> Path:
    return cache_dir / CROP_CACHE_CASES_DIRNAME


def _crop_cache_lock_path(cache_dir: Path) -> Path:
    return cache_dir / CROP_CACHE_LOCK_NAME


def _sample_signature(samples: Sequence[OrganSample]) -> str:
    fingerprints: Dict[str, Dict[str, Any]] = {}
    records = []
    for sample in samples:
        record = asdict(sample)
        for field in ("file_path", "mask_path"):
            path = Path(record[field]).expanduser().resolve()
            key = str(path)
            if key not in fingerprints:
                if path.is_dir():
                    files = []
                    for child in sorted(p for p in path.rglob("*") if p.is_file()):
                        stat = child.stat()
                        files.append([str(child.relative_to(path)), stat.st_size, stat.st_mtime_ns])
                    fingerprints[key] = {"path": key, "files": files}
                else:
                    stat = path.stat()
                    fingerprints[key] = {"path": key, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            record[field] = fingerprints[key]
        records.append(record)
    serialized_records = sorted(json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records)
    payload = {"version": CROP_CACHE_SAMPLE_SIGNATURE_VERSION, "samples": serialized_records}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _crop_cache_matches_request(
    index_payload: Dict[str, Any],
    samples: Sequence[OrganSample],
    manifest_path: Path,
    crop_size: int,
    bbox_margin: int,
    min_mask_voxels: int,
    outside_organ_mask: str,
) -> Tuple[bool, str | None]:
    expected_manifest_path = str(manifest_path)
    if str(index_payload.get("manifest_path")) != expected_manifest_path:
        return False, "manifest_path"

    if int(index_payload.get("crop_size", -1)) != int(crop_size):
        return False, "crop_size"
    if int(index_payload.get("bbox_margin", -1)) != int(bbox_margin):
        return False, "bbox_margin"
    if int(index_payload.get("min_mask_voxels", -1)) != int(min_mask_voxels):
        return False, "min_mask_voxels"
    if str(index_payload.get("outside_organ_mask", "")) != str(outside_organ_mask):
        return False, "outside_organ_mask"

    expected_sample_count = len(samples)
    if int(index_payload.get("n_requested_samples", -1)) != expected_sample_count:
        return False, "n_requested_samples"
    if expected_sample_count > 0 and int(index_payload.get("n_cached_samples", 0)) <= 0:
        return False, "n_cached_samples"

    expected_signature = _sample_signature(samples)
    cached_signature = index_payload.get("sample_signature")
    if str(cached_signature) != expected_signature:
        return False, "sample_signature"

    return True, None


def _read_crop_cache_lock_pid(lock_path: Path) -> int | None:
    try:
        return int(lock_path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, OSError, ValueError):
        return None


def _pid_is_running(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _clear_stale_crop_cache_lock(cache_dir: Path) -> bool:
    lock_path = _crop_cache_lock_path(cache_dir)
    if not lock_path.exists():
        return False

    lock_pid = _read_crop_cache_lock_pid(lock_path)
    if _pid_is_running(lock_pid):
        return False

    try:
        lock_path.unlink()
    except FileNotFoundError:
        return False

    logger.warning(
        "Removed stale crop-cache lock at %s (pid=%s)",
        lock_path,
        lock_pid if lock_pid is not None else "unknown",
    )
    return True


def _sanitize_token(value: str, max_length: int = 64) -> str:
    normalized = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    normalized = normalized.strip("_") or "case"
    return normalized[:max_length]


def _case_cache_path(cache_dir: Path, case_id: str, image_path: Path) -> Path:
    digest = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:12]
    file_name = f"{_sanitize_token(case_id)}_{digest}.pt"
    return _crop_cache_cases_dir(cache_dir) / file_name


def _load_case_crop_payload(
    image_path: Path,
    case_samples: Sequence[OrganSample],
    crop_size: int,
    bbox_margin: int,
    min_mask_voxels: int,
    outside_organ_mask: str,
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    case_image, perm = _load_case_image(image_path, case_samples[0].spacing)
    crop_tensors: List[torch.Tensor] = []
    sample_ids: List[str] = []
    failures: List[Dict[str, str]] = []

    for sample in case_samples:
        try:
            mask = _load_mask(sample, perm)
            voxel_count = int(mask.sum())
            if voxel_count < min_mask_voxels:
                raise ValueError(f"mask_too_small:{voxel_count}")
            if mask.shape != case_image.shape[1:]:
                raise ValueError(f"mask_shape_mismatch:{mask.shape}!={case_image.shape[1:]}")

            bbox = _get_bounding_box(mask, margin=bbox_margin)
            if bbox is None:
                raise ValueError("empty_mask")

            crop_tensors.append(
                _resize_crop(
                    case_image,
                    bbox,
                    crop_size,
                    mask=mask,
                    outside_organ_mask=outside_organ_mask,
                ).to(dtype=torch.float16)
            )
            sample_ids.append(sample.sample_id)
        except Exception as exc:
            failures.append({"sample_id": sample.sample_id, "reason": str(exc)})

    if crop_tensors:
        crops = torch.stack(crop_tensors, dim=0).contiguous()
    else:
        crops = torch.empty((0, 1, crop_size, crop_size, crop_size), dtype=torch.float16)

    payload = {
        "source_case_id": case_samples[0].source_case_id,
        "image_path": str(image_path),
        "sample_ids": sample_ids,
        "crops": crops,
        "outside_organ_mask": outside_organ_mask,
    }
    return payload, failures


def _write_case_crop_payload(cache_path: Path, payload: Dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(cache_path)


def _build_case_crop_cache(task: Tuple[Tuple[str, str], List[OrganSample], int, int, int, str, str, bool]) -> Dict[str, Any]:
    case_key, case_samples, crop_size, bbox_margin, min_mask_voxels, outside_organ_mask, cache_dir_str, overwrite = task
    source_case_id, image_path_str = case_key
    image_path = Path(image_path_str)
    cache_dir = Path(cache_dir_str)
    cache_path = _case_cache_path(cache_dir, source_case_id, image_path)
    requested_samples = len(case_samples)

    identity = {
        "sample_signature": _sample_signature(case_samples),
        "crop_size": crop_size,
        "bbox_margin": bbox_margin,
        "min_mask_voxels": min_mask_voxels,
        "outside_organ_mask": outside_organ_mask,
    }
    if cache_path.exists() and not overwrite:
        cached_payload = torch.load(cache_path, map_location="cpu")
        if cached_payload.get("cache_identity") == identity:
            return {
                "source_case_id": source_case_id,
                "n_requested_samples": requested_samples,
                "case_cache_path": str(cache_path),
                "n_cached_samples": len(cached_payload.get("sample_ids", [])),
                "n_failed_samples": 0,
                "failure_examples": [],
            }

    try:
        payload, failures = _load_case_crop_payload(
            image_path=image_path,
            case_samples=case_samples,
            crop_size=crop_size,
            bbox_margin=bbox_margin,
            min_mask_voxels=min_mask_voxels,
            outside_organ_mask=outside_organ_mask,
        )
    except Exception as exc:
        return {
            "source_case_id": source_case_id,
            "n_requested_samples": requested_samples,
            "case_cache_path": None,
            "n_cached_samples": 0,
            "n_failed_samples": len(case_samples),
            "failure_examples": [
                {"sample_id": sample.sample_id, "reason": f"image_load_failed: {exc}"}
                for sample in case_samples
            ],
        }

    if payload["sample_ids"]:
        payload["cache_identity"] = identity
        _write_case_crop_payload(cache_path, payload)
        case_cache_path = str(cache_path)
    else:
        case_cache_path = None

    return {
        "source_case_id": source_case_id,
        "n_requested_samples": requested_samples,
        "case_cache_path": case_cache_path,
        "n_cached_samples": len(payload["sample_ids"]),
        "n_failed_samples": len(failures),
        "failure_examples": failures[:10],
    }


def _wait_for_crop_cache(cache_dir: Path, timeout_seconds: int = 7200, poll_seconds: int = 5) -> Path:
    index_path = _crop_cache_index_path(cache_dir)
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if index_path.exists() and not _crop_cache_lock_path(cache_dir).exists():
            return index_path
        time.sleep(poll_seconds)
    raise TimeoutError(f"Timed out waiting for crop cache index at {index_path}")


def ensure_crop_cache(
    samples: Sequence[OrganSample],
    manifest_path: Path,
    crop_size: int,
    bbox_margin: int,
    min_mask_voxels: int,
    outside_organ_mask: str,
    cache_dir: Path,
    workers: int,
    overwrite: bool = False,
) -> Path:
    index_path = _crop_cache_index_path(cache_dir)
    if not overwrite:
        candidate_dirs = list(dict.fromkeys(get_reform3d_cache_path_candidates(cache_dir)))
        for candidate_dir in candidate_dirs:
            candidate_index_path = _crop_cache_index_path(candidate_dir)
            if not candidate_index_path.exists():
                continue
            try:
                index_payload = json.loads(candidate_index_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                if candidate_dir == cache_dir:
                    logger.warning("Ignoring unreadable crop-cache index at %s: %s", candidate_index_path, exc)
                    overwrite = True
                continue
            is_compatible, mismatch_key = _crop_cache_matches_request(
                index_payload=index_payload,
                samples=samples,
                manifest_path=manifest_path,
                crop_size=crop_size,
                bbox_margin=bbox_margin,
                min_mask_voxels=min_mask_voxels,
                outside_organ_mask=outside_organ_mask,
            )
            if is_compatible:
                return candidate_index_path
            if candidate_dir == cache_dir:
                logger.warning(
                    "Rebuilding incompatible crop cache at %s due to %s mismatch",
                    cache_dir,
                    mismatch_key,
                )
                overwrite = True

    cache_dir.mkdir(parents=True, exist_ok=True)
    _crop_cache_cases_dir(cache_dir).mkdir(parents=True, exist_ok=True)

    lock_path = _crop_cache_lock_path(cache_dir)
    lock_fd: int | None = None
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(lock_fd, str(os.getpid()).encode("utf-8"))
        os.close(lock_fd)
        lock_fd = None
    except FileExistsError:
        if not index_path.exists() and _clear_stale_crop_cache_lock(cache_dir):
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(lock_fd, str(os.getpid()).encode("utf-8"))
            os.close(lock_fd)
            lock_fd = None
        else:
            logger.info("Waiting for existing crop-cache build at %s", cache_dir)
            completed_index = _wait_for_crop_cache(cache_dir)
            completed_payload = json.loads(completed_index.read_text(encoding="utf-8"))
            compatible, mismatch = _crop_cache_matches_request(
                completed_payload, samples, manifest_path, crop_size, bbox_margin,
                min_mask_voxels, outside_organ_mask,
            )
            if not compatible:
                raise ValueError(f"Concurrent crop cache is incompatible: {mismatch}; use a separate cache directory")
            return completed_index

    grouped_items = list(_group_samples_by_case(samples).items())
    worker_count = max(1, int(workers))
    logger.info(
        "Building crop cache at %s for crop_size=%d across %d cases and %d samples using %d worker(s)",
        cache_dir,
        crop_size,
        len(grouped_items),
        len(samples),
        worker_count,
    )

    try:
        tasks = [
            (
                case_key,
                case_samples,
                crop_size,
                bbox_margin,
                min_mask_voxels,
                outside_organ_mask,
                str(cache_dir),
                overwrite,
            )
            for case_key, case_samples in grouped_items
        ]

        if worker_count == 1:
            iterator = (_build_case_crop_cache(task) for task in tasks)
        else:
            executor = ProcessPoolExecutor(max_workers=worker_count)
            iterator = executor.map(_build_case_crop_cache, tasks)

        case_entries: List[Dict[str, Any]] = []
        failure_examples: List[Dict[str, str]] = []
        total_cached_samples = 0
        total_failed_samples = 0

        progress = tqdm(
            total=len(samples),
            desc=f"Building crop{crop_size} cache",
            unit="sample",
            dynamic_ncols=True,
        )
        try:
            for result in iterator:
                case_entries.append(
                    {
                        "source_case_id": result["source_case_id"],
                        "case_cache_path": result["case_cache_path"],
                        "n_cached_samples": result["n_cached_samples"],
                        "n_failed_samples": result["n_failed_samples"],
                    }
                )
                total_cached_samples += int(result["n_cached_samples"])
                total_failed_samples += int(result["n_failed_samples"])
                if len(failure_examples) < 100:
                    remaining = 100 - len(failure_examples)
                    failure_examples.extend(result["failure_examples"][:remaining])
                progress.update(int(result.get("n_requested_samples", 0)))
                progress.set_postfix(cached=total_cached_samples, failed=total_failed_samples)
        finally:
            progress.close()
            if worker_count > 1:
                executor.shutdown(wait=True)

        summary = {
            "manifest_path": str(manifest_path),
            "crop_size": crop_size,
            "bbox_margin": bbox_margin,
            "min_mask_voxels": min_mask_voxels,
            "outside_organ_mask": outside_organ_mask,
            "sample_signature": _sample_signature(samples),
            "sample_signature_version": CROP_CACHE_SAMPLE_SIGNATURE_VERSION,
            "workers": worker_count,
            "cache_dir": str(cache_dir),
            "n_requested_samples": len(samples),
            "n_cached_samples": total_cached_samples,
            "n_failed_samples": total_failed_samples,
            "cases": case_entries,
            "failure_examples": failure_examples,
        }
        _write_json(index_path, summary)
        return index_path
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _extract_from_crop_cache(
    extractor: Any,
    cache_dir: Path,
    batch_size: int,
    feature_type: str,
    feature_rows: List[np.ndarray],
    feature_ids: List[str],
) -> List[Dict[str, str]]:
    index = json.loads(_crop_cache_index_path(cache_dir).read_text(encoding="utf-8"))
    failures = list(index.get("failure_examples", []))
    pending_tensors: List[torch.Tensor] = []
    pending_ids: List[str] = []

    valid_cases = [entry for entry in index.get("cases", []) if entry.get("case_cache_path")]
    total_cached_samples = sum(int(entry.get("n_cached_samples", 0)) for entry in valid_cases)
    progress = tqdm(
        total=total_cached_samples,
        desc=f"Extracting organ {feature_type} features from crop cache",
        unit="sample",
        dynamic_ncols=True,
    )
    for entry in valid_cases:
        payload = torch.load(entry["case_cache_path"], map_location="cpu")
        crops = payload["crops"]
        sample_ids = payload["sample_ids"]
        for crop, sample_id in zip(crops, sample_ids):
            pending_tensors.append(crop)
            pending_ids.append(sample_id)
            progress.update(1)
            if len(pending_tensors) >= batch_size:
                _flush_batch(extractor, pending_tensors, pending_ids, feature_type, feature_rows, feature_ids)
        progress.set_postfix(extracted=len(feature_ids), failed=len(failures))

    _flush_batch(extractor, pending_tensors, pending_ids, feature_type, feature_rows, feature_ids)
    progress.close()
    return failures


def require_complete_extraction(
    samples: Sequence[OrganSample],
    feature_ids: Sequence[str],
    failures: Sequence[Dict[str, str]],
    summary_path: Path,
) -> None:
    requested = {sample.sample_id for sample in samples}
    extracted = set(feature_ids)
    complete = (
        not failures
        and len(requested) == len(samples)
        and len(extracted) == len(feature_ids)
        and requested == extracted
    )
    if complete:
        return
    summary = {
        "status": "incomplete",
        "n_requested_samples": len(samples),
        "n_extracted_samples": len(feature_ids),
        "n_failed_samples": len(failures),
        "missing_sample_ids": sorted(requested - extracted),
        "unexpected_sample_ids": sorted(extracted - requested),
        "duplicate_manifest_ids": len(samples) - len(requested),
        "duplicate_extracted_ids": len(feature_ids) - len(extracted),
        "failures": list(failures),
        "output_npz": None,
    }
    _write_json(summary_path, summary)
    raise RuntimeError(
        "Incomplete feature extraction; no feature bundle was written by this run. "
        f"See {summary_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract organ-aware Cross-Modal embeddings for one feature family",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m reform3d.modules.cross_modal_alignment.feature_extractor -m amos22_ct_mr/core/manifest_sampled.json -c med3dino_rel_96\n"
            "  python -m reform3d.modules.cross_modal_alignment.feature_extractor -m amos22_ct_mr/core/manifest_sampled.json -c med3dino_rel_96 --max-samples 32 --batch-size 4\n"
        ),
    )
    parser.add_argument("-m", "--manifest", required=True, help="Path to Cross-Modal manifest JSON")
    parser.add_argument("-c", "--checkpoint", default="med3dino_rel_96", help="Checkpoint key or alias from the REFORM-3D registry")
    parser.add_argument(
        "--feature-type",
        default=DEFAULT_FEATURE_TYPE,
        choices=list(FEATURE_TYPES),
        help="Feature family to extract",
    )
    parser.add_argument("--crop-size", type=int, default=None, help="Optional explicit crop size for crop-cache-only runs")
    parser.add_argument("-o", "--output-npz", default=None, help="Optional output NPZ path")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--bbox-margin", type=int, default=DEFAULT_BBOX_MARGIN, help="Bounding-box margin in voxels")
    parser.add_argument(
        "--outside-organ-mask",
        default="none",
        choices=list(OUTSIDE_ORGAN_MASK_MODES),
        help="Optional outside-organ masking inside the crop bbox. Use 'zero' for context ablation.",
    )
    parser.add_argument("--min-mask-voxels", type=int, default=100, help="Skip masks smaller than this voxel count")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for smoke runs")
    parser.add_argument("--crop-cache-dir", default=None, help="Optional crop cache directory override")
    parser.add_argument("--crop-cache-workers", type=int, default=4, help="Workers used when building the crop cache")
    parser.add_argument("--overwrite-crop-cache", action="store_true", help="Rebuild crop cache even if it already exists")
    parser.add_argument("--disable-crop-cache", action="store_true", help="Disable crop-cache reuse and fall back to direct preprocessing")
    parser.add_argument("--build-crop-cache-only", action="store_true", help="Build the crop cache and exit without running model inference")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device to use. Use cpu only for debugging; the extractor is optimized for CUDA.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    feature_type = normalize_feature_type(args.feature_type)
    samples = _load_manifest_samples(manifest_path, max_samples=args.max_samples)
    if not samples:
        raise ValueError(f"No samples loaded from {manifest_path}")

    if args.crop_size is not None:
        crop_size = int(args.crop_size)
    else:
        crop_size = int(resolve_checkpoint_config(args.checkpoint).crop_size)

    outside_organ_mask = _normalize_outside_organ_mask_mode(args.outside_organ_mask)
    context_ablation_tag = _context_ablation_tag(args.bbox_margin, outside_organ_mask)

    crop_cache_dir: Path | None = None
    if not args.disable_crop_cache or args.build_crop_cache_only:
        crop_cache_dir = _resolve_crop_cache_dir(
            manifest_path,
            crop_size,
            args.crop_cache_dir,
            context_ablation_tag=context_ablation_tag,
        )
        crop_cache_index = ensure_crop_cache(
            samples=samples,
            manifest_path=manifest_path,
            crop_size=crop_size,
            bbox_margin=args.bbox_margin,
            min_mask_voxels=args.min_mask_voxels,
            outside_organ_mask=outside_organ_mask,
            cache_dir=crop_cache_dir,
            workers=args.crop_cache_workers,
            overwrite=args.overwrite_crop_cache,
        )
        crop_cache_dir = crop_cache_index.parent
        logger.info("Crop cache ready at %s", crop_cache_dir)

    if args.build_crop_cache_only:
        return

    device = args.device
    extraction_config = _build_extraction_config(
        feature_type=feature_type,
        batch_size=args.batch_size,
        device=device,
    )
    extractor = FeatureExtractor(args.checkpoint, config=extraction_config, device=device)
    crop_size = int(extractor.config.input_size)

    output_npz = (
        Path(args.output_npz).resolve()
        if args.output_npz
        else _default_output_npz(
            manifest_path,
            args.checkpoint,
            feature_type,
            context_ablation_tag=context_ablation_tag,
        )
    )
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_npz.with_name(output_npz.stem + "_summary.json")

    feature_rows: List[np.ndarray] = []
    feature_ids: List[str] = []
    failures: List[Dict[str, str]] = []

    grouped_samples = _group_samples_by_case(samples)
    logger.info("Loaded %d organ samples across %d source cases", len(samples), len(grouped_samples))
    logger.info(
        "Checkpoint=%s feature_type=%s crop_size=%d bbox_margin=%d outside_organ_mask=%s output=%s",
        args.checkpoint,
        feature_type,
        crop_size,
        args.bbox_margin,
        outside_organ_mask,
        output_npz,
    )

    if crop_cache_dir and _crop_cache_index_path(crop_cache_dir).exists():
        failures = _extract_from_crop_cache(
            extractor=extractor,
            cache_dir=crop_cache_dir,
            batch_size=args.batch_size,
            feature_type=feature_type,
            feature_rows=feature_rows,
            feature_ids=feature_ids,
        )
    else:
        pending_tensors: List[torch.Tensor] = []
        pending_ids: List[str] = []
        progress = tqdm(
            total=len(samples),
            desc=f"Extracting organ {feature_type} features",
            unit="sample",
            dynamic_ncols=True,
        )
        for (_, image_path_str), case_samples in grouped_samples.items():
            image_path = Path(image_path_str)
            try:
                case_image, perm = _load_case_image(image_path, case_samples[0].spacing)
            except Exception as exc:
                for sample in case_samples:
                    failures.append({"sample_id": sample.sample_id, "reason": f"image_load_failed: {exc}"})
                    progress.update(1)
                logger.warning("Failed to load %s: %s", image_path, exc)
                continue

            for sample in case_samples:
                try:
                    mask = _load_mask(sample, perm)
                    voxel_count = int(mask.sum())
                    if voxel_count < args.min_mask_voxels:
                        raise ValueError(f"mask_too_small:{voxel_count}")
                    if mask.shape != case_image.shape[1:]:
                        raise ValueError(f"mask_shape_mismatch:{mask.shape}!={case_image.shape[1:]}")

                    bbox = _get_bounding_box(mask, margin=args.bbox_margin)
                    if bbox is None:
                        raise ValueError("empty_mask")

                    pending_tensors.append(
                        _resize_crop(
                            case_image,
                            bbox,
                            crop_size,
                            mask=mask,
                            outside_organ_mask=outside_organ_mask,
                        )
                    )
                    pending_ids.append(sample.sample_id)
                    if len(pending_tensors) >= args.batch_size:
                        _flush_batch(extractor, pending_tensors, pending_ids, feature_type, feature_rows, feature_ids)
                except Exception as exc:
                    failures.append({"sample_id": sample.sample_id, "reason": str(exc)})
                finally:
                    progress.update(1)

            progress.set_postfix(extracted=len(feature_ids), failed=len(failures))

        _flush_batch(extractor, pending_tensors, pending_ids, feature_type, feature_rows, feature_ids)
        progress.close()

    require_complete_extraction(samples, feature_ids, failures, summary_path)

    if not feature_rows:
        raise RuntimeError("No features were extracted successfully")

    feature_matrix = np.stack(feature_rows, axis=0).astype(np.float32, copy=False)
    sample_ids = np.asarray(feature_ids, dtype=object)
    temporary_npz = output_npz.with_name(output_npz.stem + ".tmp.npz")
    np.savez_compressed(temporary_npz, features=feature_matrix, sample_ids=sample_ids)
    temporary_npz.replace(output_npz)

    summary = {
        "status": "complete",
        "manifest_path": str(manifest_path),
        "checkpoint": args.checkpoint,
        "feature_type": feature_type,
        "device": device,
        "crop_size": crop_size,
        "bbox_margin": args.bbox_margin,
        "outside_organ_mask": outside_organ_mask,
        "context_ablation_tag": context_ablation_tag,
        "min_mask_voxels": args.min_mask_voxels,
        "n_requested_samples": len(samples),
        "n_extracted_samples": len(feature_ids),
        "n_failed_samples": len(failures),
        "crop_cache_dir": str(crop_cache_dir) if crop_cache_dir else None,
        "output_npz": str(output_npz),
        "failure_examples": failures[:25],
        "feature_shape": list(feature_matrix.shape),
    }
    _write_json(summary_path, summary)

    logger.info("Wrote %d organ %s embeddings to %s", len(feature_ids), feature_type, output_npz)
    logger.info("Wrote extraction summary to %s", summary_path)


if __name__ == "__main__":
    main()
