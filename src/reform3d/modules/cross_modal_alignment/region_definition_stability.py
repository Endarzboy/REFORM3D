"""Region-definition stability evaluation for Cross-Modal representations."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing
import os
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from reform3d.core.checkpoints import resolve_checkpoint_config
from reform3d.modules.acquisition_robustness.feature_extractor import (
    ExtractionConfig,
    FeatureExtractor,
)

from .config import (
    CROSS_MODAL_CACHES,
    CROSS_MODAL_OUTPUTS,
    get_cross_modal_manifest_path,
    get_crop_cache_dir,
    normalize_cross_modal_dataset_name,
)
from .feature_extractor import (
    CROP_CACHE_INDEX_NAME,
    OrganSample,
    _binarize_mask,
    _crop_cache_matches_request,
    _get_bounding_box,
    _group_samples_by_case,
    _load_binary_mask_volume,
    _load_case_image,
    _load_manifest_samples,
    _load_png_mask_stack,
    _resize_crop,
    _sample_signature,
    _uses_png_mask_stack,
)
from .physical_mask import erode_binary_mask_mm, permute_spatial_mask
from .protocol_sampling import (
    build_anatomical_plan,
    build_retrieval_plan,
    evaluate_anatomical_plans,
    evaluate_retrieval_plan,
    validate_anatomical_plan,
    validate_retrieval_plan,
)


LOGGER = logging.getLogger(__name__)

REGION_SCHEMA_VERSION = 1
REGION_PREPROCESS_VERSION = "region_definition_v1"
SUPPORTED_DATASETS = ("amos22_ct_mr", "mmwhs_ct_mr")
DEFAULT_CHECKPOINTS = (
    "med3dino_iso_96",
    "med3dino_rel_96",
    "med3dino_sa_96",
)
DEFAULT_FEATURE_TYPES = ("cls", "avg_pool")
DEFAULT_SEEDS = (42, 123, 456)
DEFAULT_CROP_SIZE = 96
DEFAULT_MIN_MASK_VOXELS = 100
DEFAULT_SUPPORT_PER_MODALITY = 2


@dataclass(frozen=True)
class RegionSurface:
    name: str
    bbox_margin: int
    outside_organ_mask: str
    erosion_radius_mm: float | None = None


REGION_SURFACES = (
    RegionSurface("baseline_full_bbox_context", 5, "none"),
    RegionSurface("tight_bbox0_context", 0, "none"),
    RegionSurface("organ_only_bbox5", 5, "zero"),
    RegionSurface("organ_only_bbox0", 0, "zero"),
    RegionSurface("eroded_1mm_bbox5", 5, "zero", 1.0),
)
REGION_SURFACE_BY_NAME = {surface.name: surface for surface in REGION_SURFACES}
BASELINE_SURFACE = REGION_SURFACES[0]


def _initialize_region_cache_worker() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _create_region_cache_executor(workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=int(workers),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_region_cache_worker,
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary_path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        torch.save(dict(payload), temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _surface_signature(
    surface: RegionSurface,
    *,
    manifest_sha256: str,
    sample_signature: str,
) -> str:
    return _payload_sha256(
        {
            "schema_version": REGION_SCHEMA_VERSION,
            "preprocess_version": REGION_PREPROCESS_VERSION,
            "manifest_sha256": manifest_sha256,
            "sample_signature": sample_signature,
            "crop_size": DEFAULT_CROP_SIZE,
            "min_mask_voxels": DEFAULT_MIN_MASK_VOXELS,
            "surface": asdict(surface),
        }
    )


def _result_root(dataset: str) -> Path:
    return (
        CROSS_MODAL_OUTPUTS
        / "protocol_stability"
        / "region_definition"
        / dataset
        / "core"
    )


def _cache_root(dataset: str) -> Path:
    return (
        CROSS_MODAL_CACHES
        / "cross_modal_alignment"
        / dataset
        / "core"
        / "protocol_stability"
        / "region_definition"
        / "crop96"
    )


def _surface_cache_dir(dataset: str, surface_name: str) -> Path:
    return _cache_root(dataset) / surface_name


def _surface_index_path(dataset: str, surface_name: str) -> Path:
    return _surface_cache_dir(dataset, surface_name) / CROP_CACHE_INDEX_NAME


def _surface_case_path(
    dataset: str,
    surface_name: str,
    source_case_id: str,
    image_path: Path,
    cache_root_override: Path | None = None,
) -> Path:
    token = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in source_case_id
    ).strip("_")[:64] or "case"
    digest = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:12]
    cache_root = cache_root_override or _cache_root(dataset)
    return (
        cache_root
        / surface_name
        / "cases"
        / f"{token}_{digest}.pt"
    )


def _feature_bundle_path(
    dataset: str,
    surface_name: str,
    checkpoint: str,
) -> Path:
    return (
        _result_root(dataset)
        / "features"
        / surface_name
        / checkpoint
        / "organ_features.npz"
    )


def _feature_summary_path(
    dataset: str,
    surface_name: str,
    checkpoint: str,
) -> Path:
    return _feature_bundle_path(dataset, surface_name, checkpoint).with_name(
        "organ_features_summary.json"
    )


def _plan_dir(dataset: str) -> Path:
    return _result_root(dataset) / "sampling_plans"


def _evaluation_dir(
    dataset: str,
    surface_name: str,
    checkpoint: str,
    feature_type: str,
) -> Path:
    return (
        _result_root(dataset)
        / "evaluations"
        / surface_name
        / checkpoint
        / feature_type
    )


def _sample_records(samples: Sequence[OrganSample]) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": sample.sample_id,
            "file_path": str(sample.file_path),
            "mask_path": str(sample.mask_path),
            "modality": sample.modality,
            "primary_organ": sample.primary_organ,
            "source_case_id": sample.source_case_id,
        }
        for sample in samples
    ]


def _load_raw_mask(sample: OrganSample) -> np.ndarray:
    if _uses_png_mask_stack(sample.mask_loader):
        return _load_png_mask_stack(sample.mask_path)
    return _load_binary_mask_volume(sample.mask_path)


def _validate_case_payload(
    path: Path,
    *,
    expected_sample_ids: Sequence[str],
    surface_signature: str | None,
    allow_legacy_reference: bool,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    sample_ids = [str(value) for value in payload.get("sample_ids", [])]
    if sample_ids != [str(value) for value in expected_sample_ids]:
        raise ValueError(f"Case cache sample IDs do not match at {path}")
    crops = payload.get("crops")
    if not isinstance(crops, torch.Tensor):
        raise ValueError(f"Case cache has no crop tensor at {path}")
    if tuple(crops.shape) != (
        len(sample_ids),
        1,
        DEFAULT_CROP_SIZE,
        DEFAULT_CROP_SIZE,
        DEFAULT_CROP_SIZE,
    ):
        raise ValueError(f"Unexpected crop tensor shape {tuple(crops.shape)} at {path}")
    if not torch.isfinite(crops).all():
        raise ValueError(f"Nonfinite crop tensor at {path}")
    if not allow_legacy_reference:
        if str(payload.get("surface_signature")) != str(surface_signature):
            raise ValueError(f"Surface signature mismatch at {path}")
    return payload


def _case_cache_is_valid(
    path: Path,
    *,
    expected_sample_ids: Sequence[str],
    surface_signature: str,
) -> bool:
    try:
        _validate_case_payload(
            path,
            expected_sample_ids=expected_sample_ids,
            surface_signature=surface_signature,
            allow_legacy_reference=False,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return False
    return True


def _build_case_surfaces(
    task: tuple[
        str,
        tuple[str, str],
        list[OrganSample],
        tuple[RegionSurface, ...],
        dict[str, str],
        bool,
        str | None,
    ],
) -> dict[str, Any]:
    (
        dataset,
        case_key,
        case_samples,
        surfaces,
        signatures,
        overwrite,
        cache_root_override_text,
    ) = task
    source_case_id, image_path_text = case_key
    image_path = Path(image_path_text)
    cache_root_override = (
        Path(cache_root_override_text)
        if cache_root_override_text is not None
        else None
    )
    sample_ids = [sample.sample_id for sample in case_samples]
    output_paths = {
        surface.name: _surface_case_path(
            dataset,
            surface.name,
            source_case_id,
            image_path,
            cache_root_override=cache_root_override,
        )
        for surface in surfaces
    }
    missing_surfaces = [
        surface
        for surface in surfaces
        if overwrite
        or not _case_cache_is_valid(
            output_paths[surface.name],
            expected_sample_ids=sample_ids,
            surface_signature=signatures[surface.name],
        )
    ]
    if not missing_surfaces:
        return {
            "source_case_id": source_case_id,
            "image_path": str(image_path),
            "n_requested_samples": len(case_samples),
            "surfaces": {
                surface.name: {
                    "case_cache_path": str(output_paths[surface.name]),
                    "n_cached_samples": len(case_samples),
                    "n_failed_samples": 0,
                    "failure_examples": [],
                }
                for surface in surfaces
            },
        }

    try:
        case_image, permutation = _load_case_image(
            image_path,
            case_samples[0].spacing,
        )
    except Exception as exc:
        reason = f"image_load_failed:{exc}"
        return {
            "source_case_id": source_case_id,
            "image_path": str(image_path),
            "n_requested_samples": len(case_samples),
            "surfaces": {
                surface.name: {
                    "case_cache_path": None,
                    "n_cached_samples": 0,
                    "n_failed_samples": len(case_samples),
                    "failure_examples": [
                        {"sample_id": sample.sample_id, "reason": reason}
                        for sample in case_samples
                    ],
                }
                for surface in surfaces
            },
        }

    raw_masks: dict[tuple[str, str], np.ndarray] = {}
    crops: dict[str, list[torch.Tensor]] = {
        surface.name: [] for surface in missing_surfaces
    }
    crop_sample_ids: dict[str, list[str]] = {
        surface.name: [] for surface in missing_surfaces
    }
    failures: dict[str, list[dict[str, str]]] = {
        surface.name: [] for surface in missing_surfaces
    }
    for sample in case_samples:
        try:
            mask_key = (str(sample.mask_path), str(sample.mask_loader or ""))
            if mask_key not in raw_masks:
                raw_masks[mask_key] = _load_raw_mask(sample)
            original_raw = _binarize_mask(raw_masks[mask_key], sample)
            original_voxels = int(original_raw.sum())
            if original_voxels < DEFAULT_MIN_MASK_VOXELS:
                raise ValueError(f"mask_too_small:{original_voxels}")
            original_mask = permute_spatial_mask(original_raw, permutation)
            if original_mask.shape != case_image.shape[1:]:
                raise ValueError(
                    f"mask_shape_mismatch:{original_mask.shape}!={case_image.shape[1:]}"
                )
            eroded_raw: np.ndarray | None = None
            for surface in missing_surfaces:
                try:
                    content_mask = original_mask
                    if surface.erosion_radius_mm is not None:
                        if sample.spacing is None:
                            raise ValueError("missing_spacing_for_physical_erosion")
                        if eroded_raw is None:
                            eroded_raw = erode_binary_mask_mm(
                                original_raw,
                                sample.spacing,
                                surface.erosion_radius_mm,
                            )
                        if not np.any(eroded_raw):
                            raise ValueError("empty_transformed_mask")
                        content_mask = permute_spatial_mask(
                            eroded_raw,
                            permutation,
                        )
                    bbox = _get_bounding_box(
                        original_mask,
                        margin=surface.bbox_margin,
                    )
                    if bbox is None:
                        raise ValueError("empty_original_mask")
                    crop = _resize_crop(
                        case_image,
                        bbox,
                        DEFAULT_CROP_SIZE,
                        mask=content_mask,
                        outside_organ_mask=surface.outside_organ_mask,
                    ).to(dtype=torch.float16)
                    if not torch.isfinite(crop).all():
                        raise ValueError("nonfinite_crop")
                    crops[surface.name].append(crop)
                    crop_sample_ids[surface.name].append(sample.sample_id)
                except Exception as exc:
                    failures[surface.name].append(
                        {"sample_id": sample.sample_id, "reason": str(exc)}
                    )
        except Exception as exc:
            for surface in missing_surfaces:
                failures[surface.name].append(
                    {"sample_id": sample.sample_id, "reason": str(exc)}
                )

    surface_results: dict[str, dict[str, Any]] = {}
    for surface in surfaces:
        output_path = output_paths[surface.name]
        if surface not in missing_surfaces:
            surface_results[surface.name] = {
                "case_cache_path": str(output_path),
                "n_cached_samples": len(case_samples),
                "n_failed_samples": 0,
                "failure_examples": [],
            }
            continue
        surface_failures = failures[surface.name]
        if crops[surface.name]:
            payload = {
                "schema_version": REGION_SCHEMA_VERSION,
                "preprocess_version": REGION_PREPROCESS_VERSION,
                "surface": asdict(surface),
                "surface_signature": signatures[surface.name],
                "source_case_id": source_case_id,
                "image_path": str(image_path),
                "sample_ids": crop_sample_ids[surface.name],
                "crops": torch.stack(crops[surface.name], dim=0).contiguous(),
            }
            _atomic_torch_save(output_path, payload)
            cache_path: str | None = str(output_path)
        else:
            cache_path = None
        surface_results[surface.name] = {
            "case_cache_path": cache_path,
            "n_cached_samples": len(crop_sample_ids[surface.name]),
            "n_failed_samples": len(surface_failures),
            "failure_examples": surface_failures[:20],
        }
    return {
        "source_case_id": source_case_id,
        "image_path": str(image_path),
        "n_requested_samples": len(case_samples),
        "surfaces": surface_results,
    }


def _create_baseline_reference(
    dataset: str,
    *,
    samples: Sequence[OrganSample],
    manifest_path: Path,
    manifest_sha256: str,
    sample_signature: str,
    overwrite: bool,
) -> bool:
    source_dir = get_crop_cache_dir(dataset, "core", DEFAULT_CROP_SIZE)
    source_index_path = source_dir / CROP_CACHE_INDEX_NAME
    if not source_index_path.exists():
        return False
    try:
        source_index = json.loads(source_index_path.read_text(encoding="utf-8"))
        compatible, _ = _crop_cache_matches_request(
            source_index,
            samples,
            manifest_path,
            DEFAULT_CROP_SIZE,
            BASELINE_SURFACE.bbox_margin,
            DEFAULT_MIN_MASK_VOXELS,
            BASELINE_SURFACE.outside_organ_mask,
        )
        if not compatible:
            return False
        expected_by_case = {
            case_key[0]: [sample.sample_id for sample in case_samples]
            for case_key, case_samples in _group_samples_by_case(samples).items()
        }
        reference_cases: list[dict[str, Any]] = []
        seen_ids: list[str] = []
        for entry in source_index.get("cases", []):
            case_path_text = entry.get("case_cache_path")
            source_case_id = str(entry.get("source_case_id"))
            if not case_path_text or source_case_id not in expected_by_case:
                raise ValueError("Baseline index contains an unexpected case entry")
            case_path = Path(case_path_text)
            payload = _validate_case_payload(
                case_path,
                expected_sample_ids=expected_by_case[source_case_id],
                surface_signature=None,
                allow_legacy_reference=True,
            )
            seen_ids.extend(str(value) for value in payload["sample_ids"])
            reference_cases.append(
                {
                    "source_case_id": source_case_id,
                    "case_cache_path": str(case_path),
                    "n_cached_samples": len(payload["sample_ids"]),
                    "n_failed_samples": 0,
                }
            )
        expected_ids = [sample.sample_id for sample in samples]
        if sorted(seen_ids) != sorted(expected_ids) or len(seen_ids) != len(expected_ids):
            raise ValueError("Baseline cache does not cover the exact manifest cohort")
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        LOGGER.warning("Cannot reuse baseline cache at %s: %s", source_dir, exc)
        return False

    destination = _surface_index_path(dataset, BASELINE_SURFACE.name)
    if destination.exists() and not overwrite:
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if (
                str(existing.get("surface_signature"))
                == _surface_signature(
                    BASELINE_SURFACE,
                    manifest_sha256=manifest_sha256,
                    sample_signature=sample_signature,
                )
                and str(existing.get("source_cache_index")) == str(source_index_path)
            ):
                return True
        except (OSError, json.JSONDecodeError):
            pass
    _atomic_write_json(
        destination,
        {
            "schema_version": REGION_SCHEMA_VERSION,
            "preprocess_version": REGION_PREPROCESS_VERSION,
            "cache_mode": "validated_reference",
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "sample_signature": sample_signature,
            "surface": asdict(BASELINE_SURFACE),
            "surface_signature": _surface_signature(
                BASELINE_SURFACE,
                manifest_sha256=manifest_sha256,
                sample_signature=sample_signature,
            ),
            "crop_size": DEFAULT_CROP_SIZE,
            "min_mask_voxels": DEFAULT_MIN_MASK_VOXELS,
            "n_requested_samples": len(samples),
            "n_cached_samples": len(samples),
            "n_failed_samples": 0,
            "source_cache_index": str(source_index_path),
            "source_cache_preprocess_version": source_index.get("preprocess_version"),
            "source_cache_sample_signature_version": (
                source_index.get("sample_signature_version")
            ),
            "cases": reference_cases,
            "failure_examples": [],
        },
    )
    return True


def validate_surface_cache(
    dataset: str,
    surface: RegionSurface,
    *,
    samples: Sequence[OrganSample],
    manifest_path: Path,
) -> dict[str, Any]:
    index_path = _surface_index_path(dataset, surface.name)
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    manifest_sha256 = _sha256_file(manifest_path)
    sample_signature = _sample_signature(samples)
    expected_signature = _surface_signature(
        surface,
        manifest_sha256=manifest_sha256,
        sample_signature=sample_signature,
    )
    if str(payload.get("preprocess_version")) != REGION_PREPROCESS_VERSION:
        raise ValueError(f"Stale region cache preprocessing at {index_path}")
    if str(payload.get("surface_signature")) != expected_signature:
        raise ValueError(f"Region cache signature mismatch at {index_path}")
    if int(payload.get("n_requested_samples", -1)) != len(samples):
        raise ValueError(f"Region cache requested count mismatch at {index_path}")
    if int(payload.get("n_cached_samples", -1)) != len(samples):
        raise ValueError(f"Region cache coverage mismatch at {index_path}")
    if int(payload.get("n_failed_samples", -1)) != 0:
        raise ValueError(f"Region cache has failures at {index_path}")

    expected_by_case = {
        case_key[0]: [sample.sample_id for sample in case_samples]
        for case_key, case_samples in _group_samples_by_case(samples).items()
    }
    seen_ids: list[str] = []
    allow_reference = payload.get("cache_mode") == "validated_reference"
    for entry in payload.get("cases", []):
        source_case_id = str(entry["source_case_id"])
        if source_case_id not in expected_by_case:
            raise ValueError(f"Unexpected source case {source_case_id} at {index_path}")
        case_payload = _validate_case_payload(
            Path(entry["case_cache_path"]),
            expected_sample_ids=expected_by_case[source_case_id],
            surface_signature=expected_signature,
            allow_legacy_reference=allow_reference,
        )
        seen_ids.extend(str(value) for value in case_payload["sample_ids"])
    expected_ids = [sample.sample_id for sample in samples]
    if len(seen_ids) != len(expected_ids) or sorted(seen_ids) != sorted(expected_ids):
        raise ValueError(f"Region cache does not exactly cover the manifest at {index_path}")
    return payload


def _prepare_region_caches_unlocked(
    dataset: str,
    *,
    workers: int,
    overwrite: bool = False,
    samples_override: Sequence[OrganSample] | None = None,
    manifest_override: Path | None = None,
) -> dict[str, Path]:
    dataset = normalize_cross_modal_dataset_name(dataset)
    manifest_path = manifest_override or get_cross_modal_manifest_path(
        dataset,
        "sampled",
        "core",
    )
    samples = list(samples_override or _load_manifest_samples(manifest_path))
    if not samples:
        raise ValueError(f"No samples loaded from {manifest_path}")
    manifest_sha256 = _sha256_file(manifest_path)
    sample_signature = _sample_signature(samples)
    baseline_reused = _create_baseline_reference(
        dataset,
        samples=samples,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        sample_signature=sample_signature,
        overwrite=overwrite,
    )
    surfaces_to_build = tuple(
        surface
        for surface in REGION_SURFACES
        if surface != BASELINE_SURFACE or not baseline_reused
    )
    signatures = {
        surface.name: _surface_signature(
            surface,
            manifest_sha256=manifest_sha256,
            sample_signature=sample_signature,
        )
        for surface in REGION_SURFACES
    }
    grouped = list(_group_samples_by_case(samples).items())
    tasks = [
        (
            dataset,
            case_key,
            case_samples,
            surfaces_to_build,
            signatures,
            overwrite,
            None,
        )
        for case_key, case_samples in grouped
    ]
    if surfaces_to_build:
        if int(workers) <= 1:
            iterator = (_build_case_surfaces(task) for task in tasks)
            executor = None
        else:
            executor = _create_region_cache_executor(workers)
            iterator = executor.map(_build_case_surfaces, tasks)
        results: list[dict[str, Any]] = []
        try:
            progress = tqdm(
                iterator,
                total=len(tasks),
                desc=f"{dataset} region caches",
                unit="case",
                dynamic_ncols=True,
            )
            results.extend(progress)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        for surface in surfaces_to_build:
            case_entries: list[dict[str, Any]] = []
            failures: list[dict[str, str]] = []
            cached = 0
            failed = 0
            for result in results:
                surface_result = result["surfaces"][surface.name]
                case_entries.append(
                    {
                        "source_case_id": result["source_case_id"],
                        "case_cache_path": surface_result["case_cache_path"],
                        "n_cached_samples": surface_result["n_cached_samples"],
                        "n_failed_samples": surface_result["n_failed_samples"],
                    }
                )
                cached += int(surface_result["n_cached_samples"])
                failed += int(surface_result["n_failed_samples"])
                failures.extend(surface_result["failure_examples"])
            index_path = _surface_index_path(dataset, surface.name)
            _atomic_write_json(
                index_path,
                {
                    "schema_version": REGION_SCHEMA_VERSION,
                    "preprocess_version": REGION_PREPROCESS_VERSION,
                    "cache_mode": "materialized",
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": manifest_sha256,
                    "sample_signature": sample_signature,
                    "surface": asdict(surface),
                    "surface_signature": signatures[surface.name],
                    "crop_size": DEFAULT_CROP_SIZE,
                    "min_mask_voxels": DEFAULT_MIN_MASK_VOXELS,
                    "n_requested_samples": len(samples),
                    "n_cached_samples": cached,
                    "n_failed_samples": failed,
                    "cases": case_entries,
                    "failure_examples": failures[:100],
                },
            )
            if failed or cached != len(samples):
                raise RuntimeError(
                    f"Incomplete {surface.name} cache: {cached}/{len(samples)} "
                    f"samples cached and {failed} failed"
                )

    return {
        surface.name: _surface_index_path(dataset, surface.name)
        for surface in REGION_SURFACES
        if validate_surface_cache(
            dataset,
            surface,
            samples=samples,
            manifest_path=manifest_path,
        )
    }


def _process_is_running(process_id: int) -> bool:
    try:
        os.kill(int(process_id), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def prepare_region_caches(
    dataset: str,
    *,
    workers: int,
    overwrite: bool = False,
    samples_override: Sequence[OrganSample] | None = None,
    manifest_override: Path | None = None,
) -> dict[str, Path]:
    dataset = normalize_cross_modal_dataset_name(dataset)
    lock_path = _cache_root(dataset) / ".region_definition.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            process_id = int(lock_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            process_id = -1
        if process_id > 0 and _process_is_running(process_id):
            raise RuntimeError(
                f"Region cache preparation is already running for {dataset} "
                f"under process {process_id}"
            )
        lock_path.unlink(missing_ok=True)

    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
    finally:
        os.close(descriptor)
    try:
        return _prepare_region_caches_unlocked(
            dataset,
            workers=workers,
            overwrite=overwrite,
            samples_override=samples_override,
            manifest_override=manifest_override,
        )
    finally:
        lock_path.unlink(missing_ok=True)


def prepare_sampling_plans(
    dataset: str,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    samples_override: Sequence[OrganSample] | None = None,
    manifest_override: Path | None = None,
) -> list[Path]:
    dataset = normalize_cross_modal_dataset_name(dataset)
    manifest_path = manifest_override or get_cross_modal_manifest_path(
        dataset,
        "sampled",
        "core",
    )
    samples = list(samples_override or _load_manifest_samples(manifest_path))
    records = _sample_records(samples)
    supported_organs = sorted({sample.primary_organ for sample in samples})
    output_paths: list[Path] = []
    for seed in seeds:
        retrieval = build_retrieval_plan(
            records,
            seed=int(seed),
            supported_organs=supported_organs,
        )
        anatomical = build_anatomical_plan(
            records,
            seed=int(seed),
            supported_organs=supported_organs,
            support_per_modality=DEFAULT_SUPPORT_PER_MODALITY,
        )
        retrieval_path = _plan_dir(dataset) / f"retrieval_seed_{int(seed)}.json"
        anatomical_path = _plan_dir(dataset) / f"anatomical_seed_{int(seed)}.json"
        _atomic_write_json(retrieval_path, retrieval)
        _atomic_write_json(anatomical_path, anatomical)
        output_paths.extend((retrieval_path, anatomical_path))
    _atomic_write_json(
        _plan_dir(dataset) / "plan_contract.json",
        {
            "schema_version": REGION_SCHEMA_VERSION,
            "dataset": dataset,
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "seeds": [int(seed) for seed in seeds],
            "supported_organs": supported_organs,
            "support_per_modality": DEFAULT_SUPPORT_PER_MODALITY,
            "retrieval_directions": ["ct_to_mr", "mr_to_ct"],
            "case_disjoint_anatomical_transfer": True,
        },
    )
    return output_paths


def _surface_case_entries(dataset: str, surface_name: str) -> list[dict[str, Any]]:
    index = json.loads(
        _surface_index_path(dataset, surface_name).read_text(encoding="utf-8")
    )
    return list(index["cases"])


def _extract_batch_features(
    extractor: Any,
    crops: Sequence[torch.Tensor],
) -> dict[str, np.ndarray]:
    outputs = extractor.extract(torch.stack(list(crops), dim=0), normalize=True)
    return {
        feature_type: outputs[feature_type].cpu().numpy()
        for feature_type in DEFAULT_FEATURE_TYPES
    }


def _extract_surface_features(
    extractor: FeatureExtractor,
    *,
    dataset: str,
    surface: RegionSurface,
    checkpoint_sha256: str,
    manifest_path: Path,
    samples: Sequence[OrganSample],
    batch_size: int,
    overwrite: bool,
) -> Path:
    checkpoint = extractor.checkpoint_name
    output_path = _feature_bundle_path(dataset, surface.name, checkpoint)
    summary_path = _feature_summary_path(dataset, surface.name, checkpoint)
    surface_index = validate_surface_cache(
        dataset,
        surface,
        samples=samples,
        manifest_path=manifest_path,
    )
    expected_ids = [sample.sample_id for sample in samples]
    if output_path.exists() and summary_path.exists() and not overwrite:
        validate_feature_bundle(
            output_path,
            summary_path,
            expected_sample_ids=expected_ids,
            surface_signature=str(surface_index["surface_signature"]),
            checkpoint_sha256=checkpoint_sha256,
        )
        return output_path

    rows: dict[str, list[np.ndarray]] = {"cls": [], "avg_pool": []}
    sample_ids: list[str] = []
    pending_crops: list[torch.Tensor] = []
    pending_ids: list[str] = []

    def flush() -> None:
        if not pending_crops:
            return
        features = _extract_batch_features(extractor, pending_crops)
        for feature_type in DEFAULT_FEATURE_TYPES:
            rows[feature_type].extend(features[feature_type])
        sample_ids.extend(pending_ids)
        pending_crops.clear()
        pending_ids.clear()

    entries = _surface_case_entries(dataset, surface.name)
    progress = tqdm(
        total=len(samples),
        desc=f"{dataset}/{checkpoint}/{surface.name}",
        unit="sample",
        dynamic_ncols=True,
    )
    try:
        for entry in entries:
            payload = torch.load(entry["case_cache_path"], map_location="cpu")
            for crop, sample_id in zip(payload["crops"], payload["sample_ids"]):
                pending_crops.append(crop)
                pending_ids.append(str(sample_id))
                progress.update(1)
                if len(pending_crops) >= int(batch_size):
                    flush()
        flush()
    finally:
        progress.close()

    if len(sample_ids) != len(expected_ids) or set(sample_ids) != set(expected_ids):
        raise RuntimeError(
            f"Feature extraction coverage mismatch for {dataset}/{surface.name}/{checkpoint}"
        )
    order_by_id = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    order = np.asarray([order_by_id[sample_id] for sample_id in expected_ids], dtype=np.int64)
    arrays = {
        feature_type: np.stack(feature_rows, axis=0).astype(np.float32)[order]
        for feature_type, feature_rows in rows.items()
    }
    _atomic_save_npz(
        output_path,
        sample_ids=np.asarray(expected_ids, dtype=str),
        **arrays,
    )
    _atomic_write_json(
        summary_path,
        {
            "schema_version": REGION_SCHEMA_VERSION,
            "dataset": dataset,
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "surface": asdict(surface),
            "surface_signature": str(surface_index["surface_signature"]),
            "checkpoint": checkpoint,
            "checkpoint_path": str(extractor.ckpt_config.path),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_architecture": extractor.ckpt_config.arch,
            "crop_size": int(extractor.config.input_size),
            "feature_types": list(DEFAULT_FEATURE_TYPES),
            "feature_dimensions": {
                feature_type: int(array.shape[1])
                for feature_type, array in arrays.items()
            },
            "n_samples": len(expected_ids),
            "sample_ids_sha256": _payload_sha256({"sample_ids": expected_ids}),
            "output_npz": str(output_path),
        },
    )
    validate_feature_bundle(
        output_path,
        summary_path,
        expected_sample_ids=expected_ids,
        surface_signature=str(surface_index["surface_signature"]),
        checkpoint_sha256=checkpoint_sha256,
    )
    return output_path


def validate_feature_bundle(
    bundle_path: Path,
    summary_path: Path,
    *,
    expected_sample_ids: Sequence[str],
    surface_signature: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if str(summary.get("surface_signature")) != surface_signature:
        raise ValueError(f"Feature surface signature mismatch at {summary_path}")
    if str(summary.get("checkpoint_sha256")) != checkpoint_sha256:
        raise ValueError(f"Feature checkpoint hash mismatch at {summary_path}")
    with np.load(bundle_path, allow_pickle=False) as bundle:
        sample_ids = [str(value) for value in bundle["sample_ids"].tolist()]
        if sample_ids != [str(value) for value in expected_sample_ids]:
            raise ValueError(f"Feature sample order mismatch at {bundle_path}")
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"Duplicate feature sample IDs at {bundle_path}")
        for feature_type in DEFAULT_FEATURE_TYPES:
            features = np.asarray(bundle[feature_type])
            if features.ndim != 2 or features.shape[0] != len(sample_ids):
                raise ValueError(
                    f"Unexpected {feature_type} feature shape at {bundle_path}: "
                    f"{features.shape}"
                )
            if not np.isfinite(features).all():
                raise ValueError(f"Nonfinite {feature_type} features at {bundle_path}")
            expected_dim = int(summary["feature_dimensions"][feature_type])
            if features.shape[1] != expected_dim:
                raise ValueError(
                    f"{feature_type} dimension mismatch at {bundle_path}"
                )
    return summary


def extract_region_features(
    dataset: str,
    *,
    checkpoints: Sequence[str] = DEFAULT_CHECKPOINTS,
    device: str = "cuda",
    batch_size: int = 8,
    overwrite: bool = False,
) -> list[Path]:
    dataset = normalize_cross_modal_dataset_name(dataset)
    manifest_path = get_cross_modal_manifest_path(dataset, "sampled", "core")
    samples = _load_manifest_samples(manifest_path)
    outputs: list[Path] = []
    for checkpoint_name in checkpoints:
        config = resolve_checkpoint_config(checkpoint_name)
        if int(config.crop_size) != DEFAULT_CROP_SIZE:
            raise ValueError(
                f"Task 2 accepts crop96 checkpoints only: {config.checkpoint_key}"
            )
        checkpoint_sha256 = _sha256_file(config.path)
        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        extractor = FeatureExtractor(
            config.checkpoint_key,
            config=ExtractionConfig(
                extract_cls=True,
                extract_avg_pool=True,
                extract_multilayer=False,
                batch_size=int(batch_size),
                device=device,
                dtype=dtype,
            ),
            device=device,
        )
        for surface in REGION_SURFACES:
            outputs.append(
                _extract_surface_features(
                    extractor,
                    dataset=dataset,
                    surface=surface,
                    checkpoint_sha256=checkpoint_sha256,
                    manifest_path=manifest_path,
                    samples=samples,
                    batch_size=batch_size,
                    overwrite=overwrite,
                )
            )
        del extractor
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return outputs


def _load_feature_map(
    bundle_path: Path,
    feature_type: str,
) -> dict[str, np.ndarray]:
    with np.load(bundle_path, allow_pickle=False) as bundle:
        sample_ids = [str(value) for value in bundle["sample_ids"].tolist()]
        features = np.asarray(bundle[feature_type], dtype=np.float32)
    return {
        sample_id: features[index]
        for index, sample_id in enumerate(sample_ids)
    }


def evaluate_region_features(
    dataset: str,
    *,
    checkpoints: Sequence[str] = DEFAULT_CHECKPOINTS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    overwrite: bool = False,
) -> list[Path]:
    dataset = normalize_cross_modal_dataset_name(dataset)
    manifest_path = get_cross_modal_manifest_path(dataset, "sampled", "core")
    samples = _load_manifest_samples(manifest_path)
    records = _sample_records(samples)
    retrieval_plans = {
        int(seed): json.loads(
            (_plan_dir(dataset) / f"retrieval_seed_{int(seed)}.json").read_text(
                encoding="utf-8"
            )
        )
        for seed in seeds
    }
    anatomical_plans = [
        json.loads(
            (_plan_dir(dataset) / f"anatomical_seed_{int(seed)}.json").read_text(
                encoding="utf-8"
            )
        )
        for seed in seeds
    ]
    for plan in retrieval_plans.values():
        validate_retrieval_plan(plan, records)
    for plan in anatomical_plans:
        validate_anatomical_plan(plan, records)

    outputs: list[Path] = []
    for surface in REGION_SURFACES:
        for checkpoint_name in checkpoints:
            checkpoint = resolve_checkpoint_config(checkpoint_name)
            checkpoint_sha256 = _sha256_file(checkpoint.path)
            bundle_path = _feature_bundle_path(
                dataset,
                surface.name,
                checkpoint.checkpoint_key,
            )
            validate_feature_bundle(
                bundle_path,
                _feature_summary_path(
                    dataset,
                    surface.name,
                    checkpoint.checkpoint_key,
                ),
                expected_sample_ids=[sample.sample_id for sample in samples],
                surface_signature=str(
                    validate_surface_cache(
                        dataset,
                        surface,
                        samples=samples,
                        manifest_path=manifest_path,
                    )["surface_signature"]
                ),
                checkpoint_sha256=checkpoint_sha256,
            )
            for feature_type in DEFAULT_FEATURE_TYPES:
                features_by_id = _load_feature_map(bundle_path, feature_type)
                evaluation_dir = _evaluation_dir(
                    dataset,
                    surface.name,
                    checkpoint.checkpoint_key,
                    feature_type,
                )
                for seed, plan in retrieval_plans.items():
                    output_path = evaluation_dir / f"retrieval_seed_{seed}.json"
                    if not output_path.exists() or overwrite:
                        _atomic_write_json(
                            output_path,
                            {
                                "dataset": dataset,
                                "surface": surface.name,
                                "checkpoint": checkpoint.checkpoint_key,
                                "feature_type": feature_type,
                                **evaluate_retrieval_plan(
                                    features_by_id,
                                    records,
                                    plan,
                                ),
                            },
                        )
                    outputs.append(output_path)
                anatomical_path = evaluation_dir / "anatomical_transfer.json"
                if not anatomical_path.exists() or overwrite:
                    _atomic_write_json(
                        anatomical_path,
                        {
                            "dataset": dataset,
                            "surface": surface.name,
                            "checkpoint": checkpoint.checkpoint_key,
                            "feature_type": feature_type,
                            **evaluate_anatomical_plans(
                                features_by_id,
                                records,
                                anatomical_plans,
                            ),
                        },
                    )
                outputs.append(anatomical_path)
    return outputs


def write_region_contract(dataset: str) -> Path:
    dataset = normalize_cross_modal_dataset_name(dataset)
    manifest_path = get_cross_modal_manifest_path(dataset, "sampled", "core")
    output_path = _result_root(dataset) / "contract.json"
    _atomic_write_json(
        output_path,
        {
            "schema_version": REGION_SCHEMA_VERSION,
            "dataset": dataset,
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "crop_size": DEFAULT_CROP_SIZE,
            "min_mask_voxels": DEFAULT_MIN_MASK_VOXELS,
            "surfaces": [asdict(surface) for surface in REGION_SURFACES],
            "checkpoints": list(DEFAULT_CHECKPOINTS),
            "feature_types": list(DEFAULT_FEATURE_TYPES),
            "seeds": list(DEFAULT_SEEDS),
            "support_per_modality": DEFAULT_SUPPORT_PER_MODALITY,
            "surface_comparisons": {
                "crop_extent": [
                    "tight_bbox0_context",
                    "baseline_full_bbox_context",
                ],
                "outside_organ_context": [
                    "organ_only_bbox5",
                    "baseline_full_bbox_context",
                ],
                "combined_crop_and_masking": [
                    "organ_only_bbox0",
                    "baseline_full_bbox_context",
                ],
                "mask_morphology": [
                    "eroded_1mm_bbox5",
                    "organ_only_bbox5",
                ],
            },
        },
    )
    return output_path


def prepare_region_task(
    dataset: str,
    *,
    workers: int,
    overwrite: bool = False,
) -> None:
    write_region_contract(dataset)
    prepare_region_caches(
        dataset,
        workers=workers,
        overwrite=overwrite,
    )
    prepare_sampling_plans(dataset)


def validate_region_task(
    dataset: str,
    *,
    checkpoints: Sequence[str] = DEFAULT_CHECKPOINTS,
    require_evaluations: bool = False,
) -> dict[str, Any]:
    dataset = normalize_cross_modal_dataset_name(dataset)
    manifest_path = get_cross_modal_manifest_path(dataset, "sampled", "core")
    samples = _load_manifest_samples(manifest_path)
    records = _sample_records(samples)
    for surface in REGION_SURFACES:
        validate_surface_cache(
            dataset,
            surface,
            samples=samples,
            manifest_path=manifest_path,
        )
    for seed in DEFAULT_SEEDS:
        validate_retrieval_plan(
            json.loads(
                (_plan_dir(dataset) / f"retrieval_seed_{seed}.json").read_text(
                    encoding="utf-8"
                )
            ),
            records,
        )
        validate_anatomical_plan(
            json.loads(
                (_plan_dir(dataset) / f"anatomical_seed_{seed}.json").read_text(
                    encoding="utf-8"
                )
            ),
            records,
        )

    feature_count = 0
    evaluation_count = 0
    for checkpoint_name in checkpoints:
        checkpoint = resolve_checkpoint_config(checkpoint_name)
        checkpoint_sha256 = _sha256_file(checkpoint.path)
        for surface in REGION_SURFACES:
            bundle_path = _feature_bundle_path(
                dataset,
                surface.name,
                checkpoint.checkpoint_key,
            )
            if not bundle_path.exists():
                continue
            validate_feature_bundle(
                bundle_path,
                _feature_summary_path(
                    dataset,
                    surface.name,
                    checkpoint.checkpoint_key,
                ),
                expected_sample_ids=[sample.sample_id for sample in samples],
                surface_signature=str(
                    validate_surface_cache(
                        dataset,
                        surface,
                        samples=samples,
                        manifest_path=manifest_path,
                    )["surface_signature"]
                ),
                checkpoint_sha256=checkpoint_sha256,
            )
            feature_count += 1
            for feature_type in DEFAULT_FEATURE_TYPES:
                evaluation_dir = _evaluation_dir(
                    dataset,
                    surface.name,
                    checkpoint.checkpoint_key,
                    feature_type,
                )
                expected_paths = [
                    evaluation_dir / f"retrieval_seed_{seed}.json"
                    for seed in DEFAULT_SEEDS
                ] + [evaluation_dir / "anatomical_transfer.json"]
                if all(path.exists() for path in expected_paths):
                    evaluation_count += len(expected_paths)
                elif require_evaluations:
                    missing = [str(path) for path in expected_paths if not path.exists()]
                    raise FileNotFoundError(
                        f"Incomplete evaluation artifacts: {missing[:3]}"
                    )
    expected_feature_count = len(checkpoints) * len(REGION_SURFACES)
    if require_evaluations and feature_count != expected_feature_count:
        raise RuntimeError(
            f"Expected {expected_feature_count} feature bundles, found {feature_count}"
        )
    return {
        "dataset": dataset,
        "n_samples": len(samples),
        "n_surface_caches": len(REGION_SURFACES),
        "n_feature_bundles": feature_count,
        "n_evaluation_artifacts": evaluation_count,
    }


def smoke_region_extraction(
    dataset: str,
    *,
    checkpoint: str,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    dataset = normalize_cross_modal_dataset_name(dataset)
    manifest_path = get_cross_modal_manifest_path(dataset, "sampled", "core")
    samples = _load_manifest_samples(manifest_path)
    grouped = list(_group_samples_by_case(samples).values())
    if not grouped:
        raise ValueError(f"No source cases found in {manifest_path}")
    smoke_samples = grouped[0]
    with tempfile.TemporaryDirectory(prefix=f"reform3d_{dataset}_region_") as directory:
        temporary_root = Path(directory)
        manifest_payload = {
            "samples": [
                {
                    "sample_id": sample.sample_id,
                    "file_path": str(sample.file_path),
                    "mask_path": str(sample.mask_path),
                    "modality": sample.modality,
                    "primary_organ": sample.primary_organ,
                    "source_case_id": sample.source_case_id,
                    "spacing": list(sample.spacing) if sample.spacing else None,
                    "mask_label_value": sample.mask_label_value,
                    "mask_label_range": (
                        list(sample.mask_label_range)
                        if sample.mask_label_range
                        else None
                    ),
                    "mask_loader": sample.mask_loader,
                }
                for sample in smoke_samples
            ]
        }
        smoke_manifest = temporary_root / "manifest.json"
        smoke_manifest.write_text(
            json.dumps(manifest_payload),
            encoding="utf-8",
        )
        signatures = {
            surface.name: _surface_signature(
                surface,
                manifest_sha256=_sha256_file(smoke_manifest),
                sample_signature=_sample_signature(smoke_samples),
            )
            for surface in REGION_SURFACES
        }
        result = _build_case_surfaces(
            (
                dataset,
                (
                    smoke_samples[0].source_case_id,
                    str(smoke_samples[0].file_path),
                ),
                smoke_samples,
                REGION_SURFACES,
                signatures,
                True,
                str(temporary_root / "crop96"),
            )
        )
        config = resolve_checkpoint_config(checkpoint)
        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        extractor = FeatureExtractor(
            config.checkpoint_key,
            config=ExtractionConfig(
                extract_cls=True,
                extract_avg_pool=True,
                extract_multilayer=False,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            ),
            device=device,
        )
        feature_shapes: dict[str, dict[str, list[int]]] = {}
        for surface in REGION_SURFACES:
            surface_result = result["surfaces"][surface.name]
            if surface_result["n_failed_samples"]:
                raise RuntimeError(
                    f"Smoke cache failed for {surface.name}: "
                    f"{surface_result['failure_examples'][:2]}"
                )
            payload = torch.load(
                surface_result["case_cache_path"],
                map_location="cpu",
            )
            features = extractor.extract(
                payload["crops"][:batch_size],
                normalize=True,
            )
            feature_shapes[surface.name] = {
                feature_type: list(features[feature_type].shape)
                for feature_type in DEFAULT_FEATURE_TYPES
            }
        return {
            "dataset": dataset,
            "checkpoint": config.checkpoint_key,
            "n_smoke_samples": len(smoke_samples),
            "feature_shapes": feature_shapes,
            "temporary_artifacts_removed": True,
        }


def _parse_values(values: Sequence[str] | None, defaults: Sequence[str]) -> tuple[str, ...]:
    if not values:
        return tuple(defaults)
    parsed: list[str] = []
    for value in values:
        parsed.extend(item.strip() for item in value.split(",") if item.strip())
    return tuple(parsed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "extract", "evaluate", "run", "validate", "smoke"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument(
            "--dataset",
            required=True,
            choices=SUPPORTED_DATASETS,
        )
        command_parser.add_argument(
            "--checkpoints",
            nargs="+",
            default=list(DEFAULT_CHECKPOINTS),
        )
        command_parser.add_argument("--overwrite", action="store_true")
        if command in {"prepare", "run"}:
            command_parser.add_argument("--workers", type=int, default=4)
        if command in {"extract", "run", "smoke"}:
            command_parser.add_argument("--device", default="cuda")
            command_parser.add_argument("--batch-size", type=int, default=8)
        if command == "validate":
            command_parser.add_argument(
                "--require-evaluations",
                action="store_true",
            )
        if command == "smoke":
            command_parser.add_argument(
                "--checkpoint",
                default="med3dino_iso_96",
            )
    args = parser.parse_args()
    checkpoints = _parse_values(args.checkpoints, DEFAULT_CHECKPOINTS)
    dataset = args.dataset

    if args.command == "prepare":
        prepare_region_task(
            dataset,
            workers=args.workers,
            overwrite=args.overwrite,
        )
    elif args.command == "extract":
        extract_region_features(
            dataset,
            checkpoints=checkpoints,
            device=args.device,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
    elif args.command == "evaluate":
        evaluate_region_features(
            dataset,
            checkpoints=checkpoints,
            overwrite=args.overwrite,
        )
    elif args.command == "run":
        prepare_region_task(
            dataset,
            workers=args.workers,
            overwrite=args.overwrite,
        )
        extract_region_features(
            dataset,
            checkpoints=checkpoints,
            device=args.device,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
        evaluate_region_features(
            dataset,
            checkpoints=checkpoints,
            overwrite=args.overwrite,
        )
    elif args.command == "validate":
        print(
            json.dumps(
                validate_region_task(
                    dataset,
                    checkpoints=checkpoints,
                    require_evaluations=args.require_evaluations,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "smoke":
        print(
            json.dumps(
                smoke_region_extraction(
                    dataset,
                    checkpoint=args.checkpoint,
                    device=args.device,
                    batch_size=args.batch_size,
                ),
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
