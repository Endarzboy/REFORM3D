"""Load Acquisition Robustness manifests and prepare model-ready volumes.

The loader reads manifest records, resolves image paths, applies percentile
intensity normalization and foreground-aware cropping, and optionally reuses
cached preprocessed tensors. It is used by feature extraction and the native
or controlled evaluation pipelines.

Inputs:
    manifest_path: ``manifest_sampled.json`` or ``manifest_full.json``.
    crop_size: checkpoint crop size, usually 96 or 112.
    cache_root: optional preprocessing cache location.
"""

import json
import logging
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from tqdm import tqdm

from monai.transforms import (
    Compose,
    LoadImaged,
    Lambdad,
    ScaleIntensityRangePercentilesd,
    SpatialPadd,
    CenterSpatialCropd,
    ToTensord,
)

from reform3d.encoders.med3dino.spacing_aware_transforms import CropForegroundSwapSliceDimsV2
from reform3d.core.paths import get_reform3d_cache_path_candidates

try:
    from .config import (
        PREPROCESSING_CONFIG,
        ANISOTROPY_BINS,
        get_anisotropy_bin,
        get_cache_root,
        get_dataset_name_from_manifest_path,
        get_manifest_variant_from_manifest_path,
        resolve_acquisition_manifest_path,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from config import (
        PREPROCESSING_CONFIG,
        ANISOTROPY_BINS,
        get_anisotropy_bin,
        get_cache_root,
        get_dataset_name_from_manifest_path,
        get_manifest_variant_from_manifest_path,
        resolve_acquisition_manifest_path,
    )

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


OBSERVATIONAL_PREPROCESS_CACHE_VERSION = "v2"
OBSERVATIONAL_CACHE_IMAGE_DTYPE = torch.float16


def _preprocess_cache_namespace(crop_size: int, include_labels: bool) -> str:
    label_tag = "with_labels" if include_labels else "image_only"
    return (
        f"observational_preprocessed_{OBSERVATIONAL_PREPROCESS_CACHE_VERSION}_"
        f"crop{int(crop_size)}_{label_tag}"
    )


def _to_plain_tensor(value: Any) -> torch.Tensor:
    """Convert MetaTensor-like values to contiguous plain CPU tensors."""
    if hasattr(value, "as_tensor"):
        value = value.as_tensor()
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value.detach().cpu().contiguous()


class AcquisitionRobustnessDataset(Dataset):
    """
    Dataset for acquisition robustness evaluation.

    Loads volumes with preprocessing matching pretraining pipeline.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        crop_size: int = 96,
        include_labels: bool = False,
        transform: Optional[Compose] = None,
        filter_bins: Optional[List[int]] = None,
        filter_modalities: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        fail_on_load_error: Optional[bool] = None,
        cache_preprocessed: bool = True,
        cache_root: Optional[Union[str, Path]] = None,
    ):
        """
        Args:
            manifest_path: Path to manifest JSON file
            crop_size: Target crop size (matches checkpoint)
            include_labels: Whether to load label masks
            transform: Optional custom transform (overrides default)
            filter_bins: Only include volumes from these anisotropy bins
            filter_modalities: Only include volumes from these modalities
            max_samples: Maximum number of samples to include
        """
        self.manifest_path = Path(manifest_path).resolve()
        self.crop_size = crop_size
        self.include_labels = include_labels
        self.cache_preprocessed = cache_preprocessed
        self.fail_on_load_error = (
            PREPROCESSING_CONFIG.fail_on_load_error
            if fail_on_load_error is None
            else fail_on_load_error
        )

        dataset_name = get_dataset_name_from_manifest_path(self.manifest_path, fallback="default")
        manifest_variant = get_manifest_variant_from_manifest_path(
            self.manifest_path,
            fallback="original_bins",
        )
        base_cache_root = Path(cache_root) if cache_root is not None else get_cache_root(dataset_name, manifest_variant)
        self.preprocessed_cache_dir: Optional[Path] = None
        if self.cache_preprocessed:
            self.preprocessed_cache_dir = (
                base_cache_root
                / _preprocess_cache_namespace(self.crop_size, self.include_labels)
            )
            self.preprocessed_cache_dir.mkdir(parents=True, exist_ok=True)

        # Load manifest
        with open(self.manifest_path) as f:
            manifest = json.load(f)

        self.volumes = manifest.get("volumes", manifest)
        if isinstance(self.volumes, dict):
            # Handle compact manifest entries
            self.volumes = list(self.volumes.values())

        # Apply filters
        if filter_bins is not None:
            self.volumes = [v for v in self.volumes if v.get("anisotropy_bin") in filter_bins]

        if filter_modalities is not None:
            self.volumes = [v for v in self.volumes if v.get("modality") in filter_modalities]

        if max_samples is not None and len(self.volumes) > max_samples:
            np.random.seed(42)
            indices = np.random.choice(len(self.volumes), max_samples, replace=False)
            self.volumes = [self.volumes[i] for i in indices]

        logger.info(f"Loaded {len(self.volumes)} volumes from manifest")

        # Set up transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._build_default_transform()

    def _cache_path_for_volume(self, vol_info: Dict[str, Any]) -> Optional[Path]:
        if self.preprocessed_cache_dir is None:
            return None

        cache_payload = {
            "file_path": vol_info.get("file_path", ""),
            "label_path": vol_info.get("label_path") or "",
            "crop_size": int(self.crop_size),
            "include_labels": bool(self.include_labels),
            "cache_version": OBSERVATIONAL_PREPROCESS_CACHE_VERSION,
            "crop_foreground": bool(PREPROCESSING_CONFIG.crop_foreground),
            "foreground_threshold": float(PREPROCESSING_CONFIG.foreground_threshold),
            "intensity_percentile_lower": float(PREPROCESSING_CONFIG.intensity_percentile_lower),
            "intensity_percentile_upper": float(PREPROCESSING_CONFIG.intensity_percentile_upper),
            "intensity_output_min": float(PREPROCESSING_CONFIG.intensity_output_min),
            "intensity_output_max": float(PREPROCESSING_CONFIG.intensity_output_max),
        }
        digest = hashlib.md5(json.dumps(cache_payload, sort_keys=True).encode("utf-8")).hexdigest()
        return self.preprocessed_cache_dir / f"{digest}.pt"

    def _build_output(self, transformed: Dict[str, Any], vol_info: Dict[str, Any]) -> Dict[str, Any]:
        output = {
            "image": _to_plain_tensor(transformed["image"]),
            "metadata": {
                "file_path": vol_info["file_path"],
                "dataset": vol_info.get("dataset", "unknown"),
                "modality": vol_info.get("modality", "unknown"),
                "spacing": vol_info.get("spacing", [1.0, 1.0, 1.0]),
                "spacing_permuted": transformed.get("spacing_permuted"),
                "original_shape": vol_info.get("shape", [0, 0, 0]),
                "anisotropy_ratio": vol_info.get("anisotropy_ratio", 1.0),
                "anisotropy_bin": vol_info.get("anisotropy_bin", 0),
                "slice_axis_original": transformed.get("slice_axis_original"),
            },
        }

        if self.include_labels and "label" in transformed:
            output["label"] = _to_plain_tensor(transformed["label"])

        return output

    def _load_from_preprocessed_cache(self, cache_path: Path) -> Optional[Dict[str, Any]]:
        try:
            payload = torch.load(cache_path, map_location="cpu")
        except Exception as e:
            logger.warning(f"Failed to load preprocessed cache {cache_path.name}: {e}")
            return None

        if not isinstance(payload, dict) or payload.get("cache_version") != OBSERVATIONAL_PREPROCESS_CACHE_VERSION:
            return None

        sample = payload.get("sample")
        if not isinstance(sample, dict) or "image" not in sample or "metadata" not in sample:
            return None
        sample["image"] = _to_plain_tensor(sample["image"])
        if "label" in sample:
            sample["label"] = _to_plain_tensor(sample["label"])
        return sample

    def _save_to_preprocessed_cache(self, cache_path: Path, sample: Dict[str, Any]) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(f".tmp.{os.getpid()}")
        cached_sample = {
            "image": _to_plain_tensor(sample["image"]).to(dtype=OBSERVATIONAL_CACHE_IMAGE_DTYPE),
            "metadata": sample["metadata"],
        }
        if "label" in sample:
            cached_sample["label"] = _to_plain_tensor(sample["label"])
        payload = {
            "cache_version": OBSERVATIONAL_PREPROCESS_CACHE_VERSION,
            "sample": cached_sample,
        }
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, cache_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _build_default_transform(self) -> Compose:
        """Build deterministic preprocessing aligned with the verified training pipeline."""
        keys = ["image"]
        if self.include_labels:
            keys.append("label")

        transforms = [
            LoadImaged(keys=keys, ensure_channel_first=True, meta_keys=["image_meta_dict"]),
            Lambdad(
                keys=["image"],
                func=lambda x: torch.nan_to_num(x, torch.nanmean(x).item()),
            ),
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=PREPROCESSING_CONFIG.intensity_percentile_lower,
                upper=PREPROCESSING_CONFIG.intensity_percentile_upper,
                b_min=PREPROCESSING_CONFIG.intensity_output_min,
                b_max=PREPROCESSING_CONFIG.intensity_output_max,
                clip=True,
            ),
        ]

        # Match training's anisotropy-aware foreground crop + slice-axis convention.
        if PREPROCESSING_CONFIG.crop_foreground:
            transforms.append(
                CropForegroundSwapSliceDimsV2(
                    select_fn=lambda x: x > PREPROCESSING_CONFIG.foreground_threshold,
                )
            )

        # Spatial padding (ensure minimum size)
        transforms.append(
            SpatialPadd(
                keys=keys,
                spatial_size=[self.crop_size] * 3,
                mode="constant",
            )
        )

        # Center crop to target size
        transforms.append(
            CenterSpatialCropd(
                keys=keys,
                roi_size=[self.crop_size] * 3,
            )
        )

        transforms.append(ToTensord(keys=keys))

        return Compose(transforms)

    def __len__(self) -> int:
        return len(self.volumes)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single volume with metadata.

        Returns:
            Dictionary containing:
            - image: Tensor of shape (1, D, H, W)
            - metadata: Dict with spacing, anisotropy_ratio, bin, dataset, modality
            - label (optional): Tensor of shape (1, D, H, W) if include_labels=True
        """
        vol_info = self.volumes[idx]
        cache_path = self._cache_path_for_volume(vol_info)

        existing_cache_path = None
        if cache_path is not None:
            for candidate in get_reform3d_cache_path_candidates(cache_path):
                if candidate.exists():
                    existing_cache_path = candidate
                    break

        if existing_cache_path is not None:
            cached_sample = self._load_from_preprocessed_cache(existing_cache_path)
            if cached_sample is not None:
                return cached_sample
            if existing_cache_path == cache_path:
                cache_path.unlink(missing_ok=True)

        # Prepare data dict for MONAI transforms
        data = {
            "image": vol_info["file_path"],
            "spacing": vol_info.get("spacing", [1.0, 1.0, 1.0]),
            "anisotropy_ratio": vol_info.get("anisotropy_ratio", 1.0),
        }

        if self.include_labels and vol_info.get("has_label") and vol_info.get("label_path"):
            data["label"] = vol_info["label_path"]

        # Apply transforms
        try:
            transformed = self.transform(data)
        except Exception as e:
            logger.warning(f"Failed to load {vol_info['file_path']}: {e}")
            if self.fail_on_load_error:
                raise RuntimeError(f"Failed to load sample {vol_info['file_path']}") from e
            return self._get_dummy_sample(vol_info)

        output = self._build_output(transformed, vol_info)
        if cache_path is not None:
            self._save_to_preprocessed_cache(cache_path, output)

        return output

    def _get_dummy_sample(self, vol_info: Dict) -> Dict[str, Any]:
        """Return a dummy sample when loading fails."""
        dummy_image = torch.zeros(1, self.crop_size, self.crop_size, self.crop_size)
        output = {
            "image": dummy_image,
            "metadata": {
                "file_path": vol_info.get("file_path", ""),
                "dataset": vol_info.get("dataset", "unknown"),
                "modality": vol_info.get("modality", "unknown"),
                "spacing": vol_info.get("spacing", [1.0, 1.0, 1.0]),
                "original_shape": vol_info.get("shape", [0, 0, 0]),
                "anisotropy_ratio": vol_info.get("anisotropy_ratio", 1.0),
                "anisotropy_bin": vol_info.get("anisotropy_bin", 0),
                "load_failed": True,
            }
        }
        if self.include_labels:
            output["label"] = dummy_image.clone()
        return output


def create_acquisition_dataloader(
    manifest_path: Union[str, Path],
    batch_size: int = 4,
    num_workers: int = 4,
    crop_size: int = 96,
    include_labels: bool = False,
    filter_bins: Optional[List[int]] = None,
    filter_modalities: Optional[List[str]] = None,
    shuffle: bool = False,
    fail_on_load_error: Optional[bool] = None,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
    cache_preprocessed: bool = True,
    cache_root: Optional[Union[str, Path]] = None,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for acquisition robustness evaluation.

    Args:
        manifest_path: Path to manifest JSON file
        batch_size: Batch size
        num_workers: Number of data loading workers
        crop_size: Target crop size
        include_labels: Whether to load labels
        filter_bins: Only include these anisotropy bins
        filter_modalities: Only include these modalities
        shuffle: Whether to shuffle data
        pin_memory: Whether to pin host memory for GPU transfer
        persistent_workers: Whether to keep worker processes alive across epochs
        prefetch_factor: Number of batches prefetched per worker
        **kwargs: Additional arguments for AcquisitionRobustnessDataset

    Returns:
        DataLoader instance
    """
    dataset = AcquisitionRobustnessDataset(
        manifest_path=manifest_path,
        crop_size=crop_size,
        include_labels=include_labels,
        filter_bins=filter_bins,
        filter_modalities=filter_modalities,
        fail_on_load_error=fail_on_load_error,
        cache_preprocessed=cache_preprocessed,
        cache_root=cache_root,
        **kwargs
    )

    def collate_fn(batch):
        """Custom collate function to handle metadata.

        Ensures MetaTensor objects are converted to plain tensors.
        """
        # Stack images, converting MetaTensors to plain tensors
        images_list = []
        for b in batch:
            img = b["image"]
            # Convert MetaTensor to plain tensor if needed
            if hasattr(img, 'as_tensor'):
                img = img.as_tensor()
            images_list.append(img)
        images = torch.stack(images_list)

        metadata = [b["metadata"] for b in batch]

        output = {"image": images, "metadata": metadata}

        if "label" in batch[0]:
            labels_list = []
            for b in batch:
                lbl = b["label"]
                if hasattr(lbl, 'as_tensor'):
                    lbl = lbl.as_tensor()
                labels_list.append(lbl)
            labels = torch.stack(labels_list)
            output["label"] = labels

        return output

    if pin_memory is None:
        # Feature extraction is input-pipeline bound and large 3D batches can exhaust pinned host memory.
        pin_memory = False

    if persistent_workers is None:
        # Evaluation is single-pass; avoid retaining worker memory across the run.
        persistent_workers = False

    if num_workers <= 0:
        prefetch_factor = None
    elif prefetch_factor is None:
        # Keep worker parallelism while limiting queued 3D volumes per process.
        prefetch_factor = 1

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0 and persistent_workers,
        prefetch_factor=prefetch_factor,
    )


def warmup_observational_cache(
    manifest_path: Union[str, Path],
    crop_size: int,
    batch_size: int = 8,
    num_workers: int = 4,
    cache_root: Optional[Union[str, Path]] = None,
) -> int:
    """Materialize deterministic observational preprocessing cache for a crop size."""
    dataloader = create_acquisition_dataloader(
        manifest_path=manifest_path,
        batch_size=batch_size,
        num_workers=num_workers,
        crop_size=crop_size,
        shuffle=False,
        cache_preprocessed=True,
        cache_root=cache_root,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    processed = 0
    for batch in tqdm(dataloader, desc=f"Warmup observational cache c{crop_size}"):
        processed += int(batch["image"].shape[0])

    logger.info(
        "Observational preprocessing cache warmup complete for crop_size=%d (%d samples)",
        crop_size,
        processed,
    )
    return processed


def load_volume_for_visualization(
    file_path: Union[str, Path],
    label_path: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """
    Load a single volume for visualization (without preprocessing).

    Args:
        file_path: Path to NIfTI file
        label_path: Optional path to label file

    Returns:
        Tuple of (image_array, label_array, metadata_dict)
    """
    # Load image
    img = nib.load(str(file_path))
    image_data = img.get_fdata().astype(np.float32)

    # Get spacing from affine
    affine = img.affine
    spacing = tuple(float(np.abs(affine[i, i])) for i in range(3))

    # Compute anisotropy
    in_plane = min(spacing[0], spacing[1])
    anisotropy_ratio = spacing[2] / in_plane if in_plane > 0 else 1.0

    metadata = {
        "file_path": str(file_path),
        "shape": image_data.shape,
        "spacing": spacing,
        "anisotropy_ratio": anisotropy_ratio,
        "anisotropy_bin": get_anisotropy_bin(anisotropy_ratio),
        "dtype": str(image_data.dtype),
        "min_val": float(image_data.min()),
        "max_val": float(image_data.max()),
    }

    # Load label if provided
    label_data = None
    if label_path is not None and Path(label_path).exists():
        label_img = nib.load(str(label_path))
        label_data = label_img.get_fdata().astype(np.int32)
        metadata["label_path"] = str(label_path)
        metadata["num_classes"] = len(np.unique(label_data))

    return image_data, label_data, metadata


def get_random_samples_per_dataset(
    manifest_path: Union[str, Path],
    samples_per_dataset: int = 1,
    seed: int = 42,
    prefer_labeled: bool = True,
) -> Dict[str, List[Dict]]:
    """
    Get random sample volumes from each dataset in the manifest.

    Args:
        manifest_path: Path to manifest JSON
        samples_per_dataset: Number of samples per dataset
        seed: Random seed
        prefer_labeled: If True, prefer samples with labels when available

    Returns:
        Dictionary mapping dataset name to list of volume info dicts
    """
    np.random.seed(seed)

    with open(manifest_path) as f:
        manifest = json.load(f)

    volumes = manifest.get("volumes", [])

    # Group by dataset
    by_dataset = {}
    for vol in volumes:
        ds = vol.get("dataset", "unknown")
        if ds not in by_dataset:
            by_dataset[ds] = []
        by_dataset[ds].append(vol)

    # Sample from each dataset (preferring labeled samples if requested)
    samples = {}
    for ds, vols in by_dataset.items():
        n = min(samples_per_dataset, len(vols))

        if prefer_labeled:
            # Separate labeled and unlabeled
            labeled = [v for v in vols if v.get('label_path') is not None]
            unlabeled = [v for v in vols if v.get('label_path') is None]

            # Prefer labeled samples
            if len(labeled) >= n:
                indices = np.random.choice(len(labeled), n, replace=False)
                samples[ds] = [labeled[i] for i in indices]
            elif len(labeled) > 0:
                # Use all labeled + sample from unlabeled
                remaining = n - len(labeled)
                unlabeled_indices = np.random.choice(len(unlabeled), min(remaining, len(unlabeled)), replace=False)
                samples[ds] = labeled + [unlabeled[i] for i in unlabeled_indices]
            else:
                # No labeled samples, fall back to random
                indices = np.random.choice(len(vols), n, replace=False)
                samples[ds] = [vols[i] for i in indices]
        else:
            indices = np.random.choice(len(vols), n, replace=False)
            samples[ds] = [vols[i] for i in indices]

    return samples


if __name__ == "__main__":
    # Usage: python data_loader.py
    # Runs a lightweight local smoke test against acquisition_raw_data_manifest.json.
    # Test the data loader
    manifest_path = resolve_acquisition_manifest_path("acquisition_raw_data_manifest.json")

    if manifest_path.exists():
        print("Testing AcquisitionRobustnessDataset...")
        dataset = AcquisitionRobustnessDataset(manifest_path, crop_size=96, max_samples=10)
        print(f"Dataset size: {len(dataset)}")

        # Test loading a sample
        sample = dataset[0]
        print(f"Sample image shape: {sample['image'].shape}")
        print(f"Sample metadata: {sample['metadata']}")

        # Test dataloader
        print("\nTesting DataLoader...")
        loader = create_acquisition_dataloader(manifest_path, batch_size=2, num_workers=0, max_samples=10)
        batch = next(iter(loader))
        print(f"Batch image shape: {batch['image'].shape}")
        print(f"Batch metadata count: {len(batch['metadata'])}")
    else:
        print(f"Manifest not found: {manifest_path}")
        print("Run build_manifest.py or manifest_generation.py first.")
