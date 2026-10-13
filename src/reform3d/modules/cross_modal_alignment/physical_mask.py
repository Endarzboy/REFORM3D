"""Physical-space binary-mask operations for Cross-Modal evaluations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt


def normalize_spacing_mm(spacing_mm: Sequence[float]) -> tuple[float, float, float]:
    if len(spacing_mm) != 3:
        raise ValueError(f"Expected three spacing values, got {spacing_mm}")
    spacing = tuple(float(value) for value in spacing_mm)
    if any(not math.isfinite(value) or value <= 0 for value in spacing):
        raise ValueError(f"Invalid physical spacing: {spacing}")
    return spacing


def erode_binary_mask_mm(
    mask: np.ndarray,
    spacing_mm: Sequence[float],
    radius_mm: float,
) -> np.ndarray:
    """Erode a 3D binary mask using Euclidean distance in millimetres."""
    binary_mask = np.asarray(mask, dtype=bool)
    if binary_mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {binary_mask.shape}")
    radius = float(radius_mm)
    if not math.isfinite(radius) or radius < 0:
        raise ValueError(f"Invalid erosion radius: {radius_mm}")
    if radius == 0:
        return binary_mask.astype(np.uint8, copy=True)

    spacing = normalize_spacing_mm(spacing_mm)
    padded_mask = np.pad(binary_mask, 1, mode="constant", constant_values=False)
    distance_mm = distance_transform_edt(padded_mask, sampling=spacing)
    eroded = distance_mm[1:-1, 1:-1, 1:-1] > radius
    return eroded.astype(np.uint8, copy=False)


def permute_spatial_mask(mask: np.ndarray, image_permutation: Sequence[int]) -> np.ndarray:
    """Apply a channel-aware image permutation to a 3D spatial mask."""
    binary_mask = np.asarray(mask)
    if binary_mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {binary_mask.shape}")
    permutation = tuple(int(axis) for axis in image_permutation)
    if len(permutation) != 4 or permutation[0] != 0:
        raise ValueError(f"Invalid image permutation: {permutation}")
    return np.transpose(binary_mask[None, ...], axes=permutation)[0]


def resample_label_mask_to_image_grid(
    mask_path: str | Path,
    image_path: str | Path,
    *,
    label_value: int | None = None,
    label_range: tuple[int, int] | None = None,
) -> np.ndarray:
    """Map a labeled mask into an image grid using physical coordinates."""
    mask_image = sitk.ReadImage(str(mask_path))
    reference_image = sitk.ReadImage(str(image_path))
    mask_array = sitk.GetArrayFromImage(mask_image)
    if label_value is not None:
        binary_array = mask_array == int(label_value)
    elif label_range is not None:
        lower, upper = (int(label_range[0]), int(label_range[1]))
        binary_array = (mask_array >= min(lower, upper)) & (mask_array <= max(lower, upper))
    else:
        binary_array = mask_array > 0

    binary_image = sitk.GetImageFromArray(binary_array.astype(np.uint8, copy=False))
    binary_image.CopyInformation(mask_image)
    resampled = sitk.Resample(
        binary_image,
        reference_image,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    resampled_array = sitk.GetArrayFromImage(resampled)
    if resampled_array.ndim != 3:
        raise ValueError(
            f"Expected resampled 3D mask, got shape {resampled_array.shape} "
            f"for {mask_path} -> {image_path}"
        )
    return np.transpose(resampled_array, (1, 2, 0)).astype(np.uint8, copy=False)
