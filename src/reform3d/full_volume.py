"""Uncropped, physically isotropic inputs for qualitative feature examples."""
from __future__ import annotations

import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


METHOD = "full-volume-rastensor-v1"


def _fingerprint(path):
    path = Path(path).expanduser().resolve(strict=True)
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _validate_affine(affine):
    affine = np.asarray(affine, dtype=float)
    if (affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.allclose(affine[3], [0, 0, 0, 1], rtol=0, atol=1e-8)
            or abs(np.linalg.det(affine[:3, :3])) < 1e-12):
        raise ValueError("Source geometry requires a finite, invertible 3D affine")
    return affine


def _finite(array):
    # Slice-wise checks avoid another volume-sized allocation for native CTs.
    return all(np.isfinite(plane).all() for plane in array)


def _bounds(shape, affine):
    corners = np.indices((2, 2, 2)).reshape(3, -1) * np.asarray(shape)[:, None] - .5
    physical = affine[:3, :3] @ corners + affine[:3, 3, None]
    return physical.min(1), physical.max(1)


def read_source(image, mask=None):
    """Read a scalar NIfTI without cropping or changing its physical geometry.

    Arrays use closest-RAS XYZ voxel axes. Obliquity is retained in the affine;
    reorientation changes the axis order, never the sampled anatomy.
    """
    image_record = _fingerprint(image)
    im = nib.load(image_record["path"])
    if len(im.shape) != 3 or any(n <= 0 for n in im.shape):
        raise ValueError("Expected a nonempty scalar 3D image")
    if np.issubdtype(im.get_data_dtype(), np.complexfloating):
        raise ValueError("Complex-valued images are unsupported")
    original_affine = _validate_affine(im.affine)
    canonical = nib.as_closest_canonical(im)
    affine = _validate_affine(canonical.affine)
    volume = np.asarray(canonical.dataobj, dtype=np.float32)
    if not _finite(volume):
        raise ValueError("Image values must be finite")

    mask_record = _fingerprint(mask) if mask is not None else None
    if mask_record is None:
        labels = np.zeros(volume.shape, dtype=bool)
    else:
        seg = nib.load(mask_record["path"])
        if (seg.shape != im.shape or not np.allclose(_validate_affine(seg.affine), original_affine,
                                                   rtol=0, atol=1e-5)):
            raise ValueError("Image/mask geometry mismatch")
        if np.issubdtype(seg.get_data_dtype(), np.complexfloating):
            raise ValueError("Complex-valued mask labels are unsupported")
        data = np.asarray(nib.as_closest_canonical(seg).dataobj)
        if not _finite(data) or any((plane < 0).any() for plane in data):
            raise ValueError("Mask labels must be finite and nonnegative")
        labels = data > 0
    low, high = _bounds(volume.shape, affine)
    metadata = {
        "image": image_record, "mask": mask_record,
        "source_shape": list(volume.shape), "source_affine_ras": affine.tolist(),
        "source_spacing_mm": np.linalg.norm(affine[:3, :3], axis=0).tolist(),
        "source_bounds_ras_mm": {"low": low.tolist(), "high": high.tolist()},
        "original_shape": list(im.shape), "original_affine_ras": original_affine.tolist(),
        "axis_convention": "closest RAS; native obliquity retained",
    }
    return {"volume": volume, "mask": labels, "affine_ras": affine, "metadata": metadata}


def prepare_full_volume(image, mask=None, *, max_axis=224, patch_size=16, max_patches=2744):
    """Resample the complete source field of view onto an isotropic RAS grid.

    Patch-aligned symmetric padding preserves all source voxel cells. Labels
    supply display contours only; they never select the grid or intensities.
    The input is shared by all three encoders, independently of their names.
    """
    if type(patch_size) is not int or patch_size <= 0:
        raise ValueError("patch_size must be a positive integer")
    if type(max_axis) is not int or max_axis <= 0 or max_axis % patch_size:
        raise ValueError("max_axis must be a positive multiple of patch_size")
    if type(max_patches) is not int or max_patches <= 0:
        raise ValueError("max_patches must be a positive integer")
    source = read_source(image, mask)
    raw, metadata = source["volume"], source["metadata"]
    low = np.asarray(metadata["source_bounds_ras_mm"]["low"])
    high = np.asarray(metadata["source_bounds_ras_mm"]["high"])
    spacing = float(np.max(high - low) / max_axis)
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Source physical bounds must have positive finite extent")
    shape = np.ceil((high - low) / spacing - 1e-10).astype(int)
    shape = ((shape + patch_size - 1) // patch_size) * patch_size
    grid = shape // patch_size
    if np.prod(grid, dtype=np.int64) > max_patches:
        raise ValueError(f"Full-volume grid exceeds max_patches={max_patches}; reduce max_axis")
    affine = np.diag([spacing, spacing, spacing, 1.])
    affine[:3, 3] = (low + high) / 2 - shape * spacing / 2 + spacing / 2
    transform = np.linalg.solve(source["affine_ras"], affine)
    intensity_low, intensity_high = np.percentile(raw, [.05, 99.95])
    if not np.isfinite([intensity_low, intensity_high]).all() or intensity_high <= intensity_low:
        raise ValueError("Constant/degenerate intensity distribution")

    native_spacing = np.asarray(metadata["source_spacing_mm"])
    sigma = .5 * np.sqrt(np.maximum((spacing / native_spacing) ** 2 - 1, 0))
    filtered = ndimage.gaussian_filter(raw, sigma=sigma, mode="nearest") if (sigma > 0).any() else raw
    volume = ndimage.affine_transform(filtered, transform[:3, :3], transform[:3, 3],
                                     output_shape=tuple(shape), order=1, mode="constant",
                                     cval=float(intensity_low), prefilter=False)
    volume = np.ascontiguousarray((np.clip(volume, intensity_low, intensity_high) - intensity_low)
                                  / (intensity_high - intensity_low) * 2 - 1, dtype=np.float32)
    labels = ndimage.affine_transform(source["mask"].astype(np.uint8), transform[:3, :3],
                                     transform[:3, 3], output_shape=tuple(shape), order=0,
                                     mode="constant", cval=0, prefilter=False).astype(bool)
    # The display support depends only on the image, never on ground truth.
    foreground = volume > -0.95
    if not foreground.any():
        raise ValueError("Image foreground disappeared during full-volume resampling")
    model_low, model_high = _bounds(shape, affine)
    if (model_low > low + 1e-7).any() or (model_high < high - 1e-7).any():
        raise ValueError("Full-volume sampling grid truncates the source field of view")
    metadata = {
        **metadata, "method": METHOD, "no_crop": True,
        "affine_ras": affine.tolist(), "model_shape": shape.tolist(),
        "spacing_mm": spacing, "grid_shape": grid.tolist(),
        "model_bounds_ras_mm": {"low": model_low.tolist(), "high": model_high.tolist()},
        "normalization": {"percentiles": [.05, 99.95], "low": float(intensity_low),
                          "high": float(intensity_high)},
        "antialias_sigma_source_vox": sigma.tolist(),
        "max_axis": max_axis, "patch_size": patch_size, "max_patches": max_patches,
        "input_sha256": hashlib.sha256(volume.tobytes()).hexdigest(),
    }
    return {"volume": volume, "mask": labels, "foreground": foreground, "metadata": metadata}
