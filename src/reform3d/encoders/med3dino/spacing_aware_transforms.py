"""
Spacing-aware transforms for handling extreme anisotropy in medical imaging datasets.

This module provides transforms that respect physical spacing to ensure:
1. Consistent anatomical extent across varying resolutions
2. Proper handling of extreme anisotropy (up to 5×+ ratios)
3. Rotation-aware augmentation that gates cross-plane rotations
4. Unified FOV normalization (both in-plane and through-plane)
5. Cache-time foreground validation (strict rejection of invalid samples)

Architecture:
- ForegroundValidationAndBBox: Cache-time validation + bbox computation
- RandPhysicalCropd: Unconstrained random sampling with physical-scale crop
- Result: Diverse spatial views for DINO multi-crop learning

Literature references:
- nnU-Net (Isensee et al., Nature Methods 2021): Foreground-guided sampling
- MONAI CropForeground: Intensity-based foreground detection  
- Self-Training (Zoph et al., NeurIPS 2020): Rejection sampling for quality control

Author: Med3DINO team
License: CC BY-NC-ND 4.0
"""

import logging
import math
import numpy as np
import torch
from monai.transforms import MapTransform, CropForeground, Randomizable
from torch.nn.functional import interpolate

logger = logging.getLogger("dinov2")


class InvalidSampleError(ValueError):
    """Raised when a sample fails foreground validation during caching."""
    pass


class RandPhysicalCropd(Randomizable, MapTransform):
    """
    Hybrid random crop: relative in-plane + physical through-plane.
    
    Strategy:
    - In-plane (H, W): Use relative scale (% of dimension)
    - Through-plane (D): Use physical scale (mm) with anisotropy adjustment
    
    Args:
        keys: Keys to apply transform to
        inplane_relative_scale: (min, max) relative scale for H, W dimensions
        throughplane_physical_range_mm: (min, max) physical FOV in mm for D dimension
        output_size_voxels: Final crop size in voxels after resizing
        auto_adjust_anisotropy: If True, scale through-plane by sqrt(anisotropy_ratio)
        interpolation: Interpolation mode for resizing
    """

    def __init__(
        self,
        keys,
        inplane_relative_scale=(0.48, 1.0),
        throughplane_physical_range_mm=(96.0, 224.0),
        output_size_voxels=(96, 96, 96),
        auto_adjust_anisotropy=True,
        interpolation='trilinear',
        allow_missing_keys=False
    ):
        super().__init__(keys, allow_missing_keys)
        self.inplane_scale_min, self.inplane_scale_max = inplane_relative_scale
        self.throughplane_mm_min, self.throughplane_mm_max = throughplane_physical_range_mm
        self.output_size_voxels = self._parse_size(output_size_voxels, "output_size_voxels")
        self.auto_adjust_anisotropy = auto_adjust_anisotropy
        self.interpolation = interpolation
        self._slices = ()

    def _parse_size(self, val, name):
        if isinstance(val, int):
            return (val, val, val)
        if isinstance(val, (list, tuple)) and len(val) == 3:
            return tuple(val)
        raise ValueError(f"Unsupported format for {name}: {val}")

    def randomize(self, image_shape, spacing, anisotropy_ratio=1.0):
        """
        Compute random crop slices (unconstrained random location).
        
        Args:
            image_shape: (H, W, D) spatial dimensions
            spacing: [sx, sy, sz] voxel spacing in mm
            anisotropy_ratio: Ratio of max/min spacing (for auto-adjustment)
        """
        # In-plane dimensions (H, W): Use relative scale
        inplane_scale = self.R.uniform(self.inplane_scale_min, self.inplane_scale_max)
        crop_h = int(round(image_shape[0] * inplane_scale))
        crop_w = int(round(image_shape[1] * inplane_scale))
        
        # Through-plane dimension (D): Use physical scale with anisotropy adjustment
        throughplane_mm = self.R.uniform(self.throughplane_mm_min, self.throughplane_mm_max)
        
        if self.auto_adjust_anisotropy and anisotropy_ratio > 1.5:
            adjustment_factor = np.sqrt(anisotropy_ratio)
            throughplane_mm = throughplane_mm * adjustment_factor
            logger.debug(f"Anisotropy adjustment: {throughplane_mm/adjustment_factor:.1f}mm -> {throughplane_mm:.1f}mm")
        
        # Convert physical FOV to voxels
        slice_spacing = spacing[2]
        crop_d = int(round(throughplane_mm / slice_spacing)) if slice_spacing > 0 else image_shape[2]
        
        # Clamp to image dimensions
        final_crop_size = [
            max(1, min(crop_h, image_shape[0])),
            max(1, min(crop_w, image_shape[1])),
            max(1, min(crop_d, image_shape[2]))
        ]
        
        # Unconstrained random crop
        from monai.data.utils import get_random_patch, get_valid_patch_size
        valid_size = get_valid_patch_size(image_shape, final_crop_size)
        self._slices = get_random_patch(image_shape, valid_size, self.R)

    def __call__(self, data):
        """Apply random physical crop to all specified keys."""
        d = dict(data)
        
        for key in self.keys:
            img = d[key]
            spacing = d.get(f"{key}_meta_dict", {}).get("spacing", d.get("spacing", [1.0, 1.0, 1.0]))
            anisotropy_ratio = d.get('anisotropy_ratio', 1.0)
            
            # Get spatial dimensions
            if img.ndim == 4:  # (C, H, W, D)
                img_shape_spatial = img.shape[1:]
                has_channel = True
            else:  # (H, W, D)
                img_shape_spatial = img.shape
                has_channel = False

            # Sample crop location
            self.randomize(img_shape_spatial, spacing, anisotropy_ratio)
            
            # Apply crop
            if has_channel:
                cropped_img = img[:, self._slices[0], self._slices[1], self._slices[2]]
            else:
                cropped_img = img[self._slices[0], self._slices[1], self._slices[2]]

            # Resize to output size
            if has_channel:
                to_interpolate = cropped_img.unsqueeze(0)
            else:
                to_interpolate = cropped_img.unsqueeze(0).unsqueeze(0)
            
            resized = interpolate(
                to_interpolate,
                size=self.output_size_voxels,
                mode=self.interpolation,
                align_corners=None
            )
            
            if has_channel:
                d[key] = resized.squeeze(0)
            else:
                d[key] = resized.squeeze(0).squeeze(0)
        
        return d


class ForegroundValidationAndBBox(MapTransform):
    """
    Cache-time transform: validates foreground content and computes bounding box.
    
    NOTE: This class is kept for OFFLINE validation only.
    It is NOT used in the runtime training pipeline.
    Use scripts/utils/validate_manifest_foreground.py for pre-validation.
    
    Run this during the DETERMINISTIC (cached) phase of the pipeline.
    Invalid samples raise InvalidSampleError -> excluded from training.
    
    Args:
        keys: Keys to validate
        min_foreground_ratio: Minimum foreground fraction (default: 1%)
        min_intensity_variance: Minimum variance (default: 1e-5)
        background_threshold: Background intensity threshold (default: -0.95)
        reject_invalid: If True, raise error for invalid samples
    
    Stores in data dict:
        - 'foreground_bbox': ((min_h, min_w, min_d), (max_h, max_w, max_d))
        - 'foreground_ratio': float
        - 'intensity_variance': float
        - 'is_valid_sample': bool
    """
    
    def __init__(
        self,
        keys,
        min_foreground_ratio=0.01,
        min_intensity_variance=1e-5,
        background_threshold=-0.95,
        reject_invalid=True
    ):
        super().__init__(keys)
        self.min_foreground_ratio = min_foreground_ratio
        self.min_intensity_variance = min_intensity_variance
        self.background_threshold = background_threshold
        self.reject_invalid = reject_invalid
    
    def __call__(self, data):
        d = dict(data)
        
        for key in self.keys:
            img = d[key]
            
            # Handle channel dimension
            spatial_img = img[0] if img.ndim == 4 else img
            
            # Compute statistics
            flat = spatial_img.flatten().float()
            variance = flat.var().item()
            
            # Foreground mask and ratio
            fg_mask = spatial_img > self.background_threshold
            foreground_ratio = fg_mask.float().mean().item()
            
            d['foreground_ratio'] = foreground_ratio
            d['intensity_variance'] = variance
            
            # Compute bounding box
            fg_coords = torch.nonzero(fg_mask, as_tuple=False)
            
            if len(fg_coords) > 0:
                bbox_min = fg_coords.min(dim=0).values.tolist()
                bbox_max = fg_coords.max(dim=0).values.tolist()
                d['foreground_bbox'] = (tuple(bbox_min), tuple(bbox_max))
            else:
                d['foreground_bbox'] = None
            
            # Validate
            is_valid = (
                foreground_ratio >= self.min_foreground_ratio and
                variance >= self.min_intensity_variance and
                d['foreground_bbox'] is not None
            )
            d['is_valid_sample'] = is_valid
            
            if not is_valid and self.reject_invalid:
                sample_id = d.get('image_path', d.get('image', 'unknown'))
                raise InvalidSampleError(
                    f"Sample rejected: {sample_id}\n"
                    f"  foreground_ratio={foreground_ratio:.4f} (min={self.min_foreground_ratio})\n"
                    f"  variance={variance:.2e} (min={self.min_intensity_variance})\n"
                    f"  has_bbox={d['foreground_bbox'] is not None}"
                )
            
            logger.debug(f"ForegroundValidation: valid={is_valid}, fg={foreground_ratio:.4f}, var={variance:.2e}")
        
        return d


class CropForegroundSwapSliceDimsV2(CropForeground):
    """
    Enhanced CropForeground that:
    1. Identifies slice axis via spacing (largest spacing = slice axis)
    2. Swaps to (C, H, W, D) format where D is slice axis
    3. Stores anisotropy metadata for downstream transforms
    """
    
    def __init__(self, *args, anisotropy_threshold=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.anisotropy_threshold = anisotropy_threshold
        
    @staticmethod
    def get_permutation_and_metadata(spacing, isotropic_threshold=0.1):
        spacing_arr = np.array(spacing)
        
        if np.any(spacing_arr <= 0):
            raise ValueError(f"Invalid spacing: {spacing_arr}")
        
        spacing_ratio = spacing_arr.max() / spacing_arr.min()
        is_isotropic = (spacing_ratio - 1.0) <= isotropic_threshold
        
        slice_axis = 2 if is_isotropic else int(np.argmax(spacing_arr))
        inplane_axes = [ax for ax in (0, 1, 2) if ax != slice_axis]
        
        slice_spacing = spacing_arr[slice_axis]
        inplane_spacing = spacing_arr[inplane_axes]
        anisotropy_ratio = slice_spacing / np.mean(inplane_spacing)
        
        perm = [0] + [ax + 1 for ax in inplane_axes] + [slice_axis + 1]
        
        metadata = {
            'slice_axis_original': slice_axis,
            'anisotropy_ratio': float(anisotropy_ratio),
            'slice_spacing': float(slice_spacing),
            'inplane_spacing_mean': float(np.mean(inplane_spacing)),
            'is_isotropic': is_isotropic,
        }
        
        return perm, metadata
    
    def __call__(self, img_dict, mode=None, lazy=None, **pad_kwargs):
        img_spacing = img_dict.get('spacing')
        img = img_dict['image']
        
        if img_spacing is None:
            logger.warning("No spacing metadata, using shape-based permutation")
            img_spacing = img.shape[1:]
        
        perm, metadata = self.get_permutation_and_metadata(img_spacing)
        
        # if metadata['anisotropy_ratio'] > self.anisotropy_threshold:
        #     logger.warning(f"Extreme anisotropy: {metadata['anisotropy_ratio']:.2f}x")
        
        img = img.permute(*perm)
        cropped_img = super().__call__(img, mode, lazy, **pad_kwargs)
        
        result = dict(img_dict)
        result['image'] = cropped_img
        result['anisotropy_ratio'] = metadata['anisotropy_ratio']
        result['slice_axis_original'] = metadata['slice_axis_original']
        result['slice_spacing'] = metadata['slice_spacing']
        result['inplane_spacing_mean'] = metadata['inplane_spacing_mean']
        
        if 'spacing' in result:
            spacing_orig = np.array(img_spacing)
            slice_axis = metadata['slice_axis_original']
            inplane_axes = [ax for ax in (0, 1, 2) if ax != slice_axis]
            result['spacing_permuted'] = [
                float(spacing_orig[inplane_axes[0]]),
                float(spacing_orig[inplane_axes[1]]),
                float(spacing_orig[slice_axis])
            ]
        
        return result


class UnifiedFOVNormalized(MapTransform):
    """Normalize physical FOV to consistent ranges."""
    
    def __init__(self, keys, target_inplane_fov_mm=(200.0, 300.0), target_throughplane_fov_mm=(180.0, 220.0)):
        super().__init__(keys)
        self.inplane_min, self.inplane_max = target_inplane_fov_mm
        self.throughplane_min, self.throughplane_max = target_throughplane_fov_mm
        
    def __call__(self, data):
        d = dict(data)
        spacing_permuted = d.get('spacing_permuted')
        
        if spacing_permuted is None:
            logger.warning("No spacing_permuted, skipping UnifiedFOVNormalize")
            return d
        
        sx, sy, sz = spacing_permuted
        image = d["image"]
        C, H, W, D = image.shape
        
        fov_h, fov_w, fov_d = H * sx, W * sy, D * sz
        
        target_fov_h = float(np.clip(fov_h, self.inplane_min, self.inplane_max))
        target_fov_w = float(np.clip(fov_w, self.inplane_min, self.inplane_max))
        target_fov_d = float(np.clip(fov_d, self.throughplane_min, self.throughplane_max))
        
        target_H = max(1, int(round(target_fov_h / sx)))
        target_W = max(1, int(round(target_fov_w / sy)))
        target_D = max(1, int(round(target_fov_d / sz)))
        
        indices_h = np.linspace(0, H - 1, target_H).astype(np.int32)
        indices_w = np.linspace(0, W - 1, target_W).astype(np.int32)
        indices_d = np.linspace(0, D - 1, target_D).astype(np.int32)
        
        img_norm = image[:, indices_h, :, :][:, :, indices_w, :][:, :, :, indices_d]
        
        for key in self.keys:
            d[key] = img_norm
        
        d['fov_normalized'] = True
        d['target_fov'] = (target_fov_h, target_fov_w, target_fov_d)
        
        return d


class AnisotropyGatedRotation(MapTransform, Randomizable):
    """Rotation augmentation that gates cross-plane rotations based on anisotropy."""
    
    def __init__(self, keys, inplane_prob=0.5, crossplane_prob=0.3, anisotropy_threshold=1.5):
        super().__init__(keys)
        self.inplane_prob = inplane_prob
        self.crossplane_prob = crossplane_prob
        self.threshold = anisotropy_threshold
        self.R = np.random.RandomState()
        
    def randomize(self, data=None):
        self.R = np.random.RandomState()
        
    def __call__(self, data):
        from monai.transforms import RandRotate90d
        
        d = dict(data)
        anisotropy_ratio = d.get('anisotropy_ratio', 1.0)
        
        if self.R.random() < self.inplane_prob:
            d = RandRotate90d(keys=self.keys, prob=1.0, spatial_axes=(0, 1))(d)
        
        if anisotropy_ratio < self.threshold:
            if self.R.random() < self.crossplane_prob:
                d = RandRotate90d(keys=self.keys, prob=1.0, spatial_axes=(1, 2))(d)
            if self.R.random() < self.crossplane_prob:
                d = RandRotate90d(keys=self.keys, prob=1.0, spatial_axes=(0, 2))(d)
        
        return d
