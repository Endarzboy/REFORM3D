"""Acquisition Robustness protocol artifacts.

These helpers describe preprocessing, controlled resampling, sensitivity
variants, and non-spacing perturbation settings before heavy feature extraction
is launched.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Sequence

from .config import ControlledPerturbationConfig, PREPROCESSING_CONFIG


DEFAULT_EXTRA_PERTURBATIONS = (
    "kernel_blur_sigma1.0",
    "kernel_sharpen",
    "dose_noise_sigma0.05",
    "plane_reformat_coronal",
    "plane_reformat_sagittal",
)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_physical_fov_mm(values: Sequence[float] | None) -> tuple[float, float, float] | None:
    if values is None:
        return None
    if len(values) != 3:
        raise ValueError("--physical-fov-mm requires exactly three values")
    return tuple(float(value) for value in values)


def apply_controlled_perturbation_overrides(
    config: ControlledPerturbationConfig,
    *,
    orientation_policy: str = "spacing_permutation",
    physical_fov_mm: Sequence[float] | None = None,
    extra_perturbations: bool | Sequence[str] = False,
    interpolation_mode: str | None = None,
    anti_alias: bool | None = None,
) -> ControlledPerturbationConfig:
    """Return a config copy with acquisition-protocol overrides applied."""
    if extra_perturbations is True:
        extras = list(DEFAULT_EXTRA_PERTURBATIONS)
    elif extra_perturbations:
        extras = [str(name) for name in extra_perturbations]
    else:
        extras = None

    return replace(
        config,
        orientation_policy=orientation_policy,
        physical_fov_mm=parse_physical_fov_mm(physical_fov_mm),
        extra_perturbations=extras,
        interpolation_mode=interpolation_mode or config.interpolation_mode,
        anti_alias=config.anti_alias if anti_alias is None else bool(anti_alias),
    )


def build_acquisition_protocol_payload(
    *,
    dataset: str,
    manifest_variant: str,
    manifest_path: Path,
    perturbation_config: ControlledPerturbationConfig,
    checkpoints: Sequence[str],
    feature_types: Sequence[str],
) -> Dict[str, Any]:
    """Build a machine-readable acquisition protocol statement."""
    preprocessing = {
        "load": "MONAI LoadImaged(image_only=False) + EnsureChannelFirstd",
        "nan_policy": "torch.nan_to_num(image, nan=torch.nanmean(image))",
        "intensity_rescaling": {
            "transform": "ScaleIntensityRangePercentilesd",
            "lower_percentile": PREPROCESSING_CONFIG.intensity_percentile_lower,
            "upper_percentile": PREPROCESSING_CONFIG.intensity_percentile_upper,
            "output_min": PREPROCESSING_CONFIG.intensity_output_min,
            "output_max": PREPROCESSING_CONFIG.intensity_output_max,
            "clip": True,
        },
        "foreground_crop": {
            "enabled": PREPROCESSING_CONFIG.crop_foreground,
            "threshold": PREPROCESSING_CONFIG.foreground_threshold,
        },
        "model_crop_size": PREPROCESSING_CONFIG.crop_size,
        "orientation_standardization": {
            "policy": perturbation_config.orientation_policy,
            "spacing_permutation_definition": "largest physical spacing axis is moved to depth; the two remaining axes are in-plane",
            "ras_lps_affine_canonicalization": False,
        },
    }
    resampling = {
        "protocol_name": perturbation_config.protocol_name,
        "source_bin": perturbation_config.source_bin,
        "target_variants": perturbation_config.describe_targets(),
        "interpolation_mode": perturbation_config.interpolation_mode,
        "align_corners": False if perturbation_config.interpolation_mode in {"linear", "bilinear", "bicubic", "trilinear"} else None,
        "anti_alias": {
            "enabled": perturbation_config.anti_alias,
            "filter": "scipy.ndimage.gaussian_filter before downsampling",
            "sigma_source_voxels": "max((target_spacing / source_spacing - 1) / 2, 0) per axis",
        },
        "physical_fov_mm": list(perturbation_config.physical_fov_mm) if perturbation_config.physical_fov_mm else None,
        "cache_signature": perturbation_config.cache_signature(),
    }
    return {
        "artifact": "acquisition_protocol",
        "dataset": dataset,
        "manifest_variant": manifest_variant,
        "manifest_path": str(manifest_path),
        "checkpoints": list(checkpoints),
        "feature_types": list(feature_types),
        "preprocessing": preprocessing,
        "controlled_resampling": resampling,
        "non_spacing_perturbations": list(perturbation_config.extra_perturbations or []),
    }


def build_resampling_sensitivity_payload(
    *,
    dataset: str,
    manifest_variant: str,
    baseline_config: ControlledPerturbationConfig,
) -> Dict[str, Any]:
    """Enumerate controlled-resampling sensitivity variants."""
    baseline_config = replace(baseline_config, extra_perturbations=None)
    variants = [
        ("baseline", baseline_config),
        ("no_antialias", apply_controlled_perturbation_overrides(baseline_config, anti_alias=False)),
        ("nearest_interpolation", apply_controlled_perturbation_overrides(baseline_config, interpolation_mode="nearest")),
        (
            "identity_orientation",
            apply_controlled_perturbation_overrides(baseline_config, orientation_policy="identity"),
        ),
        (
            "fixed_physical_fov_192mm",
            apply_controlled_perturbation_overrides(baseline_config, physical_fov_mm=(192.0, 192.0, 192.0)),
        ),
    ]
    return {
        "artifact": "resampling_sensitivity",
        "dataset": dataset,
        "manifest_variant": manifest_variant,
        "baseline_cache_signature": baseline_config.cache_signature(),
        "sensitivity_axes": [
            "interpolation_mode",
            "anti_alias",
            "orientation_policy",
            "physical_fov_mm",
        ],
        "variants": [
            {
                "name": name,
                "protocol_name": config.protocol_name,
                "interpolation_mode": config.interpolation_mode,
                "anti_alias": config.anti_alias,
                "orientation_policy": config.orientation_policy,
                "physical_fov_mm": list(config.physical_fov_mm) if config.physical_fov_mm else None,
                "cache_signature": config.cache_signature(),
                "target_variants": config.describe_targets(),
            }
            for name, config in variants
        ],
        "interpretation": "Compare headline controlled-drift, CKA, and semantic-transfer metrics across variants; repeated-seed summaries remain separate.",
    }


def build_extra_perturbation_payload(
    *,
    dataset: str,
    manifest_variant: str,
    perturbation_config: ControlledPerturbationConfig,
) -> Dict[str, Any]:
    extras = list(perturbation_config.extra_perturbations or DEFAULT_EXTRA_PERTURBATIONS)
    return {
        "artifact": "extra_perturbation_robustness",
        "dataset": dataset,
        "manifest_variant": manifest_variant,
        "cache_signature": perturbation_config.cache_signature(),
        "perturbations": [
            {
                "name": name,
                "family": (
                    "reconstruction_kernel"
                    if "kernel" in name
                    else "dose_noise"
                    if "noise" in name
                    else "plane_reformatting"
                ),
                "deterministic": True,
            }
            for name in extras
        ],
        "metric_surface": "Included as additional variants in controlled perturbation bundles when --extra-perturbations is enabled.",
    }


def write_acquisition_protocol_artifacts(
    output_dir: Path,
    *,
    dataset: str,
    manifest_variant: str,
    manifest_path: Path,
    perturbation_config: ControlledPerturbationConfig,
    checkpoints: Sequence[str],
    feature_types: Sequence[str],
    include_sensitivity: bool,
    include_extra_perturbations: bool,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    protocol = build_acquisition_protocol_payload(
        dataset=dataset,
        manifest_variant=manifest_variant,
        manifest_path=manifest_path,
        perturbation_config=perturbation_config,
        checkpoints=checkpoints,
        feature_types=feature_types,
    )
    paths["acquisition_protocol"] = output_dir / "acquisition_protocol.json"
    _write_json(paths["acquisition_protocol"], protocol)

    if include_sensitivity:
        paths["resampling_sensitivity"] = output_dir / "resampling_sensitivity.json"
        _write_json(
            paths["resampling_sensitivity"],
            build_resampling_sensitivity_payload(
                dataset=dataset,
                manifest_variant=manifest_variant,
                baseline_config=perturbation_config,
            ),
        )

    if include_extra_perturbations:
        paths["extra_perturbation_robustness"] = output_dir / "extra_perturbation_robustness.json"
        _write_json(
            paths["extra_perturbation_robustness"],
            build_extra_perturbation_payload(
                dataset=dataset,
                manifest_variant=manifest_variant,
                perturbation_config=perturbation_config,
            ),
        )

    return paths


__all__ = [
    "DEFAULT_EXTRA_PERTURBATIONS",
    "apply_controlled_perturbation_overrides",
    "build_acquisition_protocol_payload",
    "build_extra_perturbation_payload",
    "build_resampling_sensitivity_payload",
    "parse_physical_fov_mm",
    "write_acquisition_protocol_artifacts",
]
