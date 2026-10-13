"""Evaluation protocol summaries for REFORM-3D."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


EVALUATION_SUMMARY_ITEMS = [
    {
        "id": "acquisition_resampling_pipeline",
        "description": "Controlled resampling pipeline and sensitivity to interpolation, anti-aliasing, intensity scaling, and orientation.",
        "artifacts": ["acquisition_protocol.json", "resampling_sensitivity.json"],
        "status": "implemented_as_protocol_and_sweep_surface",
    },
    {
        "id": "organ_queries_candidate_pools",
        "description": "Organ-level CT-MR query construction and controls for FOV/context artifacts.",
        "artifacts": ["retrieval_pool_diagnostics.json", "context_ablation_retrieval.json"],
        "status": "implemented_as_organ_pool_diagnostics_and_context_ablation",
    },
    {
        "id": "confidence_intervals_repeated_seed",
        "description": "Confidence intervals and repeated-seed variability for key metrics.",
        "artifacts": ["repeated_seed_summary.json", "statistical_rigor_summary.json"],
        "status": "implemented_with_separate_repeated_seed_and_bootstrap_reporting",
    },
    {
        "id": "spacing_aware_pretraining",
        "description": "Physical crop extents, anisotropy handling, multi-scale cropping, and tokenization.",
        "artifacts": ["spacing_aware_pretraining_protocol.json"],
        "status": "implemented_as_traceable_protocol_artifact",
    },
    {
        "id": "extra_acquisition_perturbations",
        "description": "Robustness beyond spacing: reconstruction kernel, noise/dose, and plane reformatting.",
        "artifacts": ["extra_perturbation_robustness.json"],
        "status": "implemented_as_deterministic_controlled_perturbation_variants",
    },
    {
        "id": "anatomical_support_size",
        "description": "Support samples per organ, imbalance handling, and trends across support sizes.",
        "artifacts": ["support_size_sweep.json", "support_size_sweep.csv"],
        "status": "implemented_as_balanced_case_disjoint_support_sweep",
    },
    {
        "id": "other_modality_pairs",
        "description": "Cross-modal alignment beyond CT-MR, including CT-PET.",
        "artifacts": ["ct_pet_alignment.json"],
        "status": "implemented_as_dataset_surface_or_corpus_constraint_report",
    },
    {
        "id": "pretraining_eval_leakage",
        "description": "Case-level leakage, duplicate patients, near-duplicates, and site-level partitioning.",
        "artifacts": ["leakage_audit.json"],
        "status": "implemented_as_manifest_level_and_optional_fingerprint_audit",
    },
]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_spacing_aware_pretraining_protocol() -> Dict[str, Any]:
    return {
        "artifact": "spacing_aware_pretraining_protocol",
        "crop_transform": "dinov2.data.spacing_aware_transforms.RandPhysicalCropd",
        "global_crops": {
            "inplane_relative_scale": [0.48, 1.0],
            "throughplane_physical_range_mm": [96.0, 224.0],
        },
        "local_crops": {
            "inplane_relative_scale": [0.16, 0.48],
            "throughplane_physical_range_mm": [32.0, 80.0],
        },
        "anisotropy_handling": {
            "slice_axis": "axis with largest spacing after CropForegroundSwapSliceDimsV2",
            "throughplane_voxels": "round(sampled_mm / slice_spacing)",
            "auto_adjust_anisotropy": "sqrt(slice_spacing / mean_inplane_spacing) when ratio > threshold",
            "cross_plane_rotation": "gated by anisotropy threshold",
        },
        "tokenization": {
            "model_input": "fixed cubic crop resized to checkpoint crop size",
            "patch_embedding": "standard ViT patchification after preprocessing; physical spacing is represented through crop geometry, not extra tokens",
        },
        "eval_mapping": {
            "orientation": "same largest-spacing-axis-to-depth convention unless --orientation-policy identity is used for sensitivity",
            "intensity": "same percentile scaling to [-1, 1]",
            "foreground": "same foreground threshold policy",
        },
    }


def build_statistical_rigor_summary(output_root: Path) -> Dict[str, Any]:
    seed_summary = output_root / "seed_summary" / "repeated_seed_summary.json"
    return {
        "artifact": "statistical_rigor_summary",
        "repeated_seed_variability": {
            "path": str(seed_summary),
            "available": seed_summary.exists(),
            "interpretation": "Mean/std/min/max/n are across full seeded module runs.",
        },
        "bootstrap_uncertainty": {
            "available": "module_metric_payload_dependent",
            "interpretation": "Bootstrap intervals are within-seed uncertainty and must not be merged with repeated-seed variability.",
        },
        "confidence_interval_policy": "Report repeated-seed variability for headline metrics; report bootstrap 95% CIs only as conditional diagnostics.",
    }


def write_evaluation_summary(
    output_root: Path,
    *,
    module_name: str,
    dataset: str,
    manifest_variant: str,
) -> Dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact": "evaluation_summary",
        "module_name": module_name,
        "dataset": dataset,
        "manifest_variant": manifest_variant,
        "items": EVALUATION_SUMMARY_ITEMS,
    }
    paths = {
        "evaluation_summary": output_root / "evaluation_summary.json",
        "statistical_rigor": output_root / "statistical_rigor_summary.json",
        "spacing_protocol": output_root / "spacing_aware_pretraining_protocol.json",
    }
    _write_json(paths["evaluation_summary"], payload)
    _write_json(paths["statistical_rigor"], build_statistical_rigor_summary(output_root.parent))
    _write_json(paths["spacing_protocol"], build_spacing_aware_pretraining_protocol())
    return paths


__all__ = [
    "EVALUATION_SUMMARY_ITEMS",
    "build_spacing_aware_pretraining_protocol",
    "build_statistical_rigor_summary",
    "write_evaluation_summary",
]
