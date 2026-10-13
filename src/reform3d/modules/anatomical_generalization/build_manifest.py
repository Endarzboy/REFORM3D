#!/usr/bin/env python

"""Build Anatomical Generalization manifests from organ-level source manifests.

The command reads a Cross-Modal organ manifest, keeps organs that satisfy the
required modality-support rule, and writes full, sampled, and metadata manifests
for case-disjoint low-shot target-organ evaluation.

Inputs:
    --dataset: dataset surface such as ``amos22_ct_mr``.
    --source-manifest: optional explicit source manifest for custom runs.
    --min-holdout-organs: minimum number of eligible organs.

Example:
    python -m reform3d.modules.anatomical_generalization.build_manifest \
        --dataset amos22_ct_mr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

from .config import (
    DEFAULT_ANATOMICAL_DATASET,
    DEFAULT_MANIFEST_VARIANT,
    DEFAULT_MIN_HOLDOUT_ORGANS,
    DEFAULT_MIN_SAMPLES_PER_MODALITY,
    DEFAULT_REQUIRED_MODALITIES,
    ANATOMICAL_DATASETS,
    get_anatomical_manifest_dir,
    get_cross_modal_manifest_path,
    get_dataset_name_from_manifest_path,
    get_manifest_variant_from_manifest_path,
    normalize_anatomical_dataset_name,
    normalize_manifest_variant,
)
from .data_loader import (
    compute_organ_modality_case_support,
    compute_organ_modality_support,
    get_eligible_holdout_organs,
    load_anatomical_manifest,
    summarize_anatomical_manifest,
    validate_anatomical_manifest,
)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_manifest_payload(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    return {"samples": payload}


def _normalize_required_modalities(values: Sequence[str]) -> list[str]:
    return [str(value).strip().lower() for value in values if str(value).strip()]


def _build_holdout_eligibility_rule(
    required_modalities: Sequence[str],
    min_samples_per_modality: int,
) -> Dict[str, Any]:
    return {
        "required_modalities": list(required_modalities),
        "min_samples_per_modality": int(min_samples_per_modality),
        "definition": (
            "An organ is target eligible iff it has at least min_samples_per_modality unique source cases "
            "in every required modality. Each eligible organ can serve as the low-shot target organ "
            "in a separate target-organ evaluation fold."
        ),
    }


def _infer_surface_stratum(samples: Sequence[Dict[str, Any]], required_modalities: Sequence[str]) -> str:
    if len(required_modalities) == 1:
        return f"{required_modalities[0]}_within_dataset_holdout_surface"
    exposures = {sample.get("pretraining_exposure") for sample in samples if sample.get("pretraining_exposure")}
    if exposures == {"pretraining_unseen_source"}:
        return "pretraining_unseen_ct_mr_validation"
    if "pretraining_seen_source" in exposures and "pretraining_unseen_source" in exposures:
        return "mixed_exposure_ct_mr_anchor"
    if exposures == {"pretraining_seen_source"}:
        return "pretraining_seen_ct_mr_surface"
    return "mixed_or_unknown_exposure_surface"


def _build_serialized_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    raw_sample = sample.get("raw_sample") or {}
    return {
        "sample_id": sample.get("sample_id"),
        "patient_id": sample.get("patient_id"),
        "modality": sample.get("modality"),
        "organ_name": sample.get("organ_name") or sample.get("primary_organ"),
        "organ_family": sample.get("organ_family"),
        "organ_status": sample.get("organ_status") or "holdout_candidate",
        "source_dataset": sample.get("source_dataset") or sample.get("dataset"),
        "pretraining_exposure": sample.get("pretraining_exposure"),
        "evidence_type": sample.get("evidence_type"),
        "image_path": sample.get("image_path") or sample.get("file_path"),
        "mask_path": sample.get("mask_path"),
        "mask_metadata": sample.get("mask_metadata"),
        "source_case_id": sample.get("source_case_id"),
        "split": sample.get("split"),
        "organs": sample.get("organs"),
        "primary_organ": sample.get("primary_organ"),
        "dataset": raw_sample.get("dataset", sample.get("dataset")),
        "file_path": raw_sample.get("file_path", sample.get("file_path")),
        "label_path": raw_sample.get("label_path"),
        "label_voxels": raw_sample.get("label_voxels"),
        "source_mask_path": raw_sample.get("mask_path"),
        "shape": raw_sample.get("shape"),
        "spacing": raw_sample.get("spacing"),
    }


def build_anatomical_manifest(
    *,
    source_manifest: Path,
    analysis_name: str | None = None,
    manifest_variant: str | None = None,
    required_modalities: Sequence[str] = DEFAULT_REQUIRED_MODALITIES,
    min_holdout_organs: int = DEFAULT_MIN_HOLDOUT_ORGANS,
    min_samples_per_modality: int = DEFAULT_MIN_SAMPLES_PER_MODALITY,
    output_dir: Path | None = None,
) -> Path:
    source_manifest = Path(source_manifest).resolve()
    resolved_analysis_name = normalize_anatomical_dataset_name(
        analysis_name or get_dataset_name_from_manifest_path(source_manifest)
    )
    resolved_manifest_variant = normalize_manifest_variant(
        manifest_variant or get_manifest_variant_from_manifest_path(source_manifest)
    )
    resolved_output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else get_anatomical_manifest_dir(resolved_analysis_name, resolved_manifest_variant)
    )
    modalities = _normalize_required_modalities(required_modalities)

    source_payload = _load_manifest_payload(source_manifest)
    source_samples = load_anatomical_manifest(source_manifest)
    validation = validate_anatomical_manifest(
        source_samples,
        required_modalities=modalities,
        min_holdout_organs=min_holdout_organs,
        min_samples_per_modality=min_samples_per_modality,
    )
    if validation["missing_modalities"]:
        raise ValueError(f"Source manifest {source_manifest} is missing required modalities: {validation['missing_modalities']}")

    eligible_holdout_organs = get_eligible_holdout_organs(
        source_samples,
        required_modalities=modalities,
        min_samples_per_modality=min_samples_per_modality,
    )
    if len(eligible_holdout_organs) < int(min_holdout_organs):
        raise ValueError(
            f"Anatomical Generalization requires at least {min_holdout_organs} eligible target organs, "
            f"but {source_manifest} provides {len(eligible_holdout_organs)} ({eligible_holdout_organs})"
        )

    organ_support = compute_organ_modality_support(source_samples, modalities)
    organ_case_support = compute_organ_modality_case_support(source_samples, modalities)
    excluded_organs = {
        organ: {
            "sample_counts": organ_support.get(organ, {}),
            "case_counts": counts,
            "reason": "insufficient_required_modality_target_case_support",
        }
        for organ, counts in organ_case_support.items()
        if organ not in set(eligible_holdout_organs)
    }
    filtered_samples = [
        sample
        for sample in source_samples
        if (sample.get("organ_name") or sample.get("primary_organ")) in set(eligible_holdout_organs)
    ]
    serialized_samples = [_build_serialized_sample(sample) for sample in filtered_samples]
    source_summary = summarize_anatomical_manifest(filtered_samples)
    surface_scope = "within_modality" if len(modalities) == 1 else "cross_modality"
    surface_stratum = _infer_surface_stratum(filtered_samples, modalities)
    holdout_eligibility_rule = _build_holdout_eligibility_rule(modalities, min_samples_per_modality)
    anatomical_protocol = {
        "target_definition": "low_shot_target_organ",
        "target_definition_note": (
            "The target organ receives fixed support examples during adaptation; evaluation uses query examples "
            "from source cases disjoint from all support cases."
        ),
        "fold_policy": "one_target_organ_per_low_shot_fold",
        "support_query_split_policy": "case_disjoint_support_query",
        "holdout_eligibility_rule": holdout_eligibility_rule,
        "surface_scope": surface_scope,
        "cross_modality_surface_stratum": surface_stratum if surface_scope == "cross_modality" else None,
        "within_modality_surface_stratum": surface_stratum if surface_scope == "within_modality" else None,
        "checkpoint_policy": "full_checkpoint_slate",
        "required_modalities": modalities,
        "eligible_holdout_organs": eligible_holdout_organs,
    }

    manifest_payload = {
        "dataset": resolved_analysis_name,
        "description": "Anatomical Generalization manifest for case-disjoint low-shot target-organ evaluation.",
        "evidence_type": source_payload.get("evidence_type", "cohort_level_same_organ_ct_mr"),
        "manifest_variant": resolved_manifest_variant,
        "reform3d_module": "anatomical_generalization",
        "required_modalities": modalities,
        "anatomical_protocol": anatomical_protocol,
        "retained_organs": eligible_holdout_organs,
        "eligible_target_organs": eligible_holdout_organs,
        "fold_eligible_holdout_organs": eligible_holdout_organs,
        "eligible_holdout_organs": eligible_holdout_organs,
        "eligible_holdout_organs_note": holdout_eligibility_rule["definition"],
        "samples": serialized_samples,
        "upstream_manifest": str(source_manifest),
        "upstream_manifest_module": source_payload.get("reform3d_module"),
        "upstream_manifest_contract": "organ_level_samples",
        "source_cross_modal_manifest": (
            str(source_manifest) if source_payload.get("reform3d_module") == "cross_modal_alignment" else None
        ),
    }
    meta_payload = {
        "dataset": resolved_analysis_name,
        "reform3d_module": "anatomical_generalization",
        "manifest_variant": resolved_manifest_variant,
        "required_modalities": modalities,
        "anatomical_protocol": anatomical_protocol,
        "upstream_manifest": str(source_manifest),
        "upstream_manifest_module": source_payload.get("reform3d_module"),
        "upstream_manifest_contract": "organ_level_samples",
        "source_cross_modal_manifest": manifest_payload["source_cross_modal_manifest"],
        "evidence_type": source_payload.get("evidence_type"),
        "eligible_target_organs": eligible_holdout_organs,
        "fold_eligible_holdout_organs": eligible_holdout_organs,
        "eligible_holdout_organs": eligible_holdout_organs,
        "eligible_holdout_organs_note": holdout_eligibility_rule["definition"],
        "excluded_organs": excluded_organs,
        "support_by_organ": organ_support,
        "case_support_by_organ": organ_case_support,
        "min_holdout_organs": int(min_holdout_organs),
        "min_samples_per_modality": int(min_samples_per_modality),
        "validation": validation,
        "summary": source_summary,
        "total_samples_full": len(serialized_samples),
        "total_samples_sampled": len(serialized_samples),
        "sampling": {"applied": False, "reason": "manifest_sampled_equals_manifest_full", "strategy": "none"},
    }

    _write_json(resolved_output_dir / "manifest_full.json", manifest_payload)
    _write_json(resolved_output_dir / "manifest_sampled.json", manifest_payload)
    _write_json(resolved_output_dir / "manifest_meta.json", meta_payload)
    return resolved_output_dir


def build_dataset_manifest(
    dataset_name: str,
    *,
    source_manifest: Path | None = None,
    analysis_name: str | None = None,
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
    required_modalities: Sequence[str] = DEFAULT_REQUIRED_MODALITIES,
    min_holdout_organs: int = DEFAULT_MIN_HOLDOUT_ORGANS,
    min_samples_per_modality: int = DEFAULT_MIN_SAMPLES_PER_MODALITY,
    output_dir: Path | None = None,
) -> Path:
    dataset_name = normalize_anatomical_dataset_name(dataset_name)
    resolved_source_manifest = (
        Path(source_manifest).resolve()
        if source_manifest is not None
        else get_cross_modal_manifest_path(dataset_name, "sampled", manifest_variant).resolve()
    )
    if not resolved_source_manifest.exists():
        raise FileNotFoundError(f"Upstream Cross-Modal manifest not found: {resolved_source_manifest}")
    return build_anatomical_manifest(
        source_manifest=resolved_source_manifest,
        analysis_name=analysis_name or dataset_name,
        manifest_variant=manifest_variant,
        required_modalities=required_modalities,
        min_holdout_organs=min_holdout_organs,
        min_samples_per_modality=min_samples_per_modality,
        output_dir=output_dir,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build Anatomical Generalization manifests from organ-level source manifests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m reform3d.modules.anatomical_generalization.build_manifest --dataset amos22_ct_mr\n"
            "  python -m reform3d.modules.anatomical_generalization.build_manifest --dataset all --skip-missing\n"
            "  python -m reform3d.modules.anatomical_generalization.build_manifest --source-manifest /path/to/manifest_sampled.json --analysis-name amos22_ct_mr"
        ),
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_ANATOMICAL_DATASET,
        choices=[*ANATOMICAL_DATASETS, "all"],
        help="Dataset surface to build from its Cross-Modal manifest",
    )
    parser.add_argument("-m", "--source-manifest", default=None, help="Explicit upstream organ manifest for a single dataset")
    parser.add_argument("-a", "--analysis-name", default=None, help="Output dataset namespace for a single dataset")
    parser.add_argument("--manifest-variant", default=DEFAULT_MANIFEST_VARIANT)
    parser.add_argument("--required-modalities", nargs="+", default=list(DEFAULT_REQUIRED_MODALITIES))
    parser.add_argument("--min-holdout-organs", type=int, default=DEFAULT_MIN_HOLDOUT_ORGANS)
    parser.add_argument("--min-samples-per-modality", type=int, default=DEFAULT_MIN_SAMPLES_PER_MODALITY)
    parser.add_argument("--output-dir", default=None, help="Explicit output directory for a single dataset")
    parser.add_argument("--skip-missing", action="store_true", help="Continue when a selected upstream manifest is missing")
    args = parser.parse_args(argv)

    if args.dataset == "all" and (args.source_manifest or args.analysis_name or args.output_dir):
        raise ValueError("--source-manifest, --analysis-name, and --output-dir are only valid for a single dataset")

    selected = list(ANATOMICAL_DATASETS) if args.dataset == "all" else [normalize_anatomical_dataset_name(args.dataset)]
    for dataset_name in selected:
        try:
            output_dir = build_dataset_manifest(
                dataset_name,
                source_manifest=Path(args.source_manifest) if args.source_manifest else None,
                analysis_name=args.analysis_name,
                manifest_variant=args.manifest_variant,
                required_modalities=args.required_modalities,
                min_holdout_organs=args.min_holdout_organs,
                min_samples_per_modality=args.min_samples_per_modality,
                output_dir=Path(args.output_dir) if args.output_dir else None,
            )
        except (FileNotFoundError, ValueError) as exc:
            if not args.skip_missing:
                raise
            print(f"Skipping {dataset_name}: {exc}")
            continue
        print(f"Wrote Anatomical Generalization manifest to {output_dir}")


if __name__ == "__main__":
    main()
