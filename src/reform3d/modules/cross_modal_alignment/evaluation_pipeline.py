#!/usr/bin/env python

"""Run Cross-Modal Anatomical Alignment manifest validation and metrics.

This CLI validates an organ-level cross-modal manifest, builds anatomical
cohorts, optionally loads a precomputed embedding bundle, and writes primary
alignment metrics under the Cross-Modal Alignment result namespace.

    Usage examples:
    python -m reform3d.modules.cross_modal_alignment.evaluation_pipeline \
        -m amos22_ct_mr/core/manifest_sampled.json

    python -m reform3d.modules.cross_modal_alignment.evaluation_pipeline \
        -m amos22_ct_mr/core/manifest_sampled.json \
        --checkpoint-name med3dino_rel_96 \
        --embeddings-npz /path/to/organ_cls_embeddings.npz
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from reform3d.core.paths import PROJECT_ROOT, get_reform3d_module_result_roots
from .config import (
    CROSS_MODAL_MODULE_NAME,
    DEFAULT_MANIFEST_VARIANT,
    DEFAULT_REQUIRED_MODALITIES,
    ensure_output_directories,
    get_checkpoint_metrics_path,
    get_dataset_name_from_manifest_path,
    get_cross_modal_manifest_path,
    get_manifest_variant_from_manifest_path,
    get_output_paths,
    normalize_cross_modal_dataset_name,
    normalize_feature_type,
    normalize_manifest_variant,
)
from .correspondence import CorrespondenceConfig, build_population_cohorts
from .alignment_metrics import compute_anatomy_over_modality_margin, compute_balanced_lisi
from .data_loader import (
    load_cross_modal_manifest,
    summarize_cross_modal_manifest,
    validate_cross_modal_manifest,
)
from .retrieval_metrics import compute_bidirectional_cross_modal_retrieval
from .retrieval_diagnostics import write_retrieval_pool_diagnostics


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MANIFEST_KIND_BY_FILE_NAME = {
    "manifest_full.json": "full",
    "manifest_sampled.json": "sampled",
    "manifest_meta.json": "meta",
}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_optional_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_existing_path(path_arg: str | Path) -> Path | None:
    candidate = Path(path_arg).expanduser()
    if candidate.exists():
        return candidate.resolve()

    if not candidate.is_absolute():
        project_candidate = (PROJECT_ROOT / candidate).resolve()
        if project_candidate.exists():
            return project_candidate

    return None


def _embedding_summary_path(npz_path: Path) -> Path:
    return npz_path.with_name(npz_path.stem + "_summary.json")


def _manifest_kind_from_path(manifest_path: Path) -> str | None:
    return MANIFEST_KIND_BY_FILE_NAME.get(manifest_path.name)


def _infer_surface_from_embeddings_path(embeddings_path: Path) -> Tuple[str | None, str | None]:
    resolved_path = embeddings_path.resolve()
    for outputs_root in get_reform3d_module_result_roots(CROSS_MODAL_MODULE_NAME):
        try:
            relative_path = resolved_path.relative_to(outputs_root)
        except ValueError:
            continue
        if len(relative_path.parts) < 2:
            continue
        return (relative_path.parts[0], relative_path.parts[1])
    return (None, None)


def _resolve_manifest_path(
    manifest_arg: str,
    analysis_name: str | None,
    manifest_variant: str | None,
    embeddings_npz: str | None,
) -> Path:
    manifest_path = Path(manifest_arg).expanduser()
    resolved_manifest_path = _resolve_existing_path(manifest_path)
    if resolved_manifest_path is not None:
        return resolved_manifest_path

    is_shorthand_manifest = not manifest_path.is_absolute() and len(manifest_path.parts) == 1
    embeddings_path = _resolve_existing_path(embeddings_npz) if embeddings_npz else None

    if is_shorthand_manifest and embeddings_path is not None:
        summary_path = _embedding_summary_path(embeddings_path)
        summary_payload = _load_optional_json(summary_path)
        summary_manifest = summary_payload.get("manifest_path") if summary_payload else None
        if summary_manifest:
            resolved_manifest = Path(str(summary_manifest)).expanduser().resolve()
            if resolved_manifest.exists():
                logger.info(
                    "Resolved shorthand manifest %s using embedding summary %s",
                    manifest_path,
                    summary_path,
                )
                return resolved_manifest

    manifest_kind = _manifest_kind_from_path(manifest_path)
    inferred_analysis_name = analysis_name
    inferred_manifest_variant = normalize_manifest_variant(manifest_variant or DEFAULT_MANIFEST_VARIANT)
    if embeddings_path is not None:
        embedding_analysis_name, embedding_manifest_variant = _infer_surface_from_embeddings_path(embeddings_path)
        if inferred_analysis_name is None:
            inferred_analysis_name = embedding_analysis_name
        if manifest_variant is None and embedding_manifest_variant:
            inferred_manifest_variant = normalize_manifest_variant(embedding_manifest_variant)

    if is_shorthand_manifest and inferred_analysis_name and manifest_kind:
        resolved_module_manifest = get_cross_modal_manifest_path(
            dataset_name=inferred_analysis_name,
            manifest_kind=manifest_kind,
            manifest_variant=inferred_manifest_variant,
        ).resolve()
        if resolved_module_manifest.exists():
            logger.info(
                "Resolved shorthand manifest %s to Cross-Modal manifest %s",
                manifest_path,
                resolved_module_manifest,
            )
            return resolved_module_manifest

    hint = ""
    if is_shorthand_manifest and inferred_analysis_name and manifest_kind:
        hinted_manifest = get_cross_modal_manifest_path(
            dataset_name=inferred_analysis_name,
            manifest_kind=manifest_kind,
            manifest_variant=inferred_manifest_variant,
        ).resolve()
        hint = f" Tried Cross-Modal manifest {hinted_manifest}."
    elif embeddings_path is not None:
        hint = f" Checked embedding summary {_embedding_summary_path(embeddings_path)}."

    raise FileNotFoundError(f"Manifest not found: {manifest_path.resolve()}.{hint}")


def _load_embedding_bundle(npz_path: Path, manifest_samples: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    from reform3d.core.features import load_organ_feature_bundle
    return load_organ_feature_bundle(npz_path, [sample["sample_id"] for sample in manifest_samples])

def _resolve_embeddings_path(embeddings_arg: str) -> Path:
    resolved_embeddings_path = _resolve_existing_path(embeddings_arg)
    if resolved_embeddings_path is not None:
        return resolved_embeddings_path
    raise FileNotFoundError(f"Embeddings bundle not found: {Path(embeddings_arg).expanduser().resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-Modal Anatomical Alignment evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m reform3d.modules.cross_modal_alignment.evaluation_pipeline -m amos22_ct_mr/core/manifest_sampled.json\n"
            "  python -m reform3d.modules.cross_modal_alignment.evaluation_pipeline -m amos22_ct_mr/core/manifest_sampled.json --checkpoint-name med3dino_rel_96 --embeddings-npz /path/to/organ_cls_embeddings.npz"
        ),
    )
    parser.add_argument("-m", "--manifest", required=True, help="Path to the Cross-Modal manifest JSON")
    parser.add_argument("-a", "--analysis-name", default=None, help="Dataset namespace for outputs")
    parser.add_argument(
        "--manifest-variant",
        default=None,
        help="Optional manifest variant label for shorthand manifest resolution",
    )
    parser.add_argument(
        "--required-modalities",
        nargs="+",
        default=list(DEFAULT_REQUIRED_MODALITIES),
        help="Required modalities for cross-modal cohort support",
    )
    parser.add_argument(
        "--min-samples-per-modality",
        type=int,
        default=5,
        help="Minimum support per organ and modality for cohort retention",
    )
    parser.add_argument(
        "--embeddings-npz",
        default=None,
        help="Optional NPZ bundle with 'features' and optional 'sample_ids'",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=None,
        help="Optional checkpoint label used to write per-checkpoint primary metrics",
    )
    parser.add_argument(
        "--feature-type",
        default="cls",
        help="Feature family label for per-checkpoint metrics output",
    )
    parser.add_argument("--metrics-output", default=None, help="Optional explicit primary metrics JSON path")
    parser.add_argument("--summary-output-dir", default=None, help="Optional manifest/cohort summary directory")
    parser.add_argument("--retrieval-diagnostics-output", default=None, help="Optional retrieval-pool diagnostics JSON path")
    parser.add_argument("--context-ablation-tag", default=None, help="Optional tag recorded for context-ablation metrics")
    parser.add_argument("--bbox-margin", type=int, default=5, help="Organ crop bbox margin used for the embedding bundle")
    parser.add_argument(
        "--outside-organ-mask",
        default="none",
        choices=("none", "zero"),
        help="Outside-organ context policy used for the embedding bundle",
    )
    parser.add_argument(
        "--max-samples-per-pool",
        type=int,
        default=None,
        help="Optional cap for per-query positive and negative pools in the anatomy-over-modality metric",
    )
    parser.add_argument(
        "--max-queries-per-organ",
        type=int,
        default=None,
        help="Optional cap for balanced retrieval queries per organ",
    )
    parser.add_argument(
        "--max-targets-per-organ",
        type=int,
        default=None,
        help="Optional cap for balanced retrieval targets per organ",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=1000,
        help="Bootstrap resamples for the anatomy-over-modality margin confidence interval",
    )
    parser.add_argument(
        "--max-lisi-samples-per-group",
        type=int,
        default=None,
        help="Optional cap for balanced LISI sampling per organ and modality group",
    )
    parser.add_argument(
        "--lisi-perplexity",
        type=float,
        default=30.0,
        help="Target LISI perplexity for local-neighborhood weighting",
    )
    parser.add_argument(
        "--lisi-k-neighbors",
        type=int,
        default=None,
        help="Optional explicit neighbor count for LISI; defaults to ceil(3 * perplexity)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for balanced sampling in primary metrics",
    )
    args = parser.parse_args()

    progress_total = 9 if args.embeddings_npz else 5
    progress = tqdm(total=progress_total, desc="Cross-Modal evaluation", unit="stage", dynamic_ncols=True)

    requested_manifest_variant = normalize_manifest_variant(args.manifest_variant) if args.manifest_variant else None
    manifest_path = _resolve_manifest_path(
        args.manifest,
        args.analysis_name,
        requested_manifest_variant,
        args.embeddings_npz,
    )
    analysis_name = normalize_cross_modal_dataset_name(args.analysis_name or get_dataset_name_from_manifest_path(manifest_path))
    manifest_variant = get_manifest_variant_from_manifest_path(
        manifest_path,
        fallback=requested_manifest_variant or DEFAULT_MANIFEST_VARIANT,
    )

    ensure_output_directories(analysis_name, manifest_variant)
    output_paths = get_output_paths(analysis_name, manifest_variant)
    summary_output_dir = Path(args.summary_output_dir).resolve() if args.summary_output_dir else output_paths["results"]
    progress.set_postfix_str("resolved paths")
    progress.update(1)

    samples = load_cross_modal_manifest(manifest_path)
    progress.set_postfix_str("loaded manifest")
    progress.update(1)
    validation = validate_cross_modal_manifest(samples, args.required_modalities)
    manifest_summary = summarize_cross_modal_manifest(samples)
    cohorts = build_population_cohorts(
        samples,
        CorrespondenceConfig(
            required_modalities=tuple(args.required_modalities),
            min_samples_per_modality=args.min_samples_per_modality,
        ),
    )
    progress.set_postfix_str("built cohorts")
    progress.update(1)

    _write_json(
        summary_output_dir / "manifest_summary.json",
        {
            "analysis_name": analysis_name,
            "manifest_path": str(manifest_path),
            "manifest_variant": manifest_variant,
            "validation": validation,
            "summary": manifest_summary,
        },
    )
    _write_json(summary_output_dir / "cohort_summary.json", cohorts)
    if args.retrieval_diagnostics_output:
        write_retrieval_pool_diagnostics(
            Path(args.retrieval_diagnostics_output).resolve(),
            samples,
            required_modalities=args.required_modalities,
            supported_organs=sorted(cohorts.get("supported_organs", {}).keys()),
            bbox_margin=args.bbox_margin,
            outside_organ_mask=args.outside_organ_mask,
        )

    logger.info("Wrote manifest summary to %s", summary_output_dir)
    progress.set_postfix_str("wrote summaries")
    progress.update(1)

    if not args.embeddings_npz:
        progress.set_postfix_str("complete")
        progress.update(1)
        progress.close()
        logger.info("No embeddings bundle provided; Cross-Modal run completed after manifest and cohort validation")
        return

    embeddings_path = _resolve_embeddings_path(args.embeddings_npz)
    checkpoint_name = args.checkpoint_name
    feature_type = normalize_feature_type(args.feature_type)
    if not checkpoint_name:
        raise ValueError(
            "--checkpoint-name is required whenever --embeddings-npz is provided. "
            "Metric files live under results/<checkpoint>/primary_metrics.json; "
            "root-level results are reserved for manifest/cohort summaries and aggregated checkpoint comparisons."
        )
    features_by_id = _load_embedding_bundle(embeddings_path, samples)
    supported_organs = sorted(cohorts.get("supported_organs", {}).keys())
    progress.set_postfix_str("loaded embeddings")
    progress.update(1)

    anatomy_margin = compute_anatomy_over_modality_margin(
        features_by_id,
        samples,
        supported_organs=supported_organs,
        max_samples_per_pool=args.max_samples_per_pool,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )

    progress.set_postfix_str("anatomy margin")
    progress.update(1)

    retrieval = compute_bidirectional_cross_modal_retrieval(
        features_by_id,
        samples,
        supported_organs=supported_organs,
        max_queries_per_organ=args.max_queries_per_organ,
        max_targets_per_organ=args.max_targets_per_organ,
        seed=args.seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )

    progress.set_postfix_str("retrieval")
    progress.update(1)

    balanced_lisi = compute_balanced_lisi(
        features_by_id,
        samples,
        supported_organs=supported_organs,
        required_modalities=args.required_modalities,
        max_samples_per_group=args.max_lisi_samples_per_group,
        perplexity=args.lisi_perplexity,
        k_neighbors=args.lisi_k_neighbors,
        seed=args.seed,
    )

    progress.set_postfix_str("balanced LISI")
    progress.update(1)

    primary_metrics = {
        "analysis_name": analysis_name,
        "checkpoint_name": checkpoint_name,
        "manifest_path": str(manifest_path),
        "embeddings_npz": str(embeddings_path),
        "feature_contract": f"{CROSS_MODAL_MODULE_NAME}_{feature_type}_v1",
        "context_ablation_tag": args.context_ablation_tag,
        "organ_context_controls": {
            "bbox_margin": int(args.bbox_margin),
            "outside_organ_mask": str(args.outside_organ_mask),
        },
        "metric_contract": {
            "primary_metrics": [
                "balanced_cross_modal_retrieval.bidirectional_mean.top@1",
                "balanced_cross_modal_retrieval.bidirectional_mean.top@5",
                "balanced_cross_modal_retrieval.bidirectional_mean.map",
                "balanced_lisi.overall.modality_ilisi.mean",
                "balanced_lisi.overall.organ_clisi.mean",
            ],
            "diagnostic_metrics": [
                "anatomy_over_modality_margin",
            ],
            "exploratory_metrics_not_run_by_default": [
                "cross_modal_cka",
                "silhouette",
                "2d_embedding_plots",
            ],
        },
        "required_modalities": list(args.required_modalities),
        "cohorts": {
            "n_supported_organs": cohorts["n_supported_organs"],
            "n_dropped_organs": cohorts["n_dropped_organs"],
            "supported_organs": supported_organs,
        },
        "anatomy_over_modality_margin": anatomy_margin,
        "balanced_cross_modal_retrieval": retrieval,
        "balanced_lisi": balanced_lisi,
    }

    metrics_output_path = (
        Path(args.metrics_output).resolve()
        if args.metrics_output
        else get_checkpoint_metrics_path(output_paths["results"], checkpoint_name, feature_type)
    )
    _write_json(metrics_output_path, primary_metrics)
    progress.set_postfix_str("wrote metrics")
    progress.update(1)
    progress.close()
    logger.info("Wrote Cross-Modal primary metrics to %s", metrics_output_path.parent)


if __name__ == "__main__":
    main()
