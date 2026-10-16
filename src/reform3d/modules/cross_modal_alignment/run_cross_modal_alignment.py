#!/usr/bin/env python

"""Run Cross-Modal Alignment from manifest generation through summaries.

The runner builds the selected manifest, extracts organ embeddings for each
checkpoint and feature family, runs alignment metrics, aggregates checkpoint
results, and optionally computes uncertainty summaries.

Inputs:
    --dataset: CT/MR dataset surface such as ``amos22_ct_mr``.
    --checkpoints: checkpoint keys or aliases.
    --feature-types: ``cls``, ``avg_pool``, or ``multilayer``.

Example:
    python -m reform3d.modules.cross_modal_alignment.run_cross_modal_alignment \
        --dataset amos22_ct_mr \
        --checkpoints med3dino_rel_96 \
        --feature-types cls
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from reform3d.core.checkpoints import CHECKPOINT_ORDER
from reform3d.core.repeated_seed_runner import repeated_seed_base_argv, run_seeded_module
from reform3d.core.repeated_seed_summary import write_repeated_seed_summary
from reform3d.core.seed_context import DEFAULT_REPEATED_SEEDS

from .config import (
    CHAOS_CROSS_MODAL_DATASET,
    CT_PET_CROSS_MODAL_DATASET,
    DEFAULT_CROSS_MODAL_DATASET,
    DEFAULT_FEATURE_TYPE,
    DEFAULT_MANIFEST_VARIANT,
    FEATURE_TYPES,
    CROSS_MODAL_DATASETS,
    get_checkpoint_feature_npz_path,
    get_cross_modal_manifest_path,
    get_output_paths,
    normalize_cross_modal_dataset_name,
    normalize_feature_type,
    normalize_manifest_variant,
)


NATIVE_CROSS_MODAL_DATASETS = tuple(
    dataset for dataset in CROSS_MODAL_DATASETS if dataset != CT_PET_CROSS_MODAL_DATASET
)


def _run(command: list[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def _python_module_command(python_bin: str, module: str, *args: str) -> list[str]:
    return [python_bin, "-u", "-m", module, *args]


def _selected_checkpoints(checkpoints: Sequence[str] | None, all_checkpoints: bool) -> list[str]:
    if all_checkpoints:
        return list(CHECKPOINT_ORDER)
    return list(checkpoints or ["med3dino_rel_96"])


def _feature_npz(
    dataset_name: str,
    manifest_variant: str,
    checkpoint_name: str,
    feature_type: str,
    context_tag: str | None = None,
) -> Path:
    output_path = get_checkpoint_feature_npz_path(
        get_output_paths(dataset_name, manifest_variant)["features"],
        checkpoint_name,
        feature_type,
    )
    if context_tag:
        return output_path.parent / context_tag / output_path.name
    return output_path


def _metrics_path(
    dataset_name: str,
    manifest_variant: str,
    checkpoint_name: str,
    feature_type: str,
    context_tag: str | None = None,
) -> Path:
    from .config import PRIMARY_METRICS_FILE_NAME, get_checkpoint_metrics_dir

    metrics_dir = get_checkpoint_metrics_dir(
        get_output_paths(dataset_name, manifest_variant)["results"],
        checkpoint_name,
        feature_type,
    )
    if context_tag:
        metrics_dir = metrics_dir / context_tag
    return metrics_dir / PRIMARY_METRICS_FILE_NAME


def _uses_single_organ_alignment(dataset_name: str) -> bool:
    return normalize_cross_modal_dataset_name(dataset_name) == CHAOS_CROSS_MODAL_DATASET


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run Cross-Modal Alignment from manifest generation through summaries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m reform3d.modules.cross_modal_alignment.run_cross_modal_alignment --dataset amos22_ct_mr --checkpoints med3dino_rel_96 --feature-types cls\n"
            "  python -m reform3d.modules.cross_modal_alignment.run_cross_modal_alignment --dataset chaos_ct_mr --checkpoints med3dino_rel_96 --feature-types cls"
        ),
    )
    parser.add_argument("--dataset", default=DEFAULT_CROSS_MODAL_DATASET, choices=list(NATIVE_CROSS_MODAL_DATASETS))
    parser.add_argument("--manifest-variant", default=DEFAULT_MANIFEST_VARIANT)
    parser.add_argument("--checkpoints", nargs="+", default=None)
    parser.add_argument("--all-checkpoints", action="store_true")
    parser.add_argument("--feature-types", nargs="+", default=[DEFAULT_FEATURE_TYPE], choices=list(FEATURE_TYPES))
    parser.add_argument("--min-voxels", type=int, default=100)
    parser.add_argument("--min-samples-per-modality", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-cache-workers", type=int, default=4)
    parser.add_argument("--min-mask-voxels", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Run repeated full evaluations for these seeds.")
    parser.add_argument("--allow-incomplete-seeds", action="store_true", help="Write repeated-seed summaries even when some seed outputs are missing.")
    parser.add_argument(
        "--fast-seeded-evaluation",
        action="store_true",
        help=(
            "Skip repeated uncertainty diagnostics during seeded sweeps "
            "while keeping primary retrieval metrics for seed summaries."
        ),
    )
    parser.add_argument("--skip-manifest-generation", action="store_true")
    parser.add_argument("--skip-feature-extraction", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--skip-aggregation", action="store_true")
    parser.add_argument("--skip-uncertainty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)

    dataset_name = normalize_cross_modal_dataset_name(args.dataset)
    manifest_variant = normalize_manifest_variant(args.manifest_variant)
    checkpoints = _selected_checkpoints(args.checkpoints, args.all_checkpoints)
    feature_types = [normalize_feature_type(feature_type) for feature_type in args.feature_types]
    bootstrap_resamples = 0 if args.fast_seeded_evaluation else args.bootstrap_resamples
    skip_uncertainty = args.skip_uncertainty or args.fast_seeded_evaluation
    uses_single_organ_alignment = _uses_single_organ_alignment(dataset_name)
    output_paths = get_output_paths(dataset_name, manifest_variant)

    if args.seeds is not None:
        selected_seeds = args.seeds or list(DEFAULT_REPEATED_SEEDS)
        base_argv = repeated_seed_base_argv(argv if argv is not None else sys.argv[1:])
        for seed in selected_seeds:
            run_seeded_module(
                module_name="reform3d.modules.cross_modal_alignment.run_cross_modal_alignment",
                base_argv=base_argv,
                seed=seed,
                dry_run=args.dry_run,
            )
        if not args.dry_run:
            write_repeated_seed_summary(
                module_name="cross_modal_alignment",
                dataset=dataset_name,
                manifest_variant=manifest_variant,
                base_root=output_paths["root"],
                seeds=selected_seeds,
                checkpoints=checkpoints,
                feature_types=feature_types,
                allow_incomplete=args.allow_incomplete_seeds,
            )
        return

    manifest_path = get_cross_modal_manifest_path(dataset_name, "sampled", manifest_variant)

    if not args.skip_manifest_generation:
        _run(
            _python_module_command(
                args.python,
                "reform3d.modules.cross_modal_alignment.build_manifests",
                "--dataset",
                dataset_name,
                "--manifest-variant",
                manifest_variant,
                "--min-voxels",
                str(args.min_voxels),
                "--min-samples-per-modality",
                str(args.min_samples_per_modality),
            ),
            args.dry_run,
        )

    for checkpoint_name in checkpoints:
        for feature_type in feature_types:
            embeddings_npz = _feature_npz(dataset_name, manifest_variant, checkpoint_name, feature_type)
            if not args.skip_feature_extraction:
                command = _python_module_command(
                    args.python,
                    "reform3d.modules.cross_modal_alignment.feature_extractor",
                    "--manifest",
                    str(manifest_path),
                    "--checkpoint",
                    checkpoint_name,
                    "--feature-type",
                    feature_type,
                    "--batch-size",
                    str(args.batch_size),
                    "--crop-cache-workers",
                    str(args.crop_cache_workers),
                    "--min-mask-voxels",
                    str(args.min_mask_voxels),
                    "--output-npz",
                    str(embeddings_npz),
                    "--device",
                    args.device,
                )
                if args.max_samples is not None:
                    command.extend(["--max-samples", str(args.max_samples)])
                _run(command, args.dry_run)

            if not args.skip_evaluation:
                _run(
                    _python_module_command(
                        args.python,
                        "reform3d.modules.cross_modal_alignment.evaluation_pipeline",
                        "--manifest",
                        str(manifest_path),
                        "--analysis-name",
                        dataset_name,
                        "--manifest-variant",
                        manifest_variant,
                        "--checkpoint-name",
                        checkpoint_name,
                        "--feature-type",
                        feature_type,
                        "--embeddings-npz",
                        str(embeddings_npz),
                        "--bootstrap-resamples",
                        str(bootstrap_resamples),
                        "--seed",
                        str(args.seed),
                    ),
                    args.dry_run,
                )
                if uses_single_organ_alignment:
                    _run(
                        _python_module_command(
                            args.python,
                            "reform3d.modules.cross_modal_alignment.single_organ_alignment",
                            "--manifest",
                            str(manifest_path),
                            "--analysis-name",
                            dataset_name,
                            "--manifest-variant",
                            manifest_variant,
                            "--checkpoint-name",
                            checkpoint_name,
                            "--feature-type",
                            feature_type,
                            "--organ",
                            "liver",
                            "--embeddings-npz",
                            str(embeddings_npz),
                        ),
                        args.dry_run,
                    )

    if not args.skip_aggregation:
        for feature_type in feature_types:
            _run(
                _python_module_command(
                    args.python,
                    "reform3d.modules.cross_modal_alignment.aggregate_checkpoint_results",
                    "--analysis-name",
                    dataset_name,
                    "--manifest-variant",
                    manifest_variant,
                    "--feature-type",
                    feature_type,
                ),
                args.dry_run,
            )

    if not skip_uncertainty:
        for feature_type in feature_types:
            if uses_single_organ_alignment:
                _run(
                    _python_module_command(
                        args.python,
                        "reform3d.modules.cross_modal_alignment.single_organ_retrieval_uncertainty",
                        "--analysis-name",
                        dataset_name,
                        "--manifest-variant",
                        manifest_variant,
                        "--feature-type",
                        feature_type,
                        "--checkpoints",
                        *checkpoints,
                        "--bootstrap-resamples",
                        str(bootstrap_resamples),
                        "--seed",
                        str(args.seed),
                    ),
                    args.dry_run,
                )
            else:
                _run(
                    _python_module_command(
                        args.python,
                        "reform3d.modules.cross_modal_alignment.retrieval_uncertainty_analysis",
                        "--analysis-name",
                        dataset_name,
                        "--manifest-variant",
                        manifest_variant,
                        "--feature-type",
                        feature_type,
                        "--checkpoints",
                        *checkpoints,
                        "--bootstrap-resamples",
                        str(bootstrap_resamples),
                        "--seed",
                        str(args.seed),
                    ),
                    args.dry_run,
                )


if __name__ == "__main__":
    main()
