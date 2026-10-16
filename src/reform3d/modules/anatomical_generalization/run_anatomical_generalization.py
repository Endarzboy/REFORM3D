#!/usr/bin/env python

"""Run Anatomical Generalization from manifest generation through summaries.

The runner builds the Anatomical manifest from the matching Cross-Modal organ
manifest, evaluates available feature bundles for each checkpoint and feature
family, and writes the statistical summary. It can optionally prepare upstream
Cross-Modal feature bundles before the Anatomical run.

Inputs:
    --dataset: dataset surface such as ``amos22_ct_mr``.
    --checkpoints: checkpoint keys or aliases.
    --feature-types: ``cls``, ``avg_pool``, or ``multilayer``.

Example:
    python -m reform3d.modules.anatomical_generalization.run_anatomical_generalization \
        --dataset amos22_ct_mr \
        --checkpoints med3dino_rel_96 \
        --feature-types cls
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Sequence

from reform3d.core.checkpoints import CHECKPOINT_ORDER
from reform3d.core.repeated_seed_runner import option_present, repeated_seed_base_argv, run_seeded_module
from reform3d.core.repeated_seed_summary import write_repeated_seed_summary
from reform3d.core.seed_context import DEFAULT_REPEATED_SEEDS

from .config import (
    DEFAULT_ANATOMICAL_DATASET,
    DEFAULT_FEATURE_TYPE,
    DEFAULT_MANIFEST_VARIANT,
    FEATURE_TYPES,
    ANATOMICAL_DATASETS,
    get_anatomical_manifest_path,
    get_output_paths,
    normalize_anatomical_dataset_name,
    normalize_feature_type,
    normalize_manifest_variant,
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


def _projection_args(args: argparse.Namespace) -> list[str]:
    command: list[str] = ["--feature-adapter", args.feature_adapter]
    if args.projection_sources:
        command.extend(["--projection-sources", *args.projection_sources])
    command.extend(
        [
            "--projection-components",
            str(args.projection_components),
            "--projection-strength",
            str(args.projection_strength),
            "--projection-context-source-surface",
            args.projection_context_source_surface,
            "--projection-context-reference-surface",
            args.projection_context_reference_surface,
            "--projection-spacing-manifest-variant",
            args.projection_spacing_manifest_variant,
        ]
    )
    if args.projection_spacing_datasets:
        command.extend(["--projection-spacing-datasets", *args.projection_spacing_datasets])
    if args.projection_spacing_reference_variant:
        command.extend(["--projection-spacing-reference-variant", args.projection_spacing_reference_variant])
    if args.projection_spacing_target_variants:
        command.extend(["--projection-spacing-target-variants", *args.projection_spacing_target_variants])
    return command


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run Anatomical Generalization from manifest generation through statistical summary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m reform3d.modules.anatomical_generalization.run_anatomical_generalization --dataset amos22_ct_mr --checkpoints med3dino_rel_96 --feature-types cls\n"
            "  python -m reform3d.modules.anatomical_generalization.run_anatomical_generalization --dataset amos22_ct_mr --prepare-cross-modal-features --checkpoints med3dino_rel_96"
        ),
    )
    parser.add_argument("--dataset", default=DEFAULT_ANATOMICAL_DATASET, choices=list(ANATOMICAL_DATASETS))
    parser.add_argument("--manifest-variant", default=DEFAULT_MANIFEST_VARIANT)
    parser.add_argument("--checkpoints", nargs="+", default=None)
    parser.add_argument("--all-checkpoints", action="store_true")
    parser.add_argument("--feature-types", nargs="+", default=[DEFAULT_FEATURE_TYPE], choices=list(FEATURE_TYPES))
    parser.add_argument("--required-modalities", nargs="+", default=["ct", "mr"])
    parser.add_argument("--min-holdout-organs", type=int, default=2)
    parser.add_argument("--min-samples-per-modality", type=int, default=5)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Run repeated full evaluations for these seeds.")
    parser.add_argument("--allow-incomplete-seeds", action="store_true", help="Write repeated-seed summaries even when some seed outputs are missing.")
    parser.add_argument(
        "--fast-seeded-evaluation",
        action="store_true",
        help=(
            "Skip repeated statistical-summary bootstrap during seeded sweeps while "
            "keeping anatomical evaluation metrics for seed summaries."
        ),
    )
    parser.add_argument("--few-shot-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--prepare-cross-modal-features", action="store_true")
    parser.add_argument("--cross-modal-batch-size", type=int, default=8)
    parser.add_argument("--cross-modal-crop-cache-workers", type=int, default=4)
    parser.add_argument("--cross-modal-device", default="cuda")
    parser.add_argument("--cross-modal-max-samples", type=int, default=None)
    parser.add_argument("--skip-manifest-generation", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--skip-statistical-summary", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--feature-adapter", choices=["none", "projection"], default="none")
    parser.add_argument("--projection-sources", nargs="+", default=None, choices=["context", "spacing", "modality"])
    parser.add_argument("--projection-components", type=int, default=4)
    parser.add_argument("--projection-strength", type=float, default=1.0)
    parser.add_argument("--projection-context-source-surface", default="baseline_full_bbox_context")
    parser.add_argument("--projection-context-reference-surface", default="organ_only_bbox0")
    parser.add_argument("--projection-spacing-datasets", nargs="+", default=None)
    parser.add_argument("--projection-spacing-manifest-variant", default="original_bins")
    parser.add_argument("--projection-spacing-reference-variant", default=None)
    parser.add_argument("--projection-spacing-target-variants", nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)

    dataset_name = normalize_anatomical_dataset_name(args.dataset)
    manifest_variant = normalize_manifest_variant(args.manifest_variant)
    checkpoints = _selected_checkpoints(args.checkpoints, args.all_checkpoints)
    feature_types = [normalize_feature_type(feature_type) for feature_type in args.feature_types]
    bootstrap_resamples = 0 if args.fast_seeded_evaluation else args.bootstrap_resamples
    skip_statistical_summary = args.skip_statistical_summary or args.fast_seeded_evaluation or args.feature_adapter == "projection"
    output_paths = get_output_paths(dataset_name, manifest_variant)
    if args.seeds is not None:
        selected_seeds = args.seeds or list(DEFAULT_REPEATED_SEEDS)
        original_argv = argv if argv is not None else sys.argv[1:]
        base_argv = repeated_seed_base_argv(original_argv)
        user_supplied_few_shot_seeds = option_present(original_argv, "--few-shot-seeds")
        for seed in selected_seeds:
            extra_args = () if user_supplied_few_shot_seeds else ("--few-shot-seeds", str(seed))
            run_seeded_module(
                module_name="reform3d.modules.anatomical_generalization.run_anatomical_generalization",
                base_argv=base_argv,
                seed=seed,
                dry_run=args.dry_run,
                extra_args=extra_args,
            )
        if not args.dry_run:
            write_repeated_seed_summary(
                module_name="anatomical_generalization",
                dataset=dataset_name,
                manifest_variant=manifest_variant,
                base_root=output_paths["root"],
                seeds=selected_seeds,
                checkpoints=checkpoints,
                feature_types=feature_types,
                allow_incomplete=args.allow_incomplete_seeds,
            )
        return

    if args.prepare_cross_modal_features:
        command = _python_module_command(
            args.python,
            "reform3d.modules.cross_modal_alignment.run_cross_modal_alignment",
            "--dataset",
            dataset_name,
            "--manifest-variant",
            manifest_variant,
            "--checkpoints",
            *checkpoints,
            "--feature-types",
            *feature_types,
            "--batch-size",
            str(args.cross_modal_batch_size),
            "--crop-cache-workers",
            str(args.cross_modal_crop_cache_workers),
            "--device",
            args.cross_modal_device,
            "--seed",
            str(args.seed),
            "--skip-aggregation",
            "--skip-uncertainty",
        )
        if args.fast_seeded_evaluation:
            command.append("--fast-seeded-evaluation")
        if args.cross_modal_max_samples is not None:
            command.extend(["--max-samples", str(args.cross_modal_max_samples)])
        _run(command, args.dry_run)

    if not args.skip_manifest_generation:
        _run(
            _python_module_command(
                args.python,
                "reform3d.modules.anatomical_generalization.build_manifest",
                "--dataset",
                dataset_name,
                "--manifest-variant",
                manifest_variant,
                "--required-modalities",
                *args.required_modalities,
                "--min-holdout-organs",
                str(args.min_holdout_organs),
                "--min-samples-per-modality",
                str(args.min_samples_per_modality),
            ),
            args.dry_run,
        )

    manifest_path = get_anatomical_manifest_path(dataset_name, "sampled", manifest_variant)

    if not args.skip_evaluation:
        command = _python_module_command(
            args.python,
            "reform3d.modules.anatomical_generalization.evaluation_pipeline",
            "--analysis-name",
            dataset_name,
            "--manifest-variant",
            manifest_variant,
            "--checkpoints",
            *checkpoints,
            "--feature-types",
            *feature_types,
            "--required-modalities",
            *args.required_modalities,
            "--min-holdout-organs",
            str(args.min_holdout_organs),
            "--min-samples-per-modality",
            str(args.min_samples_per_modality),
            "--seed",
            str(args.seed),
            *_projection_args(args),
        )
        if args.few_shot_seeds is not None:
            command.extend(["--few-shot-seeds", *[str(seed) for seed in args.few_shot_seeds]])
        if args.require_complete:
            command.append("--require-complete")
        _run(command, args.dry_run)

    if not skip_statistical_summary:
        _run(
            _python_module_command(
                args.python,
                "reform3d.modules.anatomical_generalization.statistical_rigor_summary",
                "--analysis-name",
                dataset_name,
                "--manifest-variant",
                manifest_variant,
                "--feature-types",
                *feature_types,
                "--bootstrap-resamples",
                str(bootstrap_resamples),
                "--seed",
                str(args.seed),
            ),
            args.dry_run,
        )


if __name__ == "__main__":
    main()
