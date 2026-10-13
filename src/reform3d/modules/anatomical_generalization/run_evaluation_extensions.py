#!/usr/bin/env python
"""Run anatomical generalization extension evaluations in an isolated output tree."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

from reform3d.core.checkpoints import CHECKPOINT_ORDER, resolve_checkpoint_config
from reform3d.core.evaluation_summary import write_evaluation_summary
from reform3d.core.leakage_audit import write_leakage_audit
from reform3d.core.repeated_seed_runner import repeated_seed_base_argv, run_seeded_module
from reform3d.core.repeated_seed_summary import write_extension_seed_summary
from reform3d.core.seed_context import DEFAULT_REPEATED_SEEDS, get_active_seed

from .config import (
    ANATOMICAL_DATASETS,
    DEFAULT_ANATOMICAL_DATASET,
    DEFAULT_FEATURE_TYPE,
    DEFAULT_MANIFEST_VARIANT,
    FEATURE_TYPES,
    RESULT_JSON_NAME,
    get_anatomical_manifest_path,
    get_cross_modal_feature_npz_candidates,
    get_output_paths,
    normalize_anatomical_dataset_name,
    normalize_feature_type,
    normalize_manifest_variant,
)


EXTENSION_DIR_NAME = "evaluation_extensions"


def _selected_checkpoints(checkpoints: Sequence[str] | None, all_checkpoints: bool) -> list[str]:
    selected = CHECKPOINT_ORDER if all_checkpoints else tuple(checkpoints or ["med3dino_rel_96"])
    resolved: list[str] = []
    for name in selected:
        checkpoint_key = resolve_checkpoint_config(name).checkpoint_key
        if checkpoint_key not in resolved:
            resolved.append(checkpoint_key)
    order = {name: index for index, name in enumerate(CHECKPOINT_ORDER)}
    return sorted(resolved, key=lambda name: (order.get(name, len(order)), name))


def _python_module_command(python_bin: str, module: str, *args: str) -> list[str]:
    return [python_bin, "-u", "-m", module, *args]


def _run(command: list[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def extension_root_from_output_paths(output_paths: dict[str, Path]) -> Path:
    return output_paths["root"] / EXTENSION_DIR_NAME


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        description="Run anatomical extension outputs without writing native results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default=DEFAULT_ANATOMICAL_DATASET, choices=list(ANATOMICAL_DATASETS))
    parser.add_argument("--manifest-variant", default=DEFAULT_MANIFEST_VARIANT)
    parser.add_argument("--checkpoints", nargs="+", default=None)
    parser.add_argument("--all-checkpoints", action="store_true")
    parser.add_argument("--feature-types", nargs="+", default=[DEFAULT_FEATURE_TYPE], choices=list(FEATURE_TYPES))
    parser.add_argument("--required-modalities", nargs="+", default=["ct", "mr"])
    parser.add_argument("--min-holdout-organs", type=int, default=2)
    parser.add_argument("--min-samples-per-modality", type=int, default=5)
    parser.add_argument("--few-shot-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--support-size-sweep", nargs="*", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--allow-incomplete-seeds", action="store_true")
    parser.add_argument("--allow-missing-features", action="store_true")
    parser.add_argument("--cross-modal-feature-surface", default="baseline_full_bbox_context")
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
    parser.add_argument("--leakage-audit", action="store_true")
    parser.add_argument("--write-evaluation-summary", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)

    dataset_name = normalize_anatomical_dataset_name(args.dataset)
    manifest_variant = normalize_manifest_variant(args.manifest_variant)
    checkpoints = _selected_checkpoints(args.checkpoints, args.all_checkpoints)
    feature_types = [normalize_feature_type(feature_type) for feature_type in args.feature_types]
    support_size_sweep = [1, 2, 4, 8] if args.support_size_sweep == [] else list(args.support_size_sweep)

    if args.seeds is not None:
        selected_seeds = args.seeds or list(DEFAULT_REPEATED_SEEDS)
        base_argv = repeated_seed_base_argv(argv if argv is not None else sys.argv[1:])
        for seed in selected_seeds:
            run_seeded_module(
                module_name="reform3d.modules.anatomical_generalization.run_evaluation_extensions",
                base_argv=base_argv,
                seed=seed,
                dry_run=args.dry_run,
            )
        if not args.dry_run:
            write_extension_seed_summary(
                module_name="anatomical_generalization",
                dataset=dataset_name,
                manifest_variant=manifest_variant,
                base_root=get_output_paths(dataset_name, manifest_variant)["root"],
                seeds=selected_seeds,
                checkpoints=checkpoints,
                feature_types=feature_types,
                allow_incomplete=args.allow_incomplete_seeds,
            )
        return

    output_paths = get_output_paths(dataset_name, manifest_variant)
    extension_root = extension_root_from_output_paths(output_paths)
    manifest_path = get_anatomical_manifest_path(dataset_name, "sampled", manifest_variant)

    print("Extension path check", flush=True)
    print(f"  Dataset: {dataset_name}", flush=True)
    print(f"  Manifest: [{'OK' if manifest_path.exists() else 'MISSING'}] {manifest_path}", flush=True)
    print(f"  Extension root: {extension_root}", flush=True)

    if args.dry_run:
        return
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    extension_root.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    if args.leakage_audit:
        path = extension_root / "leakage_audit.json"
        write_leakage_audit([manifest_path], path)
        artifacts["leakage_audit"] = str(path)

    metric_records = []
    few_shot_seeds = args.few_shot_seeds or [args.seed]
    for checkpoint in checkpoints:
        for feature_type in feature_types:
            feature_candidates = get_cross_modal_feature_npz_candidates(
                dataset_name,
                manifest_variant,
                checkpoint,
                feature_type,
                extension_surface=args.cross_modal_feature_surface,
            )
            embeddings_npz = next((candidate for candidate in feature_candidates if candidate.exists()), feature_candidates[0])
            if not embeddings_npz.exists():
                if args.allow_missing_features:
                    metric_records.append(
                        {
                            "checkpoint": checkpoint,
                            "feature_type": feature_type,
                            "status": "missing_feature_bundle",
                            "embeddings_npz": str(embeddings_npz),
                            "feature_candidates": [str(candidate) for candidate in feature_candidates],
                        }
                    )
                    continue
                raise FileNotFoundError(f"Missing upstream feature bundle: {embeddings_npz}")
            result_path = extension_root / "support_size_sweep" / checkpoint / feature_type / RESULT_JSON_NAME
            command = _python_module_command(
                args.python,
                "reform3d.modules.anatomical_generalization.evaluation_pipeline",
                "--manifest",
                str(manifest_path),
                "--analysis-name",
                dataset_name,
                "--manifest-variant",
                manifest_variant,
                "--checkpoint-name",
                checkpoint,
                "--feature-type",
                feature_type,
                "--embeddings-npz",
                str(embeddings_npz),
                "--required-modalities",
                *args.required_modalities,
                "--min-holdout-organs",
                str(args.min_holdout_organs),
                "--min-samples-per-modality",
                str(args.min_samples_per_modality),
                "--few-shot-seeds",
                *[str(seed) for seed in few_shot_seeds],
                "--support-size-sweep",
                *[str(value) for value in support_size_sweep],
                "--seed",
                str(args.seed),
                "--output-path",
                str(result_path),
                "--manifest-summary-output-dir",
                str(extension_root / "summaries"),
                *_projection_args(args),
            )
            _run(command, False)
            source_key = "none" if not args.projection_sources else "+".join(args.projection_sources)
            adapter_result = (
                result_path.parent / "feature_adapter" / "projection" / source_key / RESULT_JSON_NAME
                if args.feature_adapter == "projection"
                else None
            )
            support_artifact_dir = adapter_result.parent if adapter_result is not None else result_path.parent
            metric_records.append(
                {
                    "checkpoint": checkpoint,
                    "feature_type": feature_type,
                    "status": "ok",
                    "embeddings_npz": str(embeddings_npz),
                    "cross_modal_feature_surface": args.cross_modal_feature_surface,
                    "result": str(adapter_result or result_path),
                    "baseline_result_path": str(result_path),
                    "feature_adapter": args.feature_adapter,
                    "adapter_result": str(adapter_result) if adapter_result is not None else None,
                    "support_size_sweep_json": str(support_artifact_dir / "support_size_sweep.json"),
                    "support_size_sweep_csv": str(support_artifact_dir / "support_size_sweep.csv"),
                }
            )

    if args.write_evaluation_summary:
        written = write_evaluation_summary(
            extension_root,
            module_name="anatomical_generalization",
            dataset=dataset_name,
            manifest_variant=manifest_variant,
        )
        artifacts.update({name: str(path) for name, path in written.items()})

    _write_json(
        extension_root / "extension_run_manifest.json",
        {
            "artifact": "extension_run_manifest",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "active_seed": get_active_seed(),
            "dataset": dataset_name,
            "manifest_variant": manifest_variant,
            "manifest_path": str(manifest_path),
            "checkpoints": checkpoints,
            "feature_types": feature_types,
            "support_size_sweep": support_size_sweep,
            "feature_adapter": args.feature_adapter,
            "projection_sources": args.projection_sources,
            "extension_root": str(extension_root),
            "artifacts": artifacts,
            "metric_records": metric_records,
        },
    )
    print(f"Extension outputs written to {extension_root}", flush=True)


if __name__ == "__main__":
    main()
