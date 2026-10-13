#!/usr/bin/env python
"""Run REFORM-3D external-method evaluations."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from reform3d.core.repeated_seed_summary import write_repeated_seed_summary
from reform3d.core.seed_context import ACTIVE_SEED_ENV, DEFAULT_REPEATED_SEEDS, normalize_seed
from reform3d.modules.acquisition_robustness.config import (
    get_acquisition_manifest_path,
    get_output_paths as get_acquisition_output_paths,
    normalize_manifest_variant as normalize_acquisition_manifest_variant,
)
from reform3d.modules.anatomical_generalization.config import (
    get_anatomical_manifest_path,
    get_output_paths as get_anatomical_output_paths,
    normalize_anatomical_dataset_name,
    normalize_manifest_variant as normalize_anatomical_manifest_variant,
)
from reform3d.modules.cross_modal_alignment.config import (
    get_cross_modal_manifest_path,
    get_output_paths as get_cross_modal_output_paths,
    normalize_cross_modal_dataset_name,
    normalize_manifest_variant as normalize_cross_modal_manifest_variant,
)


METHODS = ("brainiac", "openmind", "voco")
METHOD_CHECKPOINT_NAMES = {
    "brainiac": "brainiac",
    "openmind": "openmind_primusm_voco",
    "voco": "voco_b_ssl_head",
}
METHOD_FEATURE_TYPES = {
    "brainiac": ("cls", "avg_pool", "multilayer"),
    "openmind": ("avg_pool", "projected"),
    "voco": ("pooled_multiscale", "final_stage_pool"),
}
METHOD_DEFAULT_FEATURE_TYPE = {
    "brainiac": "cls",
    "openmind": "avg_pool",
    "voco": "pooled_multiscale",
}
FEATURE_TYPES = tuple(sorted({feature_type for values in METHOD_FEATURE_TYPES.values() for feature_type in values}))
TASK_MODULES = {
    "acquisition-native": "native_observational",
    "acquisition-controlled": "controlled_perturbation",
    "cross-modal-extract": "cross_modal_feature_extractor",
    "cross-modal-evaluate": "cross_modal_evaluation",
    "anatomical-evaluate": "anatomical_generalization_evaluation",
}
ACQUISITION_TASKS = {"acquisition-native", "acquisition-controlled"}
CROSS_MODAL_TASKS = {"cross-modal-extract", "cross-modal-evaluate"}
ANATOMICAL_TASKS = {"anatomical-evaluate"}
ACQUISITION_NATIVE_DATASETS = ("abdomenatlas", "kits23", "totalsegmentermri")
ACQUISITION_CONTROLLED_DATASETS = ("abdomenatlas", "kits23", "imagecas", "totalsegmentermri", "jhu_stroke")
ACQUISITION_DATASETS = tuple(dict.fromkeys((*ACQUISITION_NATIVE_DATASETS, *ACQUISITION_CONTROLLED_DATASETS)))
CROSS_MODAL_DATASETS = ("totalsegmenter_ct_mr_anchor", "mmwhs_ct_mr", "amos22_ct_mr", "chaos_ct_mr")
ANATOMICAL_DATASETS = ("totalsegmenter_ct_mr_anchor", "mmwhs_ct_mr", "amos22_ct_mr")
DATASETS_BY_TASK = {
    "acquisition-native": ACQUISITION_NATIVE_DATASETS,
    "acquisition-controlled": ACQUISITION_CONTROLLED_DATASETS,
    "cross-modal-extract": CROSS_MODAL_DATASETS,
    "cross-modal-evaluate": CROSS_MODAL_DATASETS,
    "anatomical-evaluate": ANATOMICAL_DATASETS,
}
DATASETS_BY_GROUP = {
    "acquisition": ACQUISITION_DATASETS,
    "cross-modal": CROSS_MODAL_DATASETS,
    "anatomical": ANATOMICAL_DATASETS,
}
DEVICE_TASKS = {"acquisition-native", "acquisition-controlled", "cross-modal-extract"}
BATCH_SIZE_TASKS = {"acquisition-native", "cross-modal-extract"}
LOG_LEVEL_TASKS = {"acquisition-native", "acquisition-controlled", "cross-modal-extract"}
FAST_SEEDED_TASKS = {"acquisition-native", "acquisition-controlled"}


def _unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        normalized = str(item).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def _command_line(command: Sequence[str], env: dict[str, str] | None = None) -> str:
    env_parts = []
    if env:
        env_parts = [f"{key}={shlex.quote(value)}" for key, value in sorted(env.items())]
    return " ".join([*env_parts, *[shlex.quote(part) for part in command]])


def _log_name(method: str, task: str, dataset: str, feature_type: str) -> str:
    raw_name = f"{method}_{task}_{dataset}_{feature_type}"
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in raw_name)


def _feature_types_for_method(method: str, args: argparse.Namespace) -> list[str]:
    if getattr(args, "all_supported_features", False):
        return list(METHOD_FEATURE_TYPES[method])

    if args.feature_types is None:
        return [METHOD_DEFAULT_FEATURE_TYPE[method]]

    requested = _unique(args.feature_types)
    supported = set(METHOD_FEATURE_TYPES[method])
    selected = [feature_type for feature_type in requested if feature_type in supported]
    if not selected:
        valid = ", ".join(METHOD_FEATURE_TYPES[method])
        raise ValueError(f"No requested feature types are supported by {method}. Valid feature types: {valid}")

    skipped = [feature_type for feature_type in requested if feature_type not in supported]
    if skipped:
        print(
            f"[{method}] skipping unsupported feature types: {', '.join(skipped)}",
            file=sys.stderr,
            flush=True,
        )
    return selected


def _datasets_for_task(task: str, args: argparse.Namespace) -> list[str]:
    if task in TASK_MODULES:
        return args.dataset
    raise ValueError(f"Unsupported task: {task}")


def _dataset_group_for_task(task: str) -> str:
    if task in ACQUISITION_TASKS:
        return "acquisition"
    if task in CROSS_MODAL_TASKS:
        return "cross-modal"
    if task in ANATOMICAL_TASKS:
        return "anatomical"
    raise ValueError(f"Unsupported task: {task}")


def _apply_common_dataset_selection(tasks: Sequence[str], args: argparse.Namespace) -> None:
    groups = {_dataset_group_for_task(task) for task in tasks}
    if len(groups) != 1:
        joined_tasks = ", ".join(tasks)
        raise SystemExit(
            "--dataset can be used only when all selected tasks use the same dataset family. "
            f"Selected tasks: {joined_tasks}. Run separate dataset-wise commands for these tasks."
        )

    group = next(iter(groups))
    selected = _unique(args.dataset)
    valid_set = set(DATASETS_BY_TASK[tasks[0]])
    for task in tasks[1:]:
        valid_set &= set(DATASETS_BY_TASK[task])
    valid = tuple(dataset for dataset in DATASETS_BY_GROUP[group] if dataset in valid_set)
    if not valid:
        joined_tasks = ", ".join(tasks)
        raise SystemExit(f"No shared dataset set is defined for selected tasks: {joined_tasks}")
    invalid = [dataset for dataset in selected if dataset not in valid]
    if invalid:
        valid_text = ", ".join(valid)
        raise SystemExit(f"Unsupported {group} dataset(s): {', '.join(invalid)}. Valid values: {valid_text}")

    args.dataset = selected


def _manifest_variant_for_task(task: str, args: argparse.Namespace) -> str:
    if task in ACQUISITION_TASKS:
        return args.acquisition_manifest_variant
    if task in CROSS_MODAL_TASKS:
        return args.cross_modal_manifest_variant
    if task in ANATOMICAL_TASKS:
        return args.anatomical_manifest_variant
    raise ValueError(f"Unsupported task: {task}")


def _with_active_seed(active_seed: int | None, callback):
    if active_seed is None:
        return callback()

    previous_seed = os.environ.get(ACTIVE_SEED_ENV)
    os.environ[ACTIVE_SEED_ENV] = str(active_seed)
    try:
        return callback()
    finally:
        if previous_seed is None:
            os.environ.pop(ACTIVE_SEED_ENV, None)
        else:
            os.environ[ACTIVE_SEED_ENV] = previous_seed


def _acquisition_binning_scheme(manifest_variant: str) -> str:
    normalized = normalize_acquisition_manifest_variant(manifest_variant)
    if normalized == "original_bins":
        return "original"
    if normalized == "coarse_bins":
        return "coarse_bins"
    raise ValueError(f"Unsupported Acquisition manifest variant for manifest generation: {manifest_variant}")


def _manifest_command_line(command: Sequence[str], active_seed: int | None = None) -> str:
    env_updates = {ACTIVE_SEED_ENV: str(active_seed)} if active_seed is not None else None
    return _command_line(command, env_updates)


def _run_manifest_command(
    *,
    label: str,
    command: list[str],
    dry_run: bool,
    active_seed: int | None = None,
) -> None:
    print(f"[manifest:{label}] {_manifest_command_line(command, active_seed)}", flush=True)
    if dry_run:
        return

    env = os.environ.copy()
    if active_seed is not None:
        env[ACTIVE_SEED_ENV] = str(active_seed)
    subprocess.run(command, env=env, check=True)


def _ensure_manifest(
    *,
    label: str,
    path: Path,
    command: list[str],
    args: argparse.Namespace,
    active_seed: int | None = None,
) -> None:
    if path.exists():
        print(f"[manifest:{label}] ready: {path}", flush=True)
        return

    _run_manifest_command(
        label=label,
        command=command,
        dry_run=args.dry_run,
        active_seed=active_seed,
    )
    if not args.dry_run and not path.exists():
        raise FileNotFoundError(f"Manifest preparation finished but did not create: {path}")


def _prepare_acquisition_manifest(
    *,
    dataset: str,
    manifest_variant: str,
    args: argparse.Namespace,
    active_seed: int | None,
    manifest_kind: str = "sampled",
) -> None:
    manifest_variant = normalize_acquisition_manifest_variant(manifest_variant)
    manifest_path = _with_active_seed(
        active_seed,
        lambda: get_acquisition_manifest_path(dataset, manifest_kind, manifest_variant),
    )
    command = [
        args.python,
        "-u",
        "-m",
        "reform3d.modules.acquisition_robustness.build_manifest",
        "--dataset",
        dataset,
        "--output-suffix",
        dataset,
        "--binning-scheme",
        _acquisition_binning_scheme(manifest_variant),
        "--min-per-bin",
        str(args.min_per_bin),
        "--max-per-bin",
        str(args.max_per_bin),
        "--seed",
        str(args.seed),
    ]
    _ensure_manifest(
        label=f"acquisition_robustness:{dataset}:{manifest_kind}:{manifest_variant}",
        path=manifest_path,
        command=command,
        args=args,
        active_seed=active_seed,
    )


def _prepare_cross_modal_manifest(
    *,
    dataset: str,
    args: argparse.Namespace,
    manifest_variant_override: str | None = None,
) -> None:
    dataset_name = normalize_cross_modal_dataset_name(dataset)
    manifest_variant = normalize_cross_modal_manifest_variant(
        manifest_variant_override or args.cross_modal_manifest_variant
    )

    if dataset_name == "totalsegmenter_ct_mr_anchor":
        _prepare_acquisition_manifest(
            dataset="totalsegmenter_ct",
            manifest_variant="original_bins",
            args=args,
            active_seed=None,
            manifest_kind="full",
        )
        _prepare_acquisition_manifest(
            dataset="totalsegmentermri",
            manifest_variant="original_bins",
            args=args,
            active_seed=None,
            manifest_kind="full",
        )

    manifest_path = get_cross_modal_manifest_path(dataset_name, "sampled", manifest_variant)
    command = [
        args.python,
        "-u",
        "-m",
        "reform3d.modules.cross_modal_alignment.build_manifests",
        "--dataset",
        dataset_name,
        "--manifest-variant",
        manifest_variant,
        "--min-voxels",
        str(args.min_voxels),
        "--min-samples-per-modality",
        str(args.min_samples_per_modality),
    ]
    _ensure_manifest(
        label=f"cross_modal_alignment:{dataset_name}:{manifest_variant}",
        path=manifest_path,
        command=command,
        args=args,
    )


def _prepare_anatomical_manifest(*, dataset: str, args: argparse.Namespace) -> None:
    dataset_name = normalize_anatomical_dataset_name(dataset)
    manifest_variant = normalize_anatomical_manifest_variant(args.anatomical_manifest_variant)

    _prepare_cross_modal_manifest(dataset=dataset_name, args=args, manifest_variant_override=manifest_variant)

    manifest_path = get_anatomical_manifest_path(dataset_name, "sampled", manifest_variant)
    command = [
        args.python,
        "-u",
        "-m",
        "reform3d.modules.anatomical_generalization.build_manifest",
        "--dataset",
        dataset_name,
        "--manifest-variant",
        manifest_variant,
        "--required-modalities",
        *[str(modality).lower() for modality in args.required_modalities],
        "--min-holdout-organs",
        str(args.min_holdout_organs),
        "--min-samples-per-modality",
        str(args.min_samples_per_modality),
    ]
    _ensure_manifest(
        label=f"anatomical_generalization:{dataset_name}:{manifest_variant}",
        path=manifest_path,
        command=command,
        args=args,
    )


def _prepare_required_manifests(
    *,
    tasks: Sequence[str],
    args: argparse.Namespace,
    active_seed: int | None,
    include_acquisition: bool = True,
    include_shared: bool = True,
) -> None:
    if args.skip_manifest_preparation:
        return

    prepared: set[tuple[object, ...]] = set()
    for task in tasks:
        for dataset in _datasets_for_task(task, args):
            if task in ACQUISITION_TASKS:
                if not include_acquisition:
                    continue
                key = (
                    "acquisition_robustness",
                    dataset,
                    normalize_acquisition_manifest_variant(args.acquisition_manifest_variant),
                    active_seed,
                )
                if key in prepared:
                    continue
                prepared.add(key)
                _prepare_acquisition_manifest(
                    dataset=dataset,
                    manifest_variant=args.acquisition_manifest_variant,
                    args=args,
                    active_seed=active_seed,
                )
            elif task in CROSS_MODAL_TASKS:
                if not include_shared:
                    continue
                key = (
                    "cross_modal_alignment",
                    normalize_cross_modal_dataset_name(dataset),
                    normalize_cross_modal_manifest_variant(args.cross_modal_manifest_variant),
                )
                if key in prepared:
                    continue
                prepared.add(key)
                _prepare_cross_modal_manifest(dataset=dataset, args=args)
            elif task in ANATOMICAL_TASKS:
                if not include_shared:
                    continue
                key = (
                    "anatomical_generalization",
                    normalize_anatomical_dataset_name(dataset),
                    normalize_anatomical_manifest_variant(args.anatomical_manifest_variant),
                )
                if key in prepared:
                    continue
                prepared.add(key)
                _prepare_anatomical_manifest(dataset=dataset, args=args)


def _env_for_command(command_index: int, args: argparse.Namespace, active_seed: int | None) -> dict[str, str]:
    env_updates: dict[str, str] = {}
    if not args.gpus:
        gpu = None
    else:
        gpu = args.gpus[command_index % len(args.gpus)]
    if gpu is not None:
        env_updates["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if active_seed is not None:
        env_updates[ACTIVE_SEED_ENV] = str(active_seed)
    return env_updates


def _build_command(
    *,
    python_bin: str,
    method: str,
    task: str,
    dataset: str,
    feature_type: str,
    manifest_variant: str,
    args: argparse.Namespace,
    active_seed: int | None = None,
) -> list[str]:
    module = f"reform3d.external_methods.{method}.{TASK_MODULES[task]}"
    command = [
        python_bin,
        "-u",
        "-m",
        module,
        "--dataset",
        dataset,
        "--manifest-variant",
        manifest_variant,
        "--feature-type",
        feature_type,
    ]

    if task in DEVICE_TASKS:
        device = args.device
        if device is None and args.gpus:
            device = "cuda"
        if device is not None:
            command.extend(["--device", device])
        if args.dtype is not None:
            command.extend(["--dtype", args.dtype])

    if task in BATCH_SIZE_TASKS and args.batch_size is not None:
        command.extend(["--batch-size", str(args.batch_size)])
    if task == "acquisition-native" and args.num_workers is not None:
        command.extend(["--num-workers", str(args.num_workers)])
    if task == "acquisition-controlled" and args.n_source_volumes is not None:
        command.extend(["--n-source-volumes", str(args.n_source_volumes)])
    if task in FAST_SEEDED_TASKS and args.fast_seeded_evaluation:
        command.append("--fast-seeded-evaluation")
    if task == "acquisition-controlled":
        command.extend(["--seed", str(args.seed)])
    if task == "cross-modal-extract":
        if args.max_samples is not None:
            command.extend(["--max-samples", str(args.max_samples)])
        if args.crop_cache_workers is not None:
            command.extend(["--crop-cache-workers", str(args.crop_cache_workers)])
    if task == "cross-modal-evaluate":
        bootstrap_resamples = 0 if args.fast_seeded_evaluation else args.bootstrap_resamples
        command.extend(
            [
                "--required-modalities",
                *[str(modality).lower() for modality in args.required_modalities],
                "--min-samples-per-modality",
                str(args.min_samples_per_modality),
                "--bootstrap-resamples",
                str(bootstrap_resamples),
                "--seed",
                str(args.seed),
            ]
        )
    if task == "anatomical-evaluate":
        command.extend(
            [
                "--required-modalities",
                *[str(modality).lower() for modality in args.required_modalities],
                "--min-holdout-organs",
                str(args.min_holdout_organs),
                "--min-samples-per-modality",
                str(args.min_samples_per_modality),
                "--seed",
                str(args.seed),
            ]
        )
        few_shot_seeds = args.few_shot_seeds
        if few_shot_seeds is None and active_seed is not None:
            few_shot_seeds = [active_seed]
        if few_shot_seeds is not None:
            command.extend(["--few-shot-seeds", *[str(seed) for seed in few_shot_seeds]])
    if task in LOG_LEVEL_TASKS:
        command.extend(["--log-level", args.log_level])

    return command


def _run_command(
    *,
    label: str,
    command: list[str],
    env_updates: dict[str, str],
    log_root: Path | None,
    dry_run: bool,
) -> int:
    print(f"[{label}] {_command_line(command, env_updates)}", flush=True)
    if dry_run:
        return 0

    env = os.environ.copy()
    env.update(env_updates)
    if log_root is None:
        return subprocess.run(command, env=env, check=False).returncode

    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{label}.log"
    print(f"[{label}] log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {_command_line(command, env_updates)}\n\n")
        handle.flush()
        completed = subprocess.run(
            command,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(completed.returncode)


def _run_requested_commands(
    *,
    methods: Sequence[str],
    tasks: Sequence[str],
    args: argparse.Namespace,
    log_root: Path | None,
    active_seed: int | None = None,
    prepare_acquisition_manifests: bool = True,
    prepare_shared_manifests: bool = True,
) -> list[tuple[str, int]]:
    failures: list[tuple[str, int]] = []
    command_index = 0
    original_seed = args.seed
    if active_seed is not None:
        args.seed = int(active_seed)
    try:
        _prepare_required_manifests(
            tasks=tasks,
            args=args,
            active_seed=active_seed,
            include_acquisition=prepare_acquisition_manifests,
            include_shared=prepare_shared_manifests,
        )
        for method in methods:
            feature_types = _feature_types_for_method(method, args)
            for feature_type in feature_types:
                for task in tasks:
                    manifest_variant = _manifest_variant_for_task(task, args)
                    for dataset in _datasets_for_task(task, args):
                        label = _log_name(method, task, dataset, feature_type)
                        command = _build_command(
                            python_bin=args.python,
                            method=method,
                            task=task,
                            dataset=dataset,
                            feature_type=feature_type,
                            manifest_variant=manifest_variant,
                            args=args,
                            active_seed=active_seed,
                        )
                        env_updates = _env_for_command(command_index, args, active_seed)
                        returncode = _run_command(
                            label=label,
                            command=command,
                            env_updates=env_updates,
                            log_root=log_root,
                            dry_run=args.dry_run,
                        )
                        command_index += 1
                        if returncode != 0:
                            failures.append((label, returncode))
                            if not args.continue_on_error:
                                raise SystemExit(returncode)
    finally:
        args.seed = original_seed
    return failures


def _summary_base_root(module_name: str, dataset: str, args: argparse.Namespace) -> tuple[str, str, Path]:
    if module_name == "acquisition_robustness":
        manifest_variant = normalize_acquisition_manifest_variant(args.acquisition_manifest_variant)
        return dataset, manifest_variant, get_acquisition_output_paths(dataset, manifest_variant)["root"]
    if module_name == "cross_modal_alignment":
        dataset_name = normalize_cross_modal_dataset_name(dataset)
        manifest_variant = normalize_cross_modal_manifest_variant(args.cross_modal_manifest_variant)
        return dataset_name, manifest_variant, get_cross_modal_output_paths(dataset_name, manifest_variant)["root"]
    if module_name == "anatomical_generalization":
        dataset_name = normalize_anatomical_dataset_name(dataset)
        manifest_variant = normalize_anatomical_manifest_variant(args.anatomical_manifest_variant)
        return dataset_name, manifest_variant, get_anatomical_output_paths(dataset_name, manifest_variant)["root"]
    raise ValueError(f"Unsupported module for external-method summary: {module_name}")


def _write_seed_summaries(
    *,
    methods: Sequence[str],
    tasks: Sequence[str],
    seeds: Sequence[int],
    args: argparse.Namespace,
) -> None:
    summary_specs: list[tuple[str, tuple[str, ...]]] = []
    if any(task in ACQUISITION_TASKS for task in tasks):
        acquisition_tracks = []
        if "acquisition-native" in tasks:
            acquisition_tracks.append("native")
        if "acquisition-controlled" in tasks:
            acquisition_tracks.append("controlled")
        summary_specs.append(("acquisition_robustness", tuple(acquisition_tracks)))
    if "cross-modal-evaluate" in tasks:
        summary_specs.append(("cross_modal_alignment", ()))
    if "anatomical-evaluate" in tasks:
        summary_specs.append(("anatomical_generalization", ()))

    if not summary_specs:
        return

    for module_name, acquisition_tracks in summary_specs:
        for dataset in args.dataset:
            dataset_name, manifest_variant, base_root = _summary_base_root(module_name, dataset, args)
            manifest_rows: list[dict[str, object]] = []
            for method in methods:
                feature_types = _feature_types_for_method(method, args)
                summary_subdir = f"seed_summary/external_methods/{method}"
                print(
                    f"[seed summary] {module_name} {dataset_name} {method}: "
                    f"seeds={','.join(str(seed) for seed in seeds)} features={','.join(feature_types)}",
                    flush=True,
                )
                if args.dry_run:
                    manifest_rows.append(
                        {
                            "method": method,
                            "checkpoint": METHOD_CHECKPOINT_NAMES[method],
                            "feature_types": feature_types,
                            "summary_json": str(base_root / summary_subdir / "repeated_seed_summary.json"),
                        }
                    )
                    continue
                write_repeated_seed_summary(
                    module_name=module_name,
                    dataset=dataset_name,
                    manifest_variant=manifest_variant,
                    base_root=base_root,
                    seeds=seeds,
                    checkpoints=[METHOD_CHECKPOINT_NAMES[method]],
                    feature_types=feature_types,
                    allow_incomplete=args.allow_incomplete_seeds,
                    acquisition_tracks=acquisition_tracks or ("native", "controlled"),
                    summary_subdir=summary_subdir,
                )
                manifest_rows.append(
                    {
                        "method": method,
                        "checkpoint": METHOD_CHECKPOINT_NAMES[method],
                        "feature_types": feature_types,
                        "summary_json": str(base_root / summary_subdir / "repeated_seed_summary.json"),
                    }
                )
            if not args.dry_run:
                manifest_path = base_root / "seed_summary" / "external_methods" / "external_method_seed_summaries.json"
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text(
                    json.dumps(
                        {
                            "module": module_name,
                            "dataset": dataset_name,
                            "manifest_variant": manifest_variant,
                            "seeds": [int(seed) for seed in seeds],
                            "methods": manifest_rows,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run REFORM-3D external-method modules.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--task", dest="tasks", nargs="+", required=True, choices=tuple(TASK_MODULES))
    feature_group = parser.add_mutually_exclusive_group()
    feature_group.add_argument(
        "--feature-types",
        nargs="+",
        default=None,
        choices=FEATURE_TYPES,
        help="Feature types to run. Unsupported method/feature combinations are skipped.",
    )
    feature_group.add_argument(
        "--all-supported-features",
        action="store_true",
        help="Run every feature type supported by each selected method.",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        required=True,
        help="Dataset(s) for the selected task family.",
    )
    parser.add_argument("--acquisition-manifest-variant", default="original_bins")
    parser.add_argument("--cross-modal-manifest-variant", default="core")
    parser.add_argument("--anatomical-manifest-variant", default="core")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--n-source-volumes", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--crop-cache-workers", type=int, default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--min-per-bin", type=int, default=500)
    parser.add_argument("--max-per-bin", type=int, default=1000)
    parser.add_argument("--min-voxels", type=int, default=100)
    parser.add_argument("--min-samples-per-modality", type=int, default=5)
    parser.add_argument("--required-modalities", nargs="+", default=["ct", "mr"])
    parser.add_argument("--min-holdout-organs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Run repeated full evaluations for these seeds.")
    parser.add_argument("--few-shot-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--allow-incomplete-seeds", action="store_true")
    parser.add_argument(
        "--fast-seeded-evaluation",
        action="store_true",
        help="Skip repeated expensive diagnostics during seeded sweeps while keeping seeded summary metrics.",
    )
    parser.add_argument(
        "--skip-manifest-preparation",
        action="store_true",
        help="Require existing manifests instead of preparing missing ones before execution.",
    )
    parser.add_argument("--gpus", nargs="*", default=None)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)

    methods = _unique(args.methods)
    tasks = _unique(args.tasks)
    _apply_common_dataset_selection(tasks, args)

    log_root = args.log_root
    if log_root is not None and not args.dry_run:
        log_root = log_root / f"external_methods_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    failures: list[tuple[str, int]] = []
    if args.seeds is not None:
        selected_seeds = [normalize_seed(seed) for seed in (args.seeds or list(DEFAULT_REPEATED_SEEDS))]
        _prepare_required_manifests(
            tasks=tasks,
            args=args,
            active_seed=None,
            include_acquisition=False,
            include_shared=True,
        )
        for seed in selected_seeds:
            seed_log_root = log_root / f"seed_{seed}" if log_root is not None else None
            failures.extend(
                _run_requested_commands(
                    methods=methods,
                    tasks=tasks,
                    args=args,
                    log_root=seed_log_root,
                    active_seed=seed,
                    prepare_shared_manifests=False,
                )
            )
        _write_seed_summaries(methods=methods, tasks=tasks, seeds=selected_seeds, args=args)
    else:
        failures.extend(
            _run_requested_commands(
                methods=methods,
                tasks=tasks,
                args=args,
                log_root=log_root,
            )
        )

    if failures:
        details = ", ".join(f"{label}={returncode}" for label, returncode in failures)
        raise SystemExit(f"External-method runner completed with failures: {details}")


if __name__ == "__main__":
    main()
