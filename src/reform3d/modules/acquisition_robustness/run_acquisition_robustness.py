#!/usr/bin/env python
"""Run Acquisition Robustness from manifest creation through controlled evaluation."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

from reform3d.core.checkpoints import (
    CHECKPOINTS,
    CHECKPOINT_ORDER,
    resolve_checkpoint_config,
)
from reform3d.core.paths import CHECKPOINT_ROOT
from reform3d.core.repeated_seed_runner import repeated_seed_base_argv, run_seeded_module
from reform3d.core.repeated_seed_summary import write_repeated_seed_summary
from reform3d.core.seed_context import DEFAULT_REPEATED_SEEDS
from reform3d.modules.acquisition_robustness.config import (
    RAW_DATASETS,
    DEFAULT_MANIFEST_VARIANT,
    get_output_paths,
    get_acquisition_manifest_path,
    normalize_manifest_variant,
)

FEATURE_TYPES = ("cls", "avg_pool", "multilayer")
DEFAULT_CHECKPOINTS = ("med3dino_rel_96",)
NATIVE_RESULT_FILES = (
    "representation_geometry.json",
    "anisotropy_regression.json",
    "semantic_probing.json",
    "semantic_transfer.json",
)
CONTROLLED_RESULT_FILES = ("perturbation_robustness.json",)


def _selected_checkpoint_names(checkpoints: Iterable[str] | None, all_checkpoints: bool) -> list[str]:
    selected = CHECKPOINT_ORDER if all_checkpoints else tuple(checkpoints or DEFAULT_CHECKPOINTS)
    resolved: list[str] = []
    for name in selected:
        checkpoint_key = resolve_checkpoint_config(name).checkpoint_key
        if checkpoint_key not in resolved:
            resolved.append(checkpoint_key)
    order = {name: index for index, name in enumerate(CHECKPOINT_ORDER)}
    return sorted(resolved, key=lambda name: (order.get(name, len(order)), name))


def _build_manifest_binning(manifest_variant: str) -> str:
    normalized = normalize_manifest_variant(manifest_variant)
    if normalized == "original_bins":
        return "original"
    if normalized == "coarse_bins":
        return "coarse_bins"
    return normalized


def _command_line(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _print_line(message: str) -> None:
    print(message, flush=True)


def _python_module_command(module: str, *args: str) -> list[str]:
    return [sys.executable, "-u", "-m", module, *args]


def _run_step(label: str, command: list[str], log_dir: Path, dry_run: bool) -> None:
    safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label.lower())
    log_path = log_dir / f"{safe_label}.log"
    _print_line(f"[{label}] {_command_line(command)}")
    _print_line(f"[{label}] log: {log_path}")
    if dry_run:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        log_file.write(f"$ {_command_line(command)}\n\n")
        log_file.flush()
        completed = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}; see {log_path}")


def _read_manifest_volume_count(manifest_path: Path) -> int | None:
    if not manifest_path.exists():
        return None
    with manifest_path.open() as handle:
        payload = json.load(handle)
    return len(payload.get("volumes", []))


def _path_report(
    dataset: str,
    manifest_variant: str,
    checkpoints: list[str],
    feature_types: list[str],
    log_dir: Path,
) -> dict[str, object]:
    dataset_config = RAW_DATASETS[dataset]
    manifest_path = get_acquisition_manifest_path(dataset, "sampled", manifest_variant)
    output_paths = get_output_paths(dataset, manifest_variant)
    report = {
        "dataset": dataset,
        "manifest_variant": normalize_manifest_variant(manifest_variant),
        "feature_types": feature_types,
        "raw_dataset": {
            "path": str(dataset_config.path),
            "exists": dataset_config.path.exists(),
            "modality": dataset_config.modality,
            "enabled": dataset_config.is_enabled,
        },
        "manifest": {
            "path": str(manifest_path),
            "exists": manifest_path.exists(),
            "volume_count": _read_manifest_volume_count(manifest_path),
        },
        "outputs": {name: str(path) for name, path in output_paths.items()},
        "checkpoint_root": str(CHECKPOINT_ROOT),
        "checkpoints": {
            name: {
                "release_name": CHECKPOINTS[name].release_name,
                "path": str(CHECKPOINTS[name].path),
                "exists": CHECKPOINTS[name].path.exists(),
                "crop_size": CHECKPOINTS[name].crop_size,
            }
            for name in checkpoints
        },
        "log_dir": str(log_dir),
    }
    return report


def _print_path_report(report: dict[str, object]) -> None:
    raw_dataset = report["raw_dataset"]
    manifest = report["manifest"]
    _print_line("Path check")
    _print_line(f"  Dataset: {report['dataset']} ({raw_dataset['modality']})")
    _print_line(f"  Raw data: [{'OK' if raw_dataset['exists'] else 'MISSING'}] {raw_dataset['path']}")
    _print_line(f"  Manifest: [{'OK' if manifest['exists'] else 'MISSING'}] {manifest['path']}")
    for name, info in report["checkpoints"].items():
        _print_line(f"  Checkpoint {info['release_name']}: [{'OK' if info['exists'] else 'MISSING'}] {info['path']}")
    _print_line(f"  Logs: {report['log_dir']}")


def _write_plan(report: dict[str, object], log_dir: Path, dry_run: bool) -> None:
    if dry_run:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "run_plan.json").write_text(json.dumps(report, indent=2) + "\n")


def _missing_required_paths(report: dict[str, object], require_manifest: bool, allow_missing_checkpoints: bool) -> list[str]:
    missing: list[str] = []
    raw_dataset = report["raw_dataset"]
    if not raw_dataset["exists"]:
        missing.append(f"raw dataset path: {raw_dataset['path']}")
    if require_manifest and not report["manifest"]["exists"]:
        missing.append(f"manifest: {report['manifest']['path']}")
    if not allow_missing_checkpoints:
        for info in report["checkpoints"].values():
            if not info["exists"]:
                missing.append(f"checkpoint {info['release_name']}: {info['path']}")
    return missing


def _result_files(results_root: Path, checkpoint: str, feature_type: str, controlled: bool) -> list[Path]:
    file_names = CONTROLLED_RESULT_FILES if controlled else NATIVE_RESULT_FILES
    base = results_root / checkpoint / feature_type
    return [base / file_name for file_name in file_names]


def _missing_outputs(results_root: Path, checkpoints: list[str], feature_types: list[str], controlled: bool) -> list[Path]:
    missing: list[Path] = []
    for checkpoint in checkpoints:
        for feature_type in feature_types:
            for path in _result_files(results_root, checkpoint, feature_type, controlled):
                if not path.exists():
                    missing.append(path)
    return missing


def _completion_report(
    results_root: Path,
    checkpoints: list[str],
    feature_types: list[str],
    *,
    include_native: bool,
    include_controlled: bool,
) -> dict[str, list[str]]:
    report: dict[str, list[str]] = {}
    if include_native:
        report["native_missing"] = [str(path) for path in _missing_outputs(results_root, checkpoints, feature_types, False)]
    if include_controlled:
        report["controlled_missing"] = [str(path) for path in _missing_outputs(results_root, checkpoints, feature_types, True)]
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run Acquisition Robustness from path check to native and controlled evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="kits23", choices=sorted(RAW_DATASETS.keys()))
    parser.add_argument("--manifest-variant", default=DEFAULT_MANIFEST_VARIANT)
    parser.add_argument("--checkpoints", nargs="+", default=None, help="Checkpoint keys or release-name aliases.")
    parser.add_argument("--all-checkpoints", action="store_true", help="Evaluate all retained checkpoint keys, including 96 and 112 crop variants.")
    parser.add_argument("--feature-types", nargs="+", default=["cls"], choices=FEATURE_TYPES)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--semantic-label-workers", type=int, default=4)
    parser.add_argument("--semantic-cache-flush-every", type=int, default=128)
    parser.add_argument("--min-per-bin", type=int, default=500)
    parser.add_argument("--max-per-bin", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Run repeated full evaluations for these seeds.")
    parser.add_argument("--allow-incomplete-seeds", action="store_true", help="Write repeated-seed summaries even when some seed outputs are missing.")
    parser.add_argument("--validate-manifest", action="store_true")
    parser.add_argument("--force-validate", action="store_true")
    parser.add_argument("--force", action="store_true", help="Regenerate and rerun even when outputs already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print the run plan without executing heavy steps.")
    parser.add_argument("--allow-missing-checkpoints", action="store_true", help="Continue past path check without checkpoint files.")
    parser.add_argument("--allow-incomplete", action="store_true", help="Report missing output files without failing at the final check.")
    parser.add_argument(
        "--fast-seeded-evaluation",
        action="store_true",
        help=(
            "Skip repeated expensive diagnostics during seeded sweeps while keeping "
            "the seeded summary metrics: MMD/SWD, embedding geometry, t-SNE, "
            "controlled drift figures, balanced sensitivity, and controlled matched "
            "semantic probing. Native cross-bin semantic transfer and controlled "
            "matched semantic transfer remain enabled."
        ),
    )
    parser.add_argument("--skip-manifest-generation", action="store_true")
    parser.add_argument("--skip-semantic-labels", action="store_true")
    parser.add_argument("--skip-observational-cache", action="store_true")
    parser.add_argument("--skip-native-evaluation", action="store_true")
    parser.add_argument("--skip-controlled-cache", action="store_true")
    parser.add_argument("--skip-controlled-evaluation", action="store_true")
    parser.add_argument("--log-root", type=Path, default=None)
    args = parser.parse_args(argv)

    manifest_variant = normalize_manifest_variant(args.manifest_variant)
    checkpoints = _selected_checkpoint_names(args.checkpoints, args.all_checkpoints)
    feature_types = list(dict.fromkeys(args.feature_types))
    if args.seeds is not None:
        selected_seeds = args.seeds or list(DEFAULT_REPEATED_SEEDS)
        base_argv = repeated_seed_base_argv(argv if argv is not None else sys.argv[1:])
        for seed in selected_seeds:
            run_seeded_module(
                module_name="reform3d.modules.acquisition_robustness.run_acquisition_robustness",
                base_argv=base_argv,
                seed=seed,
                dry_run=args.dry_run,
            )
        if not args.dry_run:
            write_repeated_seed_summary(
                module_name="acquisition_robustness",
                dataset=args.dataset,
                manifest_variant=manifest_variant,
                base_root=get_output_paths(args.dataset, manifest_variant)["root"],
                seeds=selected_seeds,
                checkpoints=checkpoints,
                feature_types=feature_types,
                allow_incomplete=args.allow_incomplete_seeds,
            )
        return

    manifest_path = get_acquisition_manifest_path(args.dataset, "sampled", manifest_variant)
    output_paths = get_output_paths(args.dataset, manifest_variant)
    log_dir = args.log_root or (output_paths["logs"] / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    report = _path_report(args.dataset, manifest_variant, checkpoints, feature_types, log_dir)
    _print_path_report(report)
    _write_plan(report, log_dir, args.dry_run)

    require_manifest_now = args.skip_manifest_generation or manifest_path.exists()
    missing = _missing_required_paths(report, require_manifest_now, args.allow_missing_checkpoints)
    if missing and not args.dry_run:
        formatted = "\n  - ".join(missing)
        raise FileNotFoundError(f"Required inputs are missing:\n  - {formatted}")

    if not args.skip_manifest_generation:
        if args.force or not manifest_path.exists():
            command = _python_module_command(
                "reform3d.modules.acquisition_robustness.build_manifest",
                "--dataset",
                args.dataset,
                "--output-suffix",
                args.dataset,
                "--binning-scheme",
                _build_manifest_binning(manifest_variant),
                "--min-per-bin",
                str(args.min_per_bin),
                "--max-per-bin",
                str(args.max_per_bin),
                "--seed",
                str(args.seed),
            )
            if args.validate_manifest:
                command.append("--validate")
            if args.force_validate:
                command.append("--force-validate")
            _run_step("manifest_generation", command, log_dir, args.dry_run)
        else:
            _print_line(f"[manifest_generation] using existing manifest: {manifest_path}")

    if not args.dry_run and not manifest_path.exists():
        raise FileNotFoundError(f"Manifest was not created: {manifest_path}")

    checkpoint_args = ["-c", *checkpoints]
    base_eval = _python_module_command(
        "reform3d.modules.acquisition_robustness.evaluation_pipeline",
        "-m",
        str(manifest_path),
        "-a",
        args.dataset,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
    )
    if args.fast_seeded_evaluation:
        base_eval.append("--fast-seeded-evaluation")

    if not args.skip_semantic_labels:
        command = base_eval + [
            "--precompute-labels",
            "--semantic-label-workers",
            str(args.semantic_label_workers),
            "--semantic-cache-flush-every",
            str(args.semantic_cache_flush_every),
            "--transfer-dataset",
            args.dataset,
        ]
        _run_step("semantic_label_precompute", command, log_dir, args.dry_run)

    if not args.skip_observational_cache:
        command = base_eval + checkpoint_args + ["--warmup-observational-cache"]
        _run_step("observational_cache_warmup", command, log_dir, args.dry_run)

    if not args.skip_native_evaluation:
        native_missing = _missing_outputs(output_paths["results"], checkpoints, feature_types, controlled=False)
        if args.force or native_missing:
            for feature_type in feature_types:
                command = base_eval + checkpoint_args + ["--full", "--no-controlled", "-f", feature_type]
                _run_step(f"native_evaluation_{feature_type}", command, log_dir, args.dry_run)
        else:
            print("[native_evaluation] required native outputs already exist")

    if not args.skip_controlled_cache:
        crop_sizes = sorted({CHECKPOINTS[name].crop_size for name in checkpoints})
        command = _python_module_command(
            "reform3d.modules.acquisition_robustness.build_perturbation_cache",
            "-m",
            str(manifest_path),
            "-a",
            args.dataset,
            "--crop-size",
            *[str(crop_size) for crop_size in crop_sizes],
        )
        _run_step("controlled_cache_build", command, log_dir, args.dry_run)

    if not args.skip_controlled_evaluation:
        controlled_missing = _missing_outputs(output_paths["results"], checkpoints, feature_types, controlled=True)
        if args.force or controlled_missing:
            for feature_type in feature_types:
                command = base_eval + checkpoint_args + ["--controlled-only", "-f", feature_type]
                _run_step(f"controlled_evaluation_{feature_type}", command, log_dir, args.dry_run)
        else:
            _print_line("[controlled_evaluation] required controlled outputs already exist")

    completion = _completion_report(
        output_paths["results"],
        checkpoints,
        feature_types,
        include_native=not args.skip_native_evaluation,
        include_controlled=not args.skip_controlled_evaluation,
    )
    if not args.dry_run:
        (log_dir / "completion_report.json").write_text(json.dumps(completion, indent=2) + "\n")
    missing_outputs = [path for paths in completion.values() for path in paths]
    if missing_outputs:
        _print_line("Completion check found missing outputs:")
        for path in missing_outputs:
            _print_line(f"  - {path}")
        if not args.allow_incomplete and not args.dry_run:
            raise RuntimeError(f"Completion check failed; see {log_dir / 'completion_report.json'}")
    else:
        _print_line("Completion check passed.")


if __name__ == "__main__":
    main()
