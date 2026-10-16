#!/usr/bin/env python
"""Run acquisition robustness extension evaluations in an isolated output tree."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

from reform3d.core.checkpoints import CHECKPOINTS, CHECKPOINT_ORDER, resolve_checkpoint_config
from reform3d.core.evaluation_summary import write_evaluation_summary
from reform3d.core.leakage_audit import write_leakage_audit
from reform3d.core.repeated_seed_runner import repeated_seed_base_argv, run_seeded_module
from reform3d.core.repeated_seed_summary import write_extension_seed_summary
from reform3d.core.seed_context import DEFAULT_REPEATED_SEEDS, get_active_seed
from reform3d.modules.acquisition_robustness.config import (
    DEFAULT_MANIFEST_VARIANT,
    RAW_DATASETS,
    get_acquisition_manifest_path,
    get_controlled_perturbation_config_from_manifest,
    get_output_paths,
    normalize_manifest_variant,
)
from reform3d.modules.acquisition_robustness.protocol_artifacts import (
    apply_controlled_perturbation_overrides,
    write_acquisition_protocol_artifacts,
)


FEATURE_TYPES = ("cls", "avg_pool", "multilayer")
DEFAULT_CHECKPOINTS = ("med3dino_rel_96",)
EXTENSION_DIR_NAME = "evaluation_extensions"


def _selected_checkpoint_names(checkpoints: Iterable[str] | None, all_checkpoints: bool) -> list[str]:
    selected = CHECKPOINT_ORDER if all_checkpoints else tuple(checkpoints or DEFAULT_CHECKPOINTS)
    resolved: list[str] = []
    for name in selected:
        checkpoint_key = resolve_checkpoint_config(name).checkpoint_key
        if checkpoint_key not in resolved:
            resolved.append(checkpoint_key)
    order = {name: index for index, name in enumerate(CHECKPOINT_ORDER)}
    return sorted(resolved, key=lambda name: (order.get(name, len(order)), name))


def extension_root_from_output_paths(output_paths: dict[str, Path]) -> Path:
    return output_paths["root"] / EXTENSION_DIR_NAME


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_manifest(manifest_path: Path) -> dict[str, object]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _sensitivity_configs(base_config) -> list[tuple[str, object]]:
    return [
        ("baseline", base_config),
        ("no_antialias", apply_controlled_perturbation_overrides(base_config, anti_alias=False)),
        ("nearest_interpolation", apply_controlled_perturbation_overrides(base_config, interpolation_mode="nearest")),
        (
            "identity_orientation",
            apply_controlled_perturbation_overrides(base_config, orientation_policy="identity"),
        ),
        (
            "fixed_physical_fov_192mm",
            apply_controlled_perturbation_overrides(base_config, physical_fov_mm=(192.0, 192.0, 192.0)),
        ),
    ]


def _metric_surfaces(*, base_config, include_sensitivity: bool, include_extra_perturbations: bool) -> list[tuple[str, object]]:
    surfaces: list[tuple[str, object]] = []
    if include_sensitivity:
        surfaces.extend(_sensitivity_configs(base_config))
    if include_extra_perturbations:
        surfaces.append(
            (
                "extra_perturbations",
                apply_controlled_perturbation_overrides(base_config, extra_perturbations=True),
            )
        )
    return surfaces


def _checkpoints_missing(checkpoints: Iterable[str]) -> list[str]:
    return [name for name in checkpoints if not CHECKPOINTS[name].path.exists()]


def _run_metric_surface(
    *,
    manifest_path: Path,
    dataset: str,
    extension_root: Path,
    output_paths: dict[str, Path],
    surface_name: str,
    perturbation_config,
    checkpoints: list[str],
    feature_types: list[str],
    seed: int,
    fast: bool,
) -> list[dict[str, str]]:
    from reform3d.modules.acquisition_robustness.perturbation_robustness_analysis import (
        run_controlled_perturbation,
    )

    records: list[dict[str, str]] = []
    for checkpoint in checkpoints:
        for feature_type in feature_types:
            result_dir = extension_root / "controlled_perturbations" / surface_name / checkpoint / feature_type
            figures_dir = extension_root / "figures" / "controlled_perturbations" / surface_name / checkpoint / feature_type
            cache_dir = output_paths["cache_root"] / EXTENSION_DIR_NAME / surface_name
            run_controlled_perturbation(
                manifest_path=manifest_path,
                checkpoint_name=checkpoint,
                feature_type=feature_type,
                config=perturbation_config,
                output_dir=result_dir,
                figures_dir=figures_dir,
                cache_dir=cache_dir,
                semantic_cache_dir=extension_root / "semantic_label_cache",
                analysis_name=dataset,
                random_state=seed,
                compute_matched_semantics=not fast,
                compute_drift_figure=not fast,
            )
            records.append(
                {
                    "surface": surface_name,
                    "checkpoint": checkpoint,
                    "feature_type": feature_type,
                    "result": str(result_dir / "perturbation_robustness.json"),
                    "bundle": str(result_dir / "features" / "controlled_embedding_bundle.npz"),
                }
            )
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run acquisition robustness extension outputs without writing native results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="kits23", choices=sorted(RAW_DATASETS.keys()))
    parser.add_argument("--manifest-variant", default=DEFAULT_MANIFEST_VARIANT)
    parser.add_argument("--checkpoints", nargs="+", default=None, help="Checkpoint keys or release-name aliases.")
    parser.add_argument("--all-checkpoints", action="store_true", help="Use all retained checkpoint keys.")
    parser.add_argument("--feature-types", nargs="+", default=["cls"], choices=FEATURE_TYPES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Run each seed in its own extension folder.")
    parser.add_argument("--allow-incomplete-seeds", action="store_true", help="Write extension seed summaries even when some seed metrics are missing.")
    parser.add_argument("--sensitivity-sweep", action="store_true", help="Write and optionally run resampling sensitivity variants.")
    parser.add_argument("--extra-perturbations", action="store_true", help="Write and optionally run non-spacing perturbation variants.")
    parser.add_argument(
        "--orientation-policy",
        default="spacing_permutation",
        choices=("spacing_permutation", "identity"),
        help="Orientation policy for the baseline extension protocol.",
    )
    parser.add_argument(
        "--physical-fov-mm",
        nargs=3,
        type=float,
        default=None,
        metavar=("H_MM", "W_MM", "D_MM"),
        help="Optional physical FOV crop for the baseline extension protocol.",
    )
    parser.add_argument("--run-metrics", action="store_true", help="Run controlled perturbation metrics for selected extension surfaces.")
    parser.add_argument("--fast-seeded-evaluation", action="store_true", help="Skip expensive figures and matched semantic metrics.")
    parser.add_argument("--leakage-audit", action="store_true", help="Write a manifest-level leakage audit under the extension folder.")
    parser.add_argument("--write-evaluation-summary", action="store_true", help="Write neutral evaluation summary artifacts under the extension folder.")
    parser.add_argument("--dry-run", action="store_true", help="Print the extension plan without writing or running metrics.")
    args = parser.parse_args(argv)

    manifest_variant = normalize_manifest_variant(args.manifest_variant)
    checkpoints = _selected_checkpoint_names(args.checkpoints, args.all_checkpoints)
    feature_types = list(dict.fromkeys(args.feature_types))

    if args.seeds is not None:
        selected_seeds = args.seeds or list(DEFAULT_REPEATED_SEEDS)
        base_argv = repeated_seed_base_argv(argv if argv is not None else sys.argv[1:])
        for seed in selected_seeds:
            run_seeded_module(
                module_name="reform3d.modules.acquisition_robustness.run_evaluation_extensions",
                base_argv=base_argv,
                seed=seed,
                dry_run=args.dry_run,
            )
        if not args.dry_run:
            write_extension_seed_summary(
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
    extension_root = extension_root_from_output_paths(output_paths)

    print("Extension path check", flush=True)
    print(f"  Dataset: {args.dataset}", flush=True)
    print(f"  Manifest: [{'OK' if manifest_path.exists() else 'MISSING'}] {manifest_path}", flush=True)
    print(f"  Extension root: {extension_root}", flush=True)

    if not manifest_path.exists() and not args.dry_run:
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if args.run_metrics:
        missing_checkpoints = _checkpoints_missing(checkpoints)
        if missing_checkpoints and not args.dry_run:
            formatted = ", ".join(missing_checkpoints)
            raise FileNotFoundError(f"Checkpoint files are missing for: {formatted}")

    if args.dry_run:
        return

    manifest_payload = _load_manifest(manifest_path)
    base_config = apply_controlled_perturbation_overrides(
        get_controlled_perturbation_config_from_manifest(manifest_payload),
        orientation_policy=args.orientation_policy,
        physical_fov_mm=args.physical_fov_mm,
        extra_perturbations=False,
    )
    artifact_config = apply_controlled_perturbation_overrides(
        base_config,
        extra_perturbations=args.extra_perturbations,
    )

    extension_root.mkdir(parents=True, exist_ok=True)
    artifacts = write_acquisition_protocol_artifacts(
        extension_root,
        dataset=args.dataset,
        manifest_variant=manifest_variant,
        manifest_path=manifest_path,
        perturbation_config=artifact_config,
        checkpoints=checkpoints,
        feature_types=feature_types,
        include_sensitivity=args.sensitivity_sweep,
        include_extra_perturbations=args.extra_perturbations,
    )

    if args.leakage_audit:
        write_leakage_audit([manifest_path], extension_root / "leakage_audit.json")
    if args.write_evaluation_summary:
        artifacts.update(
            write_evaluation_summary(
                extension_root,
                module_name="acquisition_robustness",
                dataset=args.dataset,
                manifest_variant=manifest_variant,
            )
        )

    metric_records: list[dict[str, str]] = []
    if args.run_metrics:
        surfaces = _metric_surfaces(
            base_config=base_config,
            include_sensitivity=args.sensitivity_sweep,
            include_extra_perturbations=args.extra_perturbations,
        )
        if not surfaces:
            surfaces = [("baseline", base_config)]
        for surface_name, perturbation_config in surfaces:
            metric_records.extend(
                _run_metric_surface(
                    manifest_path=manifest_path,
                    dataset=args.dataset,
                    extension_root=extension_root,
                    output_paths=output_paths,
                    surface_name=surface_name,
                    perturbation_config=perturbation_config,
                    checkpoints=checkpoints,
                    feature_types=feature_types,
                    seed=args.seed,
                    fast=args.fast_seeded_evaluation,
                )
            )

    _write_json(
        extension_root / "extension_run_manifest.json",
        {
            "artifact": "extension_run_manifest",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "active_seed": get_active_seed(),
            "dataset": args.dataset,
            "manifest_variant": manifest_variant,
            "manifest_path": str(manifest_path),
            "checkpoints": checkpoints,
            "feature_types": feature_types,
            "extension_root": str(extension_root),
            "artifacts": {name: str(path) for name, path in artifacts.items()},
            "metric_records": metric_records,
        },
    )
    print(f"Extension outputs written to {extension_root}", flush=True)


if __name__ == "__main__":
    main()
