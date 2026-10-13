#!/usr/bin/env python
"""Run cross-modal alignment extension evaluations in an isolated output tree."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

from reform3d.core.checkpoints import CHECKPOINT_ORDER, resolve_checkpoint_config
from reform3d.core.evaluation_summary import write_evaluation_summary
from reform3d.core.leakage_audit import write_leakage_audit
from reform3d.core.repeated_seed_runner import repeated_seed_base_argv, run_seeded_module, strip_flags
from reform3d.core.repeated_seed_summary import write_extension_seed_summary
from reform3d.core.seed_context import DEFAULT_REPEATED_SEEDS, get_active_seed
from reform3d.modules.acquisition_robustness.config import RAW_DATASETS

from .config import (
    CROSS_MODAL_OUTPUTS,
    CROSS_MODAL_DATASETS,
    CT_PET_CROSS_MODAL_DATASET,
    DEFAULT_CROSS_MODAL_DATASET,
    DEFAULT_FEATURE_TYPE,
    DEFAULT_MANIFEST_VARIANT,
    FEATURE_EMBEDDING_FILE_NAMES,
    FEATURE_TYPES,
    PRIMARY_METRICS_FILE_NAME,
    get_cross_modal_manifest_path,
    get_output_paths,
    normalize_cross_modal_dataset_name,
    normalize_feature_type,
    normalize_manifest_variant,
)
from .retrieval_diagnostics import (
    DEFAULT_CONTEXT_ABLATION_SWEEP,
    build_context_ablation_summary,
    context_ablation_tag,
    write_ct_pet_corpus_constraint,
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


def _run(command: list[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        import subprocess

        subprocess.run(command, check=True)


def _python_module_command(python_bin: str, module: str, *args: str) -> list[str]:
    return [python_bin, "-u", "-m", module, *args]


def extension_root_from_output_paths(output_paths: dict[str, Path]) -> Path:
    return output_paths["root"] / EXTENSION_DIR_NAME


def shared_extension_root(dataset_name: str, manifest_variant: str) -> Path:
    return CROSS_MODAL_OUTPUTS / dataset_name / manifest_variant / EXTENSION_DIR_NAME


def _path_token(value: object) -> str:
    token = str(value).strip().lower()
    token = "".join(char if char.isalnum() or char in {"_", "-", "."} else "-" for char in token)
    return token.strip("-") or "none"


def _shared_crop_cache_dir(
    cache_root: Path,
    *,
    surface: str,
    checkpoint: str,
    bbox_margin: int,
    outside_organ_mask: str,
    min_mask_voxels: int,
    max_samples: int | None,
) -> Path:
    crop_size = int(resolve_checkpoint_config(checkpoint).crop_size)
    cache_name = (
        f"crop{crop_size}"
        f"_margin{int(bbox_margin)}"
        f"_mask-{_path_token(outside_organ_mask)}"
        f"_minvox{int(min_mask_voxels)}"
    )
    if max_samples is not None:
        cache_name = f"{cache_name}_maxsamples{int(max_samples)}"
    return cache_root / _path_token(surface) / cache_name


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _feature_npz(extension_root: Path, surface: str, checkpoint: str, feature_type: str) -> Path:
    feature_type = normalize_feature_type(feature_type)
    return extension_root / "features" / surface / checkpoint / feature_type / FEATURE_EMBEDDING_FILE_NAMES[feature_type]


def _metrics_path(extension_root: Path, surface: str, checkpoint: str, feature_type: str) -> Path:
    return extension_root / "results" / surface / checkpoint / normalize_feature_type(feature_type) / PRIMARY_METRICS_FILE_NAME


def _ct_pet_sources() -> dict[str, dict[str, object]]:
    sources = {}
    for name in ("autopet", "hecktor25"):
        config = RAW_DATASETS.get(name)
        if config is None:
            continue
        sources[name] = {
            "path": str(config.path),
            "exists": config.path.exists(),
            "modality": config.modality,
            "has_labels": config.has_labels,
            "enabled": config.is_enabled,
            "notes": config.notes,
        }
    return sources


def _prepare_manifest(args: argparse.Namespace, dataset_name: str, manifest_variant: str) -> None:
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


def _context_settings(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.context_ablation_sweep:
        return list(DEFAULT_CONTEXT_ABLATION_SWEEP)
    tag = context_ablation_tag(args.bbox_margin, args.outside_organ_mask)
    return [
        {
            "name": tag or "baseline_full_context",
            "bbox_margin": int(args.bbox_margin),
            "outside_organ_mask": str(args.outside_organ_mask),
        }
    ]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run cross-modal extension outputs without writing native results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default=DEFAULT_CROSS_MODAL_DATASET, choices=list(CROSS_MODAL_DATASETS))
    parser.add_argument("--manifest-variant", default=DEFAULT_MANIFEST_VARIANT)
    parser.add_argument("--checkpoints", nargs="+", default=None)
    parser.add_argument("--all-checkpoints", action="store_true")
    parser.add_argument("--feature-types", nargs="+", default=[DEFAULT_FEATURE_TYPE], choices=list(FEATURE_TYPES))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-cache-workers", type=int, default=4)
    parser.add_argument("--crop-cache-root", default=None)
    parser.add_argument("--min-mask-voxels", type=int, default=100)
    parser.add_argument("--bbox-margin", type=int, default=5)
    parser.add_argument("--outside-organ-mask", default="none", choices=("none", "zero"))
    parser.add_argument("--context-ablation-sweep", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--prepare-manifest", action="store_true")
    parser.add_argument("--min-voxels", type=int, default=100)
    parser.add_argument("--min-samples-per-modality", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--allow-incomplete-seeds", action="store_true")
    parser.add_argument("--fast-seeded-evaluation", action="store_true")
    parser.add_argument("--skip-feature-extraction", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--leakage-audit", action="store_true")
    parser.add_argument("--write-evaluation-summary", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)

    dataset_name = normalize_cross_modal_dataset_name(args.dataset)
    manifest_variant = normalize_manifest_variant(args.manifest_variant)
    checkpoints = _selected_checkpoints(args.checkpoints, args.all_checkpoints)
    feature_types = [normalize_feature_type(feature_type) for feature_type in args.feature_types]

    if args.prepare_manifest and dataset_name != CT_PET_CROSS_MODAL_DATASET:
        _prepare_manifest(args, dataset_name, manifest_variant)

    if args.seeds is not None:
        selected_seeds = args.seeds or list(DEFAULT_REPEATED_SEEDS)
        base_argv = repeated_seed_base_argv(argv if argv is not None else sys.argv[1:])
        base_argv = strip_flags(base_argv, {"--prepare-manifest"})
        for seed in selected_seeds:
            run_seeded_module(
                module_name="reform3d.modules.cross_modal_alignment.run_evaluation_extensions",
                base_argv=base_argv,
                seed=seed,
                dry_run=args.dry_run,
            )
        if not args.dry_run:
            write_extension_seed_summary(
                module_name="cross_modal_alignment",
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
    crop_cache_root = (
        Path(args.crop_cache_root).expanduser().resolve()
        if args.crop_cache_root
        else shared_extension_root(dataset_name, manifest_variant) / "crop_cache"
    )
    manifest_path = get_cross_modal_manifest_path(dataset_name, "sampled", manifest_variant)

    print("Extension path check", flush=True)
    print(f"  Dataset: {dataset_name}", flush=True)
    print(f"  Manifest: [{'OK' if manifest_path.exists() else 'MISSING'}] {manifest_path}", flush=True)
    print(f"  Extension root: {extension_root}", flush=True)
    print(f"  Crop cache root: {crop_cache_root}", flush=True)

    if args.dry_run:
        return

    extension_root.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    if dataset_name == CT_PET_CROSS_MODAL_DATASET:
        path = extension_root / "ct_pet_alignment.json"
        write_ct_pet_corpus_constraint(
            path,
            dataset_name=dataset_name,
            configured_sources=_ct_pet_sources(),
        )
        artifacts["ct_pet_alignment"] = str(path)
    else:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        if args.leakage_audit:
            path = extension_root / "leakage_audit.json"
            write_leakage_audit([manifest_path], path)
            artifacts["leakage_audit"] = str(path)

        metric_records = []
        bootstrap_resamples = 0 if args.fast_seeded_evaluation else args.bootstrap_resamples
        for context in _context_settings(args):
            surface = str(context["name"])
            bbox_margin = int(context["bbox_margin"])
            outside_organ_mask = str(context["outside_organ_mask"])
            tag = context_ablation_tag(bbox_margin, outside_organ_mask)
            for checkpoint in checkpoints:
                for feature_type in feature_types:
                    embeddings_npz = _feature_npz(extension_root, surface, checkpoint, feature_type)
                    metrics_path = _metrics_path(extension_root, surface, checkpoint, feature_type)
                    crop_cache_dir = _shared_crop_cache_dir(
                        crop_cache_root,
                        surface=surface,
                        checkpoint=checkpoint,
                        bbox_margin=bbox_margin,
                        outside_organ_mask=outside_organ_mask,
                        min_mask_voxels=args.min_mask_voxels,
                        max_samples=args.max_samples,
                    )
                    if not args.skip_feature_extraction:
                        command = _python_module_command(
                            args.python,
                            "reform3d.modules.cross_modal_alignment.feature_extractor",
                            "--manifest",
                            str(manifest_path),
                            "--checkpoint",
                            checkpoint,
                            "--feature-type",
                            feature_type,
                            "--batch-size",
                            str(args.batch_size),
                            "--crop-cache-workers",
                            str(args.crop_cache_workers),
                            "--bbox-margin",
                            str(bbox_margin),
                            "--outside-organ-mask",
                            outside_organ_mask,
                            "--min-mask-voxels",
                            str(args.min_mask_voxels),
                            "--output-npz",
                            str(embeddings_npz),
                            "--crop-cache-dir",
                            str(crop_cache_dir),
                            "--device",
                            args.device,
                        )
                        if args.max_samples is not None:
                            command.extend(["--max-samples", str(args.max_samples)])
                        _run(command, False)

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
                                checkpoint,
                                "--feature-type",
                                feature_type,
                                "--embeddings-npz",
                                str(embeddings_npz),
                                "--bbox-margin",
                                str(bbox_margin),
                                "--outside-organ-mask",
                                outside_organ_mask,
                                "--bootstrap-resamples",
                                str(bootstrap_resamples),
                                "--seed",
                                str(args.seed),
                                "--metrics-output",
                                str(metrics_path),
                                "--summary-output-dir",
                                str(extension_root / "summaries" / surface),
                                "--retrieval-diagnostics-output",
                                str(metrics_path.parent / "retrieval_pool_diagnostics.json"),
                                *(["--context-ablation-tag", tag] if tag else []),
                            ),
                            False,
                        )
                    metric_records.append(
                        {
                            "surface": surface,
                            "checkpoint": checkpoint,
                            "feature_type": feature_type,
                            "metrics_path": str(metrics_path),
                            "crop_cache_dir": str(crop_cache_dir),
                        }
                    )

        if args.context_ablation_sweep:
            comparisons = []
            baseline_surface = str(DEFAULT_CONTEXT_ABLATION_SWEEP[0]["name"])
            for record in metric_records:
                if record["surface"] == baseline_surface:
                    continue
                baseline_path = _metrics_path(
                    extension_root,
                    baseline_surface,
                    str(record["checkpoint"]),
                    str(record["feature_type"]),
                )
                metrics_path = Path(str(record["metrics_path"]))
                baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else None
                ablation = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
                if ablation is None:
                    continue
                comparisons.append(
                    {
                        "checkpoint_name": record["checkpoint"],
                        "feature_type": record["feature_type"],
                        "summary": build_context_ablation_summary(baseline, [ablation]),
                        "baseline_metrics_path": str(baseline_path),
                        "ablation_metrics_path": str(metrics_path),
                    }
                )
            path = extension_root / "context_ablation_retrieval.json"
            _write_json(
                path,
                {
                    "artifact": "context_ablation_retrieval",
                    "dataset": dataset_name,
                    "manifest_variant": manifest_variant,
                    "comparisons": comparisons,
                },
            )
            artifacts["context_ablation_retrieval"] = str(path)

    if args.write_evaluation_summary:
        written = write_evaluation_summary(
            extension_root,
            module_name="cross_modal_alignment",
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
            "extension_root": str(extension_root),
            "crop_cache_root": str(crop_cache_root),
            "artifacts": artifacts,
        },
    )
    print(f"Extension outputs written to {extension_root}", flush=True)


if __name__ == "__main__":
    main()
