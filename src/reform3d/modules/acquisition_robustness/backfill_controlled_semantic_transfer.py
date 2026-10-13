#!/usr/bin/env python
"""Backfill matched semantic transfer for controlled perturbation artifacts.

This utility treats ``controlled_embedding_bundle.npz`` as the authoritative
controlled-perturbation feature artifact. For each selected
``perturbation_robustness.json`` file, it aligns semantic targets to the bundle
``case_ids``, recomputes matched semantic transfer, writes the sibling
``matched_semantic_transfer.json``, and embeds the same payload into
``perturbation_robustness.json``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from .config import (
        ACQUISITION_MANIFESTS,
        ACQUISITION_OUTPUTS,
        MANIFEST_FILE_NAMES,
        normalize_manifest_variant,
    )
    from .perturbation_robustness_analysis import (
        _load_matched_semantic_targets,
        compute_matched_semantic_transfer,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    from config import (  # type: ignore
        ACQUISITION_MANIFESTS,
        ACQUISITION_OUTPUTS,
        MANIFEST_FILE_NAMES,
        normalize_manifest_variant,
    )
    from perturbation_robustness_analysis import (  # type: ignore
        _load_matched_semantic_targets,
        compute_matched_semantic_transfer,
    )


DEFAULT_FEATURE_TYPES = ("cls", "avg_pool", "multilayer")
TRANSFER_FILE_NAME = "matched_semantic_transfer.json"
CONTROLLED_BUNDLE_FALLBACK_NAME = "controlled_embedding_bundle.npz"
SKIPPED_PATH_PARTS = {"evaluation_extensions", "seed_summary"}
CRITICAL_WARNING_PREFIXES = (
    "bundle_checkpoint_mismatch:",
    "bundle_feature_type_mismatch:",
    "payload_checkpoint_mismatch:",
    "payload_feature_type_mismatch:",
    "reference_variant_mismatch:",
    "n_source_volumes_mismatch:",
    "missing_variant_features:",
)


@dataclass(frozen=True)
class MetricContext:
    metric_path: Path
    result_dir: Path
    results_dir: Path
    dataset: str
    manifest_variant: str
    seed: int | None
    checkpoint: str
    feature_type: str
    manifest_path: Path


@dataclass(frozen=True)
class BundleData:
    features_by_variant: dict[str, np.ndarray]
    variant_order: list[str]
    reference_variant: str | None
    case_ids: list[str]
    checkpoint: str | None
    feature_type: str | None


@dataclass(frozen=True)
class PerturbedVolumeShim:
    original_path: str
    dataset: str


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _scalar_text(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        if value.size == 1:
            return str(value.reshape(-1)[0])
    return str(value)


def _parse_seed_dir(seed_dir: str) -> int:
    if not seed_dir.startswith("seed_"):
        raise ValueError(f"Expected seed directory name, got {seed_dir}")
    return int(seed_dir.replace("seed_", "", 1))


def _manifest_path_for_context(
    manifest_root: Path,
    dataset: str,
    manifest_variant: str,
    seed: int | None,
    manifest_kind: str,
) -> Path:
    if manifest_kind not in MANIFEST_FILE_NAMES:
        raise ValueError(f"Unknown manifest kind: {manifest_kind}")
    base = manifest_root / dataset / normalize_manifest_variant(manifest_variant)
    if seed is not None:
        base = base / "seeds" / f"seed_{seed}"
    return base / MANIFEST_FILE_NAMES[manifest_kind]


def _context_from_metric_path(
    metric_path: Path,
    *,
    results_root: Path,
    manifest_root: Path,
    manifest_kind: str,
) -> MetricContext | None:
    if any(part in SKIPPED_PATH_PARTS for part in metric_path.parts):
        return None
    try:
        relative_parts = metric_path.resolve().relative_to(results_root.resolve()).parts
    except ValueError:
        return None
    if len(relative_parts) < 5 or relative_parts[-1] != "perturbation_robustness.json":
        return None
    try:
        results_index = relative_parts.index("results")
    except ValueError:
        return None
    if len(relative_parts) < results_index + 4:
        return None

    namespace = relative_parts[:results_index]
    checkpoint = relative_parts[results_index + 1]
    feature_type = relative_parts[results_index + 2]
    if len(namespace) >= 4 and namespace[-2] == "seeds":
        dataset = namespace[0]
        manifest_variant = namespace[1]
        seed = _parse_seed_dir(namespace[-1])
    elif len(namespace) >= 2:
        dataset = namespace[0]
        manifest_variant = namespace[1]
        seed = None
    else:
        return None

    results_dir = results_root.joinpath(*relative_parts[: results_index + 1])
    manifest_path = _manifest_path_for_context(
        manifest_root=manifest_root,
        dataset=dataset,
        manifest_variant=manifest_variant,
        seed=seed,
        manifest_kind=manifest_kind,
    )
    return MetricContext(
        metric_path=metric_path,
        result_dir=metric_path.parent,
        results_dir=results_dir,
        dataset=dataset,
        manifest_variant=normalize_manifest_variant(manifest_variant),
        seed=seed,
        checkpoint=checkpoint,
        feature_type=feature_type,
        manifest_path=manifest_path,
    )


def _discover_metric_contexts(
    *,
    results_root: Path,
    manifest_root: Path,
    manifest_kind: str,
    analysis_names: set[str] | None,
    manifest_variants: set[str] | None,
    seeds: set[int] | None,
    checkpoints: set[str] | None,
    feature_types: set[str] | None,
) -> list[MetricContext]:
    contexts: list[MetricContext] = []
    for metric_path in sorted(results_root.rglob("perturbation_robustness.json")):
        context = _context_from_metric_path(
            metric_path,
            results_root=results_root,
            manifest_root=manifest_root,
            manifest_kind=manifest_kind,
        )
        if context is None:
            continue
        if analysis_names is not None and context.dataset not in analysis_names:
            continue
        if manifest_variants is not None and context.manifest_variant not in manifest_variants:
            continue
        if seeds is not None and context.seed not in seeds:
            continue
        if checkpoints is not None and context.checkpoint not in checkpoints:
            continue
        if feature_types is not None and context.feature_type not in feature_types:
            continue
        contexts.append(context)
    return contexts


def _load_bundle(bundle_path: Path) -> BundleData:
    with np.load(bundle_path, allow_pickle=True) as bundle:
        feature_keys = sorted(key for key in bundle.files if key.startswith("features__"))
        features_by_variant = {
            key.replace("features__", "", 1): np.asarray(bundle[key], dtype=np.float32)
            for key in feature_keys
        }
        if "variant_names" in bundle:
            variant_order = [_scalar_text(value) for value in np.asarray(bundle["variant_names"]).tolist()]
        elif "spacing_names" in bundle:
            variant_order = [_scalar_text(value) for value in np.asarray(bundle["spacing_names"]).tolist()]
        else:
            variant_order = sorted(features_by_variant)

        if "reference_variant_name" in bundle:
            reference_variant = _scalar_text(bundle["reference_variant_name"])
        elif "reference_spacing_name" in bundle:
            reference_variant = _scalar_text(bundle["reference_spacing_name"])
        else:
            reference_variant = variant_order[0] if variant_order else None

        if "case_ids" in bundle:
            case_ids = [_scalar_text(value) for value in np.asarray(bundle["case_ids"]).tolist()]
        else:
            case_ids = []

        checkpoint = _scalar_text(bundle["checkpoint"]) if "checkpoint" in bundle else None
        feature_type = _scalar_text(bundle["feature_type"]) if "feature_type" in bundle else None

    return BundleData(
        features_by_variant=features_by_variant,
        variant_order=variant_order,
        reference_variant=reference_variant,
        case_ids=case_ids,
        checkpoint=checkpoint,
        feature_type=feature_type,
    )


def _expected_semantic_cache_glob(dataset: str) -> str:
    return f"semantic_labels_*_{dataset}.json"


def _select_semantic_cache_dir(context: MetricContext) -> Path:
    candidates = [
        context.results_dir,
        context.results_dir.parent / "features",
    ]
    for candidate in candidates:
        if candidate.exists() and any(candidate.glob(_expected_semantic_cache_glob(context.dataset))):
            return candidate
    return context.results_dir


def _validate_bundle_context(
    *,
    payload: dict[str, Any],
    bundle: BundleData,
    context: MetricContext,
) -> list[str]:
    warnings: list[str] = []
    if bundle.checkpoint is not None and bundle.checkpoint != context.checkpoint:
        warnings.append(f"bundle_checkpoint_mismatch:{bundle.checkpoint}")
    if bundle.feature_type is not None and bundle.feature_type != context.feature_type:
        warnings.append(f"bundle_feature_type_mismatch:{bundle.feature_type}")
    payload_checkpoint = payload.get("checkpoint")
    if payload_checkpoint is not None and payload_checkpoint != context.checkpoint:
        warnings.append(f"payload_checkpoint_mismatch:{payload_checkpoint}")
    payload_feature_type = payload.get("feature_type")
    if payload_feature_type is not None and payload_feature_type != context.feature_type:
        warnings.append(f"payload_feature_type_mismatch:{payload_feature_type}")
    payload_reference = payload.get("reference_variant")
    if payload_reference is not None and bundle.reference_variant is not None and payload_reference != bundle.reference_variant:
        warnings.append(f"reference_variant_mismatch:{payload_reference}!={bundle.reference_variant}")
    n_source_volumes = payload.get("n_source_volumes")
    if n_source_volumes is not None and bundle.case_ids and int(n_source_volumes) != len(bundle.case_ids):
        warnings.append(f"n_source_volumes_mismatch:{n_source_volumes}!={len(bundle.case_ids)}")
    for variant_name in bundle.variant_order:
        if variant_name not in bundle.features_by_variant:
            warnings.append(f"missing_variant_features:{variant_name}")
    return warnings


def _has_critical_warning(warnings: Sequence[str]) -> bool:
    return any(
        warning.startswith(prefix)
        for warning in warnings
        for prefix in CRITICAL_WARNING_PREFIXES
    )


def _backfill_one(
    context: MetricContext,
    *,
    dry_run: bool,
    random_state: int,
    n_cv_splits: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "metric_path": str(context.metric_path),
        "dataset": context.dataset,
        "manifest_variant": context.manifest_variant,
        "seed": context.seed,
        "checkpoint": context.checkpoint,
        "feature_type": context.feature_type,
    }
    if not context.manifest_path.exists():
        row.update({"status": "skipped", "skip_reason": "missing_manifest", "manifest_path": str(context.manifest_path)})
        return row

    payload = _read_json(context.metric_path)
    bundle_name = str(payload.get("controlled_embedding_bundle") or CONTROLLED_BUNDLE_FALLBACK_NAME)
    bundle_path = context.result_dir / bundle_name
    row["bundle_path"] = str(bundle_path)
    if not bundle_path.exists():
        row.update({"status": "skipped", "skip_reason": "missing_controlled_embedding_bundle"})
        return row

    bundle = _load_bundle(bundle_path)
    warnings = _validate_bundle_context(payload=payload, bundle=bundle, context=context)
    if _has_critical_warning(warnings):
        row.update({"status": "skipped", "skip_reason": "metadata_validation_failed", "warnings": warnings})
        return row
    if not bundle.case_ids:
        row.update({"status": "skipped", "skip_reason": "missing_bundle_case_ids", "warnings": warnings})
        return row

    perturbed_volumes = [
        PerturbedVolumeShim(original_path=case_id, dataset=context.dataset)
        for case_id in bundle.case_ids
    ]
    semantic_cache_dir = _select_semantic_cache_dir(context)
    semantic_labels, label_names, semantic_metadata, valid_indices = _load_matched_semantic_targets(
        context.manifest_path,
        perturbed_volumes,
        semantic_cache_dir,
    )
    if semantic_labels is None or valid_indices is None:
        row.update(
            {
                "status": "skipped",
                "skip_reason": "missing_or_insufficient_semantic_labels",
                "semantic_cache_dir": str(semantic_cache_dir),
                "warnings": warnings,
            }
        )
        return row

    ordered_features = {
        variant: bundle.features_by_variant[variant][valid_indices]
        for variant in bundle.variant_order
        if variant in bundle.features_by_variant
    }
    transfer_payload = compute_matched_semantic_transfer(
        ordered_features,
        semantic_labels,
        label_names,
        context.checkpoint,
        context.feature_type,
        n_cv_splits=n_cv_splits,
        random_state=random_state,
    )

    transfer_path = context.result_dir / TRANSFER_FILE_NAME
    row.update(
        {
            "status": "dry_run" if dry_run else "updated",
            "transfer_path": str(transfer_path),
            "semantic_cache_dir": str(semantic_cache_dir),
            "n_cases": len(bundle.case_ids),
            "n_semantic_cases": int(semantic_labels.shape[0]),
            "n_semantic_labels": int(semantic_labels.shape[1]),
            "cross_variant_accuracy": transfer_payload.get("cross_variant_accuracy"),
            "transfer_gap": transfer_payload.get("transfer_gap"),
            "warnings": warnings,
        }
    )
    if dry_run:
        return row

    updated_payload = dict(payload)
    updated_payload["matched_semantic_transfer"] = transfer_payload
    if semantic_metadata is not None:
        updated_payload["semantic_metadata"] = semantic_metadata
    _write_json(transfer_path, transfer_payload)
    _write_json(context.metric_path, updated_payload)
    return row


def _int_set(values: Sequence[int] | None) -> set[int] | None:
    if values is None:
        return None
    return {int(value) for value in values}


def _str_set(values: Sequence[str] | None) -> set[str] | None:
    if values is None:
        return None
    return {str(value) for value in values}


def _variant_set(values: Sequence[str] | None) -> set[str] | None:
    if values is None:
        return None
    return {normalize_manifest_variant(value) for value in values}


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    _write_json(path, payload)


def _log_progress(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute controlled matched semantic transfer from authoritative "
            "controlled_embedding_bundle.npz artifacts and synchronize the sibling "
            "and embedded JSON payloads."
        )
    )
    parser.add_argument("--results-root", type=Path, default=ACQUISITION_OUTPUTS)
    parser.add_argument("--manifest-root", type=Path, default=ACQUISITION_MANIFESTS)
    parser.add_argument("--manifest-kind", default="sampled", choices=tuple(MANIFEST_FILE_NAMES))
    parser.add_argument("--analysis-names", nargs="+", default=None)
    parser.add_argument("--manifest-variants", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--checkpoints", nargs="+", default=None)
    parser.add_argument("--feature-types", nargs="+", default=list(DEFAULT_FEATURE_TYPES))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Process at most N discovered metric files.")
    parser.add_argument("--dry-run", action="store_true", help="Compute and report without writing JSON artifacts.")
    parser.add_argument("--allow-failures", action="store_true", help="Exit successfully even when some items fail.")
    parser.add_argument("--report-json", type=Path, default=None)
    args = parser.parse_args(argv)

    contexts = _discover_metric_contexts(
        results_root=args.results_root,
        manifest_root=args.manifest_root,
        manifest_kind=args.manifest_kind,
        analysis_names=_str_set(args.analysis_names),
        manifest_variants=_variant_set(args.manifest_variants),
        seeds=_int_set(args.seeds),
        checkpoints=_str_set(args.checkpoints),
        feature_types=_str_set(args.feature_types),
    )
    if args.limit is not None:
        contexts = contexts[: int(args.limit)]

    _log_progress(
        "Discovered "
        f"{len(contexts)} controlled perturbation artifacts "
        f"(dry_run={bool(args.dry_run)}, cv_splits={int(args.cv_splits)}, random_state={int(args.random_state)})"
    )

    rows: list[dict[str, Any]] = []
    for index, context in enumerate(contexts, start=1):
        _log_progress(
            f"[{index}/{len(contexts)}] START "
            f"{context.dataset}/seed_{context.seed}/{context.checkpoint}/{context.feature_type}"
        )
        try:
            row = _backfill_one(
                context,
                dry_run=bool(args.dry_run),
                random_state=int(args.random_state),
                n_cv_splits=int(args.cv_splits),
            )
            rows.append(row)
            metric_text = ""
            if row.get("cross_variant_accuracy") is not None:
                metric_text = f" cross_variant_accuracy={float(row['cross_variant_accuracy']):.6f}"
            _log_progress(
                f"[{index}/{len(contexts)}] {str(row.get('status')).upper()} "
                f"{context.dataset}/seed_{context.seed}/{context.checkpoint}/{context.feature_type}"
                f"{metric_text}"
            )
        except Exception as exc:  # pragma: no cover - keeps batch runs inspectable.
            rows.append(
                {
                    "status": "failed",
                    "error": repr(exc),
                    "metric_path": str(context.metric_path),
                    "dataset": context.dataset,
                    "manifest_variant": context.manifest_variant,
                    "seed": context.seed,
                    "checkpoint": context.checkpoint,
                    "feature_type": context.feature_type,
                }
            )
            _log_progress(
                f"[{index}/{len(contexts)}] FAILED "
                f"{context.dataset}/seed_{context.seed}/{context.checkpoint}/{context.feature_type}: {exc!r}"
            )

    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    report = {
        "task": "controlled_matched_semantic_transfer_backfill",
        "results_root": str(args.results_root),
        "manifest_root": str(args.manifest_root),
        "dry_run": bool(args.dry_run),
        "n_discovered": len(contexts),
        "status_counts": status_counts,
        "rows": rows,
    }
    if args.report_json is not None:
        _write_report(args.report_json, report)
    print(json.dumps({key: report[key] for key in ("n_discovered", "status_counts", "dry_run")}, indent=2))

    failures = [row for row in rows if row.get("status") == "failed"]
    if failures and not args.allow_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
