"""Controlled perturbation geometry analysis for Acquisition Robustness embeddings.

This module evaluates the controlled perturbation embeddings produced by
Acquisition Robustness and writes a compact explanatory geometry surface:
mean reference drift, spacing-axis dominance, and collapse concentration.

Execution modes
---------------
1. Single-bundle mode
    Evaluate one ``controlled_embedding_bundle.npz`` file and write the sibling
    ``controlled_geometry.json`` result.

2. Batch mode
    Discover controlled embedding bundles under a dataset-level Acquisition Robustness
    results tree, evaluate every discovered checkpoint and feature family, and
    write a dataset-level summary that records completed and missing combinations.

The batch mode defaults to the full checkpoint registry and all supported
feature families, so the geometry analysis can be executed as a full checkpoint-by-feature
surface rather than a one-off inspection of selected runs only.

Examples
--------
    python -m reform3d.modules.explanatory_geometry.controlled_geometry \
        --analysis-name abdomenatlas \
        --manifest-variant original_bins
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from reform3d.modules.acquisition_robustness.config import (
    CHECKPOINTS,
    get_output_paths,
    normalize_manifest_variant,
)


logger = logging.getLogger(__name__)

CONTROLLED_BUNDLE_NAME = "controlled_embedding_bundle.npz"
CONTROLLED_GEOMETRY_NAME = "controlled_geometry.json"
CONTROLLED_GEOMETRY_SUMMARY_NAME = "controlled_geometry_summary.json"
DEFAULT_FEATURE_TYPES = ["cls", "avg_pool", "multilayer"]


def _ordered_checkpoint_names(checkpoint_names: Iterable[str]) -> List[str]:
    """Return checkpoint names in registry order.

    Unknown checkpoint names are kept after the known registry entries and are
    secondarily ordered alphabetically so ad hoc names remain deterministic.
    """
    known_order = {name: idx for idx, name in enumerate(CHECKPOINTS.keys())}
    unique_names = list(dict.fromkeys(checkpoint_names))
    return sorted(unique_names, key=lambda name: (known_order.get(name, len(known_order)), name))


def _load_bundle(bundle_path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Load a persisted controlled perturbation embedding bundle and its metadata."""
    with np.load(bundle_path, allow_pickle=False) as bundle:
        spacing_names = bundle["spacing_names"].tolist()
        metadata = {
            "dataset": str(bundle["dataset"].item()),
            "manifest_variant": str(bundle["manifest_variant"].item()),
            "checkpoint": str(bundle["checkpoint"].item()),
            "feature_type": str(bundle["feature_type"].item()),
            "perturbation_protocol": str(bundle["perturbation_protocol"].item()),
            "case_ids": bundle["case_ids"].tolist(),
            "spacing_names": spacing_names,
            "reference_spacing_name": str(bundle["reference_spacing_name"].item()),
            "spacing_values_z_mm": bundle["spacing_values_z_mm"].astype(float).tolist(),
            "log_spacing_severity": bundle["log_spacing_severity"].astype(float).tolist(),
        }
        features_by_spacing = {
            spacing_name: bundle[f"features__{spacing_name}"].astype(np.float32)
            for spacing_name in spacing_names
        }
    return features_by_spacing, metadata


def _rowwise_cosine_distance(features_a: np.ndarray, features_b: np.ndarray) -> np.ndarray:
    """Compute matched cosine distances between rows of two feature matrices."""
    denom = np.linalg.norm(features_a, axis=1) * np.linalg.norm(features_b, axis=1)
    cosine_similarity = np.sum(features_a * features_b, axis=1) / np.clip(denom, 1e-8, None)
    return 1.0 - np.clip(cosine_similarity, -1.0, 1.0)


def _compute_reference_pair_drifts(
    features_by_spacing: Dict[str, np.ndarray],
    spacing_names: Sequence[str],
    reference_spacing_name: str,
) -> Dict[str, float]:
    """Measure mean matched drift from the reference spacing to each other variant."""
    reference_features = features_by_spacing[reference_spacing_name]
    pair_drifts: Dict[str, float] = {}
    for spacing_name in tqdm(spacing_names, desc="Controlled spacing drifts", unit="spacing", dynamic_ncols=True, leave=False):
        if spacing_name == reference_spacing_name:
            continue
        pair_drifts[spacing_name] = float(
            np.mean(_rowwise_cosine_distance(reference_features, features_by_spacing[spacing_name]))
        )
    return pair_drifts


def _compute_pca_scores(pooled_features: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    """Project pooled controlled embeddings onto their leading principal components."""
    centered = pooled_features - pooled_features.mean(axis=0, keepdims=True)
    if not np.any(centered):
        empty_scores = np.zeros((centered.shape[0], n_components), dtype=np.float32)
        empty_variance = np.zeros(n_components, dtype=np.float32)
        return empty_scores, empty_variance

    u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    effective_components = min(n_components, singular_values.shape[0])
    scores = np.zeros((centered.shape[0], n_components), dtype=np.float32)
    if effective_components > 0:
        scores[:, :effective_components] = (
            u[:, :effective_components] * singular_values[:effective_components]
        ).astype(np.float32)

    explained_variance = (singular_values ** 2) / max(centered.shape[0] - 1, 1)
    total_variance = float(explained_variance.sum())
    explained_variance_ratio = np.zeros(n_components, dtype=np.float32)
    if total_variance > 0.0 and effective_components > 0:
        explained_variance_ratio[:effective_components] = (
            explained_variance[:effective_components] / total_variance
        ).astype(np.float32)

    return scores, explained_variance_ratio


def _r2_against_scalar(scores: np.ndarray, target: np.ndarray) -> List[float]:
    """Compute univariate $R^2$ between each principal-component score and a scalar target."""
    if scores.shape[0] != target.shape[0]:
        raise ValueError("PC scores and target severity must have the same number of rows")

    target_std = float(np.std(target))
    if target_std < 1e-8:
        return [0.0 for _ in range(scores.shape[1])]

    r2_values: List[float] = []
    for component_idx in range(scores.shape[1]):
        component_scores = scores[:, component_idx]
        component_std = float(np.std(component_scores))
        if component_std < 1e-8:
            r2_values.append(0.0)
            continue
        correlation = float(np.corrcoef(component_scores, target)[0, 1])
        if not np.isfinite(correlation):
            r2_values.append(0.0)
            continue
        r2_values.append(float(correlation ** 2))
    return r2_values


def compute_controlled_geometry_v1(bundle_path: Path) -> Dict[str, object]:
    """Compute the controlled-geometry metric surface.

    The returned structure intentionally stays small:
    mean reference drift, top-5 PC spacing-axis dominance, and top-1 variance
    concentration, plus the minimum detail needed for later plotting.
    """
    features_by_spacing, metadata = _load_bundle(bundle_path)
    spacing_names = metadata["spacing_names"]
    reference_spacing_name = metadata["reference_spacing_name"]
    log_spacing_severity = np.asarray(metadata["log_spacing_severity"], dtype=np.float32)

    pair_drifts = _compute_reference_pair_drifts(features_by_spacing, spacing_names, reference_spacing_name)
    mean_reference_drift = float(np.mean(list(pair_drifts.values()))) if pair_drifts else 0.0

    pooled_features = np.concatenate([features_by_spacing[spacing_name] for spacing_name in spacing_names], axis=0)
    n_cases = features_by_spacing[spacing_names[0]].shape[0]
    pooled_log_spacing = np.concatenate(
        [np.full(n_cases, log_spacing_severity[idx], dtype=np.float32) for idx in range(len(spacing_names))],
        axis=0,
    )

    n_components = min(5, pooled_features.shape[0], pooled_features.shape[1])
    scores, explained_variance_ratio = _compute_pca_scores(pooled_features, n_components)
    pc_log_spacing_r2 = _r2_against_scalar(scores, pooled_log_spacing)

    return {
        "analysis_version": "controlled_geometry_v1",
        "dataset": metadata["dataset"],
        "manifest_variant": metadata["manifest_variant"],
        "checkpoint": metadata["checkpoint"],
        "feature_type": metadata["feature_type"],
        "perturbation_protocol": metadata["perturbation_protocol"],
        "n_cases": len(metadata["case_ids"]),
        "spacing_names": spacing_names,
        "reference_spacing_name": reference_spacing_name,
        "spacing_values_z_mm": metadata["spacing_values_z_mm"],
        "metrics": {
            "mean_reference_drift": mean_reference_drift,
            "top5_pc_log_spacing_r2": float(max(pc_log_spacing_r2, default=0.0)),
            "top1_variance_ratio": float(explained_variance_ratio[0]) if len(explained_variance_ratio) > 0 else 0.0,
        },
        "details": {
            "reference_pair_drifts": pair_drifts,
            "pc_log_spacing_r2": pc_log_spacing_r2,
            "explained_variance_ratio_top5": explained_variance_ratio.astype(float).tolist(),
            "log_spacing_severity": [float(value) for value in metadata["log_spacing_severity"]],
        },
    }


def _default_output_for_bundle(bundle_path: Path) -> Path:
    """Return the per-bundle Controlled Geometry output path."""
    return bundle_path.with_name(CONTROLLED_GEOMETRY_NAME)


def evaluate_bundle(bundle_path: Path, output_path: Optional[Path] = None) -> Dict[str, object]:
    """Evaluate one controlled bundle and persist its geometry result JSON."""
    resolved_output = output_path or _default_output_for_bundle(bundle_path)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    results = compute_controlled_geometry_v1(bundle_path)
    with open(resolved_output, "w") as handle:
        json.dump(results, handle, indent=2)

    logger.info("Saved controlled geometry results to %s", resolved_output)
    return results


def _resolve_results_root(
    analysis_name: Optional[str],
    manifest_variant: str,
    results_root: Optional[Path],
) -> Path:
    """Resolve the dataset-level controlled perturbation results directory used for batch discovery."""
    if results_root is not None:
        return results_root
    if not analysis_name:
        raise ValueError("Batch mode requires either --results-root or --analysis-name")
    return get_output_paths(analysis_name, manifest_variant)["results"]


def _discover_bundle_map(
    results_root: Path,
    checkpoints: Sequence[str],
    feature_types: Sequence[str],
) -> Tuple[Dict[Tuple[str, str], Path], List[Tuple[str, str]]]:
    """Discover available controlled bundles and report missing expected combinations."""
    bundle_map: Dict[Tuple[str, str], Path] = {}
    missing_pairs: List[Tuple[str, str]] = []

    for checkpoint_name in checkpoints:
        for feature_type in feature_types:
            bundle_path = results_root / checkpoint_name / feature_type / CONTROLLED_BUNDLE_NAME
            pair = (checkpoint_name, feature_type)
            if bundle_path.exists():
                bundle_map[pair] = bundle_path
            else:
                missing_pairs.append(pair)

    return bundle_map, missing_pairs


def _summary_output_path(results_root: Path, requested_path: Optional[Path]) -> Path:
    """Return the default dataset-level Controlled Geometry summary path."""
    return requested_path or results_root / CONTROLLED_GEOMETRY_SUMMARY_NAME


def run_batch_evaluation(
    results_root: Path,
    checkpoints: Sequence[str],
    feature_types: Sequence[str],
    summary_output: Optional[Path] = None,
    require_complete: bool = False,
) -> Dict[str, object]:
    """Evaluate all discovered controlled perturbation bundles under one dataset-level results tree.

    The batch summary always records missing checkpoint-feature combinations.
    When ``require_complete`` is true, the function raises after writing the
    summary if any expected combination is missing.
    """
    ordered_checkpoints = _ordered_checkpoint_names(checkpoints)
    bundle_map, missing_pairs = _discover_bundle_map(results_root, ordered_checkpoints, feature_types)

    completed_entries: List[Dict[str, object]] = []
    checkpoint_iterator = tqdm(ordered_checkpoints, desc="Controlled geometry checkpoints", unit="ckpt", dynamic_ncols=True)
    for checkpoint_name in checkpoint_iterator:
        feature_iterator = tqdm(feature_types, desc=f"Controlled geometry features {checkpoint_name}", unit="feature", dynamic_ncols=True, leave=False)
        for feature_type in feature_iterator:
            pair = (checkpoint_name, feature_type)
            bundle_path = bundle_map.get(pair)
            if bundle_path is None:
                continue

            output_path = _default_output_for_bundle(bundle_path)
            results = evaluate_bundle(bundle_path, output_path)
            completed_entries.append(
                {
                    "checkpoint": checkpoint_name,
                    "feature_type": feature_type,
                    "bundle_path": str(bundle_path),
                    "output_path": str(output_path),
                    "metrics": results["metrics"],
                }
            )

    summary = {
        "analysis_version": "controlled_geometry_v1_batch",
        "results_root": str(results_root),
        "requested_checkpoints": ordered_checkpoints,
        "requested_feature_types": list(feature_types),
        "completed": completed_entries,
        "missing": [
            {"checkpoint": checkpoint_name, "feature_type": feature_type}
            for checkpoint_name, feature_type in missing_pairs
        ],
        "n_completed": len(completed_entries),
        "n_missing": len(missing_pairs),
    }

    resolved_summary_output = _summary_output_path(results_root, summary_output)
    resolved_summary_output.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved_summary_output, "w") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("Saved controlled geometry batch summary to %s", resolved_summary_output)

    if require_complete and missing_pairs:
        missing_preview = ", ".join(f"{checkpoint}/{feature_type}" for checkpoint, feature_type in missing_pairs[:10])
        raise RuntimeError(
            f"Controlled Geometry batch is incomplete: {len(missing_pairs)} bundle(s) are missing under {results_root}. "
            f"Examples: {missing_preview}"
        )

    return summary


def _build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface for single and batch execution modes."""
    parser = argparse.ArgumentParser(
        description="Controlled embedding geometry analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Single bundle:\n"
            "    python -m reform3d.modules.explanatory_geometry.controlled_geometry --bundle /path/to/controlled_embedding_bundle.npz\n\n"
            "  Full dataset batch:\n"
            "    python -m reform3d.modules.explanatory_geometry.controlled_geometry --analysis-name abdomenatlas --manifest-variant original_bins\n\n"
            "  Strict complete batch:\n"
            "    python -m reform3d.modules.explanatory_geometry.controlled_geometry --analysis-name abdomenatlas --manifest-variant original_bins --require-complete\n"
        ),
    )

    parser.add_argument("--bundle", type=Path, default=None, help="Path to one controlled_embedding_bundle.npz")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Single-bundle output JSON path (default: controlled_geometry.json next to bundle)",
    )

    parser.add_argument(
        "--analysis-name",
        type=str,
        default=None,
        help="Dataset/analysis namespace used to resolve the Acquisition Robustness results tree in batch mode",
    )
    parser.add_argument(
        "--manifest-variant",
        type=str,
        default="original_bins",
        help="Manifest variant for batch-mode output resolution (default: original_bins)",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Explicit controlled perturbation results root for batch mode (overrides --analysis-name)",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="*",
        default=None,
        help="Checkpoint names to include in batch mode (default: full checkpoint registry)",
    )
    parser.add_argument(
        "--feature-types",
        type=str,
        nargs="*",
        default=None,
        help="Feature families to include in batch mode (default: cls avg_pool multilayer)",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Batch summary JSON path (default: controlled_geometry_summary.json under the results root)",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail the batch run if any expected checkpoint-feature bundle is missing",
    )

    return parser


def main() -> None:
    """Run the single-bundle or batch Controlled Geometry workflow."""
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.bundle is not None:
        evaluate_bundle(args.bundle, args.output)
        return

    manifest_variant = normalize_manifest_variant(args.manifest_variant)
    requested_checkpoints = args.checkpoints or list(CHECKPOINTS.keys())
    requested_feature_types = args.feature_types or list(DEFAULT_FEATURE_TYPES)
    results_root = _resolve_results_root(args.analysis_name, manifest_variant, args.results_root)

    run_batch_evaluation(
        results_root=results_root,
        checkpoints=requested_checkpoints,
        feature_types=requested_feature_types,
        summary_output=args.summary_output,
        require_complete=args.require_complete,
    )


if __name__ == "__main__":
    main()
