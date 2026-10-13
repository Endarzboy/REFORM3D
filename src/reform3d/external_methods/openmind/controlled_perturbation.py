"""Run OpenMind controlled-perturbation Acquisition Robustness evaluation.

The command builds controlled spacing variants from an Acquisition manifest,
extracts OpenMind avg-pool embeddings, and writes perturbation robustness
artifacts under the Acquisition Robustness result tree.

Example:
    python -m reform3d.external_methods.openmind.controlled_perturbation \
        --dataset jhu_stroke \
        --n-source-volumes 1 \
        --device cpu
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from reform3d.external_methods._acquisition import (
    resolve_acquisition_manifest_path,
    run_controlled_perturbation_analysis,
)
from reform3d.external_methods.openmind.runtime import (
    DEFAULT_ADAPTATION_PLAN_PATH,
    DEFAULT_CHECKPOINT_NAME,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CROP_SIZE,
    DEFAULT_FEATURE_TYPE,
    DEFAULT_MODEL_NAME,
    FEATURE_SURFACES,
    OpenMindFeatureExtractor,
    SUPPORTED_FEATURE_TYPES,
    format_runtime_report,
    inspect_openmind_runtime,
    load_adaptation_plan,
    resolve_dtype,
    resolve_model_patch_size,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenMind controlled Acquisition Robustness evaluation")
    parser.add_argument("--dataset", type=str, default="jhu_stroke")
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--manifest-variant", default="original_bins")
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--adaptation-plan-path", type=Path, default=DEFAULT_ADAPTATION_PLAN_PATH)
    parser.add_argument("--checkpoint-name", type=str, default=DEFAULT_CHECKPOINT_NAME)
    parser.add_argument("--feature-type", type=str, default=DEFAULT_FEATURE_TYPE, choices=list(SUPPORTED_FEATURE_TYPES))
    parser.add_argument("--analysis-name", type=str, default="default")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--n-source-volumes", type=int, default=None)
    parser.add_argument("--crop-size", type=int, default=DEFAULT_CROP_SIZE)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast-seeded-evaluation", action="store_true")
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if args.check_runtime:
        report = inspect_openmind_runtime()
        print(format_runtime_report(report))
        raise SystemExit(1 if report["blockers"] else 0)

    adaptation_plan = load_adaptation_plan(args.checkpoint_path, args.adaptation_plan_path)
    model_patch_size = resolve_model_patch_size(adaptation_plan)
    if int(args.crop_size) != int(model_patch_size[0]):
        raise ValueError(
            f"OpenMind expects crop size {model_patch_size[0]}, got {args.crop_size}."
        )

    run_controlled_perturbation_analysis(
        manifest_path=resolve_acquisition_manifest_path(args.dataset, args.manifest_path, args.manifest_variant),
        checkpoint_path=args.checkpoint_path,
        checkpoint_name=args.checkpoint_name,
        feature_type=args.feature_type,
        extractor_factory=OpenMindFeatureExtractor,
        model_name=DEFAULT_MODEL_NAME,
        crop_size=int(model_patch_size[0]),
        analysis_name=args.analysis_name,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        cache_dir=args.cache_dir,
        n_source_volumes=args.n_source_volumes,
        device=args.device,
        dtype=resolve_dtype(args.dtype),
        extractor_kwargs={"adaptation_plan_path": args.adaptation_plan_path},
        random_state=args.seed,
        compute_matched_semantics=True,
        compute_matched_semantic_probing=not args.fast_seeded_evaluation,
        compute_matched_semantic_transfer=True,
        compute_drift_figure=not args.fast_seeded_evaluation,
        result_metadata={
            "crop_size": int(model_patch_size[0]),
            "adaptation_plan_path": str(Path(args.adaptation_plan_path).resolve()),
            "feature_surface": FEATURE_SURFACES[args.feature_type],
        },
    )


if __name__ == "__main__":
    main()
