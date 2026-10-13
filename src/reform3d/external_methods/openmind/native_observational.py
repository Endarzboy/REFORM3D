"""Run OpenMind native-spacing Acquisition Robustness evaluation.

Inputs are an Acquisition manifest, the OpenMind checkpoint, and the packaged
adaptation plan. The command extracts OpenMind features and writes the
native-spacing Acquisition Robustness artifacts.

Example:
    python -m reform3d.external_methods.openmind.native_observational \
        --dataset kits23 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from reform3d.external_methods._acquisition import (
    resolve_acquisition_manifest_path,
    run_native_observational_analysis,
)
from reform3d.external_methods.openmind.runtime import (
    DEFAULT_ADAPTATION_PLAN_PATH,
    DEFAULT_CHECKPOINT_NAME,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_FEATURE_TYPE,
    OpenMindFeatureExtractor,
    SUPPORTED_FEATURE_TYPES,
    format_runtime_report,
    inspect_openmind_runtime,
    resolve_dtype,
)


logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenMind native-spacing Acquisition Robustness evaluation")
    parser.add_argument("--dataset", type=str, default="abdomenatlas")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--manifest-variant", default="original_bins")
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--adaptation-plan-path", type=Path, default=DEFAULT_ADAPTATION_PLAN_PATH)
    parser.add_argument("--checkpoint-name", type=str, default=DEFAULT_CHECKPOINT_NAME)
    parser.add_argument("--feature-type", type=str, default=DEFAULT_FEATURE_TYPE, choices=list(SUPPORTED_FEATURE_TYPES))
    parser.add_argument("--analysis-name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32")
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

    manifest_path = resolve_acquisition_manifest_path(args.dataset, args.manifest, args.manifest_variant)
    device = "cuda:0" if args.device == "cuda" and torch.cuda.is_available() else args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dtype = resolve_dtype(args.dtype)
    if device == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32

    extractor = OpenMindFeatureExtractor(
        checkpoint_path=args.checkpoint_path,
        adaptation_plan_path=args.adaptation_plan_path,
        device=device,
        dtype=dtype,
    )
    summary = run_native_observational_analysis(
        manifest_path=manifest_path,
        checkpoint_name=args.checkpoint_name,
        feature_type=args.feature_type,
        extractor=extractor,
        analysis_name=args.analysis_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fast_seeded_evaluation=args.fast_seeded_evaluation,
    )
    logger.info("Completed OpenMind native observational run: %s", summary)


if __name__ == "__main__":
    main()
