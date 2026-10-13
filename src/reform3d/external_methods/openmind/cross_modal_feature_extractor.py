"""Extract OpenMind organ embeddings for Cross-Modal Alignment.

The command reads a Cross-Modal organ manifest, builds or reuses the organ
crop cache, and writes an OpenMind feature bundle consumable by Cross-Modal
Alignment and Anatomical Generalization evaluators.

Example:
    python -m reform3d.external_methods.openmind.cross_modal_feature_extractor \
        --dataset amos22_ct_mr \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from reform3d.external_methods._cross_modal import (
    extract_cross_modal_features,
    resolve_cross_modal_manifest_path,
)
from reform3d.external_methods.openmind.runtime import (
    DEFAULT_ADAPTATION_PLAN_PATH,
    DEFAULT_CHECKPOINT_NAME,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CROP_SIZE,
    DEFAULT_FEATURE_TYPE,
    SUPPORTED_FEATURE_TYPES,
    OpenMindFeatureExtractor,
    format_runtime_report,
    inspect_openmind_runtime,
    load_adaptation_plan,
    resolve_dtype,
    resolve_model_patch_size,
)
from reform3d.modules.cross_modal_alignment.feature_extractor import (
    DEFAULT_BBOX_MARGIN,
    OUTSIDE_ORGAN_MASK_MODES,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract OpenMind organ embeddings for Cross-Modal Alignment")
    parser.add_argument("--dataset", type=str, default="amos22_ct_mr")
    parser.add_argument("-m", "--manifest", type=Path, default=None)
    parser.add_argument("--manifest-variant", default="core")
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--adaptation-plan-path", type=Path, default=DEFAULT_ADAPTATION_PLAN_PATH)
    parser.add_argument("--checkpoint-name", type=str, default=DEFAULT_CHECKPOINT_NAME)
    parser.add_argument("--feature-type", default=DEFAULT_FEATURE_TYPE, choices=list(SUPPORTED_FEATURE_TYPES))
    parser.add_argument("--crop-size", type=int, default=DEFAULT_CROP_SIZE)
    parser.add_argument("-o", "--output-npz", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bbox-margin", type=int, default=DEFAULT_BBOX_MARGIN)
    parser.add_argument("--outside-organ-mask", default="none", choices=list(OUTSIDE_ORGAN_MASK_MODES))
    parser.add_argument("--min-mask-voxels", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--crop-cache-dir", type=Path, default=None)
    parser.add_argument("--crop-cache-workers", type=int, default=1)
    parser.add_argument("--overwrite-crop-cache", action="store_true")
    parser.add_argument("--disable-crop-cache", action="store_true")
    parser.add_argument("--build-crop-cache-only", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if args.check_runtime:
        report = inspect_openmind_runtime()
        print(format_runtime_report(report))
        raise SystemExit(1 if report["blockers"] else 0)

    manifest_path = resolve_cross_modal_manifest_path(args.dataset, args.manifest, args.manifest_variant)
    adaptation_plan = load_adaptation_plan(args.checkpoint_path, args.adaptation_plan_path)
    model_patch_size = resolve_model_patch_size(adaptation_plan)
    if int(args.crop_size) != int(model_patch_size[0]):
        raise ValueError(
            f"OpenMind expects crop size {model_patch_size[0]}, got {args.crop_size}."
        )

    extract_cross_modal_features(
        manifest_path=manifest_path,
        checkpoint_path=args.checkpoint_path,
        checkpoint_name=args.checkpoint_name,
        feature_type=args.feature_type,
        extractor_factory=OpenMindFeatureExtractor,
        crop_size=int(model_patch_size[0]),
        output_npz=Path(args.output_npz) if args.output_npz else None,
        batch_size=args.batch_size,
        bbox_margin=args.bbox_margin,
        outside_organ_mask=args.outside_organ_mask,
        min_mask_voxels=args.min_mask_voxels,
        max_samples=args.max_samples,
        crop_cache_dir=args.crop_cache_dir,
        crop_cache_workers=args.crop_cache_workers,
        overwrite_crop_cache=args.overwrite_crop_cache,
        disable_crop_cache=args.disable_crop_cache,
        build_crop_cache_only=args.build_crop_cache_only,
        device=args.device,
        dtype=resolve_dtype(args.dtype),
        extractor_kwargs={"adaptation_plan_path": args.adaptation_plan_path},
    )


if __name__ == "__main__":
    main()
