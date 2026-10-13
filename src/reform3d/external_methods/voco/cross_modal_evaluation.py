"""Run Cross-Modal Alignment metrics for VoCo embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

from reform3d.external_methods._cross_modal import (
    resolve_cross_modal_manifest_path,
    run_cross_modal_evaluation_cli,
)
from reform3d.external_methods.voco.runtime import (
    DEFAULT_CHECKPOINT_NAME,
    DEFAULT_FEATURE_TYPE,
    SUPPORTED_FEATURE_TYPES,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Cross-Modal Alignment for VoCo embeddings")
    parser.add_argument("--dataset", default="amos22_ct_mr")
    parser.add_argument("-m", "--manifest", type=Path, default=None)
    parser.add_argument("-a", "--analysis-name", default=None)
    parser.add_argument("--manifest-variant", default="core")
    parser.add_argument("--embeddings-npz", type=Path, default=None)
    parser.add_argument("--checkpoint-name", default=DEFAULT_CHECKPOINT_NAME)
    parser.add_argument("--feature-type", default=DEFAULT_FEATURE_TYPE, choices=list(SUPPORTED_FEATURE_TYPES))
    parser.add_argument("--required-modalities", nargs="+", default=["ct", "mr"])
    parser.add_argument("--min-samples-per-modality", type=int, default=5)
    parser.add_argument("--max-samples-per-pool", type=int, default=None)
    parser.add_argument("--max-queries-per-organ", type=int, default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_cross_modal_evaluation_cli(
        manifest_path=resolve_cross_modal_manifest_path(args.dataset, args.manifest, args.manifest_variant),
        checkpoint_name=args.checkpoint_name,
        feature_type=args.feature_type,
        embeddings_npz=args.embeddings_npz,
        analysis_name=args.analysis_name,
        manifest_variant=args.manifest_variant,
        required_modalities=tuple(args.required_modalities),
        min_samples_per_modality=args.min_samples_per_modality,
        max_samples_per_pool=args.max_samples_per_pool,
        max_queries_per_organ=args.max_queries_per_organ,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
