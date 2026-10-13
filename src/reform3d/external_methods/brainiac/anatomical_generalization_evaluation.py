"""Run Anatomical Generalization metrics for BrainAIC embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

from reform3d.external_methods._anatomical import (
    resolve_anatomical_manifest_path,
    run_anatomical_evaluation_cli,
)
from reform3d.external_methods.brainiac.runtime import (
    DEFAULT_CHECKPOINT_NAME,
    DEFAULT_FEATURE_TYPE,
    SUPPORTED_FEATURE_TYPES,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Anatomical Generalization for BrainAIC embeddings")
    parser.add_argument("--dataset", default="amos22_ct_mr")
    parser.add_argument("-m", "--manifest", type=Path, default=None)
    parser.add_argument("-a", "--analysis-name", default=None)
    parser.add_argument("--manifest-variant", default="core")
    parser.add_argument("--embeddings-npz", type=Path, default=None)
    parser.add_argument("--checkpoint-name", default=DEFAULT_CHECKPOINT_NAME)
    parser.add_argument("--feature-type", default=DEFAULT_FEATURE_TYPE, choices=list(SUPPORTED_FEATURE_TYPES))
    parser.add_argument("--required-modalities", nargs="+", default=None)
    parser.add_argument("--min-holdout-organs", type=int, default=None)
    parser.add_argument("--min-samples-per-modality", type=int, default=None)
    parser.add_argument("--few-shot-support-per-modality", type=int, default=None)
    parser.add_argument("--few-shot-query-per-modality", type=int, default=None)
    parser.add_argument("--few-shot-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args()

    run_anatomical_evaluation_cli(
        manifest_path=resolve_anatomical_manifest_path(args.dataset, args.manifest, args.manifest_variant),
        checkpoint_name=args.checkpoint_name,
        feature_type=args.feature_type,
        embeddings_npz=args.embeddings_npz,
        analysis_name=args.analysis_name,
        manifest_variant=args.manifest_variant,
        required_modalities=args.required_modalities,
        min_holdout_organs=args.min_holdout_organs,
        min_samples_per_modality=args.min_samples_per_modality,
        few_shot_support_per_modality=args.few_shot_support_per_modality,
        few_shot_query_per_modality=args.few_shot_query_per_modality,
        few_shot_seeds=args.few_shot_seeds,
        seed=args.seed,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
