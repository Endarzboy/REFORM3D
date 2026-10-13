#!/usr/bin/env python
"""Build a dataset-scoped Acquisition Robustness manifest.

The command scans one registered raw dataset, extracts spacing metadata, assigns
anisotropy bins, and writes ``manifest_full.json``, ``manifest_sampled.json``,
and ``manifest_meta.json`` under the configured manifest root.

Inputs:
    --dataset: key from ``RAW_DATASETS`` such as ``abdomenatlas`` or ``kits23``.
    --output-suffix: output manifest namespace; defaults to the dataset key.
    --binning-scheme: spacing-bin policy, usually ``original`` or ``coarse_bins``.

Example:
    python -m reform3d.modules.acquisition_robustness.build_manifest \
        --dataset kits23 \
        --output-suffix kits23 \
        --binning-scheme original
"""

import argparse
from pathlib import Path

try:
    from .config import (
        RAW_DATASETS,
        DEFAULT_BINNING_SCHEME,
    )
    from .manifest_generation import (
        build_single_dataset_manifest,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from config import (
        RAW_DATASETS,
        DEFAULT_BINNING_SCHEME,
    )
    from manifest_generation import (
        build_single_dataset_manifest,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build single-dataset manifest for acquisition robustness evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python build_manifest.py --dataset abdomenatlas --output-suffix abdomenatlas --binning-scheme original\n"
            "  python build_manifest.py --dataset kits23 --output-suffix kits23 --binning-scheme coarse_bins\n"
            "  python build_manifest.py --dataset totalsegmentermri --output-suffix totalsegmentermri --binning-scheme original"
        ),
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        required=True,
        help=f"Dataset name. Available: {list(RAW_DATASETS.keys())}",
    )
    parser.add_argument(
        "-s", "--output-suffix",
        type=str,
        required=True,
        help="Output suffix (e.g., 'abdomenatlas', 'kits23')",
    )
    parser.add_argument(
        "--binning-scheme",
        type=str,
        default=DEFAULT_BINNING_SCHEME,
        choices=["original", "coarse_bins"],
        help="Binning scheme / manifest variant to generate (default: original)",
    )
    parser.add_argument(
        "--min-per-bin",
        type=int,
        default=500,
        help="Minimum samples per anisotropy bin (default: 500)",
    )
    parser.add_argument(
        "--max-per-bin",
        type=int,
        default=1000,
        help="Maximum samples per anisotropy bin (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate volume integrity (skip if pre-validated)",
    )
    parser.add_argument(
        "--force-validate",
        action="store_true",
        help="Force validation even for pre-validated datasets",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: acquisition manifest root)",
    )

    args = parser.parse_args()

    build_single_dataset_manifest(
        dataset_name=args.dataset,
        output_suffix=args.output_suffix,
        binning_scheme=args.binning_scheme,
        validate_integrity=args.validate,
        force_validate=args.force_validate,
        min_per_bin=args.min_per_bin,
        max_per_bin=args.max_per_bin,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
