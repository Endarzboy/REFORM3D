"""Unified readout-sensitivity analysis for REFORM-3D.

This module is a post-hoc, CPU-only analysis that tests whether the principal
semantic-transfer and low-shot anatomical-transfer conclusions depend on
classifier optimization. It compares the current logistic readout against an
optimization-free nearest-centroid readout and against logistic regularization
values C in {0.1, 1, 10}, reusing frozen feature artifacts only. It never runs
feature extraction, crop generation, encoder inference, or CUDA.
"""

__all__ = [
    "config",
    "io_utils",
    "readouts",
    "semantic",
    "anatomical",
    "statistics",
    "tables",
]
