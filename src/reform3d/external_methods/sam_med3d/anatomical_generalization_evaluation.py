"""Run Anatomical Generalization metrics for SAM-Med3D embeddings."""

from reform3d.external_methods._primary_feature_cli import run_anatomical_cli
from reform3d.external_methods.sam_med3d.runtime import METHOD


if __name__ == "__main__":
    run_anatomical_cli(METHOD)
