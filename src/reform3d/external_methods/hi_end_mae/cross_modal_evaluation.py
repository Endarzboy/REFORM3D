"""Run Cross-Modal Alignment metrics for Hi-End-MAE embeddings."""

from reform3d.external_methods._primary_feature_cli import run_cross_modal_evaluate_cli
from reform3d.external_methods.hi_end_mae.runtime import METHOD


if __name__ == "__main__":
    run_cross_modal_evaluate_cli(METHOD)
