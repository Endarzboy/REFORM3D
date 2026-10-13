"""Run Cross-Modal Alignment metrics for MiM embeddings."""

from reform3d.external_methods._primary_feature_cli import run_cross_modal_evaluate_cli
from reform3d.external_methods.mim.runtime import METHOD


if __name__ == "__main__":
    run_cross_modal_evaluate_cli(METHOD)
