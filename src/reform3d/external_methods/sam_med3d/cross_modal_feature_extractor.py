"""Extract SAM-Med3D organ embeddings for Cross-Modal Alignment."""

from reform3d.external_methods._primary_feature_cli import run_cross_modal_extract_cli
from reform3d.external_methods.sam_med3d.runtime import METHOD


if __name__ == "__main__":
    run_cross_modal_extract_cli(METHOD)
