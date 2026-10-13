"""Extract GL-MAE organ embeddings for Cross-Modal Alignment."""

from reform3d.external_methods._primary_feature_cli import run_cross_modal_extract_cli
from reform3d.external_methods.gl_mae.runtime import METHOD


if __name__ == "__main__":
    run_cross_modal_extract_cli(METHOD)
