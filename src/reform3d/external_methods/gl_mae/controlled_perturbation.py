"""Run GL-MAE controlled Acquisition Robustness evaluation."""

from reform3d.external_methods._primary_feature_cli import run_controlled_cli
from reform3d.external_methods.gl_mae.runtime import METHOD


if __name__ == "__main__":
    run_controlled_cli(METHOD)
