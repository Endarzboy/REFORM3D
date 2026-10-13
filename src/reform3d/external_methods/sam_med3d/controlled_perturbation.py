"""Run SAM-Med3D controlled Acquisition Robustness evaluation."""

from reform3d.external_methods._primary_feature_cli import run_controlled_cli
from reform3d.external_methods.sam_med3d.runtime import METHOD


if __name__ == "__main__":
    run_controlled_cli(METHOD)
