"""Explicit output context for the public runner."""
import os
from pathlib import Path


def active_output_paths(module):
    value = os.environ.get("REFORM3D_RUN_DIR")
    if not value:
        return None
    root = Path(value)
    seed = os.environ.get("MED3DINO_REFORM3D_ACTIVE_SEED")
    if seed is not None:
        root = root / "seeds" / f"seed_{int(seed)}"
    root = root / module
    workspace = Path(os.environ["REFORM3D_WORKSPACE"])
    return {"root": root, "results": root, "figures": root / "figures", "logs": root / "logs",
            "features": workspace / "features", "cache_root": workspace / "cache",
            "crop_cache_root": workspace / "cache", "manifests": workspace / "manifests",
            "source_cross_modal_root": workspace / "features"}
