"""Explicit output context for the public runner."""
import os
from pathlib import Path


def active_output_paths(module):
    value = os.environ.get("REFORM3D_RUN_DIR")
    if not value:
        return None
    workspace = Path(os.environ["REFORM3D_WORKSPACE"])
    dataset = os.environ.get('REFORM3D_DATASET')
    if os.environ.get('REFORM3D_ENCODER_DIR'):
        root = Path(os.environ['REFORM3D_ENCODER_DIR'])
        cache = Path(os.environ['REFORM3D_CACHE_DIR'])
        return {'root': root, 'results': root, 'figures': root / 'figures',
                'logs': root / 'logs', 'features': root / 'features',
                'cache_root': cache, 'crop_cache_root': cache,
                'manifests': workspace / 'manifests' / dataset,
                'source_cross_modal_root': root / 'features'}
    root = Path(value)
    seed = os.environ.get("MED3DINO_REFORM3D_ACTIVE_SEED")
    if seed is not None:
        root = root / "seeds" / f"seed_{int(seed)}"
    root = root / module
    cache = workspace / 'cache' / dataset if dataset else workspace / 'cache'
    manifests = workspace / 'manifests' / dataset if dataset else workspace / 'manifests'
    return {"root": root, "results": root, "figures": root / "figures", "logs": root / "logs",
            "features": root / "features", "cache_root": cache,
            "crop_cache_root": cache, "manifests": manifests,
            "source_cross_modal_root": root / "features"}
