"""Construct a supported frozen encoder with its existing feature interface."""

from importlib import import_module
from pathlib import Path

import torch


EXTERNAL_EXTRACTORS = {
    "brainiac": ("brainiac", "BrainIACFeatureExtractor"),
    "voco": ("voco", "VoCoFeatureExtractor"),
    "voco_b_ssl_head": ("voco", "VoCoFeatureExtractor"),
    "openmind": ("openmind", "OpenMindFeatureExtractor"),
    "openmind_primusm_voco": ("openmind", "OpenMindFeatureExtractor"),
    "gl_mae": ("gl_mae", "GLMAEFeatureExtractor"),
    "gl_mae_vit_base_patch16": ("gl_mae", "GLMAEFeatureExtractor"),
    "hi_end_mae": ("hi_end_mae", "HiEndMAEFeatureExtractor"),
    "hi_end_mae_vit_p12_1536": ("hi_end_mae", "HiEndMAEFeatureExtractor"),
    "mambamim": ("mambamim", "MambaMIMFeatureExtractor"),
    "mambamim_mask75": ("mambamim", "MambaMIMFeatureExtractor"),
    "mim": ("mim", "MiMFeatureExtractor"),
    "mim_convvit_base": ("mim", "MiMFeatureExtractor"),
    "sam_med3d": ("sam_med3d", "SAMMed3DFeatureExtractor"),
    "sam_med3d_turbo": ("sam_med3d", "SAMMed3DFeatureExtractor"),
}


def create_feature_extractor(
    model_name: str,
    *,
    checkpoint_path: str | Path | None = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """Keep each model's native feature names and input-normalization behavior.

    The returned extractor provides ``config.input_size`` and
    ``extract(batch, normalize=True)``. Preprocess with the corresponding
    evaluation pipeline; ``normalize`` retains that runtime's existing meaning.
    Optional dependencies are imported only for the selected model.
    """
    external = EXTERNAL_EXTRACTORS.get(model_name.lower())
    if external is None:
        from reform3d.modules.acquisition_robustness.feature_extractor import (
            ExtractionConfig, FeatureExtractor,
        )

        extractor = FeatureExtractor(
            model_name,
            config=ExtractionConfig(device=device, dtype=dtype),
            checkpoint_path=checkpoint_path,
        )
    else:
        method, class_name = external
        runtime = import_module(f"reform3d.external_methods.{method}.runtime")
        path = Path(checkpoint_path).expanduser().resolve() if checkpoint_path is not None else runtime.DEFAULT_CHECKPOINT_PATH
        extractor = getattr(runtime, class_name)(
            checkpoint_path=path, device=device, dtype=dtype,
        )
    extractor.model.eval()
    extractor.model.requires_grad_(False)
    return extractor
