"""Checkpoint registry for REFORM-3D evaluation.

The registry maps checkpoint keys and aliases to model metadata, crop size, and
the expected checkpoint file under ``MED3DINO_CHECKPOINT_ROOT``.

Example:
    >>> resolve_checkpoint_config("Med3DINO-REL-96").checkpoint_key
    'med3dino_rel_96'
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

from .paths import CHECKPOINT_ROOT


CHECKPOINT_ORDER = (
    "med3dino_rel_96",
    "med3dino_iso_96",
    "med3dino_sa_96",
    "3dino",
)
FULL_CHECKPOINT_ORDER = CHECKPOINT_ORDER
DEFAULT_REFORM3D_CHECKPOINT_ORDER = CHECKPOINT_ORDER


@dataclass(frozen=True)
class CheckpointConfig:
    checkpoint_key: str
    model_name: str
    model_family: str
    geometry_assumption: str
    crop_size: int
    training_scope: str
    architecture: str
    raw_checkpoint_path: Path
    consumer_scope: tuple[str, ...]
    release_name: str
    status: str
    training_data: str = ""
    spacing_mode: str = ""
    arch: str = ""
    description: str = ""
    notes: str = ""
    aliases: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.checkpoint_key

    @property
    def path(self) -> Path:
        return self.raw_checkpoint_path

    def relative_repo_path(self) -> str:
        return self.raw_checkpoint_path.as_posix()


def _entry(
    checkpoint_key: str,
    *,
    checkpoint_relpath: str,
    crop_size: int,
    training_data: str,
    spacing_mode: str,
    model_name: str,
    model_family: str,
    geometry_assumption: str,
    training_scope: str,
    architecture: str,
    release_name: str,
    status: str,
    description: str,
    notes: str = "",
    aliases: Iterable[str] = (),
) -> CheckpointConfig:
    checkpoint_path = CHECKPOINT_ROOT / checkpoint_relpath
    return CheckpointConfig(
        checkpoint_key=checkpoint_key,
        model_name=model_name,
        model_family=model_family,
        geometry_assumption=geometry_assumption,
        crop_size=crop_size,
        training_scope=training_scope,
        architecture=architecture,
        raw_checkpoint_path=checkpoint_path,
        consumer_scope=("reform3d",),
        release_name=release_name,
        status=status,
        training_data=training_data,
        spacing_mode=spacing_mode,
        arch=architecture,
        description=description,
        notes=notes,
        aliases=tuple(aliases),
    )


def _med3dino_entry(
    variant: str,
    crop_size: int,
    *,
    model_name: str,
    training_data: str,
    spacing_mode: str,
    geometry_assumption: str,
    description: str,
    aliases: Iterable[str] = (),
) -> CheckpointConfig:
    key = f"med3dino_{variant}_{crop_size}"
    release_name = f"{model_name}-{crop_size}"
    return _entry(
        key,
        checkpoint_relpath=f"med3dino_{variant}/c{crop_size}_{variant}_teacher_checkpoint.pth",
        crop_size=crop_size,
        training_data=training_data,
        spacing_mode=spacing_mode,
        model_name=model_name,
        model_family="med3dino",
        geometry_assumption=geometry_assumption,
        training_scope=f"med3dino_{variant}",
        architecture="vit_large_3d",
        release_name=release_name,
        status="verified",
        description=description,
        aliases=aliases,
    )


CHECKPOINTS: Dict[str, CheckpointConfig] = {
    "med3dino_rel_96": _med3dino_entry(
        "rel",
        96,
        model_name="Med3DINO-REL",
        training_data="med3dino_full_release",
        spacing_mode="relative",
        geometry_assumption="relative",
        description="Med3DINO relative-spacing model, 96^3 crop.",
        aliases=("Med3DINO-REL-96", "med3dino-rel-96", "med3dino_rel_c96", "med3dino-rel-c96", "med3dino_rel", "Med3DINO-REL"),
    ),
    "med3dino_iso_96": _med3dino_entry(
        "iso",
        96,
        model_name="Med3DINO-ISO",
        training_data="med3dino_full_release",
        spacing_mode="isotropic",
        geometry_assumption="isotropic",
        description="Med3DINO isotropic-spacing model, 96^3 crop.",
        aliases=("Med3DINO-ISO-96", "med3dino-iso-96", "med3dino_iso_c96", "med3dino-iso-c96", "med3dino_iso", "Med3DINO-ISO"),
    ),
    "med3dino_sa_96": _med3dino_entry(
        "sa",
        96,
        model_name="Med3DINO-SA",
        training_data="med3dino_full_release",
        spacing_mode="spacing_aware",
        geometry_assumption="spacing_aware",
        description="Med3DINO spacing-aware model, 96^3 crop.",
        aliases=("Med3DINO-SA-96", "med3dino-sa-96", "med3dino_sa_c96", "med3dino-sa-c96", "med3dino_sa", "Med3DINO-SA"),
    ),
    "3dino": _entry(
        "3dino",
        checkpoint_relpath="3dinov2/3dinov2_teacher_checkpoint.pth",
        crop_size=112,
        training_data="3dino_original_100k",
        spacing_mode="relative",
        model_name="3DINO",
        model_family="3dino",
        geometry_assumption="relative",
        training_scope="3dino_original_100k",
        architecture="vit_large_3d",
        release_name="3DINO",
        status="verified",
        description="3DINO baseline, 112^3 crop, ViT-Large 3D.",
        aliases=("DINOv2-3D", "3DINO"),
    ),
}


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


ALIAS_TO_CHECKPOINT_KEY: Dict[str, str] = {}
for checkpoint_key, checkpoint_config in CHECKPOINTS.items():
    ALIAS_TO_CHECKPOINT_KEY[_normalize_name(checkpoint_key)] = checkpoint_key
    ALIAS_TO_CHECKPOINT_KEY[_normalize_name(checkpoint_config.release_name)] = checkpoint_key
    for alias in checkpoint_config.aliases:
        ALIAS_TO_CHECKPOINT_KEY[_normalize_name(alias)] = checkpoint_key


def resolve_checkpoint_config(checkpoint_name: str) -> CheckpointConfig:
    normalized = _normalize_name(checkpoint_name)
    try:
        return CHECKPOINTS[ALIAS_TO_CHECKPOINT_KEY[normalized]]
    except KeyError as exc:
        known = ", ".join(CHECKPOINTS[name].release_name for name in CHECKPOINT_ORDER)
        raise KeyError(f"Unknown checkpoint '{checkpoint_name}'. Known checkpoints: {known}") from exc


def checkpoint_dataset_size_label(checkpoint_name: str) -> str:
    config = resolve_checkpoint_config(checkpoint_name)
    if config.model_family == "3dino":
        return "baseline"
    return config.model_name


def build_checkpoint_summary_meta(
    checkpoint_names: Iterable[str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    selected_names = tuple(checkpoint_names) if checkpoint_names is not None else CHECKPOINT_ORDER
    summary: Dict[str, Dict[str, Any]] = {}
    for checkpoint_name in selected_names:
        config = resolve_checkpoint_config(checkpoint_name)
        summary[config.checkpoint_key] = {
            "arch": config.arch,
            "crop": config.crop_size,
            "model": config.model_name,
            "spacing_mode": config.spacing_mode,
            "status": config.status,
        }
    return summary


__all__ = [
    "ALIAS_TO_CHECKPOINT_KEY",
    "CHECKPOINT_ROOT",
    "CHECKPOINTS",
    "DEFAULT_REFORM3D_CHECKPOINT_ORDER",
    "FULL_CHECKPOINT_ORDER",
    "CHECKPOINT_ORDER",
    "CheckpointConfig",
    "build_checkpoint_summary_meta",
    "checkpoint_dataset_size_label",
    "resolve_checkpoint_config",
]
