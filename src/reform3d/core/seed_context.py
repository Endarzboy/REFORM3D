"""Repeated-seed path helpers for REFORM-3D evaluations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


ACTIVE_SEED_ENV = "MED3DINO_REFORM3D_ACTIVE_SEED"
DEFAULT_REPEATED_SEEDS = (42, 123, 456)


def normalize_seed(seed: int | str) -> int:
    """Return a validated integer seed."""
    try:
        return int(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid evaluation seed: {seed}") from exc


def get_active_seed() -> int | None:
    """Return the active repeated-evaluation seed from the environment."""
    value = os.environ.get(ACTIVE_SEED_ENV)
    if value in {None, ""}:
        return None
    return normalize_seed(value)


def seed_directory_name(seed: int | str) -> str:
    """Return the directory name used for one repeated-evaluation seed."""
    return f"seed_{normalize_seed(seed)}"


def seed_subdir(seed: int | str) -> Path:
    """Return the relative seed subdirectory."""
    return Path("seeds") / seed_directory_name(seed)


def apply_seed_to_path(path: Path | str, seed: int | str | None) -> Path:
    """Append the seed namespace to a dataset/variant path when requested."""
    path = Path(path)
    if seed is None:
        return path
    return path / seed_subdir(seed)


def apply_active_seed_to_path(path: Path | str) -> Path:
    """Append the active seed namespace when repeated-seed mode is active."""
    return apply_seed_to_path(path, get_active_seed())


def with_active_seed_env(env: Mapping[str, str] | None, seed: int | str) -> dict[str, str]:
    """Return an environment mapping with the active repeated-evaluation seed."""
    merged = dict(os.environ if env is None else env)
    merged[ACTIVE_SEED_ENV] = str(normalize_seed(seed))
    return merged


__all__ = [
    "ACTIVE_SEED_ENV",
    "DEFAULT_REPEATED_SEEDS",
    "apply_active_seed_to_path",
    "apply_seed_to_path",
    "get_active_seed",
    "normalize_seed",
    "seed_directory_name",
    "seed_subdir",
    "with_active_seed_env",
]
