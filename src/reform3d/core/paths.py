"""Path registry for REFORM-3D evaluation.

The registry resolves repository, dataset, checkpoint, manifest, cache, and
result roots from environment variables with Linux evaluation-machine defaults.
All modules use these helpers so artifacts land under the same directory
contract.

Common inputs:
    MED3DINO_EXTERNAL_DATA_ROOT, MED3DINO_CHECKPOINT_ROOT,
    MED3DINO_EXTERNAL_METHODS_CHECKPOINT_ROOT,
    MED3DINO_SHARED_MANIFEST_ROOT, MED3DINO_SHARED_CACHE_ROOT,
    MED3DINO_SHARED_RESULT_ROOT.
"""

from __future__ import annotations

import os
from pathlib import Path


def _resolve_path(env_var: str, default: Path) -> Path:
    value = os.environ.get(env_var)
    if value:
        return Path(value).expanduser().resolve()
    return Path(default).expanduser().resolve()


def _resolve_path_from_env(env_vars: tuple[str, ...], default: Path) -> Path:
    for env_var in env_vars:
        value = os.environ.get(env_var)
        if value:
            return Path(value).expanduser().resolve()
    return Path(default).expanduser().resolve()


PROJECT_ROOT = _resolve_path("MED3DINO_REPO_ROOT", Path(__file__).resolve().parents[3])

EXTERNAL_DATA_ROOT = _resolve_path(
    "MED3DINO_EXTERNAL_DATA_ROOT",
    Path.cwd() / "datasets",
)

CHECKPOINT_ROOT = _resolve_path(
    "MED3DINO_CHECKPOINT_ROOT",
    Path.cwd() / "checkpoints",
)
EXTERNAL_METHOD_CHECKPOINT_ROOT = _resolve_path_from_env(
    ("MED3DINO_EXTERNAL_METHODS_CHECKPOINT_ROOT", "MED3DINO_EXTERNAL_METHOD_CHECKPOINT_ROOT"),
    CHECKPOINT_ROOT / "external",
)
DOWNSTREAM_DATASET_ROOT = _resolve_path(
    "MED3DINO_DOWNSTREAM_DATASET_ROOT",
    EXTERNAL_DATA_ROOT / "downstreamTasks",
)
REFORM3D_DATASET_ROOT = _resolve_path(
    "MED3DINO_REFORM3D_DATASET_ROOT",
    EXTERNAL_DATA_ROOT / "3DFM_Benchmark",
)
SHARED_MANIFEST_ROOT = _resolve_path(
    "MED3DINO_SHARED_MANIFEST_ROOT",
    Path.cwd() / "workspace" / "manifests",
)
SHARED_CACHE_ROOT = _resolve_path(
    "MED3DINO_SHARED_CACHE_ROOT",
    Path.cwd() / "workspace" / "cache",
)
SHARED_RESULT_ROOT = _resolve_path(
    "MED3DINO_SHARED_RESULT_ROOT",
    Path.cwd() / "workspace" / "runs",
)

REFORM3D_MANIFEST_ROOT = SHARED_MANIFEST_ROOT
REFORM3D_CACHE_ROOT = SHARED_CACHE_ROOT
REFORM3D_RESULT_ROOT = SHARED_RESULT_ROOT


def get_reform3d_module_result_roots(
    module_name: str,
    output_directory_name: str | None = None,
) -> tuple[Path, ...]:
    return (REFORM3D_RESULT_ROOT / module_name,)


def get_reform3d_module_result_root(
    module_name: str,
    output_directory_name: str | None = None,
) -> Path:
    return REFORM3D_RESULT_ROOT / module_name


def get_reform3d_module_results_root(
    module_name: str,
    output_directory_name: str | None = None,
) -> Path:
    return get_reform3d_module_result_root(module_name, output_directory_name)


def resolve_reform3d_module_result_path(
    module_name: str,
    output_directory_name: str | None,
    relative_path: Path | str,
) -> Path:
    relative_path = Path(relative_path)
    if relative_path.is_absolute():
        return relative_path
    return get_reform3d_module_result_root(module_name, output_directory_name) / relative_path


def get_reform3d_module_manifest_roots(module_name: str) -> tuple[Path, ...]:
    return (REFORM3D_MANIFEST_ROOT / module_name,)


def get_reform3d_module_manifest_root(module_name: str) -> Path:
    return REFORM3D_MANIFEST_ROOT / module_name


def resolve_reform3d_module_manifest_path(
    module_name: str,
    relative_path: Path | str,
) -> Path:
    relative_path = Path(relative_path)
    if relative_path.is_absolute():
        return relative_path
    return get_reform3d_module_manifest_root(module_name) / relative_path


def get_reform3d_phase_result_roots(
    phase_directory_name: str,
    output_directory_name: str | None = None,
) -> tuple[Path, ...]:
    return get_reform3d_module_result_roots(phase_directory_name, output_directory_name)


def get_reform3d_phase_result_root(
    phase_directory_name: str,
    output_directory_name: str | None = None,
) -> Path:
    return get_reform3d_module_result_root(phase_directory_name, output_directory_name)


def get_reform3d_phase_results_root(
    phase_directory_name: str,
    output_directory_name: str | None = None,
) -> Path:
    return get_reform3d_module_results_root(phase_directory_name, output_directory_name)


def resolve_reform3d_phase_result_path(
    phase_directory_name: str,
    output_directory_name: str | None,
    relative_path: Path | str,
) -> Path:
    return resolve_reform3d_module_result_path(
        phase_directory_name,
        output_directory_name,
        relative_path,
    )


def get_reform3d_phase_manifest_roots(phase_directory_name: str) -> tuple[Path, ...]:
    return get_reform3d_module_manifest_roots(phase_directory_name)


def get_reform3d_phase_manifest_root(phase_directory_name: str) -> Path:
    return get_reform3d_module_manifest_root(phase_directory_name)


def resolve_reform3d_phase_manifest_path(
    phase_directory_name: str,
    relative_path: Path | str,
) -> Path:
    return resolve_reform3d_module_manifest_path(phase_directory_name, relative_path)


def get_reform3d_cache_roots() -> tuple[Path, ...]:
    return (REFORM3D_CACHE_ROOT,)


def get_reform3d_cache_path_candidates(path: Path | str) -> tuple[Path, ...]:
    path = Path(path)
    if not path.is_absolute():
        return (REFORM3D_CACHE_ROOT / path,)
    return (path.expanduser().resolve(),)


def resolve_reform3d_cache_path(path: Path | str) -> Path:
    return get_reform3d_cache_path_candidates(path)[0]


def get_reform3d_cache_root() -> Path:
    return REFORM3D_CACHE_ROOT


def get_external_method_checkpoint_root(method_name: str) -> Path:
    method_key = str(method_name).strip().lower()
    if not method_key:
        raise ValueError("External method name cannot be empty")
    return EXTERNAL_METHOD_CHECKPOINT_ROOT / method_key / "checkpoints"


def resolve_external_method_checkpoint_path(method_name: str, checkpoint_name: str) -> Path:
    checkpoint_name = str(checkpoint_name).strip()
    if not checkpoint_name:
        raise ValueError("External method checkpoint name cannot be empty")
    return get_external_method_checkpoint_root(method_name) / checkpoint_name


__all__ = [
    "CHECKPOINT_ROOT",
    "DOWNSTREAM_DATASET_ROOT",
    "EXTERNAL_METHOD_CHECKPOINT_ROOT",
    "REFORM3D_CACHE_ROOT",
    "REFORM3D_DATASET_ROOT",
    "REFORM3D_MANIFEST_ROOT",
    "REFORM3D_RESULT_ROOT",
    "SHARED_CACHE_ROOT",
    "SHARED_MANIFEST_ROOT",
    "SHARED_RESULT_ROOT",
    "EXTERNAL_DATA_ROOT",
    "PROJECT_ROOT",
    "get_external_method_checkpoint_root",
    "get_reform3d_cache_path_candidates",
    "get_reform3d_cache_root",
    "get_reform3d_cache_roots",
    "get_reform3d_module_manifest_root",
    "get_reform3d_module_manifest_roots",
    "get_reform3d_module_result_root",
    "get_reform3d_module_result_roots",
    "get_reform3d_module_results_root",
    "resolve_reform3d_cache_path",
    "resolve_external_method_checkpoint_path",
    "resolve_reform3d_module_manifest_path",
    "resolve_reform3d_module_result_path",
]
