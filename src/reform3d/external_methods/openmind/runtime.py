"""OpenMind PrimusM VoCo runtime for REFORM-3D evaluations."""

from __future__ import annotations

import importlib
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable

import torch

from reform3d.core.paths import resolve_external_method_checkpoint_path
from reform3d.encoders.checkpoints import load_encoder_state
from reform3d.external_methods._sources import source_root, add_source_path


THIS_FILE = Path(__file__).resolve()
OPENMIND_ROOT = THIS_FILE.parent
PROJECT_ROOT = THIS_FILE.parents[3]
OPENMIND_SRC = source_root("openmind", OPENMIND_ROOT / "nnssl-openneuro" / "src")

DEFAULT_MODEL_DIR = OPENMIND_ROOT / "PrimusM-OpenMind-VoCo"
DEFAULT_CHECKPOINT_PATH = resolve_external_method_checkpoint_path("openmind", "checkpoint_final.pth")
DEFAULT_ADAPTATION_PLAN_PATH = Path(os.environ.get("REFORM3D_OPENMIND_ADAPTATION_PLAN", str(DEFAULT_MODEL_DIR / "adaptation_plan.json"))).expanduser()
DEFAULT_CHECKPOINT_NAME = "openmind_primusm_voco"
DEFAULT_MODEL_NAME = "OpenMind PrimusM VoCo"
DEFAULT_FEATURE_TYPE = "avg_pool"
SUPPORTED_FEATURE_TYPES = ("avg_pool", "projected")
FEATURE_SURFACES = {
    "avg_pool": "encoder_avg_pool",
    "projected": "mean_token_projector_head",
}
DEFAULT_CROP_SIZE = 64
DEFAULT_OUTPUT_CHANNELS = 3


logger = logging.getLogger(__name__)


def ensure_import_paths(extra_paths: Iterable[Path] = ()) -> None:
    """Add OpenMind source directories needed by the runtime loader."""

    add_source_path(
        OPENMIND_SRC, "nnssl/architectures/get_network_by_name.py", ("nnssl",)
    )
    for path in extra_paths:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON object from disk."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_dtype(dtype_name: str) -> torch.dtype:
    """Resolve a CLI dtype name to a torch dtype."""

    normalized = str(dtype_name).strip().lower()
    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def strip_state_dict_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove distributed-training prefixes from checkpoint keys."""

    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        cleaned_key = key
        if cleaned_key.startswith("module."):
            cleaned_key = cleaned_key[len("module.") :]
        if cleaned_key.startswith("_orig_mod."):
            cleaned_key = cleaned_key[len("_orig_mod.") :]
        cleaned[cleaned_key] = value
    return cleaned


def resolve_dotted_attr(root: Any, path: str) -> Any:
    """Resolve a dotted attribute path on a loaded OpenMind module."""

    current = root
    for part in str(path).split("."):
        current = getattr(current, part)
    return current


def infer_embed_dim(state_dict: Dict[str, torch.Tensor]) -> int:
    """Infer the OpenMind embedding dimension from checkpoint tensors."""

    projector_key = "projector.layer1.0.weight"
    if projector_key in state_dict:
        return int(state_dict[projector_key].shape[1])

    pos_embed_key = "encoder.eva.pos_embed"
    if pos_embed_key in state_dict:
        return int(state_dict[pos_embed_key].shape[-1])

    raise KeyError(
        "Could not infer OpenMind embed_dim from checkpoint. Expected either "
        f"'{projector_key}' or '{pos_embed_key}'."
    )


def zscore_normalize_full_volume(batch: torch.Tensor) -> torch.Tensor:
    """Apply per-volume z-score normalization."""

    normalized = batch.clone()
    for batch_idx in range(normalized.shape[0]):
        sample = normalized[batch_idx]
        mean = sample.mean()
        std = sample.std(unbiased=False)
        if torch.isfinite(std) and float(std) > 0.0:
            normalized[batch_idx] = (sample - mean) / std
        else:
            normalized[batch_idx] = sample - mean
    return normalized


def inspect_openmind_runtime() -> Dict[str, Any]:
    """Inspect the local OpenMind source tree and required Python modules."""

    warnings: list[str] = []
    blockers: list[str] = []
    try:
        ensure_import_paths()
    except (FileNotFoundError, RuntimeError) as exc:
        blockers.append(str(exc))

    if not OPENMIND_SRC.exists():
        blockers.append(f"OpenMind nnssl source tree not found: {OPENMIND_SRC}")

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info < (3, 12):
        warnings.append(
            "Current Python is "
            f"{python_version}, but OpenMind nnssl declares Python >= 3.12 in its pyproject.toml."
        )

    import_checks = {
        "nnssl.architectures.get_network_by_name": "nnssl model loader",
        "nnssl.architectures.voco_architecture": "nnssl VoCo architecture",
        "dynamic_network_architectures": "dynamic-network-architectures dependency",
        "batchgenerators": "batchgenerators dependency",
    }
    for module_name, label in import_checks.items():
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            blockers.append(f"Missing {label}: {exc}")
        except Exception as exc:
            blockers.append(f"Failed to import {label}: {exc}")

    return {
        "python_version": python_version,
        "warnings": warnings,
        "blockers": blockers,
        "openmind_src": str(OPENMIND_SRC),
        "checkpoint_path": str(DEFAULT_CHECKPOINT_PATH),
    }


def format_runtime_report(report: Dict[str, Any]) -> str:
    """Format an OpenMind runtime-inspection report."""

    lines = [f"Python: {report['python_version']}", f"OpenMind src: {report['openmind_src']}"]

    warnings = report.get("warnings") or []
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in warnings)

    blockers = report.get("blockers") or []
    if blockers:
        lines.append("Blockers:")
        lines.extend(f"- {blocker}" for blocker in blockers)
    else:
        lines.append("Blockers: none")

    return "\n".join(lines)


def import_openmind_modules() -> tuple[Any, Any]:
    """Import the OpenMind model-construction symbols."""

    report = inspect_openmind_runtime()
    for warning in report["warnings"]:
        logger.warning(warning)

    if report["blockers"]:
        raise RuntimeError(
            "OpenMind runtime is not ready.\n"
            + format_runtime_report(report)
            + "\nInstall the missing OpenMind dependencies in an isolated environment before running this evaluation."
        )

    get_network_by_name = importlib.import_module(
        "nnssl.architectures.get_network_by_name"
    ).get_network_by_name
    VoCoEvaArchitecture = importlib.import_module(
        "nnssl.architectures.voco_architecture"
    ).VoCoEvaArchitecture
    return get_network_by_name, VoCoEvaArchitecture


def load_adaptation_plan(checkpoint_path: Path, adaptation_plan_path: Path) -> Dict[str, Any]:
    """Load the OpenMind adaptation plan from JSON or from the checkpoint payload."""

    adaptation_plan_path = Path(adaptation_plan_path)
    if adaptation_plan_path.exists():
        return load_json(adaptation_plan_path)

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    adaptation_plan = checkpoint.get("nnssl_adaptation_plan")
    if not adaptation_plan:
        raise FileNotFoundError(
            "Could not find OpenMind adaptation plan. Checked "
            f"'{adaptation_plan_path}' and checkpoint['nnssl_adaptation_plan']."
        )
    return adaptation_plan


def resolve_model_patch_size(adaptation_plan: Dict[str, Any]) -> tuple[int, int, int]:
    """Return the OpenMind model patch size from an adaptation plan."""

    configurations = adaptation_plan.get("pretrain_plan", {}).get("configurations", {})
    onemmiso = configurations.get("onemmiso", {})
    patch_size = onemmiso.get("patch_size")
    if patch_size is None:
        raise KeyError("OpenMind adaptation plan is missing pretrain_plan.configurations.onemmiso.patch_size")
    return tuple(int(value) for value in patch_size)


class OpenMindFeatureExtractor:
    """OpenMind feature extractor exposing avg-pool and projected descriptors."""

    def __init__(
        self,
        checkpoint_path: Path,
        device: str,
        dtype: torch.dtype,
        adaptation_plan_path: Path = DEFAULT_ADAPTATION_PLAN_PATH,
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        adaptation_plan_path = Path(adaptation_plan_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"OpenMind checkpoint not found: {checkpoint_path}")

        get_network_by_name, voco_architecture = import_openmind_modules()
        adaptation_plan = load_adaptation_plan(checkpoint_path, adaptation_plan_path)
        patch_size = resolve_model_patch_size(adaptation_plan)
        architecture_name = adaptation_plan.get("architecture_plans", {}).get("arch_class_name", "PrimusM")
        num_input_channels = int(adaptation_plan.get("pretrain_num_input_channels", 1))

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = strip_state_dict_prefixes(checkpoint["network_weights"])
        embed_dim = infer_embed_dim(state_dict)

        encoder = get_network_by_name(
            configuration_plan=SimpleNamespace(patch_size=list(patch_size)),
            architecture_name=architecture_name,
            num_input_channels=num_input_channels,
            num_output_channels=DEFAULT_OUTPUT_CHANNELS,
        )
        self.model = voco_architecture(encoder=encoder, embed_dim=embed_dim)
        load_encoder_state(self.model, state_dict)

        self.model.eval()
        self.model.to(device=device, dtype=dtype)
        self.config = SimpleNamespace(
            device=device,
            dtype=dtype,
            crop_size=int(patch_size[0]),
            input_size=int(patch_size[0]),
            checkpoint_path=str(checkpoint_path),
            adaptation_plan_path=str(adaptation_plan_path),
            architecture_name=architecture_name,
            embed_dim=embed_dim,
        )
        self.adaptation_plan = adaptation_plan

    def _prepare_batch(self, batch: torch.Tensor, normalize: bool) -> torch.Tensor:
        if batch.ndim != 5:
            raise ValueError(f"Expected [B, C, H, W, D] input, got shape {tuple(batch.shape)}")

        if normalize:
            batch = zscore_normalize_full_volume(batch)
        return batch.to(device=self.config.device, dtype=self.config.dtype, non_blocking=True)

    def _extract_encoder_avg_pool_from_prepared_batch(self, batch: torch.Tensor) -> torch.Tensor:
        encoder_output = self.model.encoder(batch)
        pooled = self.model.adaptive_pool(encoder_output)
        return torch.reshape(pooled, (pooled.shape[0], -1))

    def _extract_projected_from_prepared_batch(self, batch: torch.Tensor) -> torch.Tensor:
        stem_path = self.adaptation_plan.get("key_to_stem")
        encoder_path = self.adaptation_plan.get("key_to_encoder")
        if not stem_path or not encoder_path:
            raise KeyError("OpenMind adaptation plan is missing key_to_stem or key_to_encoder for projected extraction")

        stem = resolve_dotted_attr(self.model, stem_path)
        encoder = resolve_dotted_attr(self.model, encoder_path)

        token_grid = stem(batch)
        if token_grid.ndim != 5:
            raise ValueError(f"Expected token grid with 5 dims after stem, got shape {tuple(token_grid.shape)}")

        token_sequence = token_grid.flatten(2).transpose(1, 2).contiguous()
        encoded = encoder(token_sequence)
        if isinstance(encoded, tuple):
            encoded = encoded[0]
        if encoded.ndim != 3:
            raise ValueError(f"Expected encoded token sequence with 3 dims, got shape {tuple(encoded.shape)}")

        pooled_tokens = encoded.mean(dim=1)
        return self.model.projector(pooled_tokens)

    @torch.inference_mode()
    def extract(self, batch: torch.Tensor, normalize: bool = True) -> Dict[str, torch.Tensor]:
        batch = self._prepare_batch(batch, normalize=normalize)
        pooled = self._extract_encoder_avg_pool_from_prepared_batch(batch)
        projected = self._extract_projected_from_prepared_batch(batch)
        return {DEFAULT_FEATURE_TYPE: pooled, "projected": projected}
