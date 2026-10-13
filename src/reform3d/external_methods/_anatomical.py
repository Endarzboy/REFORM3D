"""Anatomical Generalization helpers for external baseline methods."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from reform3d.modules.anatomical_generalization.config import (
    DEFAULT_MANIFEST_VARIANT,
    get_anatomical_manifest_path,
    get_cross_modal_feature_npz_path,
    get_dataset_name_from_manifest_path,
    get_manifest_variant_from_manifest_path,
    normalize_anatomical_dataset_name,
)


def resolve_anatomical_manifest_path(
    dataset: str,
    manifest_path: Optional[Path],
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> Path:
    """Resolve an Anatomical manifest from an explicit path or dataset key."""

    if manifest_path is not None:
        resolved = Path(manifest_path).resolve()
    else:
        resolved = get_anatomical_manifest_path(
            normalize_anatomical_dataset_name(dataset),
            "sampled",
            manifest_variant,
        )
    if not resolved.exists():
        raise FileNotFoundError(
            f"Anatomical Generalization manifest not found: {resolved}. "
            "Build the manifest first or launch through reform3d/scripts/run_external_methods.sh "
            "so missing manifests are prepared before evaluation."
        )
    return resolved


def load_required_modalities(manifest_path: Path) -> List[str]:
    """Read the required modality contract from an Anatomical manifest."""

    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    required_modalities = None
    if isinstance(payload, dict):
        required_modalities = payload.get("required_modalities")
        if not required_modalities:
            protocol = payload.get("anatomical_protocol") or payload.get("phase" + "4_protocol")
            if isinstance(protocol, dict):
                required_modalities = protocol.get("required_modalities")
        if not required_modalities:
            samples = payload.get("samples")
            if isinstance(samples, list):
                ordered_modalities: list[str] = []
                seen: set[str] = set()
                for sample in samples:
                    modality = str(sample.get("modality", "")).strip().lower()
                    if modality and modality not in seen:
                        seen.add(modality)
                        ordered_modalities.append(modality)
                required_modalities = ordered_modalities
    if not required_modalities:
        raise ValueError(f"Unable to resolve required_modalities from Anatomical manifest: {manifest_path}")
    return [str(item).strip().lower() for item in required_modalities]


def run_anatomical_evaluation_cli(
    *,
    manifest_path: Path,
    checkpoint_name: str,
    feature_type: str,
    embeddings_npz: Optional[Path] = None,
    analysis_name: Optional[str] = None,
    manifest_variant: Optional[str] = None,
    required_modalities: Optional[list[str]] = None,
    min_holdout_organs: Optional[int] = None,
    min_samples_per_modality: Optional[int] = None,
    few_shot_support_per_modality: Optional[int] = None,
    few_shot_query_per_modality: Optional[int] = None,
    few_shot_seeds: Optional[list[int]] = None,
    seed: int = 42,
    output_path: Optional[Path] = None,
) -> None:
    """Run Anatomical Generalization for an external method feature bundle."""

    manifest_path = Path(manifest_path).resolve()
    analysis_name = analysis_name or get_dataset_name_from_manifest_path(manifest_path)
    manifest_variant = manifest_variant or get_manifest_variant_from_manifest_path(manifest_path)
    required_modalities = required_modalities or load_required_modalities(manifest_path)
    embeddings_npz = (
        Path(embeddings_npz).resolve()
        if embeddings_npz
        else get_cross_modal_feature_npz_path(
            analysis_name,
            manifest_variant,
            checkpoint_name,
            feature_type,
        )
    )
    if not embeddings_npz.exists():
        raise FileNotFoundError(
            f"Missing external-method Cross-Modal embedding bundle for '{analysis_name}': {embeddings_npz}. "
            "Run Cross-Modal feature extraction first."
        )

    argv = [
        sys.executable,
        "-m",
        "reform3d.modules.anatomical_generalization.evaluation_pipeline",
        "--manifest",
        str(manifest_path),
        "--analysis-name",
        analysis_name,
        "--manifest-variant",
        str(manifest_variant),
        "--embeddings-npz",
        str(embeddings_npz),
        "--checkpoint-name",
        checkpoint_name,
        "--feature-type",
        feature_type,
        "--required-modalities",
        *required_modalities,
        "--seed",
        str(seed),
    ]
    if min_holdout_organs is not None:
        argv.extend(["--min-holdout-organs", str(min_holdout_organs)])
    if min_samples_per_modality is not None:
        argv.extend(["--min-samples-per-modality", str(min_samples_per_modality)])
    if few_shot_support_per_modality is not None:
        argv.extend(["--few-shot-support-per-modality", str(few_shot_support_per_modality)])
    if few_shot_query_per_modality is not None:
        argv.extend(["--few-shot-query-per-modality", str(few_shot_query_per_modality)])
    if few_shot_seeds:
        argv.extend(["--few-shot-seeds", *[str(item) for item in few_shot_seeds]])
    if output_path:
        argv.extend(["--output-path", str(Path(output_path).resolve())])

    subprocess.run(argv, check=True)
