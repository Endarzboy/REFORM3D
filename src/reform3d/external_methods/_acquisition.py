"""Acquisition Robustness helpers for external baseline methods.

The functions in this module let an external feature extractor participate in
the same native-spacing and controlled-perturbation analyses used by the main
Acquisition Robustness module.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

from reform3d.modules.acquisition_robustness.config import (
    DEFAULT_MANIFEST_VARIANT,
    get_acquisition_manifest_path,
    get_bin_name_map,
    get_binning_scheme_from_manifest,
    get_cache_root,
    get_checkpoint_feature_dir,
    get_controlled_perturbation_config_from_manifest,
    get_dataset_name_from_manifest_path,
    get_manifest_variant_from_manifest_path,
    get_output_paths,
)
from reform3d.modules.acquisition_robustness.data_loader import (
    create_acquisition_dataloader,
    warmup_observational_cache,
)
from reform3d.modules.acquisition_robustness.evaluation_pipeline import write_results_catalog
from reform3d.modules.acquisition_robustness.perturbation_robustness_analysis import (
    _load_matched_semantic_targets,
    _resolve_controlled_dirs,
    compute_cka_between_variants,
    compute_pairwise_distances,
    extract_features_for_variants,
    plot_drift_curve,
    prepare_perturbed_dataset,
    run_matched_semantic_probing,
    run_matched_semantic_transfer,
    save_controlled_embedding_bundle,
)
from reform3d.modules.acquisition_robustness.representation_geometry_analysis import (
    run_representation_geometry_analysis,
)
from reform3d.modules.acquisition_robustness.semantic_analysis import (
    run_multilabel_cross_bin_transfer,
    run_semantic_probing,
    serialize_spacing_regression_results,
    spacing_regression_probe,
)
from reform3d.modules.acquisition_robustness.semantic_labels import compute_semantic_task_labels


logger = logging.getLogger(__name__)

ACQUISITION_DATASET_ALIASES = {
    "abdomenatlas": "abdomenatlas",
    "kits23": "kits23",
    "imagecas": "imagecas",
    "totalsegmentermri": "totalsegmentermri",
    "totalsegmenter_mr": "totalsegmentermri",
    "totalsegmentor-mr": "totalsegmentermri",
    "totalsegmentor_mr": "totalsegmentermri",
    "totalsegmenter-mr": "totalsegmentermri",
    "jhu_stroke": "jhu_stroke",
    "jhustroke": "jhu_stroke",
}


class NativeObservationalFeatureRunner:
    """Extract one feature family from a native-spacing dataloader."""

    def __init__(
        self,
        extractor: Any,
        feature_type: str,
        batch_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.extractor = extractor
        self.feature_type = feature_type
        self.batch_transform = batch_transform
        self.config = getattr(extractor, "config", SimpleNamespace(device="cpu", dtype=torch.float32))

    def extract_from_dataloader(self, dataloader, show_progress: bool = True) -> Dict[str, np.ndarray]:
        all_features = []
        iterator = tqdm(dataloader, desc="Extracting observational features") if show_progress else dataloader

        for batch in iterator:
            if isinstance(batch, dict):
                x = batch["image"]
            elif isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            if self.batch_transform is not None:
                x = self.batch_transform(x)

            x = x.to(device=self.config.device, dtype=self.config.dtype, non_blocking=True)
            with torch.inference_mode():
                try:
                    features = self.extractor.extract(x, normalize=True)
                except TypeError:
                    features = self.extractor.extract(x)
            all_features.append(features[self.feature_type].detach().cpu().numpy())

        return {self.feature_type: np.concatenate(all_features, axis=0)}


def normalize_acquisition_dataset_name(dataset_name: str) -> str:
    """Return the configured Acquisition dataset key for a CLI dataset alias."""

    normalized = str(dataset_name).strip().lower().replace(" ", "_")
    if normalized not in ACQUISITION_DATASET_ALIASES:
        valid = ", ".join(sorted(set(ACQUISITION_DATASET_ALIASES.values())))
        raise ValueError(f"Unsupported Acquisition dataset '{dataset_name}'. Expected one of: {valid}")
    return ACQUISITION_DATASET_ALIASES[normalized]


def resolve_acquisition_manifest_path(
    dataset: str,
    manifest_path: Optional[Path],
    manifest_variant: str = DEFAULT_MANIFEST_VARIANT,
) -> Path:
    """Resolve a sampled Acquisition manifest from an explicit path or dataset key."""

    if manifest_path is not None:
        resolved = Path(manifest_path)
    else:
        dataset = normalize_acquisition_dataset_name(dataset)
        resolved = get_acquisition_manifest_path(dataset, "sampled", manifest_variant)
    if not resolved.exists():
        raise FileNotFoundError(_missing_manifest_message(resolved))
    return resolved


def _missing_manifest_message(manifest_path: Path) -> str:
    return (
        f"Manifest not found: {manifest_path}. "
        "Build the manifest first with the matching module manifest builder, "
        "or launch through reform3d/scripts/run_external_methods.sh so missing manifests are prepared before evaluation."
    )


def _feature_cache_path(features_dir: Path, checkpoint_name: str, feature_type: str, manifest_path: Path) -> Path:
    manifest_hash = hashlib.md5(manifest_path.read_bytes()).hexdigest()[:8]
    cache_dir = get_checkpoint_feature_dir(features_dir, checkpoint_name, feature_type)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"native_features_{manifest_hash}.npz"


def _load_manifest_payload(manifest_path: Path) -> Dict[str, Any]:
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_compute_cka(features_by_spacing: Dict[str, Any]) -> Dict[str, Dict[str, Optional[float]]]:
    spacing_names = list(features_by_spacing.keys())
    if not spacing_names:
        return {}

    n_samples = int(features_by_spacing[spacing_names[0]].shape[0])
    if n_samples < 2:
        logger.info("Skipping CKA: fewer than 2 matched samples are available")
        return {spacing_name: {other_name: None for other_name in spacing_names} for spacing_name in spacing_names}

    raw_cka = compute_cka_between_variants(features_by_spacing)
    sanitized: Dict[str, Dict[str, Optional[float]]] = {}
    for spacing_name, row in raw_cka.items():
        sanitized[spacing_name] = {}
        for other_name, value in row.items():
            if value is None or not torch.isfinite(torch.tensor(value)):
                sanitized[spacing_name][other_name] = None
            else:
                sanitized[spacing_name][other_name] = float(value)
    return sanitized


def _compute_semantic_targets(
    manifest_path: Path,
    file_paths: list[str],
    transfer_dataset: str,
    features_dir: Path,
    semantic_label_workers: int,
    semantic_cache_flush_every: int,
) -> tuple[Optional[np.ndarray], Optional[Dict[str, Any]], Optional[list[str]]]:
    try:
        semantic_labels_full, semantic_paths, semantic_meta = compute_semantic_task_labels(
            manifest_path,
            mode="multi_binary",
            filter_dataset=transfer_dataset,
            cache_dir=features_dir,
            num_workers=semantic_label_workers,
            cache_flush_every=semantic_cache_flush_every,
        )
    except Exception as exc:
        raise RuntimeError(f"Semantic label extraction failed for {manifest_path}: {exc}") from exc

    if semantic_labels_full.ndim != 2 or semantic_labels_full.shape[1] == 0:
        return None, semantic_meta, semantic_meta.get("organ_names", [])

    semantic_path_to_index = {path: idx for idx, path in enumerate(semantic_paths)}
    semantic_labels = np.zeros((len(file_paths), semantic_labels_full.shape[1]), dtype=np.int32)
    for sample_index, file_path in enumerate(file_paths):
        if file_path in semantic_path_to_index:
            semantic_labels[sample_index] = semantic_labels_full[semantic_path_to_index[file_path]]

    return semantic_labels, semantic_meta, semantic_meta.get("organ_names", [])


def run_native_observational_analysis(
    *,
    manifest_path: Path,
    checkpoint_name: str,
    feature_type: str,
    extractor: Any,
    batch_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    analysis_name: Optional[str] = None,
    batch_size: int = 4,
    num_workers: int = 2,
    cache_features: bool = True,
    warmup_cache: bool = True,
    semantic_label_workers: int = 2,
    semantic_cache_flush_every: int = 128,
    fast_seeded_evaluation: bool = False,
) -> Dict[str, Any]:
    """Run native-spacing Acquisition Robustness analyses for an external method."""

    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(_missing_manifest_message(manifest_path))

    dataset_name = analysis_name or get_dataset_name_from_manifest_path(manifest_path)
    manifest_variant = get_manifest_variant_from_manifest_path(manifest_path)
    output_paths = get_output_paths(dataset_name, manifest_variant)
    results_dir = output_paths["results"]
    figures_dir = output_paths["figures"]
    features_dir = output_paths["features"]
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    manifest_payload = _load_manifest_payload(manifest_path)
    volumes = manifest_payload["volumes"]
    file_paths = [volume["file_path"] for volume in volumes]
    bin_labels = np.asarray([volume["anisotropy_bin"] for volume in volumes])
    anisotropy_ratios = np.asarray([volume.get("anisotropy_ratio", 1.0) for volume in volumes], dtype=np.float32)
    dataset_counts: Dict[str, int] = {}
    for volume in volumes:
        dataset_counts[volume["dataset"]] = dataset_counts.get(volume["dataset"], 0) + 1
    transfer_dataset = max(dataset_counts.items(), key=lambda item: item[1])[0]
    binning_scheme = get_binning_scheme_from_manifest(manifest_payload)
    bin_names = get_bin_name_map(binning_scheme)

    crop_size = int(getattr(extractor.config, "crop_size", getattr(extractor.config, "input_size", 96)))
    if warmup_cache:
        warmup_observational_cache(
            manifest_path=manifest_path,
            crop_size=crop_size,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    feature_cache_path = _feature_cache_path(features_dir, checkpoint_name, feature_type, manifest_path)
    features = None
    if cache_features and feature_cache_path.exists():
        cached = np.load(feature_cache_path)
        if len(cached["features"]) == len(volumes):
            features = cached["features"]

    if features is None:
        dataloader = create_acquisition_dataloader(
            manifest_path=manifest_path,
            crop_size=crop_size,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=False,
            prefetch_factor=1 if num_workers > 0 else None,
        )
        feature_runner = NativeObservationalFeatureRunner(
            extractor=extractor,
            feature_type=feature_type,
            batch_transform=batch_transform,
        )
        features = feature_runner.extract_from_dataloader(dataloader, show_progress=True)[feature_type]
        if cache_features:
            np.savez_compressed(feature_cache_path, features=features, bin_labels=bin_labels, file_paths=file_paths)

    geometry = run_representation_geometry_analysis(
        features=features,
        bin_labels=bin_labels,
        checkpoint_name=checkpoint_name,
        feature_type=feature_type,
        compute_tsne=not fast_seeded_evaluation,
        output_dir=results_dir,
        figures_dir=figures_dir,
        bin_names=bin_names,
        binning_scheme=binning_scheme,
        dataset_name=dataset_name,
        manifest_variant=manifest_variant,
        compute_distribution_metrics=not fast_seeded_evaluation,
        compute_embedding_geometry_metrics=not fast_seeded_evaluation,
        compute_balanced_sensitivity=not fast_seeded_evaluation,
    )

    spacing_regression = spacing_regression_probe(
        features=features,
        anisotropy_ratios=anisotropy_ratios,
        bin_labels=bin_labels,
    )
    checkpoint_results_dir = get_checkpoint_feature_dir(results_dir, checkpoint_name, feature_type)
    checkpoint_results_dir.mkdir(parents=True, exist_ok=True)
    with (checkpoint_results_dir / "anisotropy_regression.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": checkpoint_name,
                "feature_type": feature_type,
                "task_name": "anisotropy_regression",
                "binning_scheme": binning_scheme,
                "anisotropy_regression": serialize_spacing_regression_results(spacing_regression),
                "spacing_regression": serialize_spacing_regression_results(spacing_regression),
            },
            handle,
            indent=2,
        )

    semantic_labels, semantic_meta, label_names = _compute_semantic_targets(
        manifest_path=manifest_path,
        file_paths=file_paths,
        transfer_dataset=transfer_dataset,
        features_dir=features_dir,
        semantic_label_workers=semantic_label_workers,
        semantic_cache_flush_every=semantic_cache_flush_every,
    )

    semantic_probe = None
    semantic_transfer = None
    if semantic_labels is not None and label_names:
        transfer_mask = np.asarray([volume["dataset"] == transfer_dataset for volume in volumes])
        if int(transfer_mask.sum()) > 50:
            semantic_probe = run_semantic_probing(
                features=features[transfer_mask],
                bin_labels=bin_labels[transfer_mask],
                semantic_labels=semantic_labels[transfer_mask],
                checkpoint_name=checkpoint_name,
                feature_type=feature_type,
                label_names=label_names,
                anisotropy_ratios=anisotropy_ratios[transfer_mask],
                output_dir=results_dir,
                analysis_name=dataset_name,
                manifest_variant=manifest_variant,
            )
            semantic_transfer = run_multilabel_cross_bin_transfer(
                features=features[transfer_mask],
                bin_labels=bin_labels[transfer_mask],
                semantic_labels=semantic_labels[transfer_mask],
                checkpoint_name=checkpoint_name,
                feature_type=feature_type,
                dataset=transfer_dataset,
                output_dir=results_dir,
                analysis_name=dataset_name,
                manifest_variant=manifest_variant,
            )

    write_results_catalog(results_dir, dataset_name, manifest_path)
    summary = {
        "checkpoint": checkpoint_name,
        "feature_type": feature_type,
        "dataset_name": dataset_name,
        "manifest_variant": manifest_variant,
        "transfer_dataset": transfer_dataset,
        "representation_geometry": {
            "mean_cross_cka": geometry.cka_stats["mean_cross_cka"],
            "mean_mmd_rbf": geometry.observational_distances["summary"]["mean_mmd_rbf"],
            "mean_sliced_wasserstein": geometry.observational_distances["summary"]["mean_sliced_wasserstein"],
        },
        "semantic_transfer": semantic_transfer.cross_bin_accuracy if semantic_transfer is not None else None,
        "semantic_probing": semantic_probe.mean_balanced_accuracy if semantic_probe is not None else None,
    }
    if semantic_meta is not None:
        summary["semantic_label_count"] = len(label_names or [])
    return summary


def run_controlled_perturbation_analysis(
    *,
    manifest_path: Path,
    checkpoint_path: Path,
    checkpoint_name: str,
    feature_type: str,
    extractor_factory: Callable[..., Any],
    model_name: str,
    crop_size: int,
    analysis_name: str = "default",
    output_dir: Optional[Path] = None,
    figures_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    n_source_volumes: Optional[int] = None,
    feature_batch_size: int = 8,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    extractor_kwargs: Optional[Dict[str, Any]] = None,
    result_metadata: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
    compute_matched_semantics: bool = True,
    compute_matched_semantic_probing: Optional[bool] = None,
    compute_matched_semantic_transfer: Optional[bool] = None,
    compute_drift_figure: bool = True,
) -> Dict[str, Any]:
    """Run controlled spacing perturbation for one external method."""

    manifest_path = Path(manifest_path)
    checkpoint_path = Path(checkpoint_path)
    if not manifest_path.exists():
        raise FileNotFoundError(_missing_manifest_message(manifest_path))

    manifest_payload = _load_manifest_payload(manifest_path)
    config = replace(get_controlled_perturbation_config_from_manifest(manifest_payload))
    if n_source_volumes is not None:
        config.n_source_volumes = int(n_source_volumes)

    resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    resolved_dtype = dtype
    if resolved_device == "cpu" and resolved_dtype in {torch.float16, torch.bfloat16}:
        logger.warning("Downgrading dtype to float32 on CPU")
        resolved_dtype = torch.float32

    dataset_name = analysis_name if analysis_name != "default" else get_dataset_name_from_manifest_path(manifest_path)
    manifest_variant = get_manifest_variant_from_manifest_path(manifest_path)
    output_dir, figures_dir = _resolve_controlled_dirs(
        output_dir,
        figures_dir,
        dataset_name,
        manifest_variant,
        checkpoint_name,
        feature_type,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    effective_cache_dir = cache_dir or get_cache_root(dataset_name, manifest_variant)

    logger.info("Running external controlled perturbation: %s", model_name)
    logger.info("Manifest: %s", manifest_path)
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Protocol: %s", config.protocol_name)
    logger.info("Target variants: %s", config.describe_targets())

    perturbed_volumes = prepare_perturbed_dataset(
        manifest_path=manifest_path,
        config=config,
        crop_size=int(crop_size),
        cache_dir=effective_cache_dir,
    )
    if not perturbed_volumes:
        raise RuntimeError("No perturbed volumes were prepared")

    extractor_kwargs = extractor_kwargs or {}
    extractor = extractor_factory(
        checkpoint_path=checkpoint_path,
        device=resolved_device,
        dtype=resolved_dtype,
        **extractor_kwargs,
    )
    features_by_spacing = extract_features_for_variants(
        perturbed_volumes=perturbed_volumes,
        extractor=extractor,
        feature_type=feature_type,
        batch_size=feature_batch_size,
    )

    per_sample_dists, mean_dists, std_dists = compute_pairwise_distances(features_by_spacing)
    original_name = config.reference_variant_name(perturbed_volumes[0].original_spacing)
    representation_drift: Dict[str, float] = {}
    representation_drift_std: Dict[str, float] = {}
    for spacing_name in features_by_spacing.keys():
        if spacing_name == original_name:
            continue
        pair = tuple(sorted((original_name, spacing_name)))
        if pair in mean_dists:
            representation_drift[spacing_name] = mean_dists[pair]
            representation_drift_std[spacing_name] = std_dists[pair]

    cka_matrix = _safe_compute_cka(features_by_spacing)
    bundle_path = save_controlled_embedding_bundle(
        output_dir=output_dir,
        features_by_spacing=features_by_spacing,
        perturbed_volumes=perturbed_volumes,
        config=config,
        checkpoint_name=checkpoint_name,
        feature_type=feature_type,
        dataset_name=dataset_name,
        manifest_variant=manifest_variant,
        reference_variant=original_name,
    )

    matched_semantic_probing = None
    matched_semantic_transfer = None
    semantic_metadata = None
    if compute_matched_semantic_probing is None:
        compute_matched_semantic_probing = compute_matched_semantics
    if compute_matched_semantic_transfer is None:
        compute_matched_semantic_transfer = compute_matched_semantics
    compute_any_matched_semantics = bool(compute_matched_semantic_probing or compute_matched_semantic_transfer)
    if compute_any_matched_semantics:
        semantic_labels, label_names, semantic_metadata, valid_indices = _load_matched_semantic_targets(
            manifest_path,
            perturbed_volumes,
            get_output_paths(dataset_name, manifest_variant)["results"],
        )
    else:
        logger.info("Skipping matched controlled semantic probing/transfer")
        semantic_labels, label_names, valid_indices = None, None, None
    if semantic_labels is not None and valid_indices is not None:
        semantic_features_by_spacing = {
            spacing_name: spacing_features[valid_indices]
            for spacing_name, spacing_features in features_by_spacing.items()
        }
        if compute_matched_semantic_probing:
            matched_semantic_probing = run_matched_semantic_probing(
                semantic_features_by_spacing,
                semantic_labels,
                label_names,
                checkpoint_name,
                feature_type,
                output_dir,
                random_state=random_state,
            )
        else:
            logger.info("Skipping matched controlled semantic probing")
        if compute_matched_semantic_transfer:
            matched_semantic_transfer = run_matched_semantic_transfer(
                semantic_features_by_spacing,
                semantic_labels,
                label_names,
                checkpoint_name,
                feature_type,
                output_dir,
                random_state=random_state,
            )
        else:
            logger.info("Skipping matched controlled semantic transfer")

    if compute_drift_figure:
        plot_drift_curve(
            representation_drift=representation_drift,
            representation_drift_std=representation_drift_std,
            checkpoint_name=checkpoint_name,
            feature_type=feature_type,
            output_dir=figures_dir,
        )
    else:
        logger.info("Skipping controlled drift figure")

    results_dict: Dict[str, Any] = {
        "checkpoint": checkpoint_name,
        "feature_type": feature_type,
        "task_name": "controlled_spacing_perturbation",
        "model_name": model_name,
        "model_checkpoint_path": str(checkpoint_path),
        "manifest_path": str(manifest_path),
        "perturbation_protocol": config.protocol_name,
        "target_variants": config.describe_targets(),
        "target_spacings": None,
        "target_z_spacings": list(config.target_z_spacings) if config.target_z_spacings is not None else None,
        "source_bin": config.source_bin,
        "reference_variant": original_name,
        "controlled_embedding_bundle": bundle_path.name,
        "n_source_volumes": len(perturbed_volumes),
        "representation_drift": representation_drift,
        "representation_drift_std": representation_drift_std,
        "mean_distance_per_pair": {f"{key[0]}___{key[1]}": value for key, value in mean_dists.items()},
        "std_distance_per_pair": {f"{key[0]}___{key[1]}": value for key, value in std_dists.items()},
        "cka_matrix": cka_matrix,
        "matched_semantic_probing": matched_semantic_probing,
        "matched_semantic_transfer": matched_semantic_transfer,
        "semantic_metadata": semantic_metadata,
        "drift_figure": "drift_curve.png" if compute_drift_figure else None,
        "drift_figure_skipped": not compute_drift_figure,
        "per_sample_distances": {
            str(idx): {f"{key[0]}___{key[1]}": value for key, value in dists.items()}
            for idx, dists in per_sample_dists.items()
        },
    }
    if result_metadata:
        results_dict.update(result_metadata)
    results_path = output_dir / "perturbation_robustness.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(results_dict, handle, indent=2)
    logger.info("Saved controlled perturbation results to %s", results_path)
    return results_dict
