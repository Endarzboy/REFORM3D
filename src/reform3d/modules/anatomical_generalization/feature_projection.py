"""Feature-space projection utilities for frozen anatomical transfer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECTION_SOURCE_CONTEXT = "context"
PROJECTION_SOURCE_SPACING = "spacing"
PROJECTION_SOURCE_MODALITY = "modality"
PROJECTION_SOURCES = (
    PROJECTION_SOURCE_CONTEXT,
    PROJECTION_SOURCE_SPACING,
    PROJECTION_SOURCE_MODALITY,
)


@dataclass(frozen=True)
class ContextFeaturePair:
    source_surface: str
    reference_surface: str
    source_path: Path
    reference_path: Path
    source_by_id: Mapping[str, np.ndarray]
    reference_by_id: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class SpacingDifferenceBundle:
    dataset: str
    manifest_variant: str
    bundle_path: Path
    reference_variant: str
    target_variants: tuple[str, ...]
    differences: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class FeatureProjectionSettings:
    sources: tuple[str, ...]
    components: int = 4
    strength: float = 1.0
    context_pair: ContextFeaturePair | None = None
    spacing_bundles: tuple[SpacingDifferenceBundle, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return bool(self.sources)

    def source_key(self) -> str:
        return projection_source_key(self.sources)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "adapter": "projection",
            "sources": list(self.sources),
            "source_key": self.source_key(),
            "components": int(self.components),
            "strength": float(self.strength),
            "metadata": dict(self.metadata),
        }
        if self.context_pair is not None:
            payload["context"] = {
                "source_surface": self.context_pair.source_surface,
                "reference_surface": self.context_pair.reference_surface,
                "source_path": str(self.context_pair.source_path),
                "reference_path": str(self.context_pair.reference_path),
                "source_count": len(self.context_pair.source_by_id),
                "reference_count": len(self.context_pair.reference_by_id),
            }
        if self.spacing_bundles:
            payload["spacing"] = [
                {
                    "dataset": bundle.dataset,
                    "manifest_variant": bundle.manifest_variant,
                    "bundle_path": str(bundle.bundle_path),
                    "reference_variant": bundle.reference_variant,
                    "target_variants": list(bundle.target_variants),
                    "n_difference_blocks": len(bundle.differences),
                    "n_difference_rows": int(sum(matrix.shape[0] for matrix in bundle.differences)),
                }
                for bundle in self.spacing_bundles
            ]
        return payload


@dataclass(frozen=True)
class ProjectionFit:
    basis: np.ndarray
    strength: float
    diagnostics: Mapping[str, Any]


def normalize_projection_sources(sources: Sequence[str] | None) -> tuple[str, ...]:
    if not sources:
        return ()
    normalized: list[str] = []
    for source in sources:
        value = str(source).strip().lower().replace("-", "_")
        if value not in PROJECTION_SOURCES:
            raise ValueError(f"Unknown projection source: {source}")
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def projection_source_key(sources: Sequence[str]) -> str:
    normalized = normalize_projection_sources(sources)
    return "none" if not normalized else "+".join(normalized)


def load_feature_bundle(npz_path: Path, fallback_sample_ids: Sequence[str] | None = None) -> dict[str, np.ndarray]:
    npz_path = Path(npz_path)
    payload = np.load(npz_path, allow_pickle=True)
    if "features" not in payload:
        raise ValueError(f"Feature bundle does not contain a features array: {npz_path}")
    features = np.asarray(payload["features"], dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"Expected features to have shape [n_samples, dim], got {features.shape}: {npz_path}")
    if "sample_ids" in payload:
        sample_ids = [str(item) for item in payload["sample_ids"].tolist()]
    elif fallback_sample_ids is not None:
        sample_ids = [str(item) for item in fallback_sample_ids]
    else:
        raise ValueError(f"Feature bundle does not contain sample_ids: {npz_path}")
    if len(sample_ids) != features.shape[0]:
        raise ValueError(f"sample_ids length does not match features length: {npz_path}")
    return {sample_id: features[index] for index, sample_id in enumerate(sample_ids)}


def load_controlled_spacing_bundle(
    bundle_path: Path,
    dataset: str,
    manifest_variant: str,
    reference_variant: str | None = None,
    target_variants: Sequence[str] | None = None,
) -> SpacingDifferenceBundle:
    bundle_path = Path(bundle_path)
    payload = np.load(bundle_path, allow_pickle=True)
    variant_key = "variant_names" if "variant_names" in payload else "spacing_names"
    if variant_key not in payload:
        raise ValueError(f"Controlled bundle does not contain variant names: {bundle_path}")
    variants = [str(item) for item in payload[variant_key].tolist()]
    if not variants:
        raise ValueError(f"Controlled bundle has no variants: {bundle_path}")

    if reference_variant is None:
        if "reference_variant_name" in payload:
            reference_variant = str(payload["reference_variant_name"].item())
        elif "reference_spacing_name" in payload:
            reference_variant = str(payload["reference_spacing_name"].item())
        else:
            reference_variant = variants[0]
    if reference_variant not in variants:
        raise ValueError(f"Reference variant {reference_variant} is not present in {bundle_path}")

    reference_key = f"features__{reference_variant}"
    if reference_key not in payload:
        raise ValueError(f"Controlled bundle missing reference features: {reference_key} in {bundle_path}")
    reference_features = np.asarray(payload[reference_key], dtype=np.float32)
    if reference_features.ndim != 2:
        raise ValueError(f"Expected reference features to have shape [n_samples, dim]: {bundle_path}")

    requested_targets = [str(item) for item in target_variants] if target_variants else variants
    spacing_mask = None
    if "spacing_variant_mask" in payload:
        candidate = np.asarray(payload["spacing_variant_mask"], dtype=bool)
        if candidate.shape[0] == len(variants):
            spacing_mask = candidate

    differences: list[np.ndarray] = []
    retained_targets: list[str] = []
    for index, variant in enumerate(variants):
        if variant == reference_variant or variant not in requested_targets:
            continue
        if spacing_mask is not None and not bool(spacing_mask[index]):
            continue
        feature_key = f"features__{variant}"
        if feature_key not in payload:
            continue
        target_features = np.asarray(payload[feature_key], dtype=np.float32)
        if target_features.shape != reference_features.shape:
            raise ValueError(
                f"Controlled variant shape mismatch for {variant}: {target_features.shape} vs {reference_features.shape}"
            )
        differences.append(target_features - reference_features)
        retained_targets.append(variant)

    if not differences:
        raise ValueError(f"No controlled spacing differences available in {bundle_path}")
    return SpacingDifferenceBundle(
        dataset=str(dataset),
        manifest_variant=str(manifest_variant),
        bundle_path=bundle_path,
        reference_variant=str(reference_variant),
        target_variants=tuple(retained_targets),
        differences=tuple(differences),
    )


def _scaler_scale(scaler: Any, feature_dim: int) -> np.ndarray:
    scale = getattr(scaler, "scale_", None)
    if scale is None:
        return np.ones(feature_dim, dtype=np.float32)
    scale_array = np.asarray(scale, dtype=np.float32)
    if scale_array.shape[0] != feature_dim:
        raise ValueError(f"Scaler dimension {scale_array.shape[0]} does not match feature dimension {feature_dim}")
    return np.where(np.abs(scale_array) < 1e-8, 1.0, scale_array)


def _finite_nonzero_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D difference matrix, got shape {matrix.shape}")
    finite_mask = np.isfinite(matrix).all(axis=1)
    matrix = matrix[finite_mask]
    if matrix.size == 0:
        return matrix.reshape(0, matrix.shape[-1] if matrix.ndim == 2 else 0)
    norm_mask = np.linalg.norm(matrix, axis=1) > 1e-8
    return matrix[norm_mask]


def _fit_source_basis(matrix: np.ndarray, components: int) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = _finite_nonzero_rows(matrix)
    if matrix.shape[0] == 0:
        return np.zeros((matrix.shape[1], 0), dtype=np.float32), {
            "n_rows": 0,
            "n_components": 0,
            "explained_energy_ratio": 0.0,
        }
    component_count = max(1, min(int(components), matrix.shape[0], matrix.shape[1]))
    _, singular_values, vh = np.linalg.svd(matrix.astype(np.float64, copy=False), full_matrices=False)
    basis = vh[:component_count].T.astype(np.float32, copy=False)
    energy = np.square(singular_values)
    total_energy = float(np.sum(energy))
    explained = 0.0 if total_energy <= 0 else float(np.sum(energy[:component_count]) / total_energy)
    return basis, {
        "n_rows": int(matrix.shape[0]),
        "n_components": int(component_count),
        "explained_energy_ratio": explained,
        "singular_values": [float(value) for value in singular_values[:component_count]],
    }


def _orthonormalize(columns: Sequence[np.ndarray], feature_dim: int) -> np.ndarray:
    matrices = [matrix for matrix in columns if matrix.size and matrix.shape[1] > 0]
    if not matrices:
        return np.zeros((feature_dim, 0), dtype=np.float32)
    combined = np.concatenate(matrices, axis=1).astype(np.float64, copy=False)
    q, r = np.linalg.qr(combined, mode="reduced")
    keep = np.abs(np.diag(r)) > 1e-8
    if keep.shape[0] != q.shape[1]:
        keep = np.ones(q.shape[1], dtype=bool)
    return q[:, keep].astype(np.float32, copy=False)


def _context_difference_matrix(
    settings: FeatureProjectionSettings,
    support_records: Sequence[Mapping[str, Any]],
    feature_dim: int,
    scale: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    if settings.context_pair is None:
        raise ValueError("Context projection requires paired context feature bundles")
    rows: list[np.ndarray] = []
    missing_ids: list[str] = []
    for record in support_records:
        sample_id = str(record["sample_id"])
        source = settings.context_pair.source_by_id.get(sample_id)
        reference = settings.context_pair.reference_by_id.get(sample_id)
        if source is None or reference is None:
            missing_ids.append(sample_id)
            continue
        source = np.asarray(source, dtype=np.float32)
        reference = np.asarray(reference, dtype=np.float32)
        if source.shape[0] != feature_dim or reference.shape[0] != feature_dim:
            raise ValueError(f"Context feature dimension mismatch for sample {sample_id}")
        rows.append((source - reference) / scale)
    if not rows:
        raise ValueError("No support samples have paired context features")
    return np.stack(rows, axis=0), {
        "paired_support_samples": len(rows),
        "missing_support_samples": len(missing_ids),
        "source_surface": settings.context_pair.source_surface,
        "reference_surface": settings.context_pair.reference_surface,
    }


def _spacing_difference_matrix(
    settings: FeatureProjectionSettings,
    feature_dim: int,
    scale: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not settings.spacing_bundles:
        raise ValueError("Spacing projection requires at least one controlled embedding bundle")
    rows: list[np.ndarray] = []
    bundle_payloads: list[dict[str, Any]] = []
    for bundle in settings.spacing_bundles:
        bundle_rows = []
        for matrix in bundle.differences:
            if matrix.ndim != 2 or matrix.shape[1] != feature_dim:
                raise ValueError(
                    f"Spacing feature dimension mismatch in {bundle.bundle_path}: expected {feature_dim}, got {matrix.shape}"
                )
            bundle_rows.append(matrix / scale)
        if bundle_rows:
            stacked = np.concatenate(bundle_rows, axis=0)
            rows.append(stacked)
            bundle_payloads.append(
                {
                    "dataset": bundle.dataset,
                    "manifest_variant": bundle.manifest_variant,
                    "reference_variant": bundle.reference_variant,
                    "target_variants": list(bundle.target_variants),
                    "n_rows": int(stacked.shape[0]),
                }
            )
    if not rows:
        raise ValueError("No spacing difference rows are available")
    return np.concatenate(rows, axis=0), {"bundles": bundle_payloads}


def _modality_difference_matrix(
    support_records: Sequence[Mapping[str, Any]],
    support_features: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    grouped: dict[str, dict[str, list[int]]] = {}
    for index, record in enumerate(support_records):
        organ = str(record.get("primary_organ") or record.get("organ_name"))
        modality = str(record.get("modality")).lower()
        grouped.setdefault(organ, {}).setdefault(modality, []).append(index)

    rows: list[np.ndarray] = []
    organ_payloads: list[dict[str, Any]] = []
    for organ, modality_map in sorted(grouped.items()):
        modalities = sorted(modality_map)
        if len(modalities) < 2:
            continue
        means = {
            modality: np.mean(support_features[indices], axis=0)
            for modality, indices in modality_map.items()
            if indices
        }
        if len(means) < 2:
            continue
        if "ct" in means and "mr" in means:
            rows.append(means["ct"] - means["mr"])
            organ_payloads.append({"organ": organ, "modalities": ["ct", "mr"], "n_rows": 1})
            continue
        organ_center = np.mean(np.stack(list(means.values()), axis=0), axis=0)
        for modality, mean in sorted(means.items()):
            rows.append(mean - organ_center)
        organ_payloads.append({"organ": organ, "modalities": sorted(means), "n_rows": len(means)})

    if not rows:
        raise ValueError("No support organs have at least two modalities for modality projection")
    return np.stack(rows, axis=0), {"organs": organ_payloads, "n_organs": len(organ_payloads)}


def fit_feature_projection(
    settings: FeatureProjectionSettings,
    support_records: Sequence[Mapping[str, Any]],
    support_features: np.ndarray,
    scaler: Any,
) -> ProjectionFit:
    if not settings.enabled:
        return ProjectionFit(
            basis=np.zeros((support_features.shape[1], 0), dtype=np.float32),
            strength=0.0,
            diagnostics={"enabled": False},
        )
    if settings.components <= 0:
        raise ValueError("Projection components must be positive when projection is enabled")
    if settings.strength < 0:
        raise ValueError("Projection strength must be non-negative")

    support_features = np.asarray(support_features, dtype=np.float32)
    if support_features.ndim != 2:
        raise ValueError(f"Expected support features to have shape [n_samples, dim], got {support_features.shape}")
    feature_dim = int(support_features.shape[1])
    scale = _scaler_scale(scaler, feature_dim)

    source_bases: list[np.ndarray] = []
    source_diagnostics: dict[str, Any] = {}
    for source in settings.sources:
        if source == PROJECTION_SOURCE_CONTEXT:
            matrix, diagnostics = _context_difference_matrix(settings, support_records, feature_dim, scale)
        elif source == PROJECTION_SOURCE_SPACING:
            matrix, diagnostics = _spacing_difference_matrix(settings, feature_dim, scale)
        elif source == PROJECTION_SOURCE_MODALITY:
            matrix, diagnostics = _modality_difference_matrix(support_records, support_features)
        else:
            raise ValueError(f"Unsupported projection source: {source}")
        basis, basis_diagnostics = _fit_source_basis(matrix, settings.components)
        diagnostics.update(basis_diagnostics)
        source_bases.append(basis)
        source_diagnostics[source] = diagnostics

    basis = _orthonormalize(source_bases, feature_dim)
    return ProjectionFit(
        basis=basis,
        strength=float(settings.strength),
        diagnostics={
            "enabled": True,
            "sources": list(settings.sources),
            "source_key": settings.source_key(),
            "components_per_source": int(settings.components),
            "strength": float(settings.strength),
            "feature_dim": feature_dim,
            "combined_components": int(basis.shape[1]),
            "sources_detail": source_diagnostics,
        },
    )


def apply_feature_projection(features: np.ndarray, fit: ProjectionFit) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if fit.basis.size == 0 or fit.strength == 0:
        return features.copy()
    if features.ndim != 2:
        raise ValueError(f"Expected features to have shape [n_samples, dim], got {features.shape}")
    if features.shape[1] != fit.basis.shape[0]:
        raise ValueError(f"Feature dimension {features.shape[1]} does not match projection basis {fit.basis.shape[0]}")
    projection = (features @ fit.basis) @ fit.basis.T
    return features - float(fit.strength) * projection
