"""Path normalization helpers for Cross-Modal dataset records."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from reform3d.core.paths import DOWNSTREAM_DATASET_ROOT, EXTERNAL_DATA_ROOT, REFORM3D_DATASET_ROOT


TOTALSEGMENTER_CT_CURRENT_ROOTS = (
    DOWNSTREAM_DATASET_ROOT / "segmentation" / "totalsegmenter_ct",
    EXTERNAL_DATA_ROOT / "downstreamTasks" / "segmentation" / "totalsegmenter_ct",
)
TOTALSEGMENTER_CT_LEGACY_ROOTS = (
    REFORM3D_DATASET_ROOT / "datasets" / "ct" / "totalsegmenter_dataset",
    EXTERNAL_DATA_ROOT / "3DFM_Benchmark" / "datasets" / "ct" / "totalsegmenter_dataset",
)


def _resolve_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _first_existing(paths: tuple[Path, ...]) -> Path:
    for path in paths:
        resolved = _resolve_path(path)
        if resolved.exists():
            return resolved
    return _resolve_path(paths[0])


def get_totalsegmenter_ct_root() -> Path:
    return _first_existing(TOTALSEGMENTER_CT_CURRENT_ROOTS)


def resolve_totalsegmenter_ct_path(value: Any) -> str | None:
    if value is None:
        return None

    original_value = str(value)
    path = _resolve_path(Path(original_value))
    target_root = get_totalsegmenter_ct_root()
    for legacy_root in TOTALSEGMENTER_CT_LEGACY_ROOTS:
        try:
            relative_path = path.relative_to(_resolve_path(legacy_root))
        except ValueError:
            continue
        return str(_resolve_path(target_root / relative_path))
    return original_value


def normalize_totalsegmenter_ct_record(record: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    for key in ("file_path", "mask_path", "label_path"):
        if key in normalized and normalized[key]:
            normalized[key] = resolve_totalsegmenter_ct_path(normalized[key])
    return normalized
