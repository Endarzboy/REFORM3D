"""Case-level leakage and lightweight deduplication audits for REFORM-3D.

The audit is intentionally conservative: it flags exact normalized case IDs,
duplicate paths, split/site reuse, pretraining-exposure annotations, and
optional lightweight file fingerprints when local files are available.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_identifier(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or None


def _manifest_records(payload: Any, manifest_label: str) -> list[Dict[str, Any]]:
    if isinstance(payload, dict):
        raw_records = payload.get("samples") or payload.get("volumes") or []
        dataset = payload.get("dataset") or manifest_label
    elif isinstance(payload, list):
        raw_records = payload
        dataset = manifest_label
    else:
        return []

    records: list[Dict[str, Any]] = []
    for index, record in enumerate(raw_records):
        if not isinstance(record, dict):
            continue
        raw = record.get("raw_sample") if isinstance(record.get("raw_sample"), dict) else record
        file_path = raw.get("file_path") or raw.get("image_path") or raw.get("path")
        source_case_id = (
            raw.get("source_case_id")
            or raw.get("patient_id")
            or raw.get("subject_id")
            or raw.get("case_id")
            or raw.get("sample_id")
            or raw.get("id")
        )
        records.append(
            {
                "manifest": manifest_label,
                "dataset": raw.get("dataset") or dataset,
                "sample_id": str(raw.get("sample_id") or raw.get("id") or f"record_{index:06d}"),
                "source_case_id": source_case_id,
                "normalized_case_id": normalize_identifier(source_case_id),
                "patient_id": raw.get("patient_id") or raw.get("subject_id"),
                "file_path": str(file_path) if file_path else None,
                "normalized_file_path": str(Path(file_path).expanduser()) if file_path else None,
                "split": raw.get("split") or raw.get("fold") or "unspecified",
                "site": raw.get("site") or raw.get("center") or raw.get("institution") or "unspecified",
                "modality": raw.get("modality") or "unknown",
                "pretraining_exposure": raw.get("pretraining_exposure") or "pretraining_exposure_unknown",
            }
        )
    return records


def _group_duplicates(records: Sequence[Dict[str, Any]], key: str) -> list[Dict[str, Any]]:
    grouped: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        value = record.get(key)
        if value:
            grouped[str(value)].append(record)
    duplicates = []
    for value, group in sorted(grouped.items()):
        manifests = sorted({str(item.get("manifest")) for item in group})
        datasets = sorted({str(item.get("dataset")) for item in group})
        if len(group) > 1 and (len(manifests) > 1 or len(datasets) > 1):
            duplicates.append(
                {
                    key: value,
                    "n_records": len(group),
                    "manifests": manifests,
                    "datasets": datasets,
                    "sample_ids": [str(item.get("sample_id")) for item in group],
                }
            )
    return duplicates


def fingerprint_existing_file(path: str | Path, max_bytes: int = 1_048_576) -> str | None:
    """Return a lightweight fingerprint for an existing local file."""
    candidate = Path(path).expanduser()
    if not candidate.exists() or not candidate.is_file():
        return None
    stat = candidate.stat()
    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode())
    with candidate.open("rb") as handle:
        digest.update(handle.read(max_bytes))
        if stat.st_size > max_bytes:
            handle.seek(max(stat.st_size - max_bytes, 0))
            digest.update(handle.read(max_bytes))
    return digest.hexdigest()[:24]


def compute_leakage_audit(
    manifests: Mapping[str, Any] | Sequence[tuple[str, Any]],
    *,
    enable_fingerprints: bool = False,
) -> Dict[str, Any]:
    items = manifests.items() if isinstance(manifests, Mapping) else manifests
    records: list[Dict[str, Any]] = []
    for label, payload in items:
        records.extend(_manifest_records(payload, str(label)))

    exposure_counter = Counter(str(record["pretraining_exposure"]) for record in records)
    site_split_counter = Counter((str(record["site"]), str(record["split"])) for record in records)
    case_duplicates = _group_duplicates(records, "normalized_case_id")
    path_duplicates = _group_duplicates(records, "normalized_file_path")

    fingerprint_duplicates: list[Dict[str, Any]] = []
    if enable_fingerprints:
        for record in records:
            file_path = record.get("file_path")
            record["fingerprint"] = fingerprint_existing_file(file_path) if file_path else None
        fingerprint_duplicates = _group_duplicates(records, "fingerprint")

    return {
        "artifact": "leakage_audit",
        "n_records": len(records),
        "n_manifests": len({record["manifest"] for record in records}),
        "exact_duplicate_case_ids": case_duplicates,
        "exact_duplicate_paths": path_duplicates,
        "near_duplicate_fingerprints": fingerprint_duplicates,
        "pretraining_exposure_summary": dict(sorted(exposure_counter.items())),
        "site_split_summary": {
            f"{site}|{split}": count
            for (site, split), count in sorted(site_split_counter.items())
        },
        "has_potential_leakage": bool(case_duplicates or path_duplicates or fingerprint_duplicates),
        "fingerprints_enabled": bool(enable_fingerprints),
    }


def write_leakage_audit(
    manifest_paths: Iterable[Path],
    output_path: Path,
    *,
    enable_fingerprints: bool = False,
) -> Dict[str, Any]:
    manifests = {}
    for path in manifest_paths:
        path = Path(path)
        if path.exists():
            manifests[str(path)] = json.loads(path.read_text(encoding="utf-8"))
    payload = compute_leakage_audit(manifests, enable_fingerprints=enable_fingerprints)
    payload["manifest_paths"] = sorted(manifests)
    _write_json(output_path, payload)
    return payload


__all__ = [
    "compute_leakage_audit",
    "fingerprint_existing_file",
    "normalize_identifier",
    "write_leakage_audit",
]
