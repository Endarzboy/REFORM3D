"""Stable sample identities and immutable evaluation selections."""
from __future__ import annotations

import json
from pathlib import Path

from reform3d.settings import artifact_lock, digest, write_json


def sample_identity(record, kind="samples"):
    if kind == "samples":
        value = record.get("sample_id")
    elif kind == "volumes":
        value = record.get("case_id") or record.get("source_case_id")
        if value is None and record.get("file_path"):
            value = Path(record["file_path"]).name.removesuffix(".gz").removesuffix(".nii")
    else:
        raise ValueError(f"Unknown manifest record kind: {kind}")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Every {kind} record requires a nonempty stable identity")
    return value


def canonical_records(records, kind="samples"):
    """Sort by stable identity, rejecting ambiguous cases instead of using row indices."""
    ordered, seen = [], set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Every manifest record must be an object")
        identity = sample_identity(record, kind)
        if identity in seen:
            raise ValueError(f"Duplicate {kind} identity: {identity}; provide unique explicit IDs")
        seen.add(identity)
        record = dict(record)
        if kind == "volumes":
            record["case_id"] = identity
        ordered.append(record)
    return sorted(ordered, key=lambda record: sample_identity(record, kind))


def plan_hash(plan):
    return digest(plan)


def load_plan(path):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate key in evaluation plan: {key}")
            result[key] = value
        return result

    plan = json.loads(Path(path).read_text(), object_pairs_hook=unique_keys)
    if not isinstance(plan, dict):
        raise ValueError("Evaluation plan must be a JSON object")
    plan_hash(plan)  # Reject nonfinite values as well as malformed JSON.
    return plan


def save_plan(path, plan):
    """Reuse an identical selection plan; never replace an existing experiment."""
    path = Path(path)
    plan_hash(plan)
    with artifact_lock(path.parent / ".plans.lock"):
        if path.exists():
            if load_plan(path) != plan:
                raise ValueError(f"Evaluation plan changed: {path}")
        else:
            write_json(path, plan)
    return path
