#!/usr/bin/env python
"""Build a deterministic balanced subset of an organ-level manifest."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


def _organ_name(sample: dict[str, Any]) -> str | None:
    organ = sample.get("primary_organ") or sample.get("organ_name")
    if organ:
        return str(organ)
    organs = sample.get("organs") or []
    return str(organs[0]) if organs else None


def _case_id(sample: dict[str, Any]) -> str:
    for key in ("source_case_id", "patient_id", "case_id", "sample_id"):
        value = sample.get(key)
        if value:
            return str(value)
    raise ValueError(f"Sample does not contain a case identifier: {sample}")


def _unique_case_samples(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_cases: set[str] = set()
    for sample in sorted(samples, key=lambda row: (_case_id(row), str(row.get("sample_id", "")))):
        case_id = _case_id(sample)
        if case_id in seen_cases:
            continue
        seen_cases.add(case_id)
        selected.append(sample)
    return selected


def build_balanced_subset(
    *,
    input_manifest: Path,
    output_manifest: Path,
    dataset_name: str,
    required_modalities: Sequence[str],
    organ_count: int,
    samples_per_modality: int,
) -> dict[str, Any]:
    payload = json.loads(input_manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError(f"Expected an object with a samples list: {input_manifest}")

    modalities = tuple(str(value).strip().lower() for value in required_modalities)
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for sample in payload["samples"]:
        organ = _organ_name(sample)
        modality = str(sample.get("modality", "")).strip().lower()
        if organ and modality in modalities:
            grouped[organ][modality].append(sample)

    eligible_organs: list[str] = []
    unique_samples: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for organ in sorted(grouped):
        complete = True
        for modality in modalities:
            rows = _unique_case_samples(grouped[organ].get(modality, []))
            unique_samples[(organ, modality)] = rows
            if len(rows) < samples_per_modality:
                complete = False
        if complete:
            eligible_organs.append(organ)

    selected_organs = eligible_organs[:organ_count]
    if len(selected_organs) < organ_count:
        raise ValueError(
            f"Requested {organ_count} organs with {samples_per_modality} unique cases per modality, "
            f"but found {len(selected_organs)} in {input_manifest}"
        )

    selected_samples = [
        sample
        for organ in selected_organs
        for modality in modalities
        for sample in unique_samples[(organ, modality)][:samples_per_modality]
    ]

    output_payload = dict(payload)
    output_payload.update(
        {
            "dataset": dataset_name,
            "retained_organs": selected_organs,
            "samples": selected_samples,
            "source_manifest": str(input_manifest.resolve()),
            "sampling": {
                "applied": True,
                "strategy": "balanced_unique_cases_per_organ_modality",
                "organ_count": int(organ_count),
                "required_modalities": list(modalities),
                "samples_per_modality": int(samples_per_modality),
                "total_samples": len(selected_samples),
            },
        }
    )
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(output_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a balanced organ/modality manifest subset with unique source cases"
    )
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--required-modalities", nargs="+", default=["ct", "mr"])
    parser.add_argument("--organ-count", type=int, default=2)
    parser.add_argument("--samples-per-modality", type=int, default=5)
    args = parser.parse_args()

    if args.organ_count < 2:
        parser.error("--organ-count must be at least 2")
    if args.samples_per_modality < 3:
        parser.error("--samples-per-modality must be at least 3")

    payload = build_balanced_subset(
        input_manifest=args.input_manifest.resolve(),
        output_manifest=args.output_manifest.resolve(),
        dataset_name=args.dataset_name,
        required_modalities=args.required_modalities,
        organ_count=args.organ_count,
        samples_per_modality=args.samples_per_modality,
    )
    print(
        json.dumps(
            {
                "output_manifest": str(args.output_manifest.resolve()),
                "dataset": payload["dataset"],
                "retained_organs": payload["retained_organs"],
                "sampling": payload["sampling"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
