#!/usr/bin/env python
"""Build a deterministic Acquisition manifest subset from existing volumes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_acquisition_subset(
    *,
    input_manifest: Path,
    output_manifest: Path,
    dataset_name: str,
    anisotropy_bin: int,
    sample_count: int,
) -> dict[str, Any]:
    payload = json.loads(input_manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("volumes"), list):
        raise ValueError(f"Expected an object with a volumes list: {input_manifest}")

    candidates = [
        row
        for row in payload["volumes"]
        if int(row.get("anisotropy_bin", -1)) == anisotropy_bin
        and Path(row["file_path"]).exists()
    ]
    candidates.sort(key=lambda row: str(row["file_path"]))
    selected = candidates[:sample_count]
    if len(selected) < sample_count:
        raise ValueError(
            f"Requested {sample_count} existing volumes from anisotropy bin {anisotropy_bin}, "
            f"but found {len(selected)} in {input_manifest}"
        )

    output_payload = dict(payload)
    output_payload.update(
        {
            "dataset": dataset_name,
            "volumes": selected,
            "total_volumes": len(selected),
            "source_manifest": str(input_manifest.resolve()),
            "sampling_config": {
                "strategy": "deterministic_existing_volume_subset",
                "anisotropy_bin": int(anisotropy_bin),
                "sample_count": int(sample_count),
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
        description="Build a bounded Acquisition manifest subset from existing file paths"
    )
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--anisotropy-bin", type=int, required=True)
    parser.add_argument("--sample-count", type=int, default=10)
    args = parser.parse_args()

    if args.sample_count < 2:
        parser.error("--sample-count must be at least 2")

    payload = build_acquisition_subset(
        input_manifest=args.input_manifest.resolve(),
        output_manifest=args.output_manifest.resolve(),
        dataset_name=args.dataset_name,
        anisotropy_bin=args.anisotropy_bin,
        sample_count=args.sample_count,
    )
    print(
        json.dumps(
            {
                "output_manifest": str(args.output_manifest.resolve()),
                "dataset": payload["dataset"],
                "sampling_config": payload["sampling_config"],
                "total_volumes": payload["total_volumes"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
