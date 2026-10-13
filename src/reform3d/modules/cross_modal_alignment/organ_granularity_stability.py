"""Organ-granularity stability using Task 2 baseline feature bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reform3d.core.checkpoints import resolve_checkpoint_config

from .config import CROSS_MODAL_OUTPUTS, get_cross_modal_manifest_path, normalize_cross_modal_dataset_name
from .feature_extractor import _load_manifest_samples, _sample_signature
from .protocol_sampling import (
    build_anatomical_plan,
    build_retrieval_plan,
    evaluate_anatomical_plans,
    evaluate_retrieval_plan,
    validate_anatomical_plan,
    validate_retrieval_plan,
    validate_sample_records,
)
from .region_definition_stability import (
    BASELINE_SURFACE,
    DEFAULT_CHECKPOINTS,
    DEFAULT_FEATURE_TYPES,
    DEFAULT_SEEDS,
    DEFAULT_SUPPORT_PER_MODALITY,
    _feature_bundle_path,
    _feature_summary_path,
    _payload_sha256,
    _sample_records,
    _sha256_file,
    _surface_signature,
    validate_feature_bundle,
)


TAXONOMY_SCHEMA_VERSION = 1
SUPPORTED_DATASETS = ("amos22_ct_mr", "mmwhs_ct_mr")
DEFAULT_TAXONOMIES = ("fine_organ", "organ_family")
DEFAULT_REQUIRED_MODALITIES = ("ct", "mr")


@dataclass(frozen=True)
class TaxonomySpec:
    name: str
    mapping: Mapping[str, str]
    expected_class_count: int


DATASET_FAMILY_MAPPINGS = {
    "amos22_ct_mr": {
        "kidney_left": "kidney",
        "kidney_right": "kidney",
        "adrenal_gland_left": "adrenal_gland",
        "adrenal_gland_right": "adrenal_gland",
    },
    "mmwhs_ct_mr": {
        "left_atrium": "atrium",
        "right_atrium": "atrium",
        "left_ventricle": "ventricle",
        "right_ventricle": "ventricle",
    },
}
EXPECTED_CLASS_COUNTS = {
    "amos22_ct_mr": {"fine_organ": 13, "organ_family": 11},
    "mmwhs_ct_mr": {"fine_organ": 7, "organ_family": 5},
}


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _result_root(dataset: str) -> Path:
    return (
        CROSS_MODAL_OUTPUTS
        / "protocol_stability"
        / "organ_granularity"
        / dataset
        / "core"
    )


def _taxonomy_dir(dataset: str, taxonomy: str) -> Path:
    return _result_root(dataset) / "taxonomies" / taxonomy


def _taxonomy_contract_path(dataset: str, taxonomy: str) -> Path:
    return _taxonomy_dir(dataset, taxonomy) / "taxonomy_contract.json"


def _taxonomy_records_path(dataset: str, taxonomy: str) -> Path:
    return _taxonomy_dir(dataset, taxonomy) / "records.json"


def _plan_path(dataset: str, taxonomy: str, plan_type: str, seed: int) -> Path:
    return (
        _result_root(dataset)
        / "sampling_plans"
        / taxonomy
        / f"{plan_type}_seed_{int(seed)}.json"
    )


def _evaluation_dir(
    dataset: str,
    taxonomy: str,
    checkpoint: str,
    feature_type: str,
) -> Path:
    return (
        _result_root(dataset)
        / "evaluations"
        / taxonomy
        / checkpoint
        / feature_type
    )


def _taxonomy_spec(dataset: str, taxonomy: str) -> TaxonomySpec:
    dataset = normalize_cross_modal_dataset_name(dataset)
    if taxonomy not in DEFAULT_TAXONOMIES:
        raise ValueError(f"Unknown taxonomy: {taxonomy}")
    mapping = (
        {} if taxonomy == "fine_organ" else DATASET_FAMILY_MAPPINGS[dataset]
    )
    return TaxonomySpec(
        name=taxonomy,
        mapping=mapping,
        expected_class_count=EXPECTED_CLASS_COUNTS[dataset][taxonomy],
    )


def _representative_index(
    dataset: str,
    mapped_organ: str,
    modality: str,
    source_case_id: str,
    candidate_count: int,
) -> int:
    token = "|".join(
        (dataset, mapped_organ, modality, source_case_id, "organ_family_v1")
    )
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % candidate_count


def _record_signature(records: Sequence[Mapping[str, Any]]) -> str:
    normalized = [
        {
            "sample_id": str(record["sample_id"]),
            "source_case_id": str(record["source_case_id"]),
            "modality": str(record["modality"]),
            "fine_organ": str(record["fine_organ"]),
            "primary_organ": str(record["primary_organ"]),
        }
        for record in records
    ]
    normalized.sort(key=lambda record: record["sample_id"])
    return _payload_sha256({"records": normalized})


def build_taxonomy_records(
    dataset: str,
    canonical_records: Sequence[Mapping[str, Any]],
    taxonomy: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Map labels in memory and select one record for each merged stratum."""
    dataset = normalize_cross_modal_dataset_name(dataset)
    spec = _taxonomy_spec(dataset, taxonomy)
    mapped_records: list[dict[str, Any]] = []
    merged_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for original in canonical_records:
        fine_organ = str(original["primary_organ"])
        mapped_organ = str(spec.mapping.get(fine_organ, fine_organ))
        record = {
            **dict(original),
            "fine_organ": fine_organ,
            "primary_organ": mapped_organ,
        }
        if taxonomy == "organ_family" and mapped_organ != fine_organ:
            merged_groups[
                (
                    str(record["source_case_id"]),
                    mapped_organ,
                    str(record["modality"]),
                )
            ].append(record)
        else:
            mapped_records.append(record)

    selected_merged_ids: list[str] = []
    for group_key, candidates in sorted(merged_groups.items()):
        ordered = sorted(candidates, key=lambda record: str(record["sample_id"]))
        candidate_index = _representative_index(
            dataset,
            group_key[1],
            group_key[2],
            group_key[0],
            len(ordered),
        )
        selected = ordered[candidate_index]
        mapped_records.append(selected)
        selected_merged_ids.append(str(selected["sample_id"]))

    canonical_order = {
        str(record["sample_id"]): index
        for index, record in enumerate(canonical_records)
    }
    mapped_records.sort(key=lambda record: canonical_order[str(record["sample_id"])])
    validate_sample_records(mapped_records)
    classes = sorted({str(record["primary_organ"]) for record in mapped_records})
    if len(classes) != spec.expected_class_count:
        raise ValueError(
            f"{dataset}/{taxonomy} has {len(classes)} classes; expected "
            f"{spec.expected_class_count}"
        )
    support = _taxonomy_support(mapped_records, classes)
    _validate_taxonomy_support(support, classes)
    contract = {
        "schema_version": TAXONOMY_SCHEMA_VERSION,
        "dataset": dataset,
        "taxonomy": taxonomy,
        "mapping": dict(sorted(spec.mapping.items())),
        "fine_classes": sorted(
            {str(record["primary_organ"]) for record in canonical_records}
        ),
        "classes": classes,
        "expected_class_count": spec.expected_class_count,
        "n_canonical_records": len(canonical_records),
        "n_taxonomy_records": len(mapped_records),
        "n_merged_strata": len(merged_groups),
        "merged_representative_policy": (
            "sha256(dataset, source_case_id, mapped_organ, modality) over "
            "sorted candidate sample IDs"
            if taxonomy == "organ_family"
            else "not_applicable"
        ),
        "selected_merged_sample_ids": selected_merged_ids,
        "record_signature": _record_signature(mapped_records),
        "modality_case_support": support,
    }
    return mapped_records, contract


def _taxonomy_support(
    records: Sequence[Mapping[str, Any]],
    classes: Sequence[str],
) -> dict[str, dict[str, dict[str, int]]]:
    support: dict[str, dict[str, dict[str, int]]] = {
        organ: {
            modality: {"n_samples": 0, "n_cases": 0}
            for modality in DEFAULT_REQUIRED_MODALITIES
        }
        for organ in classes
    }
    cases: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        organ = str(record["primary_organ"])
        modality = str(record["modality"])
        if organ not in support or modality not in DEFAULT_REQUIRED_MODALITIES:
            continue
        support[organ][modality]["n_samples"] += 1
        cases[(organ, modality)].add(str(record["source_case_id"]))
    for (organ, modality), values in cases.items():
        support[organ][modality]["n_cases"] = len(values)
    return support


def _validate_taxonomy_support(
    support: Mapping[str, Mapping[str, Mapping[str, int]]],
    classes: Sequence[str],
) -> None:
    minimum_cases = DEFAULT_SUPPORT_PER_MODALITY + 1
    for organ in classes:
        for modality in DEFAULT_REQUIRED_MODALITIES:
            values = support[organ][modality]
            if int(values["n_cases"]) < minimum_cases:
                raise ValueError(
                    f"Insufficient {modality} case support for {organ}: "
                    f"{values['n_cases']} < {minimum_cases}"
                )


def _source_feature_provenance(
    dataset: str,
    checkpoint_name: str,
    samples: Sequence[Any],
    manifest_path: Path,
) -> dict[str, Any]:
    checkpoint = resolve_checkpoint_config(checkpoint_name)
    checkpoint_sha256 = _sha256_file(checkpoint.path)
    surface_signature = _surface_signature(
        BASELINE_SURFACE,
        manifest_sha256=_sha256_file(manifest_path),
        sample_signature=_sample_signature(samples),
    )
    bundle_path = _feature_bundle_path(
        dataset,
        BASELINE_SURFACE.name,
        checkpoint.checkpoint_key,
    )
    summary_path = _feature_summary_path(
        dataset,
        BASELINE_SURFACE.name,
        checkpoint.checkpoint_key,
    )
    summary = validate_feature_bundle(
        bundle_path,
        summary_path,
        expected_sample_ids=[sample.sample_id for sample in samples],
        surface_signature=surface_signature,
        checkpoint_sha256=checkpoint_sha256,
    )
    return {
        "checkpoint": checkpoint.checkpoint_key,
        "bundle_path": str(bundle_path),
        "bundle_sha256": _sha256_file(bundle_path),
        "summary_path": str(summary_path),
        "summary_sha256": _sha256_file(summary_path),
        "source_surface": BASELINE_SURFACE.name,
        "source_surface_signature": surface_signature,
        "checkpoint_sha256": checkpoint_sha256,
        "feature_dimensions": dict(summary["feature_dimensions"]),
    }


def _load_source_features(provenance: Mapping[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    with np.load(str(provenance["bundle_path"]), allow_pickle=False) as bundle:
        sample_ids = [str(value) for value in bundle["sample_ids"].tolist()]
        return {
            feature_type: {
                sample_id: np.asarray(bundle[feature_type][index], dtype=np.float32)
                for index, sample_id in enumerate(sample_ids)
            }
            for feature_type in DEFAULT_FEATURE_TYPES
        }


def _load_taxonomy_records(dataset: str, taxonomy: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = normalize_cross_modal_dataset_name(dataset)
    spec = _taxonomy_spec(dataset, taxonomy)
    contract = json.loads(
        _taxonomy_contract_path(dataset, taxonomy).read_text(encoding="utf-8")
    )
    records_payload = json.loads(
        _taxonomy_records_path(dataset, taxonomy).read_text(encoding="utf-8")
    )
    records = list(records_payload["records"])
    if str(contract.get("dataset")) != dataset or str(contract.get("taxonomy")) != taxonomy:
        raise ValueError(f"Taxonomy contract identity mismatch for {dataset}/{taxonomy}")
    if dict(contract.get("mapping", {})) != dict(spec.mapping):
        raise ValueError(f"Taxonomy mapping mismatch for {dataset}/{taxonomy}")
    if _record_signature(records) != str(contract["record_signature"]):
        raise ValueError(f"Taxonomy record signature mismatch for {dataset}/{taxonomy}")
    classes = sorted({str(record["primary_organ"]) for record in records})
    if len(classes) != int(contract["expected_class_count"]) or len(classes) != spec.expected_class_count:
        raise ValueError(f"Taxonomy class count mismatch for {dataset}/{taxonomy}")
    validate_sample_records(records)
    _validate_taxonomy_support(_taxonomy_support(records, classes), classes)
    return records, contract


def prepare_taxonomy_artifacts(
    dataset: str,
    *,
    taxonomies: Sequence[str] = DEFAULT_TAXONOMIES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> list[Path]:
    dataset = normalize_cross_modal_dataset_name(dataset)
    manifest_path = get_cross_modal_manifest_path(dataset, "sampled", "core")
    samples = _load_manifest_samples(manifest_path)
    canonical_records = _sample_records(samples)
    manifest_sha256 = _sha256_file(manifest_path)
    paths: list[Path] = []
    for taxonomy in taxonomies:
        records, contract = build_taxonomy_records(
            dataset,
            canonical_records,
            taxonomy,
        )
        contract = {
            **contract,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "canonical_sample_signature": _sample_signature(samples),
            "required_modalities": list(DEFAULT_REQUIRED_MODALITIES),
            "support_per_modality": DEFAULT_SUPPORT_PER_MODALITY,
        }
        contract_path = _taxonomy_contract_path(dataset, taxonomy)
        records_path = _taxonomy_records_path(dataset, taxonomy)
        _atomic_write_json(contract_path, contract)
        _atomic_write_json(
            records_path,
            {
                "schema_version": TAXONOMY_SCHEMA_VERSION,
                "dataset": dataset,
                "taxonomy": taxonomy,
                "record_signature": contract["record_signature"],
                "records": records,
            },
        )
        paths.extend((contract_path, records_path))

        classes = list(contract["classes"])
        for seed in seeds:
            retrieval = build_retrieval_plan(
                records,
                seed=int(seed),
                supported_organs=classes,
            )
            anatomical = build_anatomical_plan(
                records,
                seed=int(seed),
                supported_organs=classes,
                required_modalities=DEFAULT_REQUIRED_MODALITIES,
                support_per_modality=DEFAULT_SUPPORT_PER_MODALITY,
            )
            for plan_type, plan in (
                ("retrieval", retrieval),
                ("anatomical", anatomical),
            ):
                output_path = _plan_path(dataset, taxonomy, plan_type, int(seed))
                _atomic_write_json(
                    output_path,
                    {
                        "schema_version": TAXONOMY_SCHEMA_VERSION,
                        "dataset": dataset,
                        "taxonomy": taxonomy,
                        "record_signature": contract["record_signature"],
                        **plan,
                    },
                )
                paths.append(output_path)
    return paths


def _load_plans(
    dataset: str,
    taxonomy: str,
    records: Sequence[Mapping[str, Any]],
    record_signature: str,
    seeds: Sequence[int],
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    retrieval_plans: dict[int, dict[str, Any]] = {}
    anatomical_plans: list[dict[str, Any]] = []
    for seed in seeds:
        retrieval = json.loads(
            _plan_path(dataset, taxonomy, "retrieval", int(seed)).read_text(
                encoding="utf-8"
            )
        )
        anatomical = json.loads(
            _plan_path(dataset, taxonomy, "anatomical", int(seed)).read_text(
                encoding="utf-8"
            )
        )
        if str(retrieval.get("record_signature")) != record_signature:
            raise ValueError(f"Retrieval plan record signature mismatch for {taxonomy}")
        if str(anatomical.get("record_signature")) != record_signature:
            raise ValueError(f"Anatomical plan record signature mismatch for {taxonomy}")
        validate_retrieval_plan(retrieval, records)
        validate_anatomical_plan(anatomical, records)
        retrieval_plans[int(seed)] = retrieval
        anatomical_plans.append(anatomical)
    return retrieval_plans, anatomical_plans


def _validate_evaluation_artifact(
    path: Path,
    *,
    taxonomy: str,
    record_signature: str,
    feature_type: str,
    source_provenance: Mapping[str, Any],
    retrieval_plan: Mapping[str, Any] | None = None,
    anatomical_plans: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "ok":
        raise ValueError(f"Incomplete evaluation artifact: {path}")
    if str(payload.get("taxonomy")) != taxonomy:
        raise ValueError(f"Taxonomy mismatch: {path}")
    if str(payload.get("record_signature")) != record_signature:
        raise ValueError(f"Record signature mismatch: {path}")
    if str(payload.get("feature_type")) != feature_type:
        raise ValueError(f"Feature type mismatch: {path}")
    observed_provenance = payload.get("source_feature_provenance")
    provenance_keys = (
        "bundle_sha256",
        "summary_sha256",
        "source_surface_signature",
        "checkpoint_sha256",
    )
    if not isinstance(observed_provenance, Mapping) or any(
        str(observed_provenance.get(key)) != str(source_provenance.get(key))
        for key in provenance_keys
    ):
        raise ValueError(f"Source feature provenance mismatch: {path}")
    if retrieval_plan is not None:
        if int(payload.get("evaluation_seed", -1)) != int(
            retrieval_plan["evaluation_seed"]
        ):
            raise ValueError(f"Retrieval seed mismatch: {path}")
        for direction_name, plan_direction in retrieval_plan["directions"].items():
            result_direction = payload.get(direction_name, {})
            if list(result_direction.get("query_sample_ids", [])) != list(
                plan_direction["query_sample_ids"]
            ):
                raise ValueError(f"Retrieval query plan mismatch: {path}")
            if list(result_direction.get("target_sample_ids", [])) != list(
                plan_direction["target_sample_ids"]
            ):
                raise ValueError(f"Retrieval target plan mismatch: {path}")
    if anatomical_plans is not None:
        expected_seeds = [int(plan["evaluation_seed"]) for plan in anatomical_plans]
        if list(payload.get("evaluation_seeds", [])) != expected_seeds:
            raise ValueError(f"Anatomical seed plan mismatch: {path}")
        if int(payload.get("support_per_modality", -1)) != int(
            anatomical_plans[0]["support_per_modality"]
        ):
            raise ValueError(f"Anatomical support budget mismatch: {path}")
    return payload


def evaluate_taxonomy_artifacts(
    dataset: str,
    *,
    checkpoints: Sequence[str] = DEFAULT_CHECKPOINTS,
    taxonomies: Sequence[str] = DEFAULT_TAXONOMIES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    overwrite: bool = False,
) -> list[Path]:
    dataset = normalize_cross_modal_dataset_name(dataset)
    manifest_path = get_cross_modal_manifest_path(dataset, "sampled", "core")
    samples = _load_manifest_samples(manifest_path)
    output_paths: list[Path] = []
    for taxonomy in taxonomies:
        records, contract = _load_taxonomy_records(dataset, taxonomy)
        retrieval_plans, anatomical_plans = _load_plans(
            dataset,
            taxonomy,
            records,
            str(contract["record_signature"]),
            seeds,
        )
        for checkpoint_name in checkpoints:
            provenance = _source_feature_provenance(
                dataset,
                checkpoint_name,
                samples,
                manifest_path,
            )
            features = _load_source_features(provenance)
            checkpoint = str(provenance["checkpoint"])
            for feature_type in DEFAULT_FEATURE_TYPES:
                for seed, plan in retrieval_plans.items():
                    output_path = (
                        _evaluation_dir(dataset, taxonomy, checkpoint, feature_type)
                        / f"retrieval_seed_{seed}.json"
                    )
                    if output_path.exists() and not overwrite:
                        _validate_evaluation_artifact(
                            output_path,
                            taxonomy=taxonomy,
                            record_signature=str(contract["record_signature"]),
                            feature_type=feature_type,
                            source_provenance=provenance,
                            retrieval_plan=plan,
                        )
                    else:
                        _atomic_write_json(
                            output_path,
                            {
                                "schema_version": TAXONOMY_SCHEMA_VERSION,
                                "dataset": dataset,
                                "taxonomy": taxonomy,
                                "record_signature": contract["record_signature"],
                                "source_feature_provenance": provenance,
                                "feature_type": feature_type,
                                **evaluate_retrieval_plan(
                                    features[feature_type],
                                    records,
                                    plan,
                                ),
                            },
                        )
                        _validate_evaluation_artifact(
                            output_path,
                            taxonomy=taxonomy,
                            record_signature=str(contract["record_signature"]),
                            feature_type=feature_type,
                            source_provenance=provenance,
                            retrieval_plan=plan,
                        )
                    output_paths.append(output_path)
                anatomical_path = (
                    _evaluation_dir(dataset, taxonomy, checkpoint, feature_type)
                    / "anatomical_transfer.json"
                )
                if anatomical_path.exists() and not overwrite:
                    _validate_evaluation_artifact(
                        anatomical_path,
                        taxonomy=taxonomy,
                        record_signature=str(contract["record_signature"]),
                        feature_type=feature_type,
                        source_provenance=provenance,
                        anatomical_plans=anatomical_plans,
                    )
                else:
                    _atomic_write_json(
                        anatomical_path,
                        {
                            "schema_version": TAXONOMY_SCHEMA_VERSION,
                            "dataset": dataset,
                            "taxonomy": taxonomy,
                            "record_signature": contract["record_signature"],
                            "source_feature_provenance": provenance,
                            "feature_type": feature_type,
                            **evaluate_anatomical_plans(
                                features[feature_type],
                                records,
                                anatomical_plans,
                                ),
                            },
                        )
                    _validate_evaluation_artifact(
                        anatomical_path,
                        taxonomy=taxonomy,
                        record_signature=str(contract["record_signature"]),
                        feature_type=feature_type,
                        source_provenance=provenance,
                        anatomical_plans=anatomical_plans,
                    )
                output_paths.append(anatomical_path)
    return output_paths


def write_task_contract(dataset: str) -> Path:
    dataset = normalize_cross_modal_dataset_name(dataset)
    manifest_path = get_cross_modal_manifest_path(dataset, "sampled", "core")
    output_path = _result_root(dataset) / "contract.json"
    _atomic_write_json(
        output_path,
        {
            "schema_version": TAXONOMY_SCHEMA_VERSION,
            "dataset": dataset,
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "source_region_surface": BASELINE_SURFACE.name,
            "source_feature_types": list(DEFAULT_FEATURE_TYPES),
            "checkpoints": list(DEFAULT_CHECKPOINTS),
            "seeds": list(DEFAULT_SEEDS),
            "taxonomies": list(DEFAULT_TAXONOMIES),
            "support_per_modality": DEFAULT_SUPPORT_PER_MODALITY,
            "no_crop_extraction": True,
            "no_encoder_inference": True,
        },
    )
    return output_path


def prepare_granularity_task(
    dataset: str,
    *,
    taxonomies: Sequence[str] = DEFAULT_TAXONOMIES,
) -> None:
    write_task_contract(dataset)
    prepare_taxonomy_artifacts(dataset, taxonomies=taxonomies)


def validate_granularity_task(
    dataset: str,
    *,
    checkpoints: Sequence[str] = DEFAULT_CHECKPOINTS,
    taxonomies: Sequence[str] = DEFAULT_TAXONOMIES,
    require_evaluations: bool = False,
) -> dict[str, Any]:
    dataset = normalize_cross_modal_dataset_name(dataset)
    manifest_path = get_cross_modal_manifest_path(dataset, "sampled", "core")
    samples = _load_manifest_samples(manifest_path)
    checked_evaluations = 0
    for taxonomy in taxonomies:
        records, contract = _load_taxonomy_records(dataset, taxonomy)
        if len({str(record["primary_organ"]) for record in records}) != int(
            contract["expected_class_count"]
        ):
            raise ValueError(f"Invalid class count for {dataset}/{taxonomy}")
        _load_plans(
            dataset,
            taxonomy,
            records,
            str(contract["record_signature"]),
            DEFAULT_SEEDS,
        )
        for checkpoint_name in checkpoints:
            provenance = _source_feature_provenance(
                dataset,
                checkpoint_name,
                samples,
                manifest_path,
            )
            checkpoint = str(provenance["checkpoint"])
            for feature_type in DEFAULT_FEATURE_TYPES:
                evaluation_dir = _evaluation_dir(
                    dataset,
                    taxonomy,
                    checkpoint,
                    feature_type,
                )
                expected_paths = [
                    evaluation_dir / f"retrieval_seed_{seed}.json"
                    for seed in DEFAULT_SEEDS
                ] + [evaluation_dir / "anatomical_transfer.json"]
                if all(path.exists() for path in expected_paths):
                    for path in expected_paths:
                        if path.name.startswith("retrieval_seed_"):
                            seed = int(path.stem.removeprefix("retrieval_seed_"))
                            _validate_evaluation_artifact(
                                path,
                                taxonomy=taxonomy,
                                record_signature=str(contract["record_signature"]),
                                feature_type=feature_type,
                                source_provenance=provenance,
                                retrieval_plan=_load_plans(
                                    dataset,
                                    taxonomy,
                                    records,
                                    str(contract["record_signature"]),
                                    (seed,),
                                )[0][seed],
                            )
                        else:
                            _validate_evaluation_artifact(
                                path,
                                taxonomy=taxonomy,
                                record_signature=str(contract["record_signature"]),
                                feature_type=feature_type,
                                source_provenance=provenance,
                                anatomical_plans=_load_plans(
                                    dataset,
                                    taxonomy,
                                    records,
                                    str(contract["record_signature"]),
                                    DEFAULT_SEEDS,
                                )[1],
                            )
                    checked_evaluations += len(expected_paths)
                elif require_evaluations:
                    missing = [str(path) for path in expected_paths if not path.exists()]
                    raise FileNotFoundError(
                        f"Incomplete evaluation artifacts: {missing[:3]}"
                    )
    return {
        "dataset": dataset,
        "n_canonical_samples": len(samples),
        "taxonomies": list(taxonomies),
        "n_checked_evaluation_artifacts": checked_evaluations,
    }


def _parse_values(values: Sequence[str] | None, defaults: Sequence[str]) -> tuple[str, ...]:
    if not values:
        return tuple(defaults)
    parsed: list[str] = []
    for value in values:
        parsed.extend(item.strip() for item in value.split(",") if item.strip())
    return tuple(parsed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "evaluate", "run", "validate"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument(
            "--dataset",
            required=True,
            choices=SUPPORTED_DATASETS,
        )
        command_parser.add_argument(
            "--checkpoints",
            nargs="+",
            default=list(DEFAULT_CHECKPOINTS),
        )
        command_parser.add_argument(
            "--taxonomies",
            nargs="+",
            default=list(DEFAULT_TAXONOMIES),
            choices=DEFAULT_TAXONOMIES,
        )
        command_parser.add_argument("--overwrite", action="store_true")
        if command == "validate":
            command_parser.add_argument(
                "--require-evaluations",
                action="store_true",
            )
    args = parser.parse_args()
    checkpoints = _parse_values(args.checkpoints, DEFAULT_CHECKPOINTS)
    taxonomies = _parse_values(args.taxonomies, DEFAULT_TAXONOMIES)

    if args.command == "prepare":
        prepare_granularity_task(args.dataset, taxonomies=taxonomies)
    elif args.command == "evaluate":
        evaluate_taxonomy_artifacts(
            args.dataset,
            checkpoints=checkpoints,
            taxonomies=taxonomies,
            overwrite=args.overwrite,
        )
    elif args.command == "run":
        prepare_granularity_task(args.dataset, taxonomies=taxonomies)
        evaluate_taxonomy_artifacts(
            args.dataset,
            checkpoints=checkpoints,
            taxonomies=taxonomies,
            overwrite=args.overwrite,
        )
    elif args.command == "validate":
        print(
            json.dumps(
                validate_granularity_task(
                    args.dataset,
                    checkpoints=checkpoints,
                    taxonomies=taxonomies,
                    require_evaluations=args.require_evaluations,
                ),
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
