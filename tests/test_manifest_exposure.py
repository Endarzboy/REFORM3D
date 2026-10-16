"""Dataset identity does not establish a checkpoint's pretraining exposure."""

import json

import pytest

from reform3d.modules.anatomical_generalization.build_manifest import build_anatomical_manifest
from reform3d.modules.anatomical_generalization.data_loader import (
    load_anatomical_manifest,
    summarize_anatomical_manifest,
)
from reform3d.modules.anatomical_generalization.evaluation_pipeline import _build_anatomical_protocol
from reform3d.modules.cross_modal_alignment import build_manifests as cross_modal


@pytest.mark.parametrize("dataset", ["kits23", "totalsegmenter_ct", "mmwhs", "amos22", "chaos", "custom"])
@pytest.mark.parametrize("field", ["source_dataset", "dataset", "image_path", "file_path"])
def test_dataset_names_and_paths_do_not_assign_exposure(tmp_path, dataset, field):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"samples": [{
        "sample_id": "case0:liver", "modality": "ct", "organs": ["liver"],
        field: f"/data/{dataset}/case0.nii.gz",
    }]}))
    samples = load_anatomical_manifest(path)
    assert "pretraining_exposure" not in samples[0]
    assert "pretraining_exposure" not in summarize_anatomical_manifest(samples)


@pytest.mark.parametrize("modalities", [("ct",), ("ct", "mr")])
def test_supplied_annotations_do_not_change_cohort_eligibility_or_protocol(tmp_path, modalities):
    samples = [
        {"sample_id": f"{modality}:{case}:{organ}", "source_case_id": str(case),
         "modality": modality, "organs": [organ], "source_dataset": "kits23",
         "image_path": f"/kits23/{modality}_{case}.nii.gz"}
        for modality in modalities
        for case in range(2)
        for organ in ["liver", "spleen"]
    ]
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"samples": samples}))
    output = build_anatomical_manifest(
        source_manifest=source, analysis_name="mmwhs_ct_mr", required_modalities=modalities,
        min_holdout_organs=2, min_samples_per_modality=2, output_dir=tmp_path / "plain",
    )
    plain = json.loads((output / "manifest_full.json").read_text())
    assert plain["retained_organs"] == ["liver", "spleen"]
    assert all("pretraining_exposure" not in sample for sample in plain["samples"])

    samples[0]["pretraining_exposure"] = "user_verified_annotation"
    source.write_text(json.dumps({"samples": samples}))
    output = build_anatomical_manifest(
        source_manifest=source, analysis_name="mmwhs_ct_mr", required_modalities=modalities,
        min_holdout_organs=2, min_samples_per_modality=2, output_dir=tmp_path / "annotated",
    )
    annotated = json.loads((output / "manifest_full.json").read_text())
    assert annotated["samples"][0].pop("pretraining_exposure") == "user_verified_annotation"
    assert annotated == plain
    summary = json.loads((output / "manifest_meta.json").read_text())["summary"]
    assert summary["pretraining_exposure"] == {"user_verified_annotation": 1}

    protocol = _build_anatomical_protocol(modalities, plain["retained_organs"])
    scope = "within_modality" if len(modalities) == 1 else "cross_modality"
    assert protocol["surface_scope"] == scope
    assert protocol[f"{scope}_surface_stratum"] == plain["anatomical_protocol"][f"{scope}_surface_stratum"]
    assert "pretraining" not in json.dumps(protocol)

    with pytest.raises(ValueError, match="eligible target organs"):
        build_anatomical_manifest(
            source_manifest=source, analysis_name="mmwhs_ct_mr", required_modalities=modalities,
            min_holdout_organs=2, min_samples_per_modality=3, output_dir=tmp_path / "insufficient",
        )


@pytest.mark.parametrize("dataset", ["mmwhs", "amos22", "chaos"])
def test_cross_modal_builders_do_not_label_datasets_as_unseen(tmp_path, monkeypatch, dataset):
    organ = "myocardium" if dataset == "mmwhs" else "liver"
    cases = [
        {"modality": modality, "case_id": f"{modality}_0", "patient_id": "0",
         "file_path": f"/{dataset}/{modality}_0.nii.gz", "label_path": f"/{dataset}/{modality}_mask.nii.gz",
         "shape": [4, 4, 4], "spacing": [1, 1, 1], "present_organs": [organ],
         "organ_counts": {organ: 64}, "split": "train"}
        for modality in ["ct", "mr"]
    ]
    if dataset == "mmwhs":
        monkeypatch.setattr(cross_modal, "_build_mmwhs_case_records", lambda root, modality, min_voxels: (
            [case for case in cases if case["modality"] == modality], {organ: 1},
        ))
        full, sampled, metadata = cross_modal.build_mmwhs_manifest(tmp_path, 1, 1)
    elif dataset == "amos22":
        monkeypatch.setattr(cross_modal, "_build_amos22_case_records", lambda *args: (
            cases, {organ: {"ct": 1, "mr": 1}},
        ))
        full, sampled, metadata = cross_modal.build_amos22_manifest(
            tmp_path, splits=["train"], min_voxels=1, min_samples_per_modality=1,
        )
    else:
        monkeypatch.setattr(cross_modal, "_discover_chaos_case_records", lambda *args: (
            cases, {"ct": 1, "mr": 1},
        ))
        full, sampled, metadata = cross_modal.build_chaos_manifest(tmp_path, 1, 1)
    assert len(full["samples"]) == len(sampled["samples"]) == 2
    assert metadata["retained_organs"] == [organ]
    assert {sample["modality"] for sample in full["samples"]} == {"ct", "mr"}
    assert "pretraining_exposure" not in json.dumps((full, sampled, metadata))
