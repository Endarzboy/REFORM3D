"""Dependency boundaries and crop-cache integrity checks."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_acquisition_imports_do_not_inject_package_paths():
    import reform3d

    package_parent = str(Path(reform3d.__file__).resolve().parent.parent)
    code = """
import importlib, sys
sys.path.insert(0, sys.argv[1])
package_paths = lambda: tuple(p for p in sys.path if p == sys.argv[1] or p.startswith(sys.argv[1] + '/'))
before = package_paths()
for name in ('feature_extractor', 'semantic_analysis', 'representation_geometry_analysis', 'build_perturbation_cache', 'evaluation_pipeline', 'perturbation_robustness_analysis'):
    importlib.import_module('reform3d.modules.acquisition_robustness.' + name)
assert package_paths() == before, (before, package_paths())
"""
    subprocess.run([sys.executable, "-I", "-c", code, package_parent], check=True, capture_output=True, text=True, timeout=60)


@pytest.fixture
def crop_sample(tmp_path):
    from reform3d.modules.cross_modal_alignment.feature_extractor import OrganSample

    image, mask = tmp_path / "image.nii.gz", tmp_path / "mask.nii.gz"
    image.write_bytes(b"synthetic image")
    mask.write_bytes(b"synthetic mask")
    return OrganSample("sample", image, mask, "ct", "liver", "case", (1.0, 1.0, 2.0), mask_label_value=1)


def _crop_index(sample, manifest):
    from reform3d.modules.cross_modal_alignment.feature_extractor import _sample_signature

    return {
        "manifest_path": str(manifest), "crop_size": 8, "bbox_margin": 5,
        "min_mask_voxels": 1, "outside_organ_mask": "none",
        "n_requested_samples": 1, "n_cached_samples": 1,
        "sample_signature": _sample_signature([sample]),
    }


def _crop_matches(index, sample, manifest):
    from reform3d.modules.cross_modal_alignment.feature_extractor import _crop_cache_matches_request

    return _crop_cache_matches_request(index, [sample], manifest, 8, 5, 1, "none")


@pytest.mark.parametrize("field,value", [
    ("mask_label_value", 2), ("mask_label_range", (1, 4)),
    ("mask_loader", "png_mask_stack"), ("spacing", (2.0, 1.0, 1.0)),
    ("modality", "mr"), ("source_case_id", "different-case"),
])
def test_changed_crop_record_cannot_reuse_same_id_cache(tmp_path, crop_sample, field, value):
    from dataclasses import replace

    manifest = tmp_path / "manifest.json"
    index = _crop_index(crop_sample, manifest)
    assert _crop_matches(index, crop_sample, manifest) == (True, None)
    changed = replace(crop_sample, **{field: value})
    assert changed.sample_id == crop_sample.sample_id
    assert _crop_matches(index, changed, manifest) == (False, "sample_signature")


@pytest.mark.parametrize("field", ["file_path", "mask_path"])
def test_changed_crop_file_path_cannot_reuse_same_id_cache(tmp_path, crop_sample, field):
    from dataclasses import replace

    manifest = tmp_path / "manifest.json"
    index = _crop_index(crop_sample, manifest)
    replacement = tmp_path / "replacement.nii.gz"
    replacement.write_bytes(b"replacement")
    assert _crop_matches(index, replace(crop_sample, **{field: replacement}), manifest) == (False, "sample_signature")


@pytest.mark.parametrize("field", ["file_path", "mask_path"])
def test_changed_crop_file_stat_cannot_reuse_same_id_cache(tmp_path, crop_sample, field):
    manifest = tmp_path / "manifest.json"
    index = _crop_index(crop_sample, manifest)
    path = getattr(crop_sample, field)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert path.stat().st_size == stat.st_size
    assert _crop_matches(index, crop_sample, manifest) == (False, "sample_signature")


def test_crop_signature_uses_recursive_stats_without_reading_raw_files(tmp_path, crop_sample, monkeypatch):
    from dataclasses import replace
    from reform3d.modules.cross_modal_alignment.feature_extractor import _sample_signature

    mask_dir = tmp_path / "png-masks"
    mask_dir.mkdir()
    mask = mask_dir / "slice-001.png"
    mask.write_bytes(b"mask slice")
    sample = replace(crop_sample, mask_path=mask_dir, mask_loader="png_mask_stack")
    second = replace(sample, sample_id="other-organ", mask_label_value=2)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", lambda *args, **kwargs: pytest.fail("Raw files must not be read when fingerprinting"))
        before = _sample_signature([sample, second])
        assert before == _sample_signature([second, sample])
    stat = mask.stat()
    os.utime(mask, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert before != _sample_signature([sample, second])


@pytest.mark.parametrize("legacy_signature", [None, "legacy-sha1-sample-ids-only"])
def test_legacy_crop_index_is_not_reused(tmp_path, crop_sample, legacy_signature):
    manifest = tmp_path / "manifest.json"
    index = _crop_index(crop_sample, manifest)
    if legacy_signature is None:
        index.pop("sample_signature")
    else:
        index["sample_signature"] = legacy_signature
    assert _crop_matches(index, crop_sample, manifest) == (False, "sample_signature")


def test_orphan_case_crop_respects_record_and_context_identity(tmp_path, crop_sample, monkeypatch):
    from dataclasses import replace
    import torch
    from reform3d.modules.cross_modal_alignment import feature_extractor as crops

    calls = []
    def load_case(**kwargs):
        calls.append(kwargs)
        return {"sample_ids": ["sample"], "crops": torch.full((1, 1, 8, 8, 8), len(calls), dtype=torch.float16)}, []
    monkeypatch.setattr(crops, "_load_case_crop_payload", load_case)
    cache_dir = tmp_path / "cache"
    path = crops._case_cache_path(cache_dir, "case", crop_sample.file_path)
    path.parent.mkdir(parents=True)
    torch.save({"sample_ids": ["sample"], "crops": torch.zeros((1, 1, 8, 8, 8))}, path)
    def build(sample, margin=5):
        return crops._build_case_crop_cache((("case", str(sample.file_path)), [sample], 8, margin, 1, "none", str(cache_dir), False))
    build(crop_sample)
    build(crop_sample)
    assert len(calls) == 1
    build(replace(crop_sample, mask_label_value=2))
    assert len(calls) == 2
    build(replace(crop_sample, mask_label_value=2), margin=0)
    assert len(calls) == 3


def test_concurrent_crop_builder_cannot_return_incompatible_index(tmp_path, crop_sample, monkeypatch):
    from reform3d.modules.cross_modal_alignment import feature_extractor as crops

    manifest = tmp_path / "manifest.json"
    cache = tmp_path / "cache"
    cache.mkdir()
    index = _crop_index(crop_sample, manifest)
    index["sample_signature"] = "other-cohort"
    index_path = cache / crops.CROP_CACHE_INDEX_NAME
    index_path.write_text(json.dumps(index))
    (cache / crops.CROP_CACHE_LOCK_NAME).write_text("0")
    monkeypatch.setattr(crops, "_wait_for_crop_cache", lambda path: index_path)
    with pytest.raises(ValueError, match="Concurrent crop cache is incompatible"):
        crops.ensure_crop_cache([crop_sample], manifest, 8, 5, 1, "none", cache, 1)


def test_wait_for_crop_cache_does_not_return_previous_index(tmp_path, monkeypatch):
    from reform3d.modules.cross_modal_alignment import feature_extractor as crops

    index = tmp_path / crops.CROP_CACHE_INDEX_NAME
    lock = tmp_path / crops.CROP_CACHE_LOCK_NAME
    index.write_text('{"phase": "previous"}')
    lock.write_text("0")
    waits = []
    def publish(_seconds):
        waits.append(True)
        index.write_text('{"phase": "completed"}')
        lock.unlink()
    monkeypatch.setattr(crops.time, "sleep", publish)
    completed = crops._wait_for_crop_cache(tmp_path, timeout_seconds=1)
    assert waits == [True]
    assert json.loads(completed.read_text()) == {"phase": "completed"}
