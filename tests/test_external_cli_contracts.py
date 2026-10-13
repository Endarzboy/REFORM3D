"""External entrypoints must reach evaluation with their declared arguments."""

import importlib
import inspect
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from reform3d.external_methods import _acquisition, _cross_modal, _primary_feature_cli


@pytest.mark.parametrize("method_name", ["sam_med3d", "mim", "gl_mae"])
@pytest.mark.parametrize("stage", ["controlled", "cross_modal_extract"])
def test_primary_feature_cli_arguments_bind_to_runtime(tmp_path, monkeypatch, method_name, stage):
    method = importlib.import_module(f"reform3d.external_methods.{method_name}.runtime").METHOD
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{}')
    observed = {}
    if stage == "controlled":
        signature = inspect.signature(_acquisition.run_controlled_perturbation_analysis)
        function_name = "run_controlled_perturbation_analysis"
        argv = ["entrypoint", "--manifest-path", str(manifest), "--batch-size", "1"]
        run = _primary_feature_cli.run_controlled_cli
    else:
        signature = inspect.signature(_cross_modal.extract_cross_modal_features)
        function_name = "extract_cross_modal_features"
        argv = ["entrypoint", "--manifest", str(manifest), "--batch-size", "1"]
        run = _primary_feature_cli.run_cross_modal_extract_cli

    def bind_runtime(**kwargs):
        observed.update(signature.bind(**kwargs).arguments)

    monkeypatch.setattr(_primary_feature_cli, function_name, bind_runtime)
    monkeypatch.setattr(sys, "argv", argv)
    run(method)
    assert observed["result_metadata"]["feature_surface"] == method.feature_surface
    assert observed["feature_batch_size" if stage == "controlled" else "batch_size"] == 1


def test_cross_modal_metadata_does_not_change_embeddings(tmp_path, monkeypatch):
    features = _cross_modal.cross_modal_features
    sample = SimpleNamespace(sample_id="ct:case:organ", source_case_id="case", file_path=tmp_path / "image.nii.gz")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "index.json").write_text('{}')
    monkeypatch.setattr(features, "_load_manifest_samples", lambda *args, **kwargs: [sample])
    monkeypatch.setattr(features, "_resolve_crop_cache_dir", lambda *args, **kwargs: cache)
    monkeypatch.setattr(features, "ensure_crop_cache", lambda **kwargs: None)
    monkeypatch.setattr(features, "_crop_cache_index_path", lambda path: path / "index.json")
    expected = np.array([1.25, 2.5], dtype=np.float32)

    def extract_cached(**kwargs):
        kwargs["feature_rows"].append(expected.copy())
        kwargs["feature_ids"].append(sample.sample_id)
        return []

    monkeypatch.setattr(features, "_extract_from_crop_cache", extract_cached)
    outputs = []
    for index, metadata in enumerate([None, {"feature_surface": "native", "source_commit": "recorded"}]):
        output = tmp_path / f"features-{index}.npz"
        _cross_modal.extract_cross_modal_features(
            manifest_path=tmp_path / "manifest.json", checkpoint_path=tmp_path / "weights.pth",
            checkpoint_name="external", feature_type="avg_pool", crop_size=96,
            extractor_factory=lambda **kwargs: SimpleNamespace(), output_npz=output,
            device="cpu", dtype=torch.float32, result_metadata=metadata,
        )
        with np.load(output, allow_pickle=True) as bundle:
            np.testing.assert_array_equal(bundle["features"], expected[None])
            outputs.append(bundle["features"].copy())
        summary = json.loads(output.with_name(output.stem + "_summary.json").read_text())
        assert summary["status"] == "complete"
        if metadata:
            assert all(summary[key] == value for key, value in metadata.items())
    np.testing.assert_array_equal(outputs[0], outputs[1])
