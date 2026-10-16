"""Reject plausible-looking but incomplete or altered evaluation artifacts."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("module", ["cross_modal_alignment", "anatomical_generalization"])
@pytest.mark.parametrize("fault", ["duplicate_ids", "missing_case", "nonfinite", "empty_dimension", "complex"])
def test_direct_module_rejects_invalid_embeddings(tmp_path, module, fault):
    from importlib import import_module
    load = import_module(f"reform3d.modules.{module}.evaluation_pipeline")._load_embedding_bundle
    ids = ["one", "two"]
    features = np.eye(2, dtype=np.float32)
    if fault == "duplicate_ids": ids = ["one", "one"]
    if fault == "missing_case": ids = ["one", "unrelated"]
    if fault == "nonfinite": features[0, 0] = np.nan
    if fault == "complex": features = features.astype(np.complex64) + 1j
    if fault == "empty_dimension": features = np.empty((2, 0), dtype=np.float32)
    path = tmp_path / "features.npz"
    np.savez(path, sample_ids=ids, features=features)
    with pytest.raises(ValueError):
        load(path, [{"sample_id": name} for name in ["one", "two"]])


@pytest.mark.parametrize("module", ["cross_modal_alignment", "anatomical_generalization"])
def test_valid_subset_can_reuse_full_embedding_bundle(tmp_path, module):
    from importlib import import_module
    load = import_module(f"reform3d.modules.{module}.evaluation_pipeline")._load_embedding_bundle
    path = tmp_path / "features.npz"
    features = np.asarray([[1., 2.], [3., 4.]], dtype=np.float32)
    np.savez(path, sample_ids=["two", "one"], features=features)
    result = load(path, [{"sample_id": "one"}])
    np.testing.assert_array_equal(result["one"], features[1])


def stub_run(tmp_path, monkeypatch):
    from reform3d import cli
    from test_release_contracts import organ_manifest
    cfg, _, _ = organ_manifest(tmp_path)
    cfg.update(device="cpu", modules=["cross_modal"], not_evaluated=["acquisition", "anatomical"])
    weight = tmp_path / "weights.bin"
    weight.write_bytes(b"test weights")
    checkpoint = SimpleNamespace(checkpoint_key="test", path=weight, crop_size=2, architecture="test")
    monkeypatch.setattr(cli, "_checkpoints", lambda _: [checkpoint])
    calls = []
    def evaluate(module, args):
        args = list(map(str, args))
        calls.append(module)
        def value(flag): return args[args.index(flag) + 1]
        if module.endswith("feature_extractor"):
            payload = json.loads(Path(value("--manifest")).read_text())
            ids = [row["sample_id"] for row in payload["samples"]]
            np.savez(value("--output-npz"), sample_ids=ids, features=np.eye(len(ids), dtype=np.float32))
        else:
            direction = {"status": "ok", "n_organs": 2, "n_queries": 10, "n_targets": 10}
            data = {"cohorts": {"n_supported_organs": 2}, "balanced_cross_modal_retrieval": {"status": "ok", "ct_to_mr": direction, "mr_to_ct": direction, "bidirectional_mean": {"map": 0.5, "top@1": 0.5, "top@5": 1.}}, "balanced_lisi": {"status": "ok", "n_organs": 2, "n_samples": 20, "overall": {"modality_ilisi": {"mean": 1.5}, "organ_clisi": {"mean": 1.1}}}}
            Path(value("--metrics-output")).write_text(json.dumps(data))
    monkeypatch.setattr(cli, "_call", evaluate)
    return cli, cfg, calls


@pytest.mark.parametrize("artifact", ["metric", "features"])
def test_resume_rejects_altered_completed_artifact(tmp_path, monkeypatch, artifact):
    cli, cfg, _ = stub_run(tmp_path, monkeypatch)
    state = cli.run(cfg)
    if artifact == "metric":
        path = Path(state["completed"][0]["path"])
        payload = json.loads(path.read_text())
        payload["balanced_cross_modal_retrieval"]["bidirectional_mean"]["map"] = 0.95
        path.write_text(json.dumps(payload))
    else:
        path = Path(state["completed"][0]["feature_path"])
        with np.load(path) as data:
            ids, features = data["sample_ids"].copy(), data["features"].copy()
        features[0, 0] += 0.25
        np.savez(path, sample_ids=ids, features=features)
    with pytest.raises(ValueError, match="[Cc]hanged|[Cc]hecksum|[Ii]ntegrity"):
        cli.run(cfg)


def test_valid_resume_does_not_rerun_completed_evaluation(tmp_path, monkeypatch):
    cli, cfg, calls = stub_run(tmp_path, monkeypatch)
    first = cli.run(cfg)
    before = list(calls)
    second = cli.run(cfg)
    assert calls == before
    assert first["status"] == second["status"] == "complete"


@pytest.mark.parametrize("fault", ["missing_variant", "duplicate_variant", "reordered_variants"])
def test_controlled_cache_preserves_requested_protocol(tmp_path, monkeypatch, fault):
    from reform3d.modules.acquisition_robustness import perturbation_robustness_analysis as module
    from reform3d.modules.acquisition_robustness.config import ControlledPerturbationConfig
    config = ControlledPerturbationConfig()
    expected = config.describe_targets()
    names = list(expected)
    if fault == "missing_variant": names = names[:-1]
    if fault == "duplicate_variant": names[1] = names[0]
    if fault == "reordered_variants": names.reverse()
    paths = ["case_one.nii.gz", "case_two.nii.gz"]
    volumes = [module.PerturbedVolume(name, (1., 1., 1.), {key: None for key in expected}, "kits23") for name in paths]
    monkeypatch.setattr(module, "prepare_perturbed_dataset", lambda *a, **k: volumes)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"dataset": "kits23", "modality": "ct", "volumes": [{"file_path": name} for name in paths]}))
    cache = tmp_path / "features.npz"
    arrays = {f"features_{i}": np.asarray([[1., 2.], [3., 4.]], dtype=np.float32) for i in range(len(names))}
    np.savez(cache, source_paths=paths, variant_names=names, **arrays)
    def run():
        return module.run_controlled_perturbation(manifest, "med3dino_rel_96", config=config, output_dir=tmp_path / "results", figures_dir=tmp_path / "figures", cache_dir=tmp_path / "crops", feature_cache_path=cache, compute_matched_semantic_probing=False, compute_matched_semantic_transfer=False, compute_drift_figure=False, device="cpu")
    if fault != "reordered_variants":
        with pytest.raises(ValueError, match="[Vv]ariant|[Pp]rotocol"):
            run()
    else:
        run()
        result = json.loads((tmp_path / "results/perturbation_robustness.json").read_text())
        assert result["reference_variant"] == expected[0]


def test_controlled_geometry_invalidates_changed_source(tmp_path):
    import os
    from reform3d.modules.acquisition_robustness.perturbation_robustness_analysis import get_cache_path
    source = tmp_path / "source.nii"
    source.write_bytes(b"original")
    before = get_cache_path(source, cache_dir=tmp_path / "cache")
    stamp = source.stat()
    source.write_bytes(b"modified")
    os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000))
    after = get_cache_path(source, cache_dir=tmp_path / "cache")
    assert before != after


def test_anatomical_bundle_rejects_float32_overflow(tmp_path):
    from reform3d.modules.anatomical_generalization.evaluation_pipeline import _load_embedding_bundle
    path = tmp_path / "features.npz"
    np.savez(path, sample_ids=["one"], features=np.asarray([[1e300]], dtype=np.float64))
    with pytest.raises(ValueError, match="overflow"):
        _load_embedding_bundle(path, [{"sample_id": "one"}])


def test_controlled_rejects_colliding_variant_labels_before_preprocessing(tmp_path, monkeypatch):
    from reform3d.modules.acquisition_robustness import perturbation_robustness_analysis as module
    from reform3d.modules.acquisition_robustness.config import ControlledPerturbationConfig
    config = ControlledPerturbationConfig(target_spacings=[(1., 1., 1.), (1., 1., 1.01)])
    def forbidden(*args, **kwargs):
        pytest.fail("Ambiguous variants reached preprocessing")
    monkeypatch.setattr(module, "prepare_perturbed_dataset", forbidden)
    with pytest.raises(ValueError, match="uniquely named variants"):
        module.run_controlled_perturbation(tmp_path / "manifest.json", "med3dino_rel_96", config=config)
