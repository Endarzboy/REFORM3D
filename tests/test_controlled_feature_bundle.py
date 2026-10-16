"""The per-encoder analysis bundle is also the reusable inference cache."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from reform3d.modules.acquisition_robustness import perturbation_robustness_analysis as module
from reform3d.modules.acquisition_robustness import feature_extractor
from reform3d.modules.acquisition_robustness.config import ControlledPerturbationConfig


@pytest.fixture
def controlled_run(tmp_path, monkeypatch):
    config = ControlledPerturbationConfig()
    variants = config.describe_targets()
    volumes = [
        module.PerturbedVolume(f"case_{index}.nii.gz", (1., 1., 1.), dict.fromkeys(variants), "kits23")
        for index in range(4)
    ]
    arrays = {
        name: np.random.default_rng(index).normal(size=(4, 3)).astype(np.float32)
        for index, name in enumerate(variants)
    }
    calls = []
    monkeypatch.setattr(module, "prepare_perturbed_dataset", lambda *args, **kwargs: volumes)
    monkeypatch.setattr(feature_extractor, "FeatureExtractor", lambda *args, **kwargs: object())

    def extract(*args, **kwargs):
        calls.append(True)
        return arrays

    monkeypatch.setattr(module, "extract_features_for_variants", extract)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"dataset": "kits23", "modality": "ct"}))

    def run(output=None, cache=None):
        output = output or tmp_path / "results"
        module.run_controlled_perturbation(
            manifest, "med3dino_rel_96", config=config,
            output_dir=output, figures_dir=output / "figures",
            cache_dir=tmp_path / "geometry", feature_cache_path=cache,
            analysis_name="kits23", compute_matched_semantics=False,
            compute_drift_figure=False, device="cpu",
        )
        return json.loads((output / "perturbation_robustness.json").read_text())

    return run, arrays, calls


@pytest.mark.parametrize("explicit_cache", [False, True])
def test_one_bundle_serves_export_and_reuse(tmp_path, controlled_run, explicit_cache):
    run, arrays, calls = controlled_run
    bundle = tmp_path / "results/features/controlled_embedding_bundle.npz"
    cache = bundle if explicit_cache else None
    first = run(cache=cache)
    original = bundle.read_bytes()
    stamp = bundle.stat().st_mtime_ns
    second = run(cache=cache)
    assert calls == [True]
    assert first == second
    assert first["controlled_embedding_bundle"] == "features/controlled_embedding_bundle.npz"
    assert list(tmp_path.rglob("*.npz")) == [bundle]
    assert not bundle.is_symlink()
    assert bundle.read_bytes() == original
    assert bundle.stat().st_mtime_ns == stamp
    with np.load(bundle, allow_pickle=False) as saved:
        assert "source_paths" not in saved
        assert not any(key.startswith("features_") and not key.startswith("features__") for key in saved)
        assert saved["checkpoint"].item() == "med3dino_rel_96"
        for name, features in arrays.items():
            np.testing.assert_array_equal(saved[f"features__{name}"], features)


def test_reuse_from_other_seed_is_a_hardlink(tmp_path, controlled_run):
    run, _, calls = controlled_run
    first = tmp_path / "seed_42/encoder"
    second = tmp_path / "seed_123/encoder"
    run(output=first)
    source = first / "features/controlled_embedding_bundle.npz"
    result = run(output=second, cache=source)
    target = second / result["controlled_embedding_bundle"]
    assert calls == [True]
    assert target.samefile(source)
    assert not target.is_symlink()


@pytest.mark.parametrize("other_exists", [False, True])
def test_explicit_other_cache_cannot_replace_local_bundle(tmp_path, controlled_run, other_exists):
    run, _, calls = controlled_run
    run()
    bundle = tmp_path / "results/features/controlled_embedding_bundle.npz"
    original = bundle.read_bytes()
    other = tmp_path / "other.npz"
    if other_exists:
        other.write_bytes(original)
    with pytest.raises(ValueError, match="already exists"):
        run(cache=other)
    assert calls == [True]
    assert bundle.read_bytes() == original


@pytest.mark.parametrize("fault", [
    "case_order", "missing_variant", "duplicate_variant", "missing_features",
    "nonfinite", "wrong_shape", "inconsistent_dimensions", "complex",
    "checkpoint", "feature_type", "dataset", "perturbation_protocol",
    "target_spacings_mm", "reference_variant_name",
])
def test_reuse_rejects_incompatible_or_incomplete_bundle(tmp_path, controlled_run, fault):
    run, _, calls = controlled_run
    run()
    bundle = tmp_path / "results/features/controlled_embedding_bundle.npz"
    with np.load(bundle, allow_pickle=False) as saved:
        payload = {key: saved[key].copy() for key in saved}
    first_feature = f"features__{payload['variant_names'][0]}"
    if fault == "case_order":
        payload["case_ids"] = payload["case_ids"][::-1]
    elif fault == "missing_variant":
        payload["variant_names"] = payload["variant_names"][:-1]
    elif fault == "duplicate_variant":
        payload["variant_names"][1] = payload["variant_names"][0]
    elif fault == "missing_features":
        del payload[first_feature]
    elif fault == "nonfinite":
        payload[first_feature][0, 0] = np.nan
    elif fault == "wrong_shape":
        payload[first_feature] = payload[first_feature][:-1]
    elif fault == "inconsistent_dimensions":
        payload[first_feature] = payload[first_feature][:, :-1]
    elif fault == "complex":
        payload[first_feature] = payload[first_feature].astype(np.complex64) + 1j
    elif fault == "target_spacings_mm":
        payload[fault][0, 0] += 0.5
    else:
        payload[fault] = np.asarray("different")
    np.savez_compressed(bundle, **payload)
    with pytest.raises(ValueError):
        run()
    assert calls == [True], "An invalid bundle must fail instead of silently re-extracting"


def test_failed_write_preserves_existing_bundle(tmp_path, controlled_run, monkeypatch):
    run, arrays, _ = controlled_run
    run()
    directory = tmp_path / "results/features"
    bundle = directory / "controlled_embedding_bundle.npz"
    original = bundle.read_bytes()

    def interrupted_save(path, **kwargs):
        path.write_bytes(b"incomplete archive")
        raise OSError("interrupted write")

    monkeypatch.setattr(module.np, "savez_compressed", interrupted_save)
    with pytest.raises(OSError, match="interrupted write"):
        module.save_controlled_embedding_bundle(
            directory, arrays,
            [module.PerturbedVolume(f"case_{i}.nii.gz", (1., 1., 1.)) for i in range(4)],
            ControlledPerturbationConfig(), "med3dino_rel_96", "cls", "kits23", "full",
            next(iter(arrays)),
        )
    assert bundle.read_bytes() == original
    assert list(directory.iterdir()) == [bundle]


def test_cli_acquisition_reuses_run_local_bundle_across_seeds(tmp_path, monkeypatch):
    from reform3d import cli, prepare
    from reform3d.settings import digest, file_hash, load_settings
    from reform3d.workspace import run_directory

    variants = ControlledPerturbationConfig().describe_targets()
    volumes = [
        module.PerturbedVolume(f"case_{i}.nii.gz", (1., 1., 1.), dict.fromkeys(variants), "kits23")
        for i in range(12)
    ]
    payload = {
        "dataset": "kits23", "modality": "ct",
        "volumes": [{"file_path": item.original_path, "anisotropy_bin": 0} for item in volumes],
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload))
    settings = tmp_path / "config.yaml"
    settings.write_text(json.dumps({
        "dataset": "kits23", "manifest": str(manifest), "workspace": str(tmp_path / "workspace"),
        "checkpoints": ["med3dino_rel_96"], "seeds": [42, 123], "source_bin": 0,
        "feature_types": ["cls"], "device": "cpu",
    }))
    cfg = load_settings(settings)
    weight = tmp_path / "checkpoint.bin"
    weight.write_bytes(b"test checkpoint")
    checkpoint = SimpleNamespace(
        checkpoint_key="med3dino_rel_96", path=weight, crop_size=96, architecture="vit_large",
    )
    monkeypatch.setattr(cli, "_checkpoints", lambda _: [checkpoint])
    monkeypatch.setattr(prepare, "prepare_manifest", lambda _: (manifest, payload, digest(payload)))
    monkeypatch.setattr(module, "prepare_perturbed_dataset", lambda *args, **kwargs: volumes)
    monkeypatch.setattr(feature_extractor, "FeatureExtractor", lambda *args, **kwargs: object())
    arrays = {name: np.random.default_rng(i).normal(size=(12, 4)).astype(np.float32) for i, name in enumerate(variants)}
    calls = []

    def extract(*args, **kwargs):
        calls.append(True)
        return arrays

    monkeypatch.setattr(module, "extract_features_for_variants", extract)
    labels = np.asarray([[i % 2] for i in range(12)])
    monkeypatch.setattr(module, "_load_matched_semantic_targets", lambda *args, **kwargs: (
        labels, ["cyst"], {"mode": "label_id_multilabel", "organ_names": ["cyst"]}, np.arange(12),
    ))
    expected_semantics = {
        seed: module.compute_matched_semantic_transfer(
            arrays, labels, ["cyst"], checkpoint.checkpoint_key, "cls", random_state=seed,
        ) for seed in cfg["seeds"]
    }
    _, expected_drift, _ = module.compute_pairwise_distances(arrays)
    expected_cka = module.compute_cka_between_variants(arrays)

    state = cli.run(cfg)
    assert state["status"] == "complete"
    assert calls == [True]
    assert len(state["completed"]) == 2
    artifacts = []
    for entry in state["completed"]:
        result_path = Path(entry["path"])
        bundle = result_path.parent / "features/controlled_embedding_bundle.npz"
        assert Path(entry["feature_path"]) == bundle
        assert not bundle.is_symlink()
        assert entry["feature_sha256"] == file_hash(bundle)
        assert json.loads((bundle.parent / "identity.json").read_text())["feature_sha256"] == file_hash(bundle)
        result = json.loads(result_path.read_text())
        assert result["controlled_embedding_bundle"] == "features/controlled_embedding_bundle.npz"
        assert result["matched_semantic_transfer"] == expected_semantics[entry["seed"]]
        assert result["cka_matrix"] == expected_cka
        for variant, value in result["representation_drift"].items():
            assert value == expected_drift[tuple(sorted((variants[0], variant)))]
        artifacts.append(bundle)
    assert artifacts[0].samefile(artifacts[1])
    workspace = Path(cfg["workspace"])
    assert sorted(workspace.rglob("*.npz")) == sorted(artifacts)
    assert not (workspace / "cache/kits23/features").exists()
    run_dir = run_directory(cfg)
    scores = (run_dir / "scores.csv").read_bytes()
    cli.refresh_report(run_dir)
    assert (run_dir / "scores.csv").read_bytes() == scores
    assert (run_dir / "report.html").is_file()
    cli.run(cfg)
    assert calls == [True], "A completed run must also resume without feature extraction"
