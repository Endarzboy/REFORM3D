"""Uncertainty intervals must use the primary evaluator's paired-row CKA."""
import json

import numpy as np
import pytest

from reform3d.modules.acquisition_robustness import controlled_uncertainty_analysis as controlled
from reform3d.modules.acquisition_robustness import native_uncertainty_analysis as native
from reform3d.modules.acquisition_robustness.representation_geometry_analysis import cka_linear


@pytest.mark.parametrize("module", [controlled, native])
@pytest.mark.parametrize("shape", [(20, 6, 6), (8, 25, 19), (40, 4, 7)])
def test_uncertainty_backends_match_primary_cka(module, shape):
    n, dx, dy = shape
    rng = np.random.default_rng(0)
    source, target = rng.normal(size=(n, dx)), rng.normal(size=(n, dy))
    expected = cka_linear(source, target)
    for method in ("gram", "feature", "auto"):
        assert module._linear_cka(source, target, method) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("module", [controlled, native])
@pytest.mark.parametrize("method", ["gram", "feature", "auto"])
def test_case_permutation_breaks_alignment_but_rotation_preserves_it(module, method):
    rng = np.random.default_rng(0)
    source = rng.normal(size=(20, 6))
    permuted = source[rng.permutation(len(source))]
    rotation, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    assert module._linear_cka(source, permuted, method) == pytest.approx(0.26749398677394576)
    assert module._linear_cka(source, source @ rotation, method) == pytest.approx(1.0)


@pytest.mark.parametrize("module", [controlled, native])
def test_bootstrap_duplicates_preserve_primary_statistic(module):
    rng = np.random.default_rng(4)
    source, target = rng.normal(size=(12, 5)), rng.normal(size=(12, 5))
    indices = np.array([0, 0, 2, 3, 3, 3, 6, 6, 7, 9, 10, 11])
    source, target = source[indices], target[indices]
    for method in ("gram", "feature", "auto"):
        assert module._linear_cka(source, target, method) == pytest.approx(cka_linear(source, target))


@pytest.mark.parametrize("module", [controlled, native])
@pytest.mark.parametrize("method", ["gram", "feature", "auto"])
def test_degenerate_inputs_and_invalid_pairs(module, method):
    source = np.arange(24, dtype=float).reshape(8, 3)
    for target in (np.ones_like(source), source * 1e-10):
        assert module._linear_cka(source, target, method) == cka_linear(source, target) == 0.0
    assert module._linear_cka(source[:1], source[:1], method) == 0.0
    assert module._linear_cka(source[:0], source[:0], method) == 0.0
    with pytest.raises(ValueError, match="same number of rows"):
        module._linear_cka(source, source[:3], method)
    invalid = source.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        module._linear_cka(source, invalid, method)


def test_bootstrap_interval_matches_explicit_primary_resamples():
    rng = np.random.default_rng(5)
    features = {name: rng.normal(size=(12, 4)) for name in ("a", "b")}
    expected_samples = []
    rng = np.random.default_rng(42)
    for _ in range(25):
        indices = rng.integers(0, 12, size=12)
        expected_samples.append(cka_linear(features["a"][indices], features["b"][indices]))
    result = controlled._bootstrap_compact_cka(features, ["a", "b"], 25, 42, "auto")
    assert result["mean"] == pytest.approx(cka_linear(features["a"], features["b"]))
    assert result["ci_lower"] == pytest.approx(np.percentile(expected_samples, 2.5))
    assert result["ci_upper"] == pytest.approx(np.percentile(expected_samples, 97.5))


@pytest.mark.parametrize("module,filename", [
    (controlled, "perturbation_robustness.json"),
    (native, "representation_geometry.json"),
])
@pytest.mark.parametrize("seed_directory", ["", "seeds/seed_42/"])
def test_direct_result_discovery(module, filename, seed_directory, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "OUTPUTS_ROOT", tmp_path)
    path = tmp_path / f"custom/coarse_bins/{seed_directory}results/encoder/cls/{filename}"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    assert module._discover_metric_jsons(["custom"], ["cls"], ["encoder"]) == [path]
    assert module._manifest_variant_from_metric_path(path) == f"coarse_bins/{seed_directory}".rstrip("/")
    assert module._discover_metric_jsons(["other"], ["cls"], ["encoder"]) == []


def test_run_local_bundle_uses_recorded_paths_and_current_statistic(tmp_path):
    run_dir = tmp_path / "runs/custom/acquisition/example"
    leaf = run_dir / "seeds/seed_42/encoder--cls"
    features_dir = leaf / "features"
    features_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({
        "configuration": {"dataset": "custom"}, "manifest": str(tmp_path / "manifest.json"),
    }))
    rng = np.random.default_rng(0)
    source = rng.normal(size=(20, 6)).astype(np.float32)
    target = source[rng.permutation(20)]
    bundle = features_dir / "controlled_embedding_bundle.npz"
    np.savez(bundle, features__a=source, features__b=target,
             variant_names=np.array(["a", "b"]), reference_variant_name=np.array("a"))
    path = leaf / "perturbation_robustness.json"
    path.write_text(json.dumps({"checkpoint": "encoder", "feature_type": "cls",
                                "controlled_embedding_bundle": "features/controlled_embedding_bundle.npz"}))
    original_bundle = bundle.read_bytes()
    result = controlled._analyze_metric_path(path, 5, 42, "auto", 5)
    assert result["status"] == "ok"
    assert result["analysis_name"] == "custom"
    assert result["checkpoint_name"] == "encoder"
    assert result["feature_type"] == "cls"
    assert result["evaluation_seed"] == 42
    assert result["run_name"] == "example"
    assert result["bundle_path"] == str(bundle)
    assert result["compact_cka_mean"] == pytest.approx(cka_linear(source, target), abs=1e-7)
    assert bundle.read_bytes() == original_bundle
    assert list(tmp_path.rglob("*.npz")) == [bundle]


def test_uncertainty_command_refuses_to_replace_input(tmp_path, monkeypatch):
    path = tmp_path / "perturbation_robustness.json"
    path.write_text("{}")
    monkeypatch.setattr("sys.argv", ["uncertainty", "--metric-jsons", str(path),
                                    "--output-json", str(path)])
    with pytest.raises(SystemExit) as exc:
        controlled.main()
    assert exc.value.code == 2
    assert path.read_text() == "{}"
