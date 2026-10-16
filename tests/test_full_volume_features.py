"""Full-volume caches share inputs and never request attention or cropped caches."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from reform3d import full_volume_features as features
from reform3d.encoders import factory


@pytest.fixture
def example(tmp_path, monkeypatch):
    calls = {"prepare": [], "factory": [], "forward": []}
    cases = []
    for i, modality in enumerate(("CT", "MR")):
        image, mask = tmp_path / f"{modality}.nii", tmp_path / f"{modality}-mask.nii"
        image.write_bytes(bytes([i + 1])); mask.write_bytes(b"mask")
        cases.append({"id": f"{modality.lower()}-example", "modality": modality,
                      "image": str(image), "mask": str(mask), "reference_ras_mm": [1., 2., 3.]})
    checkpoints = {}
    for i, name in enumerate(features.MODELS):
        path = tmp_path / f"{name}.pth"
        path.write_bytes(bytes([i + 1]))
        checkpoints[name] = str(path)
    cfg = {"cases": cases, "checkpoints": checkpoints, "device": "cpu",
           "output_dir": str(tmp_path / "full-volume"), "max_axis": 32, "max_patches": 8}

    def prepare(image, mask=None, **options):
        calls["prepare"].append((image, mask, options))
        value = Path(image).read_bytes()[0]
        volume = np.full((32, 16, 16), value / 10, dtype=np.float32)
        return {"volume": volume, "mask": np.zeros(volume.shape, bool),
                "foreground": np.ones(volume.shape, bool), "metadata": {
                    "affine_ras": np.diag([2., 2., 2., 1.]).tolist(), "grid_shape": [2, 1, 1],
                    "input_sha256": hashlib.sha256(volume.tobytes()).hexdigest(),
                    "no_crop": True, "model_shape": list(volume.shape)}}

    class Model(torch.nn.Module):
        patch_size, embed_dim = 16, 4

        def __init__(self, name):
            super().__init__()
            self.name = name
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward_features(self, x):
            calls["forward"].append((self.name, x.detach().cpu().numpy().copy()))
            count = int(np.prod([n // 16 for n in x.shape[2:]]))
            value = 10 * (features.MODELS.index(self.name) + 1) + x.mean()
            tokens = torch.arange(count * 4, dtype=torch.float32).reshape(1, count, 4) + value
            return {"x_norm_patchtokens": tokens}

        def get_self_attention(self, *args, **kwargs):
            raise AssertionError("Full-volume similarity must not extract attention")

    def make(name, **kwargs):
        calls["factory"].append((name, kwargs))
        return SimpleNamespace(model=Model(name))

    monkeypatch.setattr(features.full_volume, "prepare_full_volume", prepare)
    monkeypatch.setattr(factory, "create_feature_extractor", make)
    return cfg, calls


def test_one_forward_per_model_case_and_exact_shared_inputs(example):
    cfg, calls = example
    arrays, receipt = features.load_or_extract_full_volume_features(cfg)
    assert len(calls["prepare"]) == 2
    assert len(calls["factory"]) == 3
    assert len(calls["forward"]) == 6
    assert receipt["forward_count"] == 6
    assert receipt["attention_extracted"] is False
    assert receipt["feature_schema"] == features.SCHEMA
    assert receipt["embedding_dimensions"] == dict.fromkeys(features.MODELS, 4)
    assert len(arrays) == 12
    assert not any("attention" in name for name in arrays)
    for i in range(2):
        assert receipt["cases"][i]["case_id"] == cfg["cases"][i]["id"]
        assert receipt["cases"][i]["no_crop"] is True
        for m in range(3):
            np.testing.assert_array_equal(calls["forward"][2 * m + i][1][0, 0], arrays[f"volume_{i}"])
            assert arrays[f"tokens_{m}_{i}"].shape == (2, 4)
    for name, info in receipt["signature"]["checkpoints"].items():
        assert receipt["checkpoint_sha256"][name] == features._sha256(cfg["checkpoints"][name]) == info["sha256"]


def test_cache_and_new_reference_points_do_not_rerun_inference(example):
    cfg, calls = example
    arrays, receipt = features.load_or_extract_full_volume_features(cfg)
    before = deepcopy(calls)
    changed = deepcopy(cfg)
    changed["cases"][0]["reference_ras_mm"] = [999., -20., 14.]
    changed["num_slices"], changed["opacity"] = 5, .1
    cached, same = features.load_or_extract_full_volume_features(changed)
    assert len(calls["forward"]) == len(before["forward"])
    assert calls["prepare"] == before["prepare"]
    assert calls["factory"] == before["factory"]
    assert same == receipt
    for key in arrays:
        np.testing.assert_array_equal(cached[key], arrays[key])
    features.load_full_volume_features(changed)
    assert len(calls["forward"]) == 6


def test_prepare_freezes_inputs_without_loading_encoders(example):
    cfg, calls = example
    arrays, receipt = features.prepare_full_volume_inputs(cfg)
    assert len(calls["prepare"]) == 2
    assert calls["factory"] == calls["forward"] == []
    same, second = features.prepare_full_volume_inputs(cfg)
    assert len(calls["prepare"]) == 2
    assert second == receipt
    assert all(np.array_equal(arrays[k], same[k]) for k in arrays)
    assert not (Path(cfg["output_dir"]) / "maps.npz").exists()


def test_cache_only_does_not_prepare_or_extract_missing_inputs(example):
    cfg, calls = example
    with pytest.raises(FileNotFoundError, match="Prepare"):
        features.load_full_volume_features(cfg)
    assert calls == {"prepare": [], "factory": [], "forward": []}


def test_cache_only_does_not_extract_missing_features(example):
    cfg, calls = example
    features.prepare_full_volume_inputs(cfg)
    with pytest.raises(FileNotFoundError, match="Extract"):
        features.load_full_volume_features(cfg)
    assert calls["factory"] == []


@pytest.mark.parametrize("change", ["image", "mask", "case_id", "case_order", "geometry", "checkpoint", "device"])
def test_cache_identity_changes_fail_without_overwriting(example, change):
    cfg, calls = example
    features.load_or_extract_full_volume_features(cfg)
    before = {p: p.read_bytes() for p in Path(cfg["output_dir"]).rglob("*") if p.is_file()}
    changed = deepcopy(cfg)
    if change in {"image", "mask"}:
        Path(changed["cases"][0][change]).write_bytes(b"changed")
    elif change == "case_id":
        changed["cases"][0]["id"] = "different"
    elif change == "case_order":
        changed["cases"].reverse()
    elif change == "geometry":
        changed["max_axis"] = 48
    elif change == "checkpoint":
        Path(changed["checkpoints"][features.MODELS[1]]).write_bytes(b"changed")
    elif change == "device":
        changed["device"] = "cuda:0"
    with pytest.raises(ValueError, match="changed"):
        features.load_or_extract_full_volume_features(changed)
    assert len(calls["forward"]) == 6
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize("location", ["inputs", "features"])
def test_partial_caches_fail_without_inference(example, location):
    cfg, calls = example
    if location == "features":
        features.prepare_full_volume_inputs(cfg)
        folder = Path(cfg["output_dir"])
    else:
        folder = Path(cfg["output_dir"]) / "inputs"
        folder.mkdir(parents=True)
    path = folder / ("maps.npz" if location == "features" else "inputs.npz")
    path.write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="Incomplete"):
        features.load_or_extract_full_volume_features(cfg)
    assert path.read_bytes() == b"incomplete"
    assert calls["factory"] == []


@pytest.mark.parametrize("target", ["maps.npz", "receipt.json", "inputs/inputs.npz", "inputs/receipt.json"])
def test_corrupt_cache_fails_explicitly(example, target):
    cfg, calls = example
    features.load_or_extract_full_volume_features(cfg)
    (Path(cfg["output_dir"]) / target).write_bytes(b"bad")
    with pytest.raises(ValueError, match="changed|receipt"):
        features.load_or_extract_full_volume_features(cfg)
    assert len(calls["forward"]) == 6


def test_modified_frozen_array_with_updated_archive_hash_still_fails(example):
    cfg, _ = example
    features.prepare_full_volume_inputs(cfg)
    folder = Path(cfg["output_dir"]) / "inputs"
    arrays = features._load_archive(folder / "inputs.npz")
    arrays["volume_0"][0, 0, 0] = .99
    np.savez_compressed(folder / "inputs.npz", **arrays)
    receipt = json.loads((folder / "receipt.json").read_text())
    receipt["sha256"] = features._sha256(folder / "inputs.npz")
    (folder / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="hash mismatch"):
        features.prepare_full_volume_inputs(cfg)


@pytest.mark.parametrize("invalid", ["missing", "zero", "nan", "shape"])
def test_invalid_patch_outputs_fail_without_publishing_cache(example, monkeypatch, invalid):
    cfg, calls = example
    original = factory.create_feature_extractor

    def make(*args, **kwargs):
        extractor = original(*args, **kwargs)
        forward = extractor.model.forward_features

        def malformed(x):
            output = forward(x)
            if invalid == "missing":
                return {}
            if invalid == "zero":
                output["x_norm_patchtokens"].zero_()
            elif invalid == "nan":
                output["x_norm_patchtokens"].fill_(float("nan"))
            else:
                output["x_norm_patchtokens"] = torch.ones(1, 3, 4)
            return output
        extractor.model.forward_features = malformed
        return extractor
    monkeypatch.setattr(factory, "create_feature_extractor", make)
    with pytest.raises((ValueError, KeyError), match="patch|feature|x_norm_patchtokens"):
        features.load_or_extract_full_volume_features(cfg)
    out = Path(cfg["output_dir"])
    assert not (out / "maps.npz").exists()
    assert not (out / "receipt.json").exists()
    assert not (out / ".features.lock").exists()
    assert (out / "inputs/inputs.npz").exists()


@pytest.mark.parametrize("key,value", [
    ("max_axis", True), ("max_axis", 0), ("max_axis", 225), ("max_axis", 240),
    ("max_patches", 0), ("max_patches", 2745), ("max_patches", True),
    ("memory_fraction", 0), ("memory_fraction", float("nan")), ("memory_fraction", True),
])
def test_invalid_bounds_fail_before_encoder_loading(example, key, value):
    cfg, calls = example
    cfg[key] = value
    with pytest.raises(ValueError):
        features.load_or_extract_full_volume_features(cfg)
    assert calls["factory"] == []


def test_duplicate_case_ids_and_unknown_checkpoint_fail(example):
    cfg, calls = example
    invalid = deepcopy(cfg)
    invalid["cases"][1]["id"] = invalid["cases"][0]["id"]
    with pytest.raises(ValueError, match="unique"):
        features.prepare_full_volume_inputs(invalid)
    cfg["checkpoints"]["unexpected"] = cfg["checkpoints"][features.MODELS[0]]
    with pytest.raises(ValueError, match="checkpoint"):
        features.load_or_extract_full_volume_features(cfg)
    assert calls["factory"] == []


def test_unrecognized_directory_is_never_overwritten(example):
    cfg, calls = example
    out = Path(cfg["output_dir"])
    out.mkdir()
    preserved = out / "cropped-pca-results.txt"
    preserved.write_text("keep me")
    with pytest.raises(ValueError, match="Unrecognized"):
        features.load_or_extract_full_volume_features(cfg)
    assert preserved.read_text() == "keep me"
    assert calls["prepare"] == calls["factory"] == []
