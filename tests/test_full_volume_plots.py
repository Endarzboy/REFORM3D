"""Full-volume feature maps retain native frames, physical seeds and source context."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("full_volume_views", ROOT / "notebooks/visualization_utils.py")
views = importlib.util.module_from_spec(spec)
spec.loader.exec_module(views)


@pytest.fixture
def example(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    from reform3d import full_volume

    monkeypatch.setattr(Figure, "savefig", lambda fig, path, **kw: Path(path).write_bytes(b"figure"))
    shape = (8, 6, 10)
    source_affine = np.diag([2., 3., 4., 1.])
    source_affine[:3, 3] = [100., -20., 5.]
    raw = np.indices(shape).sum(0).astype(np.float32)
    mask = np.zeros(shape, bool)
    mask[2:6, 2:5, 2:9] = True
    source_files = [tmp_path / "image.nii.gz", tmp_path / "mask.nii.gz"]
    for filename in source_files:
        filename.write_bytes(filename.name.encode())
    image_record, mask_record = [{"path": str(filename), "bytes": filename.stat().st_size,
                                  "mtime_ns": filename.stat().st_mtime_ns} for filename in source_files]
    source = {"volume": raw, "mask": mask, "affine_ras": source_affine,
              "metadata": {"source_shape": list(shape), "source_affine_ras": source_affine.tolist(),
                           "source_bounds_ras_mm": {"low": [99., -21.5, 3.], "high": [115., -3.5, 43.]},
                           "image": image_record, "mask": mask_record}}
    monkeypatch.setattr(full_volume, "read_source", lambda image, mask: source)
    point = (source_affine @ [3, 3, 5, 1])[:3]
    cfg = {"cases": [{"id": "MR_example", "modality": "MR", "image": "source.nii.gz",
                       "mask": "mask.nii.gz", "reference_ras_mm": point.tolist()}],
           "output_dir": str(tmp_path / "output")}
    model = np.ones((32, 32, 32), dtype=np.float32)
    model_affine = np.diag([2., 2., 2., 1.])
    model_affine[:3, 3] = [76., -42., -6.]
    metadata = {"case_id": "MR_example", "modality": "MR", "affine_ras": model_affine.tolist(),
                "input_sha256": hashlib.sha256(model.tobytes()).hexdigest(), "grid_shape": [2, 2, 2],
                "normalization": {"low": 0., "high": 25.}}
    arrays = {"volume_0": model, "mask_0": np.ones_like(model, bool),
              "foreground_0": np.ones_like(model, bool)}
    tokens = np.array([[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.], [1., 1., 0.],
                       [1., -1., 0.], [1., 0., 0.], [0., 0., 1.], [-1., 1., 0.]])
    for model_index in range(3):
        arrays[f"tokens_{model_index}_0"] = np.roll(tokens, model_index, axis=0)
    receipt = {"cases": [metadata], "maps_sha256": "feature-cache",
               "checkpoint_sha256": {policy: policy + "-sha" for policy in views.POLICIES},
               "signature": {"inputs": {"cases": [{
                   "image": {**image_record, "sha256": views._digest(image_record["path"])},
                   "mask": {**mask_record, "sha256": views._digest(mask_record["path"])}}]}}}
    yield cfg, arrays, receipt, source
    plt.close("all")


def report(cfg):
    paths = list((Path(cfg["output_dir"]) / "full_volume_similarity").glob("*_display.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text())


def axes_for(gallery, rows):
    return np.asarray(gallery.axes[:rows * 4], object).reshape(rows, 4)


def test_full_source_frame_keeps_native_pixels_cell_bounds_and_image_values(example):
    cfg, arrays, receipt, source = example
    before = {key: value.copy() for key, value in arrays.items()}
    figures = views.plot_full_volume_similarity(cfg, arrays, receipt, case_id="MR_example",
                                                slice_indices=[5, 1], interpolation="nearest")
    assert len(figures) == 2
    record = report(cfg)
    assert record["reference_ras_mm"] == [106., -11., 25.]
    assert record["reference_patch"] == [0, 1, 1]
    assert record["source_shape"] == [8, 6, 10]
    assert record["native_plane_shape"] == [6, 8]
    assert record["source_slice_indices"] == [5, 1]
    assert record["slice_z_mm"] == [25., 9.]
    assert record["maps_sha256"] == receipt["maps_sha256"]
    assert record["checkpoint_sha256"] == receipt["checkpoint_sha256"]
    for row, z in zip(axes_for(figures[1], 2), [5, 1]):
        expected = 2 * source["volume"][::-1, :, z].T / 25 - 1
        for ax in row:
            assert ax.images[0].get_array().shape == (6, 8)
            assert ax.images[0].get_extent() == [115., 99., -21.5, -3.5]
            np.testing.assert_allclose(ax.images[0].get_array(), expected, atol=1e-7)
            assert ax.get_xlim() == (115., 99.)
            assert ax.get_ylim() == (-21.5, -3.5)
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], before[key])


def test_native_cosine_grids_are_sampled_once_directly_on_source_plane(example):
    cfg, arrays, receipt, source = example
    _, gallery = views.plot_full_volume_similarity(cfg, arrays, receipt, case_id="MR_example",
                                                    slice_indices=[5], interpolation="nearest")
    record = report(cfg)
    world, _ = views._plane_coordinates(source, 25.)
    with np.load(Path(cfg["output_dir"]) / record["files"]["maps"]) as maps:
        for model, (policy, ax) in enumerate(zip(views.POLICIES, axes_for(gallery, 1)[0][1:])):
            features = arrays[f"tokens_{model}_0"]
            unit = features / np.linalg.norm(features, axis=1, keepdims=True)
            expected_grid = (unit @ unit[3]).reshape(2, 2, 2)
            np.testing.assert_allclose(maps[policy], expected_grid, atol=1e-7)
            assert maps[policy][0, 1, 1] == 1.
            expected_plane = views._sample_plane(expected_grid, record["patch_affine_ras"],
                                                 world, order=0, mode="nearest")
            np.testing.assert_allclose(ax.images[1].get_array(), expected_plane, atol=1e-7)
            np.testing.assert_allclose(ax.images[1].get_alpha(), .65 * np.clip(expected_plane, 0, 1))
            assert ax.images[1].get_clim() == (-1., 1.)
    for relative in record["files"].values():
        assert (Path(cfg["output_dir"]) / relative).is_file()


def test_annotations_only_draw_contours_and_do_not_select_slices_or_feature_values(example):
    cfg, arrays, receipt, source = example
    _, first = views.plot_full_volume_similarity(cfg, arrays, receipt, case_id="MR_example", num_slices=3)
    initial_report = report(cfg)
    initial_planes = [ax.images[1].get_array().copy() for row in axes_for(first, 3) for ax in row[1:]]
    assert any(ax.collections for row in axes_for(first, 3) for ax in row)
    source["mask"][:] = False
    arrays["mask_0"][:] = False
    arrays["foreground_0"][:] = False
    arrays["volume_0"][:] = 0.
    # The model image is only validated against its receipt, never used as the background.
    receipt["cases"][0]["input_sha256"] = hashlib.sha256(arrays["volume_0"].tobytes()).hexdigest()
    _, second = views.plot_full_volume_similarity(cfg, arrays, receipt, case_id="MR_example", num_slices=3)
    for expected, ax in zip(initial_planes, [ax for row in axes_for(second, 3) for ax in row[1:]]):
        np.testing.assert_array_equal(ax.images[1].get_array(), expected)
        assert len(ax.collections) == 0
    paths = list((Path(cfg["output_dir"]) / "full_volume_similarity").glob("*_display.json"))
    for path in paths:
        assert json.loads(path.read_text())["source_slice_indices"] == initial_report["source_slice_indices"]


def test_physical_point_and_patch_footprint_only_appear_at_true_levels(example):
    cfg, arrays, receipt, source = example
    preview, gallery = views.plot_full_volume_similarity(cfg, arrays, receipt, case_id="MR_example",
                                                        slice_indices=[5, 1])
    record = report(cfg)
    assert record["reference_patch_bounds_ras_mm"] == {"low": [75., -11., 25.], "high": [107., 21., 57.]}
    assert len(preview.axes[0].patches) == 1
    assert len(preview.axes[0].lines) == 1
    rows = axes_for(gallery, 2)
    assert all(len(ax.patches) == 1 and len(ax.lines) == 1 for ax in rows[0])
    assert all(not ax.patches and not ax.lines for ax in rows[1])


def test_default_slices_follow_query_and_ignore_feature_maps(example):
    cfg, arrays, receipt, source = example
    views.plot_full_volume_similarity(cfg, arrays, receipt, case_id="MR_example", num_slices=1,
                                      show_contours=False)
    assert report(cfg)["source_slice_indices"] == [5]
    reference = np.array(cfg["cases"][0]["reference_ras_mm"])
    reference[2] = 13.
    _, gallery = views.plot_full_volume_similarity(cfg, arrays, receipt, case_id="MR_example",
                                                  reference_ras_mm=reference, num_slices=1,
                                                  show_contours=False)
    assert axes_for(gallery, 1)[0, 0].get_ylabel() == "Source slice 2\nz = 13.0 mm"
    assert all(not ax.collections for ax in axes_for(gallery, 1)[0])


def test_cases_are_isolated_by_unique_id_even_with_shared_modality(example):
    cfg, arrays, receipt, source = example
    cfg["cases"].insert(0, {"id": "other_mr", "modality": "MR", "image": "unused"})
    receipt["cases"].insert(0, {"case_id": "other_mr", "modality": "MR"})
    receipt["signature"]["inputs"]["cases"].insert(0, {})
    moved = {key[:-1] + "1": value for key, value in arrays.items()}
    views.plot_full_volume_similarity(cfg, moved, receipt, case_id="MR_example", num_slices=1)
    assert report(cfg)["case_index"] == 1


@pytest.mark.parametrize("kwargs", [
    {"case_id": "missing"}, {"case_id": "MR_example", "num_slices": 0},
    {"case_id": "MR_example", "num_slices": 6}, {"case_id": "MR_example", "num_slices": True},
    {"case_id": "MR_example", "slice_indices": []}, {"case_id": "MR_example", "slice_indices": [1, 1]},
    {"case_id": "MR_example", "slice_indices": [10]}, {"case_id": "MR_example", "slice_indices": [False]},
    {"case_id": "MR_example", "reference_ras_mm": [0, 0, 0]},
    {"case_id": "MR_example", "reference_ras_mm": [100, np.nan, 5]},
    {"case_id": "MR_example", "reference_ras_mm": [100, 5]},
    {"case_id": "MR_example", "opacity": 1.1}, {"case_id": "MR_example", "opacity": True},
    {"case_id": "MR_example", "interpolation": "cubic"},
])
def test_invalid_display_inputs_fail_explicitly(example, kwargs):
    cfg, arrays, receipt, source = example
    with pytest.raises(ValueError):
        views.plot_full_volume_similarity(cfg, arrays, receipt, **kwargs)


@pytest.mark.parametrize("change", ["source", "input", "case", "grid", "normalization"])
def test_inconsistent_source_or_feature_receipts_fail(example, change):
    cfg, arrays, receipt, source = example
    if change == "source":
        source["metadata"]["image"]["mtime_ns"] += 1
    elif change == "input":
        arrays["volume_0"][0, 0, 0] = 0
    elif change == "case":
        receipt["cases"][0]["case_id"] = "other"
    elif change == "grid":
        receipt["cases"][0]["grid_shape"] = [1, 1, 1]
    else:
        receipt["cases"][0]["normalization"]["high"] = 0
    with pytest.raises(ValueError):
        views.plot_full_volume_similarity(cfg, arrays, receipt, case_id="MR_example", num_slices=1)


def test_display_window_changes_only_raw_image_intensity(example):
    cfg, arrays, receipt, source = example
    _, first = views.plot_full_volume_similarity(cfg, arrays, receipt, case_id="MR_example", num_slices=1)
    first_axes = axes_for(first, 1)[0]
    cfg["cases"][0]["display_window"] = [-160, 240]
    _, second = views.plot_full_volume_similarity(cfg, arrays, receipt, case_id="MR_example", num_slices=1)
    for old, new in zip(first_axes[1:], axes_for(second, 1)[0][1:]):
        assert not np.array_equal(old.images[0].get_array(), new.images[0].get_array())
        np.testing.assert_array_equal(old.images[1].get_array(), new.images[1].get_array())
    records = [json.loads(path.read_text()) for path in
               (Path(cfg["output_dir"]) / "full_volume_similarity").glob("*_display.json")]
    assert sorted(record["display_window"] for record in records) == [[-160., 240.], [0., 25.]]



def shared_config(checkpoint, cases, **options):
    return {"checkpoints": {f"med3dino_{policy}_96": str(checkpoint) for policy in ("iso", "rel", "sa")},
            "device": "cpu", "similarity": {"cases": cases, "output_dir": "full", **options}}


def test_setup_resolves_config_relative_paths_independently_of_kernel_directory(tmp_path, monkeypatch):
    directory = tmp_path / "examples"
    directory.mkdir()
    for filename in ("image.nii.gz", "mask.nii.gz", "weights.pth"):
        (directory / filename).write_bytes(b"file")
    path = directory / "visualization_config.json"
    case = {"id": "ct", "modality": "CT", "image": "image.nii.gz", "mask": "mask.nii.gz"}
    config = shared_config("weights.pth", [case])
    path.write_text(json.dumps(config))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REFORM3D_VIZ_CONFIG", str(path))
    resolved = views.setup(workflow="similarity")
    for role in ("image", "mask"):
        assert resolved["cases"][0][role] == str(directory / case[role])
    assert set(resolved["checkpoints"].values()) == {str(directory / "weights.pth")}
    assert resolved["output_dir"] == str(directory / "full")


@pytest.mark.parametrize("case", [
    {"id": "bad", "modality": "US", "image": "missing"},
    {"modality": "MR", "image": "missing"},
])
def test_setup_rejects_ambiguous_cases_before_loading_inputs(tmp_path, case):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(shared_config("missing.pth", [case])))
    with pytest.raises(ValueError, match="CT/MR|unique id"):
        views.setup(workflow="similarity", config_path=path)


def test_setup_missing_configuration_explains_the_example_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="visualization_config.example.json"):
        views.setup(workflow="similarity", config_path=tmp_path / "missing.json")


def test_setup_inherits_checkpoints_and_device_without_reusing_crop_output(tmp_path):
    image = tmp_path / "image.nii.gz"
    checkpoint = tmp_path / "weights.pth"
    image.touch(); checkpoint.touch()
    path = tmp_path / "config.json"
    cfg = shared_config(checkpoint, [{"id": "MR", "modality": "MR", "image": str(image)}])
    cfg["pca"] = {"output_dir": "crop"}
    path.write_text(json.dumps(cfg))
    loaded = views.setup(workflow="similarity", config_path=path)
    assert loaded["checkpoints"] == cfg["checkpoints"]
    assert loaded["device"] == "cpu"
    assert loaded["output_dir"] == str(tmp_path / "full")
    assert cfg["pca"]["output_dir"] == "crop"
    del cfg["similarity"]["output_dir"]
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="explicit output_dir"):
        views.setup(workflow="similarity", config_path=path)


def test_setup_rejects_aliases_of_the_same_output_directory(tmp_path):
    cfg = shared_config("unused.pth", [])
    cfg["pca"] = {"output_dir": "nested/../full"}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="separate output directories"):
        views.setup(workflow="similarity", config_path=path)


def test_setup_requires_the_selected_workflow_section(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"pca": {"output_dir": "crop"}}))
    with pytest.raises(ValueError, match="requires a similarity section"):
        views.setup(workflow="similarity", config_path=path)
    with pytest.raises(ValueError, match="workflow must be"):
        views.setup(workflow="attention", config_path=path)


def test_setup_validates_only_selected_inputs_and_preserves_device_override(tmp_path):
    image = tmp_path / "image.nii.gz"; image.touch()
    checkpoint = tmp_path / "weights.pth"; checkpoint.touch()
    cfg = shared_config(checkpoint, [{"id": "MR", "modality": "MR", "image": "image.nii.gz"}],
                        device="cuda:1")
    cfg["pca"] = {"output_dir": "crop", "cases": [{"modality": "CT", "image": "missing.nii.gz"}]}
    path = tmp_path / "config.json"; path.write_text(json.dumps(cfg))
    resolved = views.setup(workflow="similarity", config_path=path)
    assert resolved["device"] == "cuda:1" and resolved["cases"][0]["mask"] is None
    with pytest.raises(ValueError, match="PCA requires one CT and one MR"):
        views.setup(workflow="pca", config_path=path)
    checkpoint.unlink()
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        views.setup(workflow="similarity", config_path=path)


def test_pca_setup_does_not_require_similarity_scans(tmp_path):
    image = tmp_path / "image.nii.gz"; image.touch()
    checkpoint = tmp_path / "weights.pth"; checkpoint.touch()
    cfg = shared_config(checkpoint, [{"id": "MR", "modality": "MR", "image": "missing.nii.gz"}])
    cfg["pca"] = {"output_dir": "crop", "cases": [
        {"modality": modality, "image": "image.nii.gz"} for modality in ("CT", "MR")]}
    path = tmp_path / "config.json"; path.write_text(json.dumps(cfg))
    resolved = views.setup(workflow="pca", config_path=path)
    assert resolved["output_dir"] == str(tmp_path / "crop")
    assert all(case["image"] == str(image) for case in resolved["cases"])
