"""Whole-volume sampling preserves anatomy and physical geometry."""
import hashlib

import nibabel as nib
import numpy as np
import pytest

from reform3d.full_volume import prepare_full_volume, read_source


def save_case(tmp_path, shape=(24, 20, 16), affine=None):
    affine = np.diag([1.5, 1., 2., 1.]) if affine is None else affine
    xyz = np.indices(shape)
    image = (2 * xyz[0] + xyz[1] + 3 * xyz[2]).astype(np.float32)
    labels = np.zeros(shape, np.uint8)
    labels[6:18, 5:15, 4:12] = 2
    image_path, mask_path = tmp_path / "image.nii.gz", tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(image, affine), image_path)
    nib.save(nib.Nifti1Image(labels, affine), mask_path)
    return image_path, mask_path, image, labels


def corners(shape, affine):
    points = np.indices((2, 2, 2)).reshape(3, -1) * np.asarray(shape)[:, None] - .5
    return affine[:3, :3] @ points + affine[:3, 3, None]


def test_native_source_reorientation_preserves_values_labels_and_physical_coordinates(tmp_path):
    affine = np.array([[0, 0, -2, 15], [0, -1., 0, 20], [-1.5, 0, 0, 30], [0, 0, 0, 1.]])
    image_path, mask_path, raw, labels = save_case(tmp_path, affine=affine)
    original_image = hashlib.sha256(image_path.read_bytes()).hexdigest()
    source = read_source(image_path, mask_path)
    assert nib.aff2axcodes(source["affine_ras"]) == ("R", "A", "S")
    assert source["volume"].dtype == np.float32
    assert source["mask"].dtype == bool
    np.testing.assert_array_equal(np.sort(source["volume"].ravel()), np.sort(raw.ravel()))
    assert source["mask"].sum() == np.count_nonzero(labels)
    original_corners = corners(raw.shape, affine)
    canonical_corners = corners(source["volume"].shape, source["affine_ras"])
    np.testing.assert_allclose(canonical_corners.min(1), original_corners.min(1))
    np.testing.assert_allclose(canonical_corners.max(1), original_corners.max(1))
    assert hashlib.sha256(image_path.read_bytes()).hexdigest() == original_image


@pytest.mark.parametrize("affine", [
    np.diag([1.5, 1., 2., 1.]),
    np.array([[1.3, -.4, .1, 31], [.75, .69, 0, -22], [0, .2, 2., 17], [0, 0, 0, 1.]]),
    np.array([[0, 0, -2., 15], [0, -1., 0, 20], [-1.5, 0, 0, 30], [0, 0, 0, 1.]]),
])
def test_model_box_covers_all_source_voxel_cells_without_aspect_distortion(tmp_path, affine):
    image_path, mask_path, raw, _ = save_case(tmp_path, shape=(48, 20, 16), affine=affine)
    result = prepare_full_volume(image_path, mask_path, max_axis=64)
    metadata = result["metadata"]
    output_affine = np.asarray(metadata["affine_ras"])
    spacing = np.diag(output_affine)[:3]
    assert np.all(spacing == spacing[0])
    assert np.all(np.array(result["volume"].shape) % 16 == 0)
    assert max(result["volume"].shape) == 64
    assert len(set(result["volume"].shape)) > 1  # No forced cubic stretching.
    source_corners = corners(raw.shape, nib.load(image_path).affine)
    model_corners = corners(result["volume"].shape, output_affine)
    np.testing.assert_allclose(source_corners.mean(1), model_corners.mean(1), atol=1e-6)
    assert np.all(model_corners.min(1) <= source_corners.min(1) + 1e-6)
    assert np.all(model_corners.max(1) >= source_corners.max(1) - 1e-6)
    assert metadata["no_crop"] is True
    assert metadata["grid_shape"] == (np.array(result["volume"].shape) // 16).tolist()
    assert metadata["input_sha256"] == hashlib.sha256(result["volume"].tobytes()).hexdigest()
    assert result["volume"].dtype == np.float32 and result["volume"].flags.c_contiguous
    assert -1 <= result["volume"].min() <= result["volume"].max() <= 1


def test_labels_and_their_absence_cannot_select_image_geometry_or_features(tmp_path):
    image_path, mask_path, _, _ = save_case(tmp_path)
    annotated = prepare_full_volume(image_path, mask_path, max_axis=32)
    unannotated = prepare_full_volume(image_path, max_axis=32)
    for key in ("volume", "foreground"):
        np.testing.assert_array_equal(annotated[key], unannotated[key])
    assert annotated["metadata"]["input_sha256"] == unannotated["metadata"]["input_sha256"]
    np.testing.assert_array_equal(annotated["metadata"]["affine_ras"],
                                  unannotated["metadata"]["affine_ras"])
    assert annotated["mask"].dtype == bool and annotated["mask"].any()
    assert not unannotated["mask"].any()


def test_nearest_neighbor_labels_follow_the_same_physical_sampling_grid(tmp_path):
    from scipy.ndimage import affine_transform

    image_path, mask_path, _, _ = save_case(tmp_path)
    result = prepare_full_volume(image_path, mask_path, max_axis=32)
    source = read_source(image_path, mask_path)
    transform = np.linalg.solve(source["affine_ras"], result["metadata"]["affine_ras"])
    expected = affine_transform(source["mask"].astype(np.uint8), transform[:3, :3], transform[:3, 3],
                                output_shape=result["volume"].shape, order=0,
                                mode="constant", cval=0, prefilter=False).astype(bool)
    np.testing.assert_array_equal(result["mask"], expected)


def test_same_physical_reference_survives_uncropping(tmp_path):
    image_path, _, _, _ = save_case(tmp_path)
    result = prepare_full_volume(image_path, max_axis=32)
    source = read_source(image_path)
    physical = source["affine_ras"] @ np.array([12., 10., 8., 1.])
    model_voxel = np.linalg.solve(np.asarray(result["metadata"]["affine_ras"]), physical)
    assert np.all(model_voxel[:3] >= 0)
    assert np.all(model_voxel[:3] < result["volume"].shape)
    np.testing.assert_allclose(np.asarray(result["metadata"]["affine_ras"]) @ model_voxel, physical)


def test_antialiasing_is_recorded_and_repeated_runs_match(tmp_path):
    image_path, mask_path, _, _ = save_case(tmp_path, shape=(64, 48, 40))
    first = prepare_full_volume(image_path, mask_path, max_axis=32)
    second = prepare_full_volume(image_path, mask_path, max_axis=32)
    assert any(s > 0 for s in first["metadata"]["antialias_sigma_source_vox"])
    np.testing.assert_array_equal(first["volume"], second["volume"])
    assert first["metadata"] == second["metadata"]


@pytest.mark.parametrize("options", [
    {"max_axis": 0}, {"max_axis": -16}, {"max_axis": 31}, {"max_axis": True},
    {"max_axis": 32.}, {"patch_size": 0}, {"patch_size": True}, {"patch_size": 2.},
    {"max_patches": 0}, {"max_patches": True}, {"max_patches": 4.5},
])
def test_invalid_model_grid_settings_fail_explicitly(tmp_path, options):
    with pytest.raises(ValueError):
        prepare_full_volume(tmp_path / "unused.nii.gz", **options)


def test_patch_budget_is_enforced_before_model_allocation(tmp_path):
    image_path, _, _, _ = save_case(tmp_path)
    with pytest.raises(ValueError, match="max_patches"):
        prepare_full_volume(image_path, max_axis=32, max_patches=1)


@pytest.mark.parametrize("bad_kind", ["singular", "nan", "homogeneous"])
def test_invalid_source_affines_fail_before_reorientation(tmp_path, monkeypatch, bad_kind):
    from reform3d import full_volume

    image_path, _, _, _ = save_case(tmp_path)
    loaded = nib.load(image_path)
    affine = loaded.affine.copy()
    if bad_kind == "singular":
        affine[:, 0] = 0
    elif bad_kind == "nan":
        affine[0, 0] = np.nan
    else:
        affine[3, 0] = 1
    loaded._affine = affine
    monkeypatch.setattr(full_volume.nib, "load", lambda path: loaded)
    with pytest.raises(ValueError, match="affine"):
        read_source(image_path)


@pytest.mark.parametrize("bad_kind", ["shape", "affine", "negative", "nan"])
def test_mask_geometry_and_labels_fail_explicitly(tmp_path, bad_kind):
    image_path, mask_path, _, labels = save_case(tmp_path)
    affine = np.diag([1.5, 1., 2., 1.])
    data = labels.astype(np.float32)
    if bad_kind == "shape":
        data = data[:-1]
    elif bad_kind == "affine":
        affine[0, 3] += 2
    else:
        data[0, 0, 0] = -1 if bad_kind == "negative" else np.nan
    nib.save(nib.Nifti1Image(data, affine), mask_path)
    with pytest.raises(ValueError, match="geometry|labels"):
        read_source(image_path, mask_path)


@pytest.mark.parametrize("bad_kind", ["nan", "constant", "four_dimensional", "complex"])
def test_invalid_source_images_fail_explicitly(tmp_path, bad_kind):
    data = np.arange(8**3, dtype=np.float32).reshape((8,) * 3)
    if bad_kind == "nan":
        data[0, 0, 0] = np.nan
    elif bad_kind == "constant":
        data.fill(1)
    elif bad_kind == "four_dimensional":
        data = data[..., None]
    else:
        data = data.astype(np.complex64)
    path = tmp_path / "invalid.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    with pytest.raises(ValueError, match="finite|degenerate|3D|Complex"):
        prepare_full_volume(path, max_axis=16)
