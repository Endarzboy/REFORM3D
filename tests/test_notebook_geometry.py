"""Shared-input freezing, 3D geometry, and model-independent notebook displays."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("notebook_views", ROOT / "notebooks/visualization_utils.py")
views = importlib.util.module_from_spec(spec)
spec.loader.exec_module(views)


def input_view(value=0., shape=(16, 16, 16)):
    volume = np.full(shape, value, np.float32)
    affine = np.diag([2., 2., 2., 1.]); affine[:3, 3] = [-20., -30., 7.]
    return {"volume": volume, "mask": np.zeros(shape, bool), "foreground": np.ones(shape, bool),
            "slice": shape[2] // 2, "metadata": {"affine_ras": affine.tolist(),
            "input_sha256": hashlib.sha256(volume.tobytes()).hexdigest()}}


@pytest.fixture
def small_plot_data(tmp_path):
    cfg = {"cases": [{"modality": "CT"}, {"modality": "MR"}], "output_dir": str(tmp_path)}
    arrays, cases = {}, {}
    rng = np.random.default_rng(7)
    for i in range(2):
        view = input_view(float(i))
        for m in range(3):
            suffix = f"{m}_{i}"
            cases[suffix] = copy.deepcopy(view["metadata"])
            for key in ("volume", "mask", "foreground"):
                arrays[f"{key}_{suffix}"] = view[key].copy()
            arrays[f"tokens_{suffix}"] = rng.normal(size=(8, 6))
            arrays[f"cls_{suffix}"] = np.ones(6)
            arrays[f"gap_{suffix}"] = np.ones(6)
            for key in ("attention", "patch_attention"):
                attention = rng.random((2, 2, 2, 2))
                attention /= attention.sum((1, 2, 3), keepdims=True)
                arrays[f"{key}_{suffix}"] = attention
    return cfg, arrays, {"cases": cases, "maps_sha256": "synthetic-test"}


def test_frozen_inputs_are_sampled_once_and_shared_by_all_models(tmp_path, monkeypatch):
    cfg = {"cases": [], "output_dir": str(tmp_path / "out"), "seed": 42, "crop": {"fov_mm": 192.}}
    for modality in ("CT", "MR"):
        path = tmp_path / modality; path.write_bytes(b"source")
        cfg["cases"].append({"modality": modality, "image": str(path), "mask": str(path)})
    calls = []
    def prepare(image, mask, **kwargs):
        calls.append(kwargs)
        return input_view(float(kwargs['seed']), (96,) * 3)
    monkeypatch.setattr(views.viz, 'prepare_shared_case', prepare)
    first = views.prepare_inputs(cfg)
    second = views.prepare_inputs(cfg)
    assert len(calls) == 2
    assert [c['seed'] for c in calls] == [42, 43]
    for i in range(2):
        for m in range(3):
            assert first[m, i] is first[0, i]
            np.testing.assert_array_equal(second[m, i]['volume'], first[0, i]['volume'])
    with pytest.raises(ValueError, match='Frozen inputs changed'):
        views.prepare_inputs({**cfg, 'seed': 43})
    with pytest.raises(ValueError, match='Frozen inputs changed'):
        views.prepare_inputs({**cfg, 'crop': {'fov_mm': 160.}})
    (Path(cfg['output_dir']) / 'inputs/inputs.npz').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='Frozen inputs changed'):
        views.prepare_inputs(cfg)
    with pytest.raises(ValueError, match='Frozen inputs changed'):
        views.extract_features(cfg)


def test_partial_input_archive_is_not_overwritten(tmp_path):
    cfg = {"cases": [], "output_dir": str(tmp_path)}
    folder = tmp_path / 'inputs'; folder.mkdir(); (folder / 'inputs.npz').write_bytes(b'partial')
    with pytest.raises(ValueError, match='Incomplete frozen inputs'):
        views.prepare_inputs(cfg)
    assert (folder / 'inputs.npz').read_bytes() == b'partial'


def test_curated_ct_center_is_forwarded_frozen_and_leaves_mr_defaults(tmp_path, monkeypatch):
    cfg = {"cases": [], "output_dir": str(tmp_path / "out"), "seed": 42}
    for modality in ("CT", "MR"):
        path = tmp_path / modality; path.write_bytes(b"source")
        cfg["cases"].append({"modality": modality, "image": str(path)})
    default_mr = views._crop_options(cfg)[1][1]
    cfg["cases"][0]["crop"] = {"fov_mm": 144., "center_ras_mm": [12, -34, 56]}
    calls = []
    def prepare(image, mask, **kwargs):
        calls.append(kwargs)
        return input_view(float(kwargs['seed']), (96,) * 3)
    monkeypatch.setattr(views.viz, 'prepare_shared_case', prepare)
    first, second = views.prepare_inputs(cfg), views.prepare_inputs(cfg)
    assert len(calls) == 2
    assert calls[0] == {"seed": 42, "size": 96, "fov_mm": 144.,
                        "center_ras_mm": [12., -34., 56.]}
    assert calls[1] == {"seed": 43, **default_mr}
    for m in range(3):
        np.testing.assert_array_equal(first[m, 0]['volume'], second[m, 0]['volume'])
    changed = copy.deepcopy(cfg)
    changed["cases"][0]["crop"]["center_ras_mm"][2] += 1
    with pytest.raises(ValueError, match='Frozen inputs changed'):
        views.prepare_inputs(changed)


@pytest.mark.parametrize('center', [None, [], [1, 2], [1, 2, 3, 4],
                                    [True, 0, 0], ['1', 2, 3], [0, float('nan'), 0],
                                    [0, 0, float('inf')], [[1], [2], [3]]])
def test_invalid_curated_centers_fail_before_sampling(center):
    cfg = {'cases': [{'modality': 'CT', 'crop': {'center_ras_mm': center}}]}
    with pytest.raises(ValueError, match='center_ras_mm'):
        views._crop_options(cfg)


def test_display_options_do_not_change_extraction_signature(monkeypatch):
    cfg = {'cases': [{'modality': 'CT'}, {'modality': 'MR'}]}
    before = views._preprocessing(cfg)
    monkeypatch.setattr(views, 'plot_features', lambda *a, **k: None)
    monkeypatch.setattr(views, 'select_axial_slices', lambda *a, **k: [])
    assert views._preprocessing(cfg) == before
    assert views._preprocessing({**cfg, 'seed': 43}) != before
    assert views._preprocessing({**cfg, 'crop': {'fov_mm': 160.}}) != before
    assert views._preprocessing({**cfg, 'crop': {'center_ras_mm': [12., -34., 56.]}}) != before


def test_axial_reslice_preserves_physical_z_and_radiological_direction():
    view = input_view()
    x, y, z = np.indices(view['volume'].shape)
    physical_z = (7 + 2 * z).astype(np.float32)
    plane, extent = views._axial_plane(physical_z, view['metadata'], z_mm=19.)
    np.testing.assert_allclose(plane, 19.)
    horizontal, _ = views._axial_plane(x.astype(np.float32), view['metadata'], z_mm=19.)
    assert horizontal[0, 0] == 15 and horizontal[0, -1] == 0
    assert extent[0] > extent[1] and extent[2] < extent[3]
    rgb = np.repeat(physical_z[..., None], 3, axis=-1)
    color_plane, _ = views._axial_plane(rgb, view['metadata'], z_mm=19.)
    np.testing.assert_array_equal(color_plane[..., 0], plane)


def test_slice_selection_uses_foreground_and_ignores_annotations_and_features(small_plot_data):
    cfg, arrays, receipt = small_plot_data
    for key in arrays:
        if key.startswith('foreground_'):
            arrays[key][:] = False; arrays[key][2:14, 2:14, 4:12] = True
    selection = views.select_axial_slices(cfg, arrays, receipt, 5)
    for s in selection:
        assert s['indices'] == [4, 6, 8, 9, 11]
        assert len(s['z_mm']) == 5 and np.all(np.diff(s['z_mm']) > 0)
    for key in arrays:
        if key.startswith(('mask_', 'attention_', 'tokens_', 'patch_attention_')):
            arrays[key][...] = 1
    assert views.select_axial_slices(cfg, arrays, receipt, 5) == selection
    assert len(views.select_axial_slices(cfg, arrays, receipt, 1)[0]['z_mm']) == 1


@pytest.mark.parametrize('count', [0, 6, -1, 2.5, True])
def test_slice_count_fails_explicitly(count):
    with pytest.raises(ValueError, match='num_slices'):
        views.select_axial_slices({}, {}, {}, count)


def test_slice_selection_rejects_different_inputs_and_empty_foreground(small_plot_data):
    cfg, arrays, receipt = small_plot_data
    arrays['volume_2_0'][0, 0, 0] = 3
    with pytest.raises(ValueError, match='identical inputs'):
        views.select_axial_slices(cfg, arrays, receipt)
    arrays['volume_2_0'][0, 0, 0] = arrays['volume_0_0'][0, 0, 0]
    for key in arrays:
        if key.startswith('foreground_'): arrays[key][:] = False
    with pytest.raises(ValueError, match='no anatomy'):
        views.select_axial_slices(cfg, arrays, receipt)


def test_pca_context_fusion_is_normalized_and_not_a_constant_shift():
    rng = np.random.default_rng(5)
    tokens, cls = rng.normal(size=(24, 8)), rng.normal(size=8)
    patch = tokens / np.linalg.norm(tokens, axis=1, keepdims=True)
    fused = views.contextual_tokens(tokens, cls, .5)
    np.testing.assert_allclose(np.linalg.norm(fused, axis=1), 1)
    np.testing.assert_allclose(views.contextual_tokens(tokens, cls, 0), patch)
    assert not np.allclose(fused - fused.mean(0), patch - patch.mean(0))
    with pytest.raises(ValueError, match='no spatial variation'):
        views.contextual_tokens(tokens, cls, 1)


def test_simple_pca_plot_call_returns_separate_four_column_figures(small_plot_data, monkeypatch):
    import matplotlib.pyplot as plt
    cfg, arrays, receipt = small_plot_data
    monkeypatch.setattr(views, 'load_features', lambda c: (arrays, receipt))
    figures = views.plot_features(cfg, arrays, kind="pca", num_slices=1)
    assert len(figures) == 2
    for fig in figures:
        assert 'patch features' in fig._suptitle.get_text()
        assert [ax.get_title() for ax in fig.axes[:4]] == ['Shared input', 'ISO', 'REL', 'SA']
        plt.close(fig)


def test_implicit_receipt_rejects_arrays_from_another_cache(small_plot_data, monkeypatch):
    cfg, arrays, receipt = small_plot_data
    monkeypatch.setattr(views, 'load_features', lambda c: (arrays, receipt))
    changed = dict(arrays); changed['tokens_0_0'] = arrays['tokens_0_0'] + 1
    with pytest.raises(ValueError, match='do not match this cache'):
        views.plot_features(cfg, changed, kind='pca')


@pytest.mark.parametrize('kwargs', [
    {'kind': 'attention'}, {'mode': 'cls'}, {'mode': 'unknown'},
    {'interpolation': 'cubic'}, {'mode': 'combined', 'cls_weight': -0.1},
    {'mode': 'combined', 'cls_weight': 1.0},
])
def test_invalid_pca_display_options_fail_clearly(small_plot_data, kwargs):
    cfg, arrays, receipt = small_plot_data
    with pytest.raises(ValueError):
        views.plot_features(cfg, arrays, receipt, **kwargs)


@pytest.mark.parametrize('mode', ['patch', 'combined'])
def test_pca_rgb_planes_use_one_3d_basis_and_ignore_ground_truth(small_plot_data, mode):
    import matplotlib.pyplot as plt

    cfg, arrays, receipt = small_plot_data
    tokens_before = {key: value.copy() for key, value in arrays.items()
                     if key.startswith(('tokens_', 'cls_'))}
    figures = views.plot_features(cfg, arrays, receipt, mode=mode,
                                  interpolation='nearest', num_slices=3)
    suffix = f'pca_{mode}_nearest_slices3' + ('_cls0.5' if mode == 'combined' else '')
    record = json.loads((Path(cfg['output_dir']) / f'{suffix}_display.json').read_text())
    with np.load(Path(cfg['output_dir']) / f'{suffix}_bases.npz') as data:
        fitted = {key: value.copy() for key, value in data.items()}
    displayed = []
    for i, figure in enumerate(figures):
        for m in range(3):
            key = f'{m}_{i}'
            tokens = arrays[f'tokens_{key}']
            if mode == 'combined':
                tokens = views.contextual_tokens(tokens, arrays[f'cls_{key}'], .5)
            colors, basis = views.viz.pca_colors(
                [tokens], [arrays[f'foreground_{key}']], [(2, 2, 2)],
            )
            for field, value in basis.items():
                np.testing.assert_allclose(fitted[f'{key}_{field}'], value)
            # Nearest interpolation replicates each native patch in XYZ order.
            rgb = colors[0]
            for axis in range(3):
                rgb = np.repeat(rgb, 8, axis=axis)
            for row, index in enumerate(record['slice_selection'][i]['indices']):
                pixels = np.asarray(figure.axes[4 * row + m + 1].images[-1].get_array())
                expected = rgb[:, :, index].swapaxes(0, 1)[:, ::-1]
                np.testing.assert_allclose(pixels, expected, atol=1e-6)
                displayed.append(pixels.copy())
        plt.close(figure)

    # Annotation contours and cached attention values cannot alter PCA colors.
    for key, value in arrays.items():
        if key.startswith('mask_'):
            value[3:10, 3:10, :] = True
        elif key.startswith(('attention_', 'patch_attention_')):
            value[:] = 0
    changed = views.plot_features(cfg, arrays, receipt, mode=mode,
                                 interpolation='nearest', num_slices=3)
    after = []
    for figure in changed:
        for m in range(3):
            for row in range(3):
                after.append(np.asarray(figure.axes[4 * row + m + 1].images[-1].get_array()))
        plt.close(figure)
    for before, current in zip(displayed, after):
        np.testing.assert_array_equal(current, before)
    with np.load(Path(cfg['output_dir']) / f'{suffix}_bases.npz') as data:
        for key, value in fitted.items():
            np.testing.assert_array_equal(data[key], value)
    for key, value in tokens_before.items():
        np.testing.assert_array_equal(arrays[key], value)
