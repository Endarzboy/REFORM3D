"""Qualitative examples must preserve spatial alignment and display semantics."""
import json
import numpy as np
import pytest
import SimpleITK as sitk
import torch

from reform3d.visualization import prepare_case, prepare_shared_case, spatial_maps, pca_colors, load_config, MODELS
from reform3d.visualization import spatial_features, attention_summaries
from reform3d.encoders.med3dino.vision_transformer import DinoVisionTransformer3d


def write_pair(tmp_path, *, empty=False, mismatch=False):
    shape = (24, 28, 32)
    image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    mask = np.zeros(shape, dtype=np.uint8)
    if not empty:
        mask[6:18, 8:20, 10:22] = 1
    paths = []
    for name, array in [('image', image), ('mask', mask)]:
        volume = sitk.GetImageFromArray(array)
        volume.SetSpacing((1., 1.5, 2.))
        if mismatch and name == 'mask':
            volume.SetOrigin((3., 0., 0.))
        path = tmp_path / (name + '.nii.gz')
        sitk.WriteImage(volume, str(path))
        paths.append(path)
    return paths


def test_physical_crop_preserves_annotation_and_normalization(tmp_path):
    case = prepare_case(*write_pair(tmp_path), size=32, padding_mm=4, min_fov_mm=32)
    assert case['volume'].shape == case['mask'].shape == (32, 32, 32)
    assert case['volume'].dtype == np.float32
    assert -1 <= case['volume'].min() <= case['volume'].max() <= 1
    assert case['mask'].any()
    assert case['slice'] == int(case['mask'].sum(axis=(0, 1)).argmax())
    assert case['metadata']['fov_mm'] >= 32
    # Annotations may select the view but not PCA's intensity foreground.
    np.testing.assert_array_equal(case['foreground'], case['volume'] > -.95)


@pytest.mark.parametrize('empty,mismatch,match', [(True, False, 'empty'), (False, True, 'geometry')])
def test_invalid_annotation_fails(tmp_path, empty, mismatch, match):
    with pytest.raises(ValueError, match=match):
        prepare_case(*write_pair(tmp_path, empty=empty, mismatch=mismatch))


def write_foreground_case(tmp_path):
    z, y, x = np.indices((64, 64, 64))
    foreground = ((x - 32) ** 2 + (y - 32) ** 2 + (z - 32) ** 2) < 23 ** 2
    raw = np.where(foreground, 20 + x + y / 2 + z / 3, -1000).astype(np.float32)
    im = sitk.GetImageFromArray(raw)
    im.SetSpacing((1.5,) * 3)
    path = tmp_path / 'foreground.nii.gz'
    sitk.WriteImage(im, str(path))
    return path, im


def test_shared_crop_is_seeded_and_annotation_independent(tmp_path):
    import hashlib
    image, im = write_foreground_case(tmp_path)
    masks = []
    for index in range(2):
        mask = np.zeros((64, 64, 64), np.uint8)
        if index:
            mask[24:40, 24:40, 24:40] = 1
        seg = sitk.GetImageFromArray(mask); seg.CopyInformation(im)
        path = tmp_path / f'reference{index}.nii.gz'; sitk.WriteImage(seg, str(path))
        masks.append(path)
    cases = [prepare_shared_case(image, path, seed=17, fov_mm=36, size=32) for path in [None, *masks]]
    for case in cases:
        assert case['volume'].shape == case['mask'].shape == case['foreground'].shape == (32,) * 3
        assert case['volume'].dtype == np.float32
        assert case['mask'].dtype == case['foreground'].dtype == bool
        assert -1 <= case['volume'].min() <= case['volume'].max() <= 1
        assert case['metadata']['input_sha256'] == hashlib.sha256(case['volume'].tobytes()).hexdigest()
        np.testing.assert_array_equal(case['volume'], cases[0]['volume'])
        np.testing.assert_array_equal(case['foreground'], cases[0]['foreground'])
        np.testing.assert_array_equal(case['metadata']['affine_ras'], cases[0]['metadata']['affine_ras'])
    assert not cases[0]['mask'].any() and not cases[1]['mask'].any()
    different = prepare_shared_case(image, seed=18, fov_mm=36, size=32)
    assert different['metadata']['input_sha256'] != cases[0]['metadata']['input_sha256']


@pytest.mark.parametrize('explicit_center', [False, True])
def test_shared_crop_oblique_source_preserves_ras_geometry_and_labels(tmp_path, explicit_center):
    z, y, x = np.indices((48, 56, 64))
    raw = (x + 3 * y + 7 * z).astype(np.float32)
    annotation = ((x > 14) & (x < 45) & (y > 12) & (y < 44) & (z > 8) & (z < 36)).astype(np.uint8)
    angle = np.deg2rad(30)
    direction = [np.cos(angle), -np.sin(angle), 0., np.sin(angle), np.cos(angle), 0., 0., 0., 1.]
    paths = []
    for name, values in [('oblique', raw), ('oblique-mask', annotation)]:
        volume = sitk.GetImageFromArray(values)
        volume.SetDirection(direction); volume.SetSpacing((1., 1.5, 2.)); volume.SetOrigin((20., -10., 30.))
        path = tmp_path / f'{name}.nii.gz'; sitk.WriteImage(volume, str(path)); paths.append(path)
    center = np.asarray(volume.TransformContinuousIndexToPhysicalPoint((32., 28., 24.))) * [-1, -1, 1]
    kwargs = {'center_ras_mm': center} if explicit_center else {}
    case = prepare_shared_case(*paths, seed=8, fov_mm=24, size=32, **kwargs)
    if explicit_center:
        np.testing.assert_array_equal(case['metadata']['crop_center_ras_mm'], center)
        assert case['metadata']['center_selection'] == 'explicit RAS center'
    affine = np.asarray(case['metadata']['affine_ras'])
    np.testing.assert_allclose(affine[:3, :3], np.eye(3) * .75)
    np.testing.assert_allclose(affine @ np.array([15.5, 15.5, 15.5, 1]), [*case['metadata']['crop_center_ras_mm'], 1])
    indices = np.asarray(np.meshgrid(*([np.arange(32)] * 3), indexing='ij'))
    ras = np.einsum('ij,j...->i...', affine[:3, :3], indices) + affine[:3, 3, None, None, None]
    inverse = np.linalg.inv(np.asarray(case['metadata']['source_affine_ras']))
    source = np.einsum('ij,j...->i...', inverse[:3, :3], ras) + inverse[:3, 3, None, None, None]
    assert source.min() >= -1e-6
    assert np.all(source <= np.array([63, 55, 47])[:, None, None, None] + 1e-6)
    low, high = case['metadata']['normalization']['bounds']
    expected = (np.clip(source[0] + 3 * source[1] + 7 * source[2], low, high) - low) / (high - low) * 2 - 1
    interior = np.all((source >= 1e-6) & (source <= np.array([63, 55, 47])[:, None, None, None] - 1e-6), axis=0)
    np.testing.assert_allclose(case['volume'][interior], expected[interior], atol=5e-7)
    nearest = np.rint(source).astype(int)
    expected_mask = annotation[nearest[2], nearest[1], nearest[0]] > 0
    np.testing.assert_array_equal(case['mask'][interior], expected_mask[interior])


def test_explicit_shared_center_uses_same_input_for_every_seed(tmp_path):
    image, _ = write_foreground_case(tmp_path)
    center = [-48., -48., 48.]
    cases = [prepare_shared_case(image, seed=seed, fov_mm=36, size=32, center_ras_mm=center)
             for seed in [0, 42, 123]]
    for case in cases:
        np.testing.assert_array_equal(case['volume'], cases[0]['volume'])
        np.testing.assert_array_equal(case['foreground'], cases[0]['foreground'])
        assert case['metadata']['input_sha256'] == cases[0]['metadata']['input_sha256']
        affine = np.asarray(case['metadata']['affine_ras'])
        np.testing.assert_allclose(affine @ [15.5, 15.5, 15.5, 1], [*center, 1])
    default = prepare_shared_case(image, seed=42, fov_mm=36, size=32)
    explicit_none = prepare_shared_case(image, seed=42, fov_mm=36, size=32, center_ras_mm=None)
    np.testing.assert_array_equal(default['volume'], explicit_none['volume'])
    assert default['metadata']['center_selection'] == 'seeded image-foreground sampling'


def test_explicit_shared_center_preserves_requested_center_with_padding(tmp_path):
    image, _ = write_foreground_case(tmp_path)
    center = [-18., -48., 48.]
    case = prepare_shared_case(image, fov_mm=72, size=32, center_ras_mm=center)
    np.testing.assert_array_equal(case['metadata']['crop_center_ras_mm'], center)
    affine = np.asarray(case['metadata']['affine_ras'])
    np.testing.assert_allclose(affine[:3, :3], np.eye(3) * 2.25)
    np.testing.assert_allclose(affine @ [15.5, 15.5, 15.5, 1], [*center, 1])
    assert 0 < case['metadata']['source_coverage_fraction'] < 1
    assert case['foreground'].any()
    # The high-R edge lies beyond source x=0 and keeps the requested padding.
    np.testing.assert_array_equal(case['volume'][-1], -np.ones((32, 32), np.float32))
    assert not case['foreground'][-1].any()


@pytest.mark.parametrize('center', [[], [1, 2], [[1, 2, 3]], [1, 2, float('nan')],
                                   [1, 2, float('inf')], ['left', 2, 3], 4])
def test_explicit_shared_center_rejects_invalid_coordinates(tmp_path, center):
    image, _ = write_foreground_case(tmp_path)
    with pytest.raises(ValueError, match='RAS center'):
        prepare_shared_case(image, center_ras_mm=center)


def test_shared_crop_reports_unavoidable_padding(tmp_path):
    image, _ = write_foreground_case(tmp_path)
    case = prepare_shared_case(image, fov_mm=160, size=32)
    assert 0 < case['metadata']['source_coverage_fraction'] < .5
    assert case['foreground'].any()
    assert np.isfinite(case['volume']).all()
    assert np.count_nonzero(case['volume'] == -1) > case['volume'].size / 2


def test_shared_crop_fails_if_source_foreground_disappears_when_resampled(tmp_path):
    raw = np.full((64, 64, 64), -1000, dtype=np.float32)
    raw[32, 24:40, 24:40] = 100
    image = tmp_path / 'thin-foreground.nii.gz'
    sitk.WriteImage(sitk.GetImageFromArray(raw), str(image))
    with pytest.raises(ValueError, match='foreground disappeared'):
        prepare_shared_case(image, fov_mm=512, size=32)


@pytest.mark.parametrize('kwargs', [{'seed': -1}, {'seed': True}, {'fov_mm': 0}, {'fov_mm': float('nan')}, {'size': 31}])
def test_shared_crop_rejects_invalid_parameters(tmp_path, kwargs):
    image, _ = write_foreground_case(tmp_path)
    with pytest.raises(ValueError):
        prepare_shared_case(image, **kwargs)


@pytest.mark.parametrize('chunks', [0, 1, 2])
def test_patch_tokens_and_attention_use_same_spatial_grid(chunks):
    torch.manual_seed(7)
    model = DinoVisionTransformer3d(img_size=16, patch_size=4, in_chans=1,
        embed_dim=24, depth=2, num_heads=3, block_chunks=chunks, init_values=1e-5).eval()
    volume = torch.randn(1, 1, 16, 16, 16)
    tokens, attention = spatial_maps(model, volume)
    assert tokens.shape == (64, 24)
    assert attention.shape == (3, 4, 4, 4)
    np.testing.assert_allclose(attention.sum(axis=(1, 2, 3)), 1, atol=1e-6)
    with torch.inference_mode():
        expected = model.forward_features(volume)['x_norm_patchtokens'][0].numpy()
    np.testing.assert_array_equal(tokens, expected)
    with pytest.raises(ValueError, match='Expected'):
        spatial_maps(model, volume[0])


@pytest.mark.parametrize('chunks', [0, 1, 2])
def test_cls_and_mean_patch_attention_match_the_full_matrix(chunks):
    torch.manual_seed(17)
    model = DinoVisionTransformer3d(img_size=16, patch_size=4, in_chans=1,
        embed_dim=24, depth=2, num_heads=3, block_chunks=chunks, init_values=1e-5).eval()
    volume = torch.randn(1, 1, 16, 16, 16)
    result = spatial_features(model, volume)
    with torch.inference_mode():
        features = model.forward_features(volume)
        full = model.get_self_attention(volume, reshape=False)[0]
    rows = full[..., 1:]
    rows = rows / rows.sum(-1, keepdim=True)
    np.testing.assert_allclose(result['attention'], rows[:, 0].reshape(3, 4, 4, 4), atol=1e-7)
    np.testing.assert_allclose(result['patch_attention'], rows[:, 1:].mean(1).reshape(3, 4, 4, 4), atol=1e-7)
    np.testing.assert_array_equal(result['cls'], features['x_norm_clstoken'][0])
    np.testing.assert_array_equal(result['gap'], features['x_norm_patchtokens'][0].mean(0))
    for name in ['attention', 'patch_attention']:
        np.testing.assert_allclose(result[name].sum((1, 2, 3)), 1, atol=1e-6)
    with pytest.raises(ValueError, match='mass'):
        attention_summaries(torch.zeros(1, 3, 65, 65), (4, 4, 4))


def test_pca_uses_one_basis_for_identical_features_across_cases():
    rng = np.random.default_rng(4)
    values = rng.normal(size=(8, 6)).astype(np.float32)
    foreground = np.ones((4, 4, 4), bool)
    colors, basis = pca_colors([values, values], [foreground, foreground], [(2, 2, 2)] * 2)
    np.testing.assert_array_equal(colors[0], colors[1])
    assert colors[0].shape == (2, 2, 2, 3)
    assert 0 <= colors[0].min() <= colors[0].max() <= 1
    assert basis['components'].shape == (3, 6)
    with pytest.raises(ValueError, match='foreground'):
        pca_colors([values], [np.zeros_like(foreground)], [(2, 2, 2)])
    with pytest.raises(ValueError, match='norm'):
        pca_colors([values * 0], [foreground], [(2, 2, 2)])
    with pytest.raises(ValueError, match='matching'):
        pca_colors([values], [], [])


def test_config_resolves_relative_paths_and_requires_ct_mr(tmp_path):
    cfg = {'checkpoints': {name: f'{name}.pth' for name in MODELS},
           'cases': [{'modality': m, 'image': 'image.nii.gz', 'mask': 'mask.nii.gz'} for m in ['CT','MR']],
           'output_dir': 'results'}
    p = tmp_path / 'config.json'; p.write_text(json.dumps(cfg))
    actual = load_config(p)
    assert actual['cases'][0]['image'] == str(tmp_path / 'image.nii.gz')
    cfg['cases'][1]['modality'] = 'CT'; p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='CT'):
        load_config(p)


def test_config_allows_reference_masks_to_be_omitted(tmp_path):
    cfg = {'checkpoints': {name: f'{name}.pth' for name in MODELS},
           'cases': [{'modality': m, 'image': 'image.nii.gz'} for m in ['CT', 'MR']],
           'output_dir': 'results'}
    path = tmp_path / 'config.json'; path.write_text(json.dumps(cfg))
    assert all(case['mask'] is None for case in load_config(path)['cases'])


def test_shared_input_integrity_rejects_different_tensors_or_geometry(tmp_path):
    import copy
    import hashlib
    from reform3d.visualization import _validate_shared_views
    image, _ = write_foreground_case(tmp_path)
    case = prepare_shared_case(image, fov_mm=36, size=32)
    views = {(m, 0): copy.deepcopy(case) for m in range(3)}
    _validate_shared_views(views, 1)
    views[2, 0]['volume'][16, 16, 16] += .1
    with pytest.raises(ValueError, match='hash'):
        _validate_shared_views(views, 1)
    views[2, 0]['metadata']['input_sha256'] = hashlib.sha256(views[2, 0]['volume'].tobytes()).hexdigest()
    with pytest.raises(ValueError, match='identical'):
        _validate_shared_views(views, 1)
    views[2, 0] = copy.deepcopy(case)
    views[2, 0]['metadata']['affine_ras'][0][3] += 1
    with pytest.raises(ValueError, match='affine'):
        _validate_shared_views(views, 1)


def test_shared_collect_accepts_frozen_inputs_but_rejects_unrelated_files(tmp_path):
    from reform3d.visualization import collect_maps
    cfg = {'device': 'cpu', 'cases': [], 'checkpoints': {}, 'output_dir': str(tmp_path)}
    (tmp_path / 'inputs').mkdir()
    def prepare(cfg):
        raise RuntimeError('frozen input loader was called')
    preprocessing = {'method': 'ct-mr-shared-crop-v1'}
    with pytest.raises(RuntimeError, match='frozen input loader'):
        collect_maps(cfg, prepare_views=prepare, preprocessing=preprocessing)
    (tmp_path / 'unrecognized.txt').write_text('partial output')
    with pytest.raises(ValueError, match='Incomplete'):
        collect_maps(cfg, prepare_views=prepare, preprocessing=preprocessing)


def test_loaded_shared_maps_recheck_tensor_identity_even_with_updated_archive_hash(tmp_path):
    import copy
    from reform3d.visualization import load_maps, _signature, _sha256
    image, _ = write_foreground_case(tmp_path)
    case = prepare_shared_case(image, fov_mm=36, size=32)
    out = tmp_path / 'shared'; out.mkdir()
    cfg = {'device': 'cpu', 'cases': [{'modality': 'CT', 'image': str(image), 'mask': None}],
           'checkpoints': {}, 'output_dir': str(out)}
    preprocessing = {'method': 'ct-mr-shared-crop-v1'}
    arrays, metadata = {}, {}
    for m in range(3):
        for key in ['volume', 'mask', 'foreground']:
            arrays[f'{key}_{m}_0'] = case[key].copy()
        arrays[f'tokens_{m}_0'] = np.ones((8, 4))
        arrays[f'attention_{m}_0'] = np.ones((2, 2, 2, 2)) / 8
        metadata[f'{m}_0'] = copy.deepcopy(case['metadata'])
    receipt = {'signature': _signature(cfg, preprocessing), 'cases': metadata,
               'shared_input_sha256': {'0': case['metadata']['input_sha256']}}
    def save():
        np.savez_compressed(out / 'maps.npz', **arrays)
        receipt['maps_sha256'] = _sha256(out / 'maps.npz')
        (out / 'receipt.json').write_text(json.dumps(receipt))
    save()
    load_maps(cfg, preprocessing=preprocessing)
    arrays['volume_2_0'][16, 16, 16] += .1
    save()
    with pytest.raises(ValueError, match='hash'):
        load_maps(cfg, preprocessing=preprocessing)


def test_completed_maps_detect_tampering_and_incomplete_keys(tmp_path):
    from reform3d.visualization import load_maps, _signature, _sha256
    cfg = {'device': 'cpu', 'cases': [], 'checkpoints': {}}
    for modality in ['CT', 'MR', 'US']:
        image = tmp_path / f'{modality}.nii.gz'; image.write_bytes(b'image')
        mask = tmp_path / f'{modality}-mask.nii.gz'; mask.write_bytes(b'mask')
        cfg['cases'].append({'modality': modality, 'image': str(image), 'mask': str(mask)})
    for name in MODELS:
        weight = tmp_path / f'{name}.pth'; weight.write_bytes(b'weights')
        cfg['checkpoints'][name] = str(weight)
    out = tmp_path / 'maps'; out.mkdir(); cfg['output_dir'] = str(out)
    arrays = {}
    for i in range(3):
        for key in ['volume', 'mask', 'foreground']:
            arrays[f'{key}_{i}'] = np.zeros((2, 2, 2))
        for m in range(3):
            arrays[f'tokens_{m}_{i}'] = np.ones((8, 4))
            arrays[f'attention_{m}_{i}'] = np.ones((2, 2, 2, 2)) / 8
    def save():
        np.savez_compressed(out / 'maps.npz', **arrays)
        (out / 'receipt.json').write_text(json.dumps({'signature': _signature(cfg), 'maps_sha256': _sha256(out / 'maps.npz')}))
    save()
    loaded, _ = load_maps(cfg)
    assert loaded.keys() == arrays.keys()
    (out / 'maps.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='changed'):
        load_maps(cfg)
    del arrays['tokens_2_2']; save()
    with pytest.raises(ValueError, match='Incomplete'):
        load_maps(cfg)


def test_collect_rejects_partial_directory_before_inference(tmp_path):
    from reform3d.visualization import collect_maps
    cfg = {'device': 'cpu', 'cases': [], 'checkpoints': {}, 'output_dir': str(tmp_path)}
    (tmp_path / 'partial.npz').write_bytes(b'partial')
    with pytest.raises(ValueError, match='Incomplete'):
        collect_maps(cfg)


def test_patch_flatten_order_preserves_spatial_marker():
    from reform3d.encoders.med3dino.layers.patch_embed3d import PatchEmbed3d
    embed = PatchEmbed3d(img_size=(4, 6, 8), patch_size=2, in_chans=1, embed_dim=1)
    with torch.no_grad():
        embed.proj.weight.fill_(1); embed.proj.bias.zero_()
        x = torch.zeros(1, 1, 4, 6, 8); x[0, 0, 2, 0, 6] = 1
        grid = embed(x)[0, :, 0].reshape(2, 3, 4).numpy()
    assert np.argwhere(grid).tolist() == [[1, 0, 3]]
