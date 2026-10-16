import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from reform3d.encoders.checkpoints import load_encoder_state
from reform3d.prepare import prepare_manifest
from reform3d.settings import load_settings
from reform3d.workspace import run_directory


def test_encoder_requires_every_backbone_weight():
    model = nn.Linear(3, 2)
    with pytest.raises(RuntimeError, match='missing'):
        load_encoder_state(model, {'weight': model.weight.clone()})


def test_encoder_only_ignores_named_unused_heads():
    model = nn.Linear(3, 2)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    state['dino_head.weight'] = torch.zeros(2, 2)
    with pytest.raises(RuntimeError, match='unexpected'):
        load_encoder_state(model, state)
    load_encoder_state(model, state, allowed_extra_prefixes=('dino_head.',))


def test_encoder_rejects_wrong_weight_shape():
    model = nn.Linear(3, 2)
    state = model.state_dict()
    state['weight'] = torch.zeros(3, 2)
    with pytest.raises(RuntimeError, match='size mismatch'):
        load_encoder_state(model, state)


def config_path(tmp_path, **changes):
    values = {'dataset': 'mmwhs_ct_mr', 'checkpoints': ['med3dino_rel_96'], 'manifest': 'manifest.json'}
    values.update(changes)
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(values))
    return path


@pytest.mark.parametrize('changes', [
    {'unknown_setting': True}, {'seeds': [42, 42]}, {'seeds': [True]},
    {'feature_types': ['invented']}, {'batch_size': 0},
    {'run_name': '../outside'}, {'modules': ['acquisition']},
])
def test_invalid_config_is_rejected(tmp_path, changes):
    with pytest.raises(ValueError):
        load_settings(config_path(tmp_path, **changes))


def test_config_paths_are_relative_to_config(tmp_path, monkeypatch):
    monkeypatch.chdir('/')
    cfg = load_settings(config_path(tmp_path, workspace='outputs'))
    assert Path(cfg['workspace']) == tmp_path / 'outputs'
    assert Path(cfg['manifest']) == tmp_path / 'manifest.json'
    assert cfg['not_evaluated'] == ['acquisition']


def organ_manifest(tmp_path, *, affine=None, missing_image=False):
    import nibabel as nib
    image = tmp_path / 'image.nii.gz'
    mask = tmp_path / 'mask.nii.gz'
    if not missing_image:
        nib.save(nib.Nifti1Image(np.ones((5, 5, 5)), np.eye(4)), image)
    nib.save(nib.Nifti1Image(np.ones((5, 5, 5)), np.eye(4) if affine is None else affine), mask)
    row = {'sample_id': 'ct:case:aorta', 'source_case_id': 'case', 'primary_organ': 'aorta', 'modality': 'ct', 'file_path': str(image), 'mask_path': str(mask), 'mask_label_value': 1}
    manifest = tmp_path / 'manifest.json'
    rows = []
    for modality in ['ct', 'mr']:
        for case in range(5):
            case_image = image if (modality, case) == ('ct', 0) else tmp_path / f'{modality}-{case}.nii.gz'
            if case_image != image:
                nib.save(nib.Nifti1Image(np.ones((5, 5, 5)), np.eye(4)), case_image)
            for organ in ['aorta', 'heart']:
                rows.append({**row, 'sample_id': f'{modality}:case{case}:{organ}', 'source_case_id': f'{modality}:case{case}', 'primary_organ': organ, 'modality': modality, 'file_path': str(case_image)})
    manifest.write_text(json.dumps({'samples': rows}))
    cfg = load_settings(config_path(tmp_path, workspace='outputs'))
    return cfg, manifest, row


def test_missing_input_fails_before_features(tmp_path):
    cfg, _, _ = organ_manifest(tmp_path, missing_image=True)
    with pytest.raises(FileNotFoundError, match='missing'):
        prepare_manifest(cfg)


def test_image_mask_affine_mismatch_is_rejected(tmp_path):
    affine = np.eye(4)
    affine[0, 3] = 2
    cfg, _, _ = organ_manifest(tmp_path, affine=affine)
    with pytest.raises(ValueError, match='geometry mismatch'):
        prepare_manifest(cfg)


def test_duplicate_sample_identity_is_rejected(tmp_path):
    cfg, manifest, row = organ_manifest(tmp_path)
    manifest.write_text(json.dumps({'samples': [row, row]}))
    with pytest.raises(ValueError, match='duplicate'):
        prepare_manifest(cfg)


def test_changed_input_invalidates_manifest_identity(tmp_path):
    import nibabel as nib
    cfg, _, _ = organ_manifest(tmp_path)
    first, _, first_id = prepare_manifest(cfg)
    nib.save(nib.Nifti1Image(np.full((5, 5, 5), 2.0), np.eye(4)), tmp_path / 'image.nii.gz')
    second, _, second_id = prepare_manifest(cfg)
    assert first_id != second_id
    assert first != second
    assert first.is_file() and second.is_file()


@pytest.mark.parametrize('ids,failures', [(['one'], []), (['one', 'one'], []), (['one', 'two', 'extra'], []), (['one', 'two'], [{'sample_id': 'one', 'reason': 'bad mask'}])])
def test_partial_or_ambiguous_extraction_cannot_succeed(tmp_path, ids, failures):
    from types import SimpleNamespace
    from reform3d.modules.cross_modal_alignment.feature_extractor import require_complete_extraction
    samples = [SimpleNamespace(sample_id='one'), SimpleNamespace(sample_id='two')]
    output = tmp_path / 'summary.json'
    with pytest.raises(RuntimeError, match='Incomplete'):
        require_complete_extraction(samples, ids, failures, output)
    summary = json.loads(output.read_text())
    assert summary['status'] == 'incomplete'
    assert summary['output_npz'] is None


@pytest.mark.parametrize('mutation', ['weights', 'seed'])
def test_resume_rejects_changed_run_identity(tmp_path, monkeypatch, mutation):
    from types import SimpleNamespace
    from reform3d import cli
    cfg, _, _ = organ_manifest(tmp_path)
    checkpoint = tmp_path / 'checkpoint.bin'
    checkpoint.write_bytes(b'first')
    monkeypatch.setattr(cli, '_checkpoints', lambda _: [SimpleNamespace(checkpoint_key='test', path=checkpoint)])
    for name in ['MED3DINO_CHECKPOINT_ROOT', 'MED3DINO_EXTERNAL_DATA_ROOT', 'MED3DINO_SHARED_MANIFEST_ROOT', 'MED3DINO_SHARED_CACHE_ROOT', 'MED3DINO_SHARED_RESULT_ROOT', 'REFORM3D_WORKSPACE', 'REFORM3D_RUN_DIR']:
        monkeypatch.setenv(name, '')
    cli.run(cfg, dry_run=True)
    if mutation == 'weights':
        checkpoint.write_bytes(b'second')
    else:
        cfg['seeds'] = [123]
    with pytest.raises(ValueError, match='different inputs/configuration/software'):
        cli.run(cfg, dry_run=True)


def test_check_preserves_completed_run_state(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from reform3d import cli
    cfg, _, _ = organ_manifest(tmp_path)
    checkpoint = tmp_path / 'checkpoint.bin'
    checkpoint.write_bytes(b'weights')
    monkeypatch.setattr(cli, '_checkpoints', lambda _: [SimpleNamespace(checkpoint_key='test', path=checkpoint)])
    cli.run(cfg, dry_run=True)
    path = run_directory(cfg) / 'run.json'
    state = json.loads(path.read_text())
    state['status'] = 'complete'
    state['completed'] = [{'module': 'anatomical', 'seed': 42}]
    path.write_text(json.dumps(state))
    before = path.read_bytes()
    cli.run(cfg, dry_run=True)
    assert path.read_bytes() == before


def test_unavailable_cuda_records_failed_state(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from reform3d import cli
    cfg, _, _ = organ_manifest(tmp_path)
    checkpoint = tmp_path / 'checkpoint.bin'
    checkpoint.write_bytes(b'weights')
    monkeypatch.setattr(cli, '_checkpoints', lambda _: [SimpleNamespace(checkpoint_key='test', path=checkpoint)])
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA'):
        cli.run(cfg)
    state = json.loads((run_directory(cfg) / 'run.json').read_text())
    assert state['status'] == 'failed'
    assert 'CUDA' in state['error']


def test_core_imports_do_not_require_original_checkout(tmp_path):
    import subprocess
    import sys
    code = '''
import importlib, importlib.abc, sys
class BlockOriginal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'dinov2', 'reform3d_eval'}:
            raise ImportError('Original project import forbidden: ' + fullname)
sys.meta_path.insert(0, BlockOriginal())
for name in ('acquisition_robustness.feature_extractor', 'cross_modal_alignment.evaluation_pipeline', 'anatomical_generalization.evaluation_pipeline'):
    importlib.import_module('reform3d.modules.' + name)
assert not any(n.split('.')[0] in {'dinov2', 'reform3d_eval'} for n in sys.modules)
'''
    result = subprocess.run([sys.executable, '-I', '-c', code], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('invalid', ['retrieval_status', 'lisi_nan', 'missing_direction'])
def test_report_rejects_incomplete_cross_modal_metrics(tmp_path, invalid):
    from reform3d.cli import read_metrics
    direction = {'status': 'ok', 'n_organs': 2, 'n_queries': 4, 'n_targets': 4}
    data = {'cohorts': {'n_supported_organs': 2},
            'balanced_cross_modal_retrieval': {'status': 'ok', 'bidirectional_mean': {'map': 0.5, 'top@1': 0.5, 'top@5': 1.0}, 'ct_to_mr': dict(direction), 'mr_to_ct': dict(direction)},
            'balanced_lisi': {'status': 'ok', 'n_organs': 2, 'n_samples': 8, 'overall': {'modality_ilisi': {'mean': 1.5}, 'organ_clisi': {'mean': 1.1}}}}
    if invalid == 'retrieval_status':
        data['balanced_cross_modal_retrieval']['status'] = 'incomplete'
    elif invalid == 'lisi_nan':
        data['balanced_lisi']['overall']['modality_ilisi']['mean'] = float('nan')
    else:
        data['balanced_cross_modal_retrieval']['mr_to_ct']['n_queries'] = 0
    path = tmp_path / 'metrics.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        read_metrics('cross_modal', path)


def test_report_rejects_missing_anatomical_folds(tmp_path):
    from reform3d.cli import read_metrics
    data = {'manifest_validation': {'is_valid': True}, 'surface_b_few_shot_transfer_evidence': {'status': 'ok'},
            'headline_metrics': {'task_2_low_shot_transfer': {'primary_metric_value': 0.8, 'n_supported_organs': 7, 'n_evaluated_organs': 6}, 'task_1_low_shot_target_adaptation_gain': {'baseline_metric_value': 0.7, 'primary_metric_value': 0.1}}}
    path = tmp_path / 'metrics.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='omitted'):
        read_metrics('anatomical', path)


def test_checkpoint_registry_matches_release_manifest():
    from reform3d.core.checkpoints import CHECKPOINTS, DEFAULT_REFORM3D_CHECKPOINT_ORDER, resolve_checkpoint_config
    expected = {'med3dino_iso_96': 96, 'med3dino_rel_96': 96, 'med3dino_sa_96': 96, '3dino': 112}
    assert set(CHECKPOINTS) == set(DEFAULT_REFORM3D_CHECKPOINT_ORDER) == set(expected)
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'reproduction/checkpoints.json').read_text())
    assert {entry['checkpoint_key'] for entry in manifest['checkpoints']} == set(expected)
    for entry in manifest['checkpoints']:
        config = resolve_checkpoint_config(entry['checkpoint_key'])
        assert config.crop_size == entry['crop_size'] == expected[config.checkpoint_key]
        assert config.architecture == entry['architecture'] == 'vit_large_3d'
        for alias in entry['aliases']:
            assert resolve_checkpoint_config(alias) == config


@pytest.mark.parametrize('variant,size', [('base', 96), ('base', 112), ('iso', 112), ('rel', 112), ('sa', 112)])
def test_unsupported_checkpoint_variants_are_rejected(variant, size):
    from reform3d.core.checkpoints import resolve_checkpoint_config
    names = [f'med3dino_{variant}_{size}', f'Med3DINO-{variant.upper()}-{size}', f'med3dino_{variant}_c{size}', f'med3dino-{variant}-c{size}']
    if variant == 'base':
        names.append(f'med3dino_{variant}')
    for name in names:
        with pytest.raises(KeyError):
            resolve_checkpoint_config(name)


def test_acquisition_runner_defaults_are_registered():
    from reform3d.core.checkpoints import resolve_checkpoint_config
    from reform3d.modules.acquisition_robustness.run_acquisition_robustness import DEFAULT_CHECKPOINTS
    for name in DEFAULT_CHECKPOINTS:
        assert resolve_checkpoint_config(name).checkpoint_key == name
