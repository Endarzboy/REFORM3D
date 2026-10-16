"""Feature files belong to encoder results and remain reusable without aliases."""
import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from reform3d import cli, workspace
from reform3d.settings import file_hash
from test_artifact_integrity import stub_run


def _identity():
    return {
        'bundle_version': 1,
        'geometry': {'manifest': 'manifest-a', 'crop': 96, 'software': 'geometry-a'},
        'checkpoint_sha256': 'weights-a',
        'architecture': 'vit_large_3d',
        'encoder_software': 'encoder-a',
        'feature': 'cls',
        'execution': {'precision': 'float32', 'batch_size': 1, 'device_type': 'cpu'},
    }


def _stored_features(tmp_path, *, dataset='kits23', checkpoint='model', feature='cls'):
    cfg = {
        'workspace': str(tmp_path), 'dataset': dataset,
        'modules': ['acquisition'], 'run_name': 'first',
    }
    directory = workspace.feature_directory(cfg, checkpoint, feature, 42, 'acquisition')
    directory.mkdir(parents=True)
    artifact = directory / 'controlled_embedding_bundle.npz'
    artifact.write_bytes(b'complete frozen embeddings')
    identity = _identity()
    identity['feature'] = feature
    (directory / 'identity.json').write_text(json.dumps({
        **identity, 'feature_sha256': file_hash(artifact),
    }))
    return cfg, directory, artifact, identity


def test_run_local_reuse_survives_removal_of_earlier_seed(tmp_path):
    cfg, source, artifact, identity = _stored_features(tmp_path)
    destination = workspace.feature_directory(cfg, 'model', 'cls', 123, 'acquisition')
    workspace.reuse_features(cfg, 'model', 'cls', destination, identity, artifact.name)
    reused = destination / artifact.name
    assert reused.samefile(artifact)
    assert not reused.is_symlink()
    assert not (destination / 'identity.json').is_symlink()
    assert json.loads((destination / 'identity.json').read_text())['feature_sha256'] == file_hash(reused)
    shutil.rmtree(source)
    assert reused.read_bytes() == b'complete frozen embeddings'
    assert not (tmp_path / 'cache').exists()


@pytest.mark.parametrize('change', [
    'weights', 'geometry', 'software', 'execution', 'architecture',
    'dataset', 'checkpoint', 'readout',
])
def test_reuse_does_not_cross_feature_identity_boundaries(tmp_path, change):
    cfg, _, artifact, identity = _stored_features(tmp_path)
    requested = copy.deepcopy(identity)
    checkpoint, feature = 'model', 'cls'
    if change == 'weights':
        requested['checkpoint_sha256'] = 'weights-b'
    elif change == 'geometry':
        requested['geometry']['crop'] = 112
    elif change == 'software':
        requested['encoder_software'] = 'encoder-b'
    elif change == 'execution':
        requested['execution']['precision'] = 'float16'
    elif change == 'architecture':
        requested['architecture'] = 'vit_base_3d'
    elif change == 'dataset':
        cfg = {**cfg, 'dataset': 'another_dataset'}
    elif change == 'checkpoint':
        checkpoint = 'other_model'
    else:
        feature = 'avg_pool'
        requested['feature'] = feature
    cfg = {**cfg, 'run_name': 'second'}
    destination = workspace.feature_directory(cfg, checkpoint, feature, 42, 'acquisition')
    workspace.reuse_features(cfg, checkpoint, feature, destination, requested, artifact.name)
    assert not (destination / artifact.name).exists()
    assert artifact.read_bytes() == b'complete frozen embeddings'


def test_matching_corrupt_features_fail_before_reuse(tmp_path):
    cfg, _, artifact, identity = _stored_features(tmp_path)
    artifact.write_bytes(b'changed after sealing')
    destination = workspace.feature_directory(cfg, 'model', 'cls', 123, 'acquisition')
    with pytest.raises(ValueError, match='[Cc]hecksum|[Cc]hanged|[Ii]ntegrity'):
        workspace.reuse_features(cfg, 'model', 'cls', destination, identity, artifact.name)
    assert not (destination / artifact.name).exists()


def test_unsealed_features_are_not_reused(tmp_path):
    cfg, source, artifact, identity = _stored_features(tmp_path)
    (source / 'identity.json').write_text(json.dumps(identity))
    destination = workspace.feature_directory(cfg, 'model', 'cls', 123, 'acquisition')
    workspace.reuse_features(cfg, 'model', 'cls', destination, identity, artifact.name)
    assert not (destination / artifact.name).exists()


def test_existing_destination_is_never_replaced_by_reuse(tmp_path):
    cfg, _, artifact, identity = _stored_features(tmp_path)
    destination = workspace.feature_directory(cfg, 'model', 'cls', 123, 'acquisition')
    destination.mkdir(parents=True)
    current = destination / artifact.name
    current.write_bytes(b'unrelated existing output')
    with pytest.raises((ValueError, FileExistsError)):
        workspace.reuse_features(cfg, 'model', 'cls', destination, identity, artifact.name)
    assert current.read_bytes() == b'unrelated existing output'
    assert artifact.read_bytes() == b'complete frozen embeddings'


def _anatomical_evaluator(monkeypatch, calls):
    original = cli._call

    def evaluate(module, arguments):
        if not module.startswith('modules.anatomical'):
            return original(module, arguments)
        calls.append(module)
        output = Path(arguments[arguments.index('--output-path') + 1])
        output.write_text(json.dumps({
            'headline_metrics': {
                'task_2_low_shot_transfer': {
                    'n_evaluated_organs': 2, 'n_supported_organs': 2,
                    'primary_metric_value': .7,
                },
                'task_1_low_shot_target_adaptation_gain': {
                    'baseline_metric_value': .6, 'primary_metric_value': .1,
                },
            },
            'surface_b_few_shot_transfer_evidence': {'status': 'ok'},
            'manifest_validation': {'is_valid': True},
        }))

    monkeypatch.setattr(cli, '_call', evaluate)


def _assert_local_artifacts(cfg, state):
    artifacts = []
    for entry in state['completed']:
        artifact = Path(entry['feature_path'])
        assert artifact.parent == Path(entry['path']).parent / 'features'
        assert artifact.is_file() and not artifact.is_symlink()
        identity = artifact.parent / 'identity.json'
        assert identity.is_file() and not identity.is_symlink()
        assert file_hash(artifact) == entry['feature_sha256']
        assert json.loads(identity.read_text())['feature_sha256'] == entry['feature_sha256']
        artifacts.append(artifact)
    assert not (Path(cfg['workspace']) / 'cache' / cfg['dataset'] / 'features').exists()
    return artifacts


def test_two_seeds_and_two_modules_share_one_extraction_in_result_folders(tmp_path, monkeypatch):
    _, cfg, calls = stub_run(tmp_path, monkeypatch)
    cfg.update(modules=['cross_modal', 'anatomical'], seeds=[42, 123], not_evaluated=['acquisition'])
    _anatomical_evaluator(monkeypatch, calls)
    state = cli.run(cfg)
    artifacts = _assert_local_artifacts(cfg, state)
    assert len(artifacts) == 4
    assert len({str(path) for path in artifacts}) == 4
    assert all(path.samefile(artifacts[0]) for path in artifacts)
    assert sum(name.endswith('feature_extractor') for name in calls) == 1
    before = list(calls)
    cli.run(cfg)
    cli.refresh_report(workspace.run_directory(cfg))
    assert calls == before


def test_later_run_reuses_sealed_features_without_cache_feature_storage(tmp_path, monkeypatch):
    _, cfg, calls = stub_run(tmp_path, monkeypatch)
    first = cli.run(cfg)
    previous = Path(first['completed'][0]['feature_path'])
    cfg = {**cfg, 'run_name': 'second'}
    second = cli.run(cfg)
    current = _assert_local_artifacts(cfg, second)[0]
    assert current != previous and current.samefile(previous)
    assert sum(name.endswith('feature_extractor') for name in calls) == 1
    shutil.rmtree(workspace.run_directory({**cfg, 'run_name': first['configuration']['run_name']}))
    cli.refresh_report(workspace.run_directory(cfg))
    assert current.is_file()


def test_runner_does_not_claim_an_unowned_existing_feature_file(tmp_path, monkeypatch):
    _, cfg, calls = stub_run(tmp_path, monkeypatch)
    first = cli.run(cfg)
    source = Path(first['completed'][0]['feature_path'])
    cfg = {**cfg, 'run_name': 'second'}
    destination = workspace.feature_directory(cfg, 'test', 'cls', 42, 'cross_modal')
    destination.mkdir(parents=True)
    artifact = destination / source.name
    shutil.copyfile(source, artifact)
    checksum = file_hash(artifact)
    before = list(calls)
    with pytest.raises((ValueError, FileExistsError)):
        cli.run(cfg)
    assert file_hash(artifact) == checksum
    assert not (destination / 'identity.json').exists()
    assert calls == before


@pytest.mark.parametrize('change', ['weights', 'readout', 'geometry'])
def test_later_run_extracts_again_for_changed_feature_inputs(tmp_path, monkeypatch, change):
    _, cfg, calls = stub_run(tmp_path, monkeypatch)
    first = cli.run(cfg)
    previous = Path(first['completed'][0]['feature_path'])
    cfg = {**cfg, 'run_name': 'second'}
    if change == 'weights':
        (tmp_path / 'weights.bin').write_bytes(b'new checkpoint weights')
    elif change == 'readout':
        cfg['feature_types'] = ['avg_pool']
    else:
        cfg['bbox_margin'] += 1
    second = cli.run(cfg)
    current = _assert_local_artifacts(cfg, second)[0]
    assert not current.samefile(previous)
    assert sum(name.endswith('feature_extractor') for name in calls) == 2


def test_external_acquisition_writes_one_bundle_inside_features(tmp_path, monkeypatch):
    from reform3d.external_methods import _acquisition
    from reform3d.modules.acquisition_robustness.config import ControlledPerturbationConfig

    config = ControlledPerturbationConfig()
    names = config.describe_targets()
    arrays = {
        name: np.asarray([[1., 0.], [0., 1.], [1., 1.]], dtype=np.float32)
        for name in names
    }
    volumes = [
        SimpleNamespace(original_path=f'case-{index}.nii.gz', original_spacing=(1., 1., 1.))
        for index in range(3)
    ]
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'dataset': 'kits23', 'modality': 'ct'}))
    output = tmp_path / 'run/seeds/seed_42/external--avg_pool'
    monkeypatch.setattr(_acquisition, 'get_controlled_perturbation_config_from_manifest', lambda _: config)
    monkeypatch.setattr(_acquisition, 'prepare_perturbed_dataset', lambda **_: volumes)
    monkeypatch.setattr(_acquisition, 'extract_features_for_variants', lambda **_: arrays)
    result = _acquisition.run_controlled_perturbation_analysis(
        manifest_path=manifest, checkpoint_path=tmp_path / 'weights.pth',
        checkpoint_name='external', feature_type='avg_pool',
        extractor_factory=lambda **_: SimpleNamespace(), model_name='external',
        crop_size=96, analysis_name='kits23', output_dir=output,
        figures_dir=output / 'figures', cache_dir=tmp_path / 'cache',
        device='cpu', compute_matched_semantics=False, compute_drift_figure=False,
    )
    assert result['controlled_embedding_bundle'] == 'features/controlled_embedding_bundle.npz'
    bundle = output / result['controlled_embedding_bundle']
    assert list(output.rglob('*.npz')) == [bundle]
    assert not bundle.is_symlink()
    with np.load(bundle, allow_pickle=False) as payload:
        for name in names:
            np.testing.assert_array_equal(payload[f'features__{name}'], arrays[name])


@pytest.mark.parametrize('available', ['current', 'legacy', 'both', 'neither'])
def test_optional_spacing_projection_prefers_current_bundle_with_legacy_fallback(tmp_path, monkeypatch, available):
    from reform3d.modules.acquisition_robustness import config
    from reform3d.modules.anatomical_generalization.evaluation_pipeline import _resolve_spacing_bundle_path

    result_dir = tmp_path / 'model/cls'
    current = result_dir / 'features/controlled_embedding_bundle.npz'
    legacy = result_dir / 'controlled_embedding_bundle.npz'
    current.parent.mkdir(parents=True)
    if available in {'current', 'both'}:
        current.write_bytes(b'current bundle')
    if available in {'legacy', 'both'}:
        legacy.write_bytes(b'legacy bundle')
    monkeypatch.setattr(config, 'get_output_paths', lambda *args: {'results': tmp_path})
    monkeypatch.setattr(config, 'get_checkpoint_feature_dir', lambda root, checkpoint, feature: root / checkpoint / feature)
    found = _resolve_spacing_bundle_path('kits23', 'core', 'model', 'cls')
    assert found == (legacy if available == 'legacy' else current)
