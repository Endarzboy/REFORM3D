"""Public output routing, shared features and offline report regeneration."""
import json
from pathlib import Path

import pytest

from reform3d import cli
from reform3d.settings import configure_environment, file_hash
from reform3d.workspace import (
    cache_directory, feature_directory,
    module_directory, run_directory,
)
from test_artifact_integrity import stub_run


def test_dataset_and_module_paths_do_not_collide(tmp_path):
    cfg = {'workspace': str(tmp_path), 'dataset': 'kits23', 'modules': ['acquisition'], 'run_name': 'experiment'}
    assert run_directory(cfg) == tmp_path / 'runs/kits23/acquisition/experiment'
    other = {**cfg, 'dataset': 'mmwhs_ct_mr', 'modules': ['cross_modal', 'anatomical']}
    assert run_directory(other) == tmp_path / 'runs/mmwhs_ct_mr/_runs/experiment'
    assert module_directory(other, 'anatomical') == tmp_path / 'runs/mmwhs_ct_mr/anatomical/experiment'
    assert cache_directory(cfg, 'acquisition', 'same') != cache_directory(other, 'organs', 'same')
    assert feature_directory(cfg, 'model', 'cls', 42, 'acquisition') != feature_directory(cfg, 'model', 'avg_pool', 42, 'acquisition')


def test_completed_runner_exposes_features_and_refreshes_without_inference(tmp_path, monkeypatch):
    _, cfg, calls = stub_run(tmp_path, monkeypatch)
    state = cli.run(cfg)
    root = run_directory(cfg)
    entry = state['completed'][0]
    features = Path(entry['path']).parent / 'features/organ_features.npz'
    assert features.samefile(entry['feature_path'])
    assert Path(entry['feature_path']) == features
    assert not features.is_symlink()
    assert not (Path(cfg['workspace']) / 'cache/mmwhs_ct_mr/features').exists()
    before = list(calls)
    receipt = (root / 'run.json').read_bytes()
    cli.refresh_report(root)
    assert calls == before
    assert (root / 'run.json').read_bytes() == receipt
    assert (root / 'report.html').is_file()
    Path(entry['path']).write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        cli.refresh_report(root)


def test_legacy_run_does_not_silently_start_a_duplicate(tmp_path, monkeypatch):
    _, cfg, _ = stub_run(tmp_path, monkeypatch)
    old = Path(cfg['workspace']) / 'runs' / cfg['run_name']
    old.mkdir(parents=True)
    (old / 'run.json').write_text('{}')
    with pytest.raises(ValueError, match='Legacy run'):
        cli.run(cfg, dry_run=True)
    assert not run_directory(cfg).exists()


def test_module_output_context_matches_encoder_leaf(tmp_path, monkeypatch):
    from reform3d.core.layout import active_output_paths
    cfg = {'workspace': str(tmp_path), 'dataset': 'kits23', 'modules': ['acquisition'],
           'run_name': 'experiment', 'checkpoint_root': str(tmp_path / 'weights')}
    configure_environment(cfg)
    leaf = run_directory(cfg) / 'seeds/seed_42/model--cls'
    cache = cache_directory(cfg, 'acquisition', 'hash')
    monkeypatch.setenv('REFORM3D_ENCODER_DIR', str(leaf))
    monkeypatch.setenv('REFORM3D_CACHE_DIR', str(cache))
    actual = active_output_paths('acquisition_robustness')
    assert actual['features'] == leaf / 'features'
    assert actual['results'] == leaf
    assert actual['crop_cache_root'] == cache
    assert actual['manifests'] == tmp_path / 'manifests/kits23'


def test_multimodule_run_shares_features_and_keeps_reports_separate(tmp_path, monkeypatch):
    _, cfg, calls = stub_run(tmp_path, monkeypatch)
    cfg.update(modules=['cross_modal', 'anatomical'], seeds=[42, 123], not_evaluated=['acquisition'])
    original = cli._call

    def evaluate(module, args):
        if not module.startswith('modules.anatomical'):
            return original(module, args)
        calls.append(module)
        output = Path(args[args.index('--output-path') + 1])
        output.write_text(json.dumps({
            'headline_metrics': {
                'task_2_low_shot_transfer': {'n_evaluated_organs': 2, 'n_supported_organs': 2, 'primary_metric_value': .7},
                'task_1_low_shot_target_adaptation_gain': {'baseline_metric_value': .6, 'primary_metric_value': .1},
            },
            'surface_b_few_shot_transfer_evidence': {'status': 'ok'},
            'manifest_validation': {'is_valid': True},
        }))

    monkeypatch.setattr(cli, '_call', evaluate)
    state = cli.run(cfg)
    assert len(state['completed']) == 4
    assert len([c for c in calls if c.endswith('feature_extractor')]) == 1
    shared = {entry['feature_path'] for entry in state['completed']}
    assert len(shared) == 4
    assert all(Path(path).samefile(next(iter(shared))) for path in shared)
    for module in cfg['modules']:
        root = module_directory(cfg, module)
        assert (root / 'run.json').samefile(run_directory(cfg) / 'run.json')
        assert (root / 'report.html').is_file()
        profile = json.loads((root / 'profile.json').read_text())
        assert {r['module'] for r in profile['metrics']} == {module}
        for seed in cfg['seeds']:
            leaf = root / f'seeds/seed_{seed}/test--cls'
            assert (leaf / f'{module}.json').is_file()
            assert (leaf / 'features/organ_features.npz').samefile(next(iter(shared)))
        cli.refresh_report(root)
        assert {r['module'] for r in json.loads((root / 'profile.json').read_text())['metrics']} == {module}
    before = list(calls)
    cli.run(cfg)
    assert calls == before


def test_multimodule_run_cannot_overwrite_single_module_run(tmp_path, monkeypatch):
    _, cfg, calls = stub_run(tmp_path, monkeypatch)
    first = cli.run(cfg)
    root = run_directory(cfg)
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    cfg.update(modules=['cross_modal', 'anatomical'], not_evaluated=['acquisition'])
    previous_calls = list(calls)
    with pytest.raises(ValueError, match='belongs to another run'):
        cli.run(cfg)
    assert calls == previous_calls
    assert all(path.read_bytes() == data for path, data in before.items())


def test_multimodule_run_locks_every_module_before_evaluation(tmp_path, monkeypatch):
    from reform3d.settings import artifact_lock
    _, cfg, calls = stub_run(tmp_path, monkeypatch)
    cfg.update(modules=['cross_modal', 'anatomical'], not_evaluated=['acquisition'])
    with artifact_lock(module_directory(cfg, 'cross_modal') / '.run.lock'):
        with pytest.raises(RuntimeError, match='Another run'):
            cli.run(cfg)
    assert calls == []
