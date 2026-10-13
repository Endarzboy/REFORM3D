"""Prepare and evaluate a frozen-encoder cohort from one configuration."""
from __future__ import annotations
import argparse
import contextlib
import csv
import html
import importlib
import json
import math
import os
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from .settings import artifact_lock, configure_environment, digest, file_hash, load_settings, write_json


def _call(module, arguments):
    old = sys.argv
    try:
        sys.argv = [module, *map(str, arguments)]
        importlib.import_module('reform3d.' + module).main()
    finally:
        sys.argv = old


def _software_hash():
    root = Path(__file__).parent
    return digest({str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob('*.py'))})


def _checkpoints(cfg):
    from .core.checkpoints import CHECKPOINTS, resolve_checkpoint_config
    found = []
    for name in cfg['checkpoints']:
        item = resolve_checkpoint_config(name)
        override = cfg['checkpoint_paths'].get(name, cfg['checkpoint_paths'].get(item.checkpoint_key))
        if override:
            item = replace(item, raw_checkpoint_path=Path(override))
            CHECKPOINTS[item.checkpoint_key] = item
        if not item.path.is_file():
            raise FileNotFoundError(f'Checkpoint {name} missing: {item.path}; set checkpoint_root or checkpoint_paths')
        if any(c.checkpoint_key == item.checkpoint_key for c in found):
            raise ValueError(f'Two aliases select the same checkpoint: {name}')
        found.append(item)
    return found


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Required metric {name} is missing or nonfinite: {value}')
    return float(value)


def read_metrics(module, path):
    data = json.loads(Path(path).read_text())
    if module == 'cross_modal':
        retrieval = data['balanced_cross_modal_retrieval']
        lisi = data['balanced_lisi']
        supported = data['cohorts']['n_supported_organs']
        if retrieval.get('status') != 'ok' or lisi.get('status') != 'ok' or supported < 1:
            raise ValueError('Cross-modal evaluation did not complete successfully')
        for direction in ('ct_to_mr', 'mr_to_ct'):
            result = retrieval[direction]
            if result.get('status') != 'ok' or result.get('n_organs') != supported or result.get('n_queries', 0) < 1 or result.get('n_targets', 0) < 1:
                raise ValueError(f'Incomplete cross-modal retrieval: {direction}')
        if lisi.get('n_organs') != supported or lisi.get('n_samples', 0) < 1:
            raise ValueError('Incomplete balanced LISI evaluation')
        scores = retrieval['bidirectional_mean']
        return {**{k: _finite(scores[k], k) for k in ['map', 'top@1', 'top@5']},
                'modality_ilisi': _finite(lisi['overall']['modality_ilisi']['mean'], 'modality_ilisi'),
                'organ_clisi': _finite(lisi['overall']['organ_clisi']['mean'], 'organ_clisi')}
    if module == 'anatomical':
        headline = data['headline_metrics']
        transfer = headline['task_2_low_shot_transfer']
        gain = headline['task_1_low_shot_target_adaptation_gain']
        evidence = data['surface_b_few_shot_transfer_evidence']
        if evidence.get('status') != 'ok' or not data['manifest_validation']['is_valid']:
            raise ValueError('Anatomical evaluation did not complete successfully')
        if transfer.get('n_evaluated_organs', 0) < 1 or transfer['n_evaluated_organs'] != transfer.get('n_supported_organs'):
            raise ValueError('Anatomical evaluation omitted supported organs')
        return {'probe_ba': _finite(transfer['primary_metric_value'], 'probe_ba'),
                'centroid_ba': _finite(gain['baseline_metric_value'], 'centroid_ba'),
                'adaptation_gain': _finite(gain['primary_metric_value'], 'adaptation_gain')}
    drift = data['representation_drift']
    cka = data['cka_matrix']
    variants = data['target_variants']
    reference = data['reference_variant']
    if data.get('n_source_volumes', 0) < 2 or len(variants) < 2 or set(drift) != set(variants) - {reference}:
        raise ValueError('Controlled evaluation has incomplete cases or spacing variants')
    d = statistics.mean(_finite(v, 'drift') for v in drift.values())
    if set(cka) != set(variants) or any(set(row) != set(variants) for row in cka.values()):
        raise ValueError('Controlled CKA matrix is incomplete')
    values = [_finite(v, 'cka') for outer, row in cka.items() for inner, v in row.items() if inner != outer]
    transfer = data.get('matched_semantic_transfer')
    if not transfer or not isinstance(transfer.get('cross_variant_accuracy'), (int, float)):
        raise ValueError('Controlled semantic transfer is unavailable; this run cannot claim a complete acquisition profile')
    matrix = transfer.get('transfer_matrix', {})
    if set(matrix) != set(variants) or any(set(row) != set(variants) for row in matrix.values()):
        raise ValueError('Controlled semantic-transfer matrix is incomplete')
    for row in matrix.values():
        for value in row.values():
            _finite(value, 'semantic transfer')
    return {'drift': d, 'cka': statistics.mean(values), 'transfer': _finite(transfer['cross_variant_accuracy'], 'transfer')}



def _verify_checksum(path, expected):
    path = Path(path)
    if not expected or not path.is_file() or file_hash(path) != expected:
        raise ValueError(f"Artifact checksum changed or integrity record is missing: {path}")


def _verified_completed(previous):
    verified = set()
    checked_features = set()
    for entry in previous.get('completed', []):
        _verify_checksum(entry['path'], entry.get('sha256'))
        feature = (entry.get('feature_path'), entry.get('feature_sha256'))
        if not all(feature):
            raise ValueError('Completed result is missing feature integrity provenance')
        if feature not in checked_features:
            _verify_checksum(*feature)
            checked_features.add(feature)
        verified.add(str(Path(entry['path']).resolve()))
    return verified


def _feature_provenance(path, identity, artifact, *, seal=False):
    saved = json.loads(path.read_text()) if path.exists() else dict(identity)
    previous_hash = saved.pop('feature_sha256', None)
    if saved != identity:
        raise ValueError(f'Feature identity changed: {path}')
    if previous_hash:
        _verify_checksum(artifact, previous_hash)
    if seal:
        if not artifact.is_file():
            raise ValueError(f'Missing completed feature artifact: {artifact}')
        previous_hash = file_hash(artifact)
    payload = {**identity, **({'feature_sha256': previous_hash} if previous_hash else {})}
    if not path.exists() or (seal and json.loads(path.read_text()) != payload):
        write_json(path, payload)
    return previous_hash


def _report(run_dir, cfg, rows, status='complete'):
    groups = {}
    for row in rows:
        key = (row['checkpoint'], row['feature'], row['module'], row['metric'])
        groups.setdefault(key, []).append(row['value'])
    summary = [{'checkpoint': k[0], 'feature': k[1], 'module': k[2], 'metric': k[3], 'mean': statistics.mean(v), 'std': statistics.stdev(v) if len(v) > 1 else None, 'n_seeds': len(v)} for k, v in sorted(groups.items())]
    columns = ['checkpoint', 'feature', 'module', 'metric', 'mean', 'std', 'n_seeds']
    with (run_dir / 'scores.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(summary)
    best = {}
    for row in summary:
        key = row['module'] + '/' + row['metric']
        current = best.get(key)
        if current is None or (row['mean'] < current['mean'] if row['metric'] in {'drift', 'organ_clisi'} else row['mean'] > current['mean']):
            best[key] = row
    profile = {'status': status, 'dataset': cfg['dataset'], 'not_evaluated': cfg['not_evaluated'], 'metrics': summary, 'highest_observed_per_property': best, 'interpretation': 'Descriptive frozen-feature comparisons for this cohort and protocol. No overall score or guaranteed downstream winner. SD describes evaluation seeds, not pretraining runs.'}
    write_json(run_dir / 'profile.json', profile)
    esc = lambda value: html.escape(str(value))
    cells = ''.join('<tr>' + ''.join('<td>' + esc(row[c] if row[c] is not None else 'n/a') + '</td>' for c in columns) + '</tr>' for row in summary)
    (run_dir / 'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>REFORM-3D profile</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#142635}table{border-collapse:collapse;width:100%}th,td{padding:10px;text-align:left;border-bottom:1px solid #ddd}th{background:#eef4f7}</style><h1>REFORM-3D: ' + esc(cfg['dataset']) + '</h1><p>Status: ' + esc(status) + '</p><p>' + esc(profile['interpretation']) + '</p><p>Not evaluated: ' + esc(', '.join(cfg['not_evaluated']) or 'none') + '</p><table><tr>' + ''.join('<th>'+esc(c)+'</th>' for c in columns) + '</tr>' + cells + '</table></html>')
    return profile


def _run(cfg, *, dry_run=False):
    from .prepare import prepare_manifest
    run_dir = configure_environment(cfg)
    checkpoints = _checkpoints(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'logs').mkdir(exist_ok=True)
    print('Checking inputs and preparing manifest...', flush=True)
    with (run_dir / 'logs' / 'preparation.log').open('a') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        manifest, payload, manifest_hash = prepare_manifest(cfg)
    from importlib.metadata import version, PackageNotFoundError
    runtime_versions = {}
    for name in ('torch', 'numpy', 'monai', 'scipy', 'nibabel', 'SimpleITK', 'Pillow', 'scikit-learn', 'xformers'):
        try:
            runtime_versions[name] = version(name)
        except PackageNotFoundError:
            runtime_versions[name] = None
    software = _software_hash()
    package = Path(__file__).parent
    geometry_sources = ['encoders/med3dino/spacing_aware_transforms.py', 'modules/acquisition_robustness/data_loader.py', 'modules/acquisition_robustness/perturbation_robustness_analysis.py', 'modules/acquisition_robustness/config.py', 'modules/acquisition_robustness/semantic_labels.py', 'modules/cross_modal_alignment/feature_extractor.py', 'modules/cross_modal_alignment/physical_mask.py']
    geometry_software = digest({'sources': {name: file_hash(package / name) for name in geometry_sources}, 'runtime': runtime_versions})
    encoder_software = digest({'sources': {str(p.relative_to(package)): file_hash(p) for p in [*sorted((package / 'encoders').rglob('*.py')), package / 'modules/acquisition_robustness/feature_extractor.py']}, 'runtime': runtime_versions})
    weights = {c.checkpoint_key: file_hash(c.path) for c in checkpoints}
    import torch
    device = torch.device(cfg['device'])
    execution = {'precision': 'float16' if device.type == 'cuda' else 'float32', 'batch_size': cfg['batch_size'], 'device_type': device.type}
    if device.type == 'cuda' and torch.cuda.is_available():
        execution.update(device_name=torch.cuda.get_device_name(device), capability=list(torch.cuda.get_device_capability(device)), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    identity = {'configuration': cfg, 'manifest_hash': manifest_hash, 'checkpoint_hashes': weights, 'software_hash': software, 'runtime_versions': runtime_versions, 'execution': execution}
    identity_hash = digest(identity)
    state_path = run_dir / 'run.json'
    previous = {}
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        if previous.get('identity_hash') != identity_hash:
            raise ValueError(f'Run {cfg["run_name"]} already exists with different inputs/configuration/software. Choose a new run_name.')
    state = {**identity, 'identity_hash': identity_hash, 'manifest': str(manifest), 'status': 'checked' if dry_run else 'prepared', 'completed': [], 'started_unix': time.time()}
    if dry_run and state_path.exists():
        write_json(run_dir / "check.json", state)
    else:
        write_json(state_path, state)
    print(f"Dataset: {cfg['dataset']}; modules: {', '.join(cfg['modules'])}; checkpoints: {len(checkpoints)}; seeds: {cfg['seeds']}", flush=True)
    print(f"Manifest: {manifest}; output: {run_dir}", flush=True)
    if dry_run:
        return state
    workspace = Path(cfg['workspace'])
    rows = []
    try:
        import numpy as np
        import torch
        if cfg['device'].startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; choose device: cpu or install a compatible PyTorch build')
        verified_metrics = _verified_completed(previous)
        torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
        state['status'] = 'running'
        write_json(state_path, state)
        for checkpoint in checkpoints:
            for feature in cfg['feature_types']:
                geometry = {'manifest': manifest_hash, 'crop': checkpoint.crop_size, 'bbox_margin': cfg['bbox_margin'], 'outside_organ_mask': cfg['outside_organ_mask'], 'min_mask_voxels': cfg['min_mask_voxels'], 'source_bin': cfg['source_bin'], 'n_source_volumes': cfg['n_source_volumes'], 'software': geometry_software}
                geometry_id = digest(geometry)[:20]
                cache = workspace / 'cache' / geometry_id
                feature_identity = {'bundle_version': 1, 'geometry': geometry, 'checkpoint_sha256': weights[checkpoint.checkpoint_key], 'architecture': checkpoint.architecture, 'encoder_software': encoder_software, 'feature': feature, 'execution': execution}
                feature_id = digest(feature_identity)[:20]
                bundle = workspace / 'features' / feature_id
                bundle.mkdir(parents=True, exist_ok=True)
                with artifact_lock(bundle / '.lock'), artifact_lock(cache / '.lock'):
                    organ_npz = bundle / 'organ_features.npz'
                    feature_artifact = bundle / 'controlled_features.npz' if 'acquisition' in cfg['modules'] else organ_npz
                    _feature_provenance(bundle / 'identity.json', feature_identity, feature_artifact)
                    if 'acquisition' not in cfg['modules']:
                        if not organ_npz.exists():
                            print(f'Extracting {checkpoint.checkpoint_key}/{feature} organ features...', flush=True)
                            args = ['--manifest', manifest, '--checkpoint', checkpoint.checkpoint_key, '--feature-type', feature, '--output-npz', organ_npz, '--crop-cache-dir', cache, '--batch-size', cfg['batch_size'], '--crop-cache-workers', cfg['workers'], '--device', cfg['device'], '--bbox-margin', cfg['bbox_margin'], '--outside-organ-mask', cfg['outside_organ_mask'], '--min-mask-voxels', cfg['min_mask_voxels']]
                            with (run_dir / 'logs' / f'{checkpoint.checkpoint_key}-{feature}-extraction.log').open('a') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                                _call('modules.cross_modal_alignment.feature_extractor', args)
                        from .core.features import load_organ_feature_bundle
                        load_organ_feature_bundle(organ_npz, [row['sample_id'] for row in payload['samples']], exact=True)
                        _feature_provenance(bundle / 'identity.json', feature_identity, feature_artifact, seal=True)
                    for seed in cfg['seeds']:
                        os.environ['MED3DINO_REFORM3D_ACTIVE_SEED'] = str(seed)
                        leaf = run_dir / 'seeds' / f'seed_{seed}' / f'{checkpoint.checkpoint_key}--{feature}'
                        leaf.mkdir(parents=True, exist_ok=True)
                        for module in cfg['modules']:
                            metric_file = leaf / (module + '.json')
                            print(f'Evaluating {module}: {checkpoint.checkpoint_key}/{feature}, seed {seed}', flush=True)
                            with (run_dir / 'logs' / f'{checkpoint.checkpoint_key}-{feature}-{module}-seed{seed}.log').open('a') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                                if str(metric_file.resolve()) not in verified_metrics:
                                    if module == 'acquisition':
                                        from .modules.acquisition_robustness.config import ControlledPerturbationConfig, MRI_CONTROLLED_PERTURBATION_CONFIG
                                        from .modules.acquisition_robustness.perturbation_robustness_analysis import run_controlled_perturbation
                                        protocol = replace(MRI_CONTROLLED_PERTURBATION_CONFIG) if payload.get('modality') == 'mr' else ControlledPerturbationConfig()
                                        protocol.source_bin = cfg['source_bin']; protocol.n_source_volumes = cfg['n_source_volumes']
                                        result = run_controlled_perturbation(manifest, checkpoint.checkpoint_key, feature_type=feature, config=protocol, output_dir=leaf, figures_dir=leaf / 'figures', cache_dir=cache, semantic_cache_dir=cache / 'semantics', analysis_name=cfg['dataset'], random_state=seed, compute_matched_semantic_probing=False, compute_drift_figure=False, feature_cache_path=bundle / 'controlled_features.npz', device=cfg['device'], batch_size=cfg['batch_size'])
                                        expected = sum(v['anisotropy_bin'] == protocol.source_bin for v in payload['volumes'])
                                        if protocol.n_source_volumes:
                                            expected = min(expected, protocol.n_source_volumes)
                                        if result.n_source_volumes != expected:
                                            raise ValueError(f'Controlled cohort incomplete: {result.n_source_volumes}/{expected}')
                                        original = leaf / 'perturbation_robustness.json'
                                        write_json(metric_file, json.loads(original.read_text()))
                                    elif module == 'cross_modal':
                                        if cfg['dataset'] == 'chaos_ct_mr':
                                            raise ValueError('CHAOS uses the separate single-organ protocol; use its reproduction entry point')
                                        _call('modules.cross_modal_alignment.evaluation_pipeline', ['--manifest', manifest, '--analysis-name', cfg['dataset'], '--manifest-variant', 'core', '--embeddings-npz', organ_npz, '--checkpoint-name', checkpoint.checkpoint_key, '--feature-type', feature, '--metrics-output', metric_file, '--summary-output-dir', leaf / 'cohort', '--retrieval-diagnostics-output', leaf / 'retrieval_diagnostics.json', '--seed', seed, '--bootstrap-resamples', cfg['bootstrap_resamples'], '--bbox-margin', cfg['bbox_margin'], '--outside-organ-mask', cfg['outside_organ_mask'], '--required-modalities', *cfg['required_modalities']])
                                    else:
                                        _call('modules.anatomical_generalization.evaluation_pipeline', ['--manifest', manifest, '--analysis-name', cfg['dataset'], '--manifest-variant', 'core', '--embeddings-npz', organ_npz, '--checkpoint-name', checkpoint.checkpoint_key, '--feature-type', feature, '--output-path', metric_file, '--manifest-summary-output-dir', leaf / 'cohort', '--seed', seed, '--few-shot-seeds', seed, '--few-shot-support-per-modality', cfg['support_per_modality'], '--required-modalities', *cfg['required_modalities'], '--min-samples-per-modality', cfg['min_samples_per_modality']])
                                feature_sha = _feature_provenance(bundle / 'identity.json', feature_identity, feature_artifact, seal=True)
                                metrics = read_metrics(module, metric_file)
                            rows.extend({'checkpoint': checkpoint.checkpoint_key, 'feature': feature, 'module': module, 'metric': key, 'seed': seed, 'value': value} for key, value in metrics.items())
                            state['completed'].append({'checkpoint': checkpoint.checkpoint_key, 'feature': feature, 'module': module, 'seed': seed, 'path': str(metric_file), 'sha256': file_hash(metric_file), 'feature_path': str(feature_artifact), 'feature_sha256': feature_sha})
                            write_json(state_path, state)
        _report(run_dir, cfg, rows)
        state['status'] = 'complete'; state['elapsed_seconds'] = time.time() - state['started_unix']
        write_json(state_path, state)
        print(f'Complete: {run_dir / "report.html"}', flush=True)
        return state
    except BaseException as exc:
        state['status'] = 'failed'; state['error'] = str(exc)
        state['elapsed_seconds'] = time.time() - state['started_unix']
        write_json(state_path, state)
        _report(run_dir, cfg, rows, status='incomplete')
        raise
    finally:
        os.environ.pop('MED3DINO_REFORM3D_ACTIVE_SEED', None)



def run(cfg, *, dry_run=False):
    import fcntl
    run_dir = Path(cfg['workspace']) / 'runs' / cfg['run_name']
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another process is writing run {cfg["run_name"]}') from exc
        try:
            return _run(cfg, dry_run=dry_run)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description='REFORM-3D frozen-encoder evaluation')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ['run', 'check']:
        command = sub.add_parser(name, help='Prepare and evaluate' if name == 'run' else 'Validate configuration and input cohort without inference')
        command.add_argument('--config', required=True, type=Path)
        if name == 'run':
            command.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        run(load_settings(args.config), dry_run=args.command == 'check' or getattr(args, 'dry_run', False))
    except Exception as exc:
        parser.exit(1, f'REFORM-3D failed: {exc}\n')


if __name__ == '__main__':
    main()
