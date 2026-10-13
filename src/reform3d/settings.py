"""Validated configuration and artifact identities for public runs."""
from __future__ import annotations
import hashlib
from contextlib import contextmanager
import json
import os
import re
from pathlib import Path
import yaml

FIELDS = {'dataset', 'data_root', 'checkpoint_root', 'checkpoint_paths', 'checkpoints', 'workspace', 'run_name', 'preset', 'seeds', 'feature_types', 'device', 'batch_size', 'workers', 'manifest', 'modules', 'bbox_margin', 'outside_organ_mask', 'min_mask_voxels', 'bootstrap_resamples', 'source_bin', 'n_source_volumes', 'modality', 'label_root', 'file_pattern', 'required_modalities', 'min_samples_per_modality', 'support_per_modality'}
ORGAN_DATASETS = {'mmwhs_ct_mr', 'amos22_ct_mr', 'chaos_ct_mr', 'totalsegmenter_ct_mr_anchor', 'custom_organs'}
MODULES = {'acquisition', 'cross_modal', 'anatomical'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_settings(path):
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError('Configuration must be a YAML mapping')
    unknown = set(raw) - FIELDS
    if unknown:
        raise ValueError(f'Unknown settings: {sorted(unknown)}')
    for key in ('dataset', 'checkpoints'):
        if not raw.get(key):
            raise ValueError(f'Missing required setting: {key}')
    if not isinstance(raw['dataset'], str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', raw['dataset']):
        raise ValueError('dataset must be a simple identifier')
    if not isinstance(raw['checkpoints'], list) or not all(isinstance(x, str) for x in raw['checkpoints']):
        raise ValueError('checkpoints must be a nonempty list of registered names')
    cfg = {'preset': 'demo', 'workspace': './workspace', 'checkpoint_root': './checkpoints', 'seeds': [42], 'feature_types': ['cls'], 'device': 'cuda:0', 'batch_size': 1, 'workers': 2, 'bbox_margin': 5, 'outside_organ_mask': 'none', 'min_mask_voxels': 100, 'bootstrap_resamples': 1000, 'source_bin': None, 'n_source_volumes': None, 'required_modalities': ['ct', 'mr'], 'min_samples_per_modality': 5, 'support_per_modality': 2}
    cfg.update(raw)
    if cfg['preset'] not in {'demo', 'screening', 'paper'}:
        raise ValueError('preset must be demo, screening, or paper')
    if cfg['preset'] == 'paper' and 'seeds' not in raw:
        cfg['seeds'] = [42, 123, 456]
    for field in ['seeds', 'feature_types']:
        if not isinstance(cfg[field], list) or not cfg[field] or len(set(cfg[field])) != len(cfg[field]):
            raise ValueError(f'{field} must be a nonempty list without duplicates')
    if not all(isinstance(s, int) and not isinstance(s, bool) and s >= 0 for s in cfg['seeds']):
        raise ValueError('seeds must be nonnegative integers')
    if not set(cfg['feature_types']) <= {'cls', 'avg_pool', 'multilayer'}:
        raise ValueError('Supported feature_types: cls, avg_pool, multilayer')
    if len(set(cfg['checkpoints'])) != len(cfg['checkpoints']):
        raise ValueError('Duplicate checkpoint names')
    if cfg['outside_organ_mask'] not in {'none', 'zero'}:
        raise ValueError('outside_organ_mask must be none or zero')
    for key in ['batch_size', 'workers', 'min_mask_voxels', 'bootstrap_resamples', 'min_samples_per_modality', 'support_per_modality']:
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(cfg['bbox_margin']) is not int or cfg['bbox_margin'] < 0:
        raise ValueError('bbox_margin must be a nonnegative integer')
    if cfg['n_source_volumes'] is not None and (type(cfg['n_source_volumes']) is not int or cfg['n_source_volumes'] < 1):
        raise ValueError('n_source_volumes must be null or a positive integer')
    if cfg['source_bin'] is not None and (type(cfg['source_bin']) is not int or cfg['source_bin'] < 0):
        raise ValueError('source_bin must be null or a nonnegative integer')
    if cfg.get('modality') is not None and cfg['modality'] not in {'ct', 'mr'}:
        raise ValueError('modality must be ct or mr')
    required = cfg['required_modalities']
    if not isinstance(required, list) or len(required) != 2 or not all(isinstance(m, str) for m in required) or set(required) != {'ct', 'mr'}:
        raise ValueError('required_modalities must contain ct and mr exactly once')
    name = cfg.get('run_name', cfg['dataset'] + '-' + cfg['preset'])
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
        raise ValueError('run_name must be a simple directory name')
    cfg['run_name'] = name
    for key in ['workspace', 'checkpoint_root', 'data_root', 'manifest', 'label_root']:
        if cfg.get(key):
            p = Path(os.path.expandvars(str(cfg[key]))).expanduser()
            cfg[key] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    cfg['checkpoint_paths'] = {k: str((path.parent / Path(os.path.expandvars(v)).expanduser()).resolve()) for k, v in cfg.get('checkpoint_paths', {}).items()}
    if not cfg.get('manifest') and not cfg.get('data_root'):
        raise ValueError('Provide data_root for preparation or manifest for existing/custom data')
    eligible = ['cross_modal', 'anatomical'] if cfg['dataset'] in ORGAN_DATASETS else ['acquisition']
    if cfg['dataset'] == 'chaos_ct_mr':
        eligible = ['cross_modal']
    modules = cfg.get('modules', eligible)
    if not isinstance(modules, list) or not modules or not set(modules) <= MODULES or len(set(modules)) != len(modules):
        raise ValueError('modules must contain acquisition, cross_modal, and/or anatomical without duplicates')
    if not set(modules) <= set(eligible):
        raise ValueError(f'{cfg["dataset"]} supports {eligible}; unsupported requested modules: {sorted(set(modules)-set(eligible))}')
    cfg['modules'] = modules
    cfg['not_evaluated'] = sorted(MODULES - set(modules))
    return cfg


def configure_environment(cfg):
    workspace = Path(cfg['workspace'])
    values = {'MED3DINO_CHECKPOINT_ROOT': cfg['checkpoint_root'], 'MED3DINO_EXTERNAL_DATA_ROOT': cfg.get('data_root', workspace / 'data'), 'MED3DINO_SHARED_MANIFEST_ROOT': workspace / 'manifests', 'MED3DINO_SHARED_CACHE_ROOT': workspace / 'cache', 'MED3DINO_SHARED_RESULT_ROOT': workspace / 'runs', 'REFORM3D_WORKSPACE': workspace, 'REFORM3D_RUN_DIR': workspace / 'runs' / cfg['run_name']}
    for key, value in values.items():
        os.environ[key] = str(value)
    os.environ.pop('MED3DINO_REPO_ROOT', None)
    os.environ.pop('MED3DINO_REFORM3D_ACTIVE_SEED', None)
    return Path(values['REFORM3D_RUN_DIR'])


@contextmanager
def artifact_lock(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another run is preparing this artifact: {path.parent}') from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
