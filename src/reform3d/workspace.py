"""Dataset-scoped storage with embeddings kept in encoder result folders."""
from __future__ import annotations

import os
from pathlib import Path


def module_directory(cfg, module):
    return Path(cfg['workspace']) / 'runs' / cfg['dataset'] / module / cfg['run_name']


def run_directory(cfg):
    module = cfg['modules'][0] if len(cfg['modules']) == 1 else '_runs'
    return module_directory(cfg, module)


def manifest_directory(cfg):
    return Path(cfg['workspace']) / 'manifests' / cfg['dataset']


def cache_directory(cfg, namespace, identity):
    return Path(cfg['workspace']) / 'cache' / cfg['dataset'] / namespace / identity


def feature_directory(cfg, checkpoint, feature, seed, module):
    return module_directory(cfg, module) / 'seeds' / f'seed_{seed}' / f'{checkpoint}--{feature}' / 'features'


def link_artifact(source, destination):
    """Expose one shared artifact without copying bytes or replacing other files."""
    source, destination = Path(source).resolve(), Path(destination)
    if destination.is_symlink() or destination.exists():
        if destination.resolve() != source:
            raise FileExistsError(f'Artifact destination already exists: {destination}')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source, destination.parent))


def reuse_features(cfg, checkpoint, feature, directory, identity, artifact_name):
    """Reuse a sealed run-local bundle without another physical copy of its arrays."""
    import json
    from .settings import file_hash, write_json

    directory = Path(directory)
    target = directory / artifact_name
    if directory.is_symlink() or target.is_symlink():
        raise FileExistsError(f'Feature storage must be an ordinary file directory: {directory}')
    if target.exists():
        receipt = directory / 'identity.json'
        if not target.is_file() or receipt.is_symlink() or not receipt.is_file():
            raise FileExistsError(f'Existing feature artifact has no ordinary identity receipt: {target}')
        saved = json.loads(receipt.read_text())
        checksum = saved.pop('feature_sha256', None)
        if saved != identity or (checksum and file_hash(target) != checksum):
            raise ValueError(f'Existing feature artifact failed identity or checksum verification: {target}')
        return
    root = Path(cfg['workspace']) / 'runs' / cfg['dataset']
    pattern = f'*/*/seeds/seed_*/{checkpoint}--{feature}/features/identity.json'
    for receipt in sorted(root.glob(pattern)):
        if receipt.parent == directory or receipt.is_symlink():
            continue
        saved = json.loads(receipt.read_text())
        checksum = saved.pop('feature_sha256', None)
        if saved != identity or not checksum:
            continue
        source = receipt.parent / artifact_name
        if (source.is_symlink() or not source.is_file()
                or not source.resolve().is_relative_to(root.resolve())
                or file_hash(source) != checksum):
            raise ValueError(f'Compatible feature bundle failed integrity verification: {source}')
        directory.mkdir(parents=True, exist_ok=True)
        os.link(source, target)
        if file_hash(target) != checksum:
            target.unlink()
            raise ValueError(f'Feature bundle changed while being reused: {source}')
        write_json(directory / 'identity.json', {**identity, 'feature_sha256': checksum})
        return
