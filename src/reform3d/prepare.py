"""Dataset preparation with explicit cohort and input provenance."""
from __future__ import annotations
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from .settings import digest, write_json

PATH_FIELDS = ('file_path', 'mask_path', 'label_path')


def prepare_manifest(cfg):
    import nibabel as nib
    import numpy as np
    workspace = Path(cfg['workspace'])
    if cfg.get('manifest'):
        source = Path(cfg['manifest'])
        if source.suffix.lower() == '.csv':
            records = []
            for row in csv.DictReader(source.open()):
                required = {'case_id', 'modality', 'image', 'mask', 'organ', 'label_id'}
                if not required <= row.keys():
                    raise ValueError(f'Organ CSV requires columns {sorted(required)}')
                image = (source.parent / row['image']).resolve()
                records.append({'sample_id': f"{row['modality']}:{row['case_id']}:{row['organ']}", 'source_case_id': row['case_id'], 'patient_id': row.get('patient_id') or row['case_id'], 'file_path': str(image), 'mask_path': str((source.parent / row['mask']).resolve()), 'modality': row['modality'], 'primary_organ': row['organ'], 'mask_label_value': int(row['label_id']), 'spacing': list(map(float, nib.load(image).header.get_zooms()[:3]))})
            payload = {'samples': records, 'dataset': cfg['dataset']}
        else:
            payload = json.loads(source.read_text())
        for record in payload.get('samples', payload.get('volumes', [])):
            for key in PATH_FIELDS:
                if record.get(key):
                    record[key] = str((source.parent / record[key]).resolve())
    elif cfg['dataset'] in {'mmwhs_ct_mr', 'amos22_ct_mr', 'chaos_ct_mr'}:
        from .modules.cross_modal_alignment import build_manifests as builder
        root = Path(cfg['data_root'])
        args = (root, cfg['min_mask_voxels'], cfg['min_samples_per_modality'])
        if cfg['dataset'] == 'mmwhs_ct_mr':
            trio = builder.build_mmwhs_manifest(*args)
        elif cfg['dataset'] == 'amos22_ct_mr':
            trio = builder.build_amos22_manifest(root, ['training', 'validation'], *args[1:])
        else:
            trio = builder.build_chaos_manifest(*args)
        payload = trio[1]
    elif cfg['modules'] == ['acquisition']:
        from .modules.acquisition_robustness import config as ac
        from .modules.acquisition_robustness.manifest_generation import build_single_dataset_manifest
        if cfg['dataset'] not in ac.RAW_DATASETS:
            raise ValueError('Unknown acquisition dataset; provide a compatible volume manifest')
        root = Path(cfg['data_root'])
        old = ac.RAW_DATASETS[cfg['dataset']]
        images = root / 'images' if (root / 'images').is_dir() else root
        labels = Path(cfg['label_root']) if cfg.get('label_root') else (root / 'masks' if (root / 'masks').is_dir() else old.label_path)
        ac.RAW_DATASETS[cfg['dataset']] = replace(old, path=images, label_path=labels, file_pattern=cfg.get('file_pattern', old.file_pattern), is_enabled=True)
        staging = workspace / 'manifests' / (cfg['dataset'] + '-scan')
        build_single_dataset_manifest(cfg['dataset'], cfg['dataset'], validate_integrity=True, seed=cfg['seeds'][0], output_dir=staging)
        payload = json.loads((staging / 'manifest_sampled.json').read_text())
    else:
        raise ValueError('This dataset needs an explicit manifest; see docs/data.md')
    records = payload.get('samples') if 'samples' in payload else payload.get('volumes')
    if not isinstance(records, list) or not records:
        raise ValueError('Manifest contains no samples or volumes')
    if ('acquisition' in cfg['modules']) != ('volumes' in payload):
        raise ValueError('Acquisition needs volumes; cross-modal/anatomical need organ samples')
    if 'volumes' in payload:
        from .modules.acquisition_robustness.config import RAW_DATASETS, compute_anisotropy_ratio, get_anisotropy_bin
        modalities = [cfg.get('modality'), payload.get('modality')]
        modalities.extend(row.get('modality') for row in records if isinstance(row, dict))
        adapter = RAW_DATASETS.get(cfg['dataset'])
        if adapter is not None and adapter.modality in {'ct', 'mr', 'cta'}:
            modalities.append(adapter.modality)
        resolved = {str(value).lower() for value in modalities if value is not None}
        resolved = {'ct' if value == 'cta' else value for value in resolved}
        if len(resolved) != 1 or not resolved <= {'ct', 'mr'}:
            raise ValueError('Acquisition modality must be an unambiguous ct or mr; configuration, manifest and records must agree')
        payload['modality'] = resolved.pop()
        if cfg['source_bin'] is None:
            cfg['source_bin'] = 1 if payload['modality'] == 'mr' else 0
    fingerprints, seen, headers = {}, set(), {}
    image_cases, patient_cases, image_modalities = {}, {}, {}
    scientific_samples, volume_images = set(), set()
    voxel_spacings = {}
    cohort = {}
    for row in records:
        if not isinstance(row, dict):
            raise ValueError('Every manifest record must be an object')
        if not row.get('file_path'):
            raise ValueError('Every manifest record requires file_path')
        if 'samples' in payload:
            for key in ('sample_id', 'modality', 'primary_organ'):
                if not row.get(key):
                    raise ValueError(f'Organ record requires {key}')
            case = row.get('source_case_id') or row.get('patient_id')
            if not case:
                raise ValueError('Organ record requires source_case_id or patient_id')
            if cfg['dataset'] == 'custom_organs' and row.get('patient_id'):
                row['source_case_id'] = row['patient_id']
                case = row['source_case_id']
            image = str(Path(row['file_path']).resolve())
            if image in image_cases and image_cases[image] != case:
                raise ValueError(f'One image has inconsistent case identities: {image}')
            image_cases[image] = case
            modality = row['modality']
            if modality not in cfg['required_modalities']:
                raise ValueError(f'Unsupported organ modality: {modality}')
            if image in image_modalities and image_modalities[image] != modality:
                raise ValueError(f'One image cannot represent different modalities: {image}')
            image_modalities[image] = modality
            scientific_sample = (image, modality, row['primary_organ'])
            if scientific_sample in scientific_samples:
                raise ValueError(f'Duplicate image/organ record: {scientific_sample}')
            scientific_samples.add(scientific_sample)
            if row.get('patient_id'):
                patient = (row['modality'], row['patient_id'])
                if patient in patient_cases and patient_cases[patient] != case:
                    raise ValueError('Repeated scans of one patient must use a consistent split group')
                patient_cases[patient] = case
            cohort.setdefault(row['primary_organ'], {}).setdefault(row['modality'], set()).add(case)
        else:
            spacing = np.asarray(row.get('spacing', []), dtype=float)
            if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
                raise ValueError('Acquisition record requires three positive finite spacing values')
            if type(row.get('anisotropy_bin')) is not int or row['anisotropy_bin'] < 0:
                raise ValueError('Acquisition record requires nonnegative integer anisotropy_bin')
            if row['anisotropy_bin'] != get_anisotropy_bin(compute_anisotropy_ratio(tuple(spacing))):
                raise ValueError('Manifest anisotropy bin disagrees with its spacing')
            image = str(Path(row['file_path']).resolve())
            if image in volume_images:
                raise ValueError(f'Duplicate acquisition source image: {image}')
            volume_images.add(image)
            if not row.get('label_path') or row.get('has_label') is False:
                raise ValueError('Controlled semantic evaluation requires label_path for every volume')
            row['has_label'] = True
        identity = row.get('sample_id') or row.get('case_id') or row.get('file_path')
        if not identity or identity in seen:
            raise ValueError(f'Missing or duplicate record identity: {identity}')
        seen.add(identity)
        if 'samples' in payload and not (row.get('source_case_id') or row.get('patient_id')):
            raise ValueError(f'Missing case identity for {identity}; case-disjoint evaluation requires it')
        for key in PATH_FIELDS:
            value = row.get(key)
            if not value:
                continue
            path = Path(value)
            if not path.exists():
                raise FileNotFoundError(f'{key} is missing for {identity}: {path}')
            if path.is_dir():
                children = sorted(p for p in path.rglob('*') if p.is_file())
                if not children:
                    raise ValueError(f'Empty image/mask directory: {path}')
                fingerprints[str(path)] = [[str(p.relative_to(path)), p.stat().st_size, p.stat().st_mtime_ns] for p in children]
            else:
                st = path.stat()
                fingerprints[str(path)] = [st.st_size, st.st_mtime_ns]
                if str(path).endswith(('.nii', '.nii.gz')) and str(path) not in headers:
                    img = nib.load(path)
                    spacing = np.asarray(img.header.get_zooms()[:3], dtype=float)
                    if len(img.shape) < 3 or not np.isfinite(spacing).all() or (spacing <= 0).any():
                        raise ValueError(f'Invalid volume geometry: {path}')
                    headers[str(path)] = (img.shape[:3], img.affine.tolist())
                    voxel_spacings[str(path)] = spacing
        image = row.get('file_path')
        mask = row.get('mask_path') or row.get('label_path')
        if image in voxel_spacings and row.get('spacing') is not None:
            declared = np.asarray(row['spacing'], dtype=float)
            if declared.shape != (3,) or not np.allclose(declared, voxel_spacings[image], rtol=1e-5, atol=1e-4):
                raise ValueError(f'Manifest spacing disagrees with volume geometry for {identity}')
        if image in headers and mask in headers:
            ishape, iaff = headers[image]; mshape, maff = headers[mask]
            if ishape != mshape or not np.allclose(iaff, maff, rtol=1e-5, atol=1e-4):
                raise ValueError(f'Image/mask geometry mismatch for {identity}')
        if 'samples' in payload and not mask:
            raise ValueError(f'Organ sample requires an annotation: {identity}')
    if 'samples' in payload:
        required = cfg['required_modalities']
        eligible = [organ for organ, by_modality in cohort.items() if all(len(by_modality.get(m, set())) >= cfg['min_samples_per_modality'] for m in required)]
        if len(eligible) < 2 or len(eligible) != len(cohort):
            raise ValueError(f'Need at least two organs with {cfg["min_samples_per_modality"]} unique cases in every required modality; ineligible organs: {sorted(set(cohort) - set(eligible))}')
        if 'anatomical' in cfg['modules'] and cfg['support_per_modality'] >= min(len(cohort[o][m]) for o in eligible for m in required):
            raise ValueError('Support budget leaves no case-disjoint query cases')
    else:
        eligible_sources = sum(row['anisotropy_bin'] == cfg['source_bin'] for row in records)
        if cfg['n_source_volumes'] is not None:
            eligible_sources = min(eligible_sources, cfg['n_source_volumes'])
        if eligible_sources < 2:
            raise ValueError('Controlled source cohort needs at least two distinct eligible volumes')
    payload['dataset'] = cfg['dataset']
    payload['manifest_variant'] = payload.get('manifest_variant', 'core' if 'samples' in payload else 'original_bins')
    identity = digest({'payload': payload, 'files': fingerprints})
    path = workspace / 'manifests' / f"{cfg['dataset']}-{identity[:16]}.json"
    write_json(path, payload)
    write_json(path.with_suffix('.provenance.json'), {'manifest_hash': identity, 'fingerprint_policy': 'resolved paths, file sizes and nanosecond modification times; headers validated', 'files': fingerprints, 'records': len(records)})
    return path, payload, identity
