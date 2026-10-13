"""Adversarial input and annotation tests using small synthetic NIfTI volumes."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import yaml

from reform3d.prepare import prepare_manifest
from reform3d.settings import load_settings


def _settings(tmp_path, **changes):
    config = {'dataset': 'kits23', 'manifest': 'manifest.json',
              'checkpoints': ['med3dino_rel_96'], 'workspace': 'workspace',
              'device': 'cpu'}
    config.update(changes)
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    return load_settings(path)


def _volume(path, *, spacing=(1, 1, 1), value=1):
    nib.save(nib.Nifti1Image(np.full((6, 6, 6), value, dtype=np.int16),
                           np.diag([*spacing, 1])), path)
    return str(path)


def _acquisition(tmp_path):
    records = []
    for case in range(2):
        records.append({'case_id': str(case), 'file_path': _volume(tmp_path / f'image{case}.nii'),
                        'label_path': _volume(tmp_path / f'mask{case}.nii'),
                        'has_label': True, 'spacing': [1, 1, 1], 'anisotropy_bin': 0})
    return records


def _manifest(tmp_path, records, key='volumes'):
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({key: records, 'dataset': 'kits23', 'modality': 'ct'}))
    return path


@pytest.mark.parametrize('changes', [
    {'source_bin': True}, {'source_bin': -1}, {'source_bin': '0'},
    {'batch_size': True}, {'bbox_margin': False}, {'n_source_volumes': True},
    {'dataset': 'custom_organs', 'required_modalities': []},
    {'dataset': 'custom_organs', 'required_modalities': ['ct', 'ct']},
    {'dataset': 'custom_organs', 'required_modalities': ['ct', 'pet']},
])
def test_ambiguous_protocol_settings_are_rejected(tmp_path, changes):
    with pytest.raises(ValueError):
        _settings(tmp_path, **changes)


def test_acquisition_cannot_count_same_image_as_two_cases(tmp_path):
    records = _acquisition(tmp_path)
    records[1]['file_path'] = records[0]['file_path']
    _manifest(tmp_path, records)
    with pytest.raises(ValueError, match='duplicate|Duplicate|same image'):
        prepare_manifest(_settings(tmp_path))


def test_declared_spacing_must_match_volume_geometry(tmp_path):
    records = _acquisition(tmp_path)
    records[0]['spacing'] = [1, 1, 3]
    records[0]['anisotropy_bin'] = 2
    _manifest(tmp_path, records)
    with pytest.raises(ValueError, match='spacing|geometry'):
        prepare_manifest(_settings(tmp_path))


def test_source_bin_must_contain_enough_distinct_cases(tmp_path):
    records = _acquisition(tmp_path)
    _manifest(tmp_path, records)
    with pytest.raises(ValueError, match='source|Source|eligible'):
        prepare_manifest(_settings(tmp_path, source_bin=1))


def test_controlled_semantic_cohort_cannot_omit_annotations(tmp_path):
    records = _acquisition(tmp_path)
    records[0].pop('label_path')
    records[0]['has_label'] = False
    _manifest(tmp_path, records)
    with pytest.raises(ValueError, match='annotation|label|semantic'):
        prepare_manifest(_settings(tmp_path))


def _organ_records(tmp_path):
    records = []
    for modality in ('ct', 'mr'):
        for case in range(5):
            image = _volume(tmp_path / f'{modality}{case}.nii')
            mask = _volume(tmp_path / f'{modality}{case}-mask.nii')
            for organ in ('kidney', 'heart'):
                records.append({'sample_id': f'{modality}{case}-{organ}',
                                'source_case_id': str(case), 'patient_id': str(case),
                                'modality': modality, 'primary_organ': organ,
                                'file_path': image, 'mask_path': mask,
                                'mask_label_value': 1})
    return records


def test_one_image_cannot_be_both_ct_and_mr(tmp_path):
    records = _organ_records(tmp_path)
    records[10]['file_path'] = records[0]['file_path']
    _manifest(tmp_path, records, 'samples')
    with pytest.raises(ValueError, match='modality|modalities'):
        prepare_manifest(_settings(tmp_path, dataset='custom_organs'))


def test_same_image_and_organ_cannot_be_repeated_under_new_sample_id(tmp_path):
    records = _organ_records(tmp_path)
    records.append({**records[0], 'sample_id': 'duplicate-under-new-name'})
    _manifest(tmp_path, records, 'samples')
    with pytest.raises(ValueError, match='duplicate|Duplicate|repeated'):
        prepare_manifest(_settings(tmp_path, dataset='custom_organs'))


@pytest.mark.parametrize('method', ['extract_organ_presence', 'extract_dominant_organ', 'extract_foreground_voxel_count'])
def test_corrupt_mask_cannot_become_an_absent_organ(tmp_path, method):
    from reform3d.modules.acquisition_robustness import semantic_labels
    mask = tmp_path / 'truncated-mask.nii'
    _volume(mask)
    with mask.open('r+b') as stream:
        stream.truncate(352)
    assert nib.load(mask).shape == (6, 6, 6)
    arguments = (mask,) if method == 'extract_foreground_voxel_count' else (mask, {1: 'kidney'})
    with pytest.raises((ValueError, OSError, RuntimeError)):
        getattr(semantic_labels, method)(*arguments)


def test_corrupt_present_structure_mask_cannot_become_absent(tmp_path):
    from reform3d.modules.acquisition_robustness.semantic_labels import extract_directory_structure_presence
    (tmp_path / 'kidney.nii.gz').write_bytes(b'corrupted supplied annotation')
    with pytest.raises((ValueError, OSError, RuntimeError, nib.filebasedimages.ImageFileError)):
        extract_directory_structure_presence(tmp_path, ['kidney'])


def test_true_absent_organ_preserves_zero_target(tmp_path):
    from reform3d.modules.acquisition_robustness.semantic_labels import extract_organ_presence
    mask = _volume(tmp_path / 'empty-mask.nii', value=0)
    presence, volumes = extract_organ_presence(mask, {1: 'kidney'})
    np.testing.assert_array_equal(presence, [0])
    assert volumes == {1: 0}


def test_semantic_cache_cardinality_cannot_substitute_for_case_identity(tmp_path):
    from reform3d.modules.acquisition_robustness.semantic_labels import extract_labels_for_manifest
    from reform3d.modules.acquisition_robustness.config import resolve_semantic_target_spec
    records = _acquisition(tmp_path)
    manifest = _manifest(tmp_path, records[:1])
    cache = tmp_path / 'labels.json'
    cache.write_text(json.dumps({'unrelated-case': {'organ_presence': [1, 1],
                    'label_path': records[1]['label_path'], 'extraction_mode': 'label_id_multilabel',
                    'semantic_target': [1, 1]}}))
    result = extract_labels_for_manifest(manifest, cache_path=cache,
                                        target_spec=resolve_semantic_target_spec('kits23', 'ct'))
    assert set(result) == {records[0]['file_path']}


def test_paired_custom_patient_groups_remain_disjoint_in_low_shot_split(tmp_path):
    from reform3d.modules.anatomical_generalization.few_shot_transfer_analysis import (
        _build_case_disjoint_support_and_query_sets, _group_by_organ_and_modality)
    records = _organ_records(tmp_path)
    for row in records:
        row['source_case_id'] = row['modality'] + row['patient_id']
    _manifest(tmp_path, records, 'samples')
    _, payload, _ = prepare_manifest(_settings(tmp_path, dataset='custom_organs'))
    groups = _group_by_organ_and_modality(payload['samples'], ['ct', 'mr'])
    for seed in (42, 123, 456):
        support, query, _ = _build_case_disjoint_support_and_query_sets(
            groups, ['heart', 'kidney'], ['ct', 'mr'], 2, None, np.random.default_rng(seed))
        assert support and query
        assert {row['patient_id'] for row in support}.isdisjoint(row['patient_id'] for row in query)


@pytest.mark.parametrize('variant', ['config_disagrees', 'record_disagrees', 'missing_unknown'])
def test_acquisition_modality_cannot_select_the_wrong_protocol(tmp_path, variant):
    records = _acquisition(tmp_path)
    manifest = _manifest(tmp_path, records)
    changes = {}
    if variant == 'config_disagrees':
        changes['modality'] = 'mr'
    elif variant == 'record_disagrees':
        records[0]['modality'] = 'mr'
        _manifest(tmp_path, records)
    else:
        payload = json.loads(manifest.read_text())
        payload.pop('modality')
        payload['dataset'] = 'unregistered_scans'
        manifest.write_text(json.dumps(payload))
        changes['dataset'] = 'unregistered_scans'
    with pytest.raises(ValueError, match='modality|modalities'):
        prepare_manifest(_settings(tmp_path, **changes))


def test_mri_uses_its_protocol_source_bin_unless_explicitly_overridden(tmp_path):
    records = _acquisition(tmp_path)
    for record in records:
        record['file_path'] = _volume(Path(record['file_path']), spacing=(1, 1, 2))
        record['label_path'] = _volume(Path(record['label_path']), spacing=(1, 1, 2))
        record.update(spacing=[1, 1, 2], anisotropy_bin=1, modality='mr')
    _manifest(tmp_path, records)
    payload = {'volumes': records, 'dataset': 'unregistered_mri', 'modality': 'mr'}
    (tmp_path / 'manifest.json').write_text(json.dumps(payload))
    cfg = _settings(tmp_path, dataset='unregistered_mri')
    _, prepared, _ = prepare_manifest(cfg)
    assert cfg['source_bin'] == 1
    assert prepared['modality'] == 'mr'
    with pytest.raises(ValueError, match='source|Source|eligible'):
        prepare_manifest(_settings(tmp_path, dataset='unregistered_mri', source_bin=0))


@pytest.mark.parametrize('presence', [[1, 0], [1, float('nan'), 0]])
def test_malformed_semantic_cache_cannot_be_reused(tmp_path, presence):
    from reform3d.modules.acquisition_robustness.semantic_labels import extract_labels_for_manifest
    from reform3d.modules.acquisition_robustness.config import resolve_semantic_target_spec
    records = _acquisition(tmp_path)
    manifest = _manifest(tmp_path, records[:1])
    cache = tmp_path / 'labels.json'
    cache.write_text(json.dumps({records[0]['file_path']: {'organ_presence': presence,
                    'label_path': records[0]['label_path'], 'extraction_mode': 'label_id_multilabel',
                    'semantic_target': presence}}))
    with pytest.raises(ValueError, match='cache|Cache|semantic'):
        extract_labels_for_manifest(manifest, cache_path=cache,
                                    target_spec=resolve_semantic_target_spec('kits23', 'ct'))


def test_manifest_anisotropy_bin_must_match_spacing(tmp_path):
    records = _acquisition(tmp_path)
    for record in records:
        record['anisotropy_bin'] = 2
    _manifest(tmp_path, records)
    with pytest.raises(ValueError, match='anisotropy|spacing'):
        prepare_manifest(_settings(tmp_path, source_bin=2))


@pytest.mark.parametrize('method', ['extract_organ_presence', 'extract_dominant_organ', 'extract_foreground_voxel_count'])
def test_nonfinite_annotation_is_not_a_negative_semantic_target(tmp_path, method):
    from reform3d.modules.acquisition_robustness import semantic_labels
    mask = tmp_path / 'nonfinite-mask.nii'
    nib.save(nib.Nifti1Image(np.full((6, 6, 6), np.nan), np.eye(4)), mask)
    arguments = (mask,) if method == 'extract_foreground_voxel_count' else (mask, {1: 'kidney'})
    with pytest.raises(ValueError, match='nonfinite|finite|annotation'):
        getattr(semantic_labels, method)(*arguments)


def test_different_deterministic_noise_seeds_cannot_share_geometry_cache():
    import torch
    from dataclasses import replace
    from reform3d.modules.acquisition_robustness.config import ControlledPerturbationConfig
    from reform3d.modules.acquisition_robustness.perturbation_robustness_analysis import _apply_extra_acquisition_perturbation
    first = ControlledPerturbationConfig(extra_perturbations=['noise'], extra_perturbation_seed=17)
    second = replace(first, extra_perturbation_seed=23)
    image = torch.full((1, 5, 5, 5), 0.5)
    left = _apply_extra_acquisition_perturbation(image, 'noise', first.extra_perturbation_seed)
    right = _apply_extra_acquisition_perturbation(image, 'noise', second.extra_perturbation_seed)
    assert not torch.equal(left, right)
    assert first.cache_signature() != second.cache_signature()


@pytest.mark.parametrize('changes', [
    {'target_spacings': [(1.0, 1.0, 1.000001), (1.0, 1.0, 3.0)]},
    {'target_z_spacings': [2.000001, 4.0]},
    {'physical_fov_mm': (96.000001, 96.0, 96.0)},
])
def test_geometry_cache_preserves_full_precision_protocol_values(changes):
    from dataclasses import replace
    from reform3d.modules.acquisition_robustness.config import ControlledPerturbationConfig
    first = ControlledPerturbationConfig(target_spacings=[(1.0, 1.0, 1.0), (1.0, 1.0, 3.0)],
                                        target_z_spacings=[2.0, 4.0], physical_fov_mm=(96.0, 96.0, 96.0),
                                        protocol_name='native_inplane_z' if 'target_z_spacings' in changes else 'fixed_absolute')
    second = replace(first, **changes)
    assert first.cache_signature() != second.cache_signature()


def test_per_volume_geometry_cache_remains_shared_across_cohort_limits():
    from dataclasses import replace
    from reform3d.modules.acquisition_robustness.config import ControlledPerturbationConfig
    first = ControlledPerturbationConfig(n_source_volumes=3)
    assert first.cache_signature() == replace(first, n_source_volumes=10).cache_signature()
    assert first.cache_signature() == replace(first, n_source_volumes=None).cache_signature()
