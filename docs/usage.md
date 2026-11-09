# Usage

Additional options for the [README evaluation workflow](../README.md#evaluation).

## Configuration

Paths resolve beside the YAML file and expand environment variables. `manifest` takes precedence over `data_root`.

| Setting | Meaning |
|---|---|
| `dataset`, `data_root` / `manifest` | Dataset adapter and inputs |
| `checkpoint_root` / `checkpoint_paths` | Registered weight layout or explicit file mapping |
| `checkpoints`, `feature_types` | Encoder keys; `cls`, `avg_pool`, or `multilayer` readouts |
| `modules` | Supported subset of `acquisition`, `cross_modal`, `anatomical` |
| `workspace`, `run_name`, `seeds` | Output location, experiment name, evaluation seeds |
| `device`, `batch_size`, `workers` | Compute resources |

`demo` and `screening` use the selected cohort. `paper` defaults to seeds `[42, 123, 456]` when omitted; datasets and encoders remain explicitly selected.

## Run preparation and reuse

`check` and `run --dry-run` validate inputs and prepare manifests and evaluation plans without inference. They require checkpoint files and read dataset contents. `run` extracts features, evaluates each encoder/readout/seed, and writes reports. Repeat the same command to resume verified outputs; use a new `run_name` after changing configuration, inputs, weights, or software.

## Integrity checks

Invalid inputs, geometry, weights, or features fail explicitly; missing results are identified. Inspect `logs/` and `run.json`. Completed artifacts are checksum-verified before reuse; concurrent writes to one run are rejected.

## Reports and saved outputs

Reports summarize seeds by mean and sample SD; CSV/JSON retain full precision. Multilayer tables are hidden by default. Rankings compare individual metrics; read acquisition stability alongside semantic transfer.

Each encoder/readout's `features/` contains `identity.json` and `controlled_embedding_bundle.npz` (acquisition) or `organ_features.npz`. Compatible bundles share storage through hard links. `cache/<dataset>/` holds preprocessed inputs only.

## Reproducibility

One plan per module and seed fixes selections across encoders and readouts:

- **Acquisition:** source cases, semantic targets, and patient-disjoint folds across spacing variants.
- **Cross-modal:** queries, galleries, comparison pools, diagnostic cohorts, and bootstrap seeds/order.
- **Anatomical:** case-disjoint support/query examples.

Plans live under `workspace/manifests/<dataset>/` and use stable record identities. Unchanged inputs and protocol regenerate the same selections; different seeds intentionally vary them.

Compare matching seeds with identical data, weights, code, dependencies, precision, batch size, and hardware/backend. Strict deterministic execution can reduce throughput. `run.json` records hashes, plans, software, and numerical settings. Feature reuse verifies these identities and the readout; a plan hash alone does not establish equivalence.

## Custom datasets

Use [custom-organs.yaml](../configs/custom-organs.yaml) to evaluate frozen encoders on your own annotated CT and MR data with the cross-modal and anatomical modules. It does not run acquisition robustness.

Provide a CSV manifest with one row per image–organ pair:

```csv
case_id,patient_id,modality,image,mask,organ,label_id
ct01,p01,ct,images/ct01.nii.gz,masks/ct01.nii.gz,kidney,1
mr01,p02,mr,images/mr01.nii.gz,masks/mr01.nii.gz,kidney,1
```

`label_id` identifies the organ in the mask. Image and mask paths resolve beside the CSV, and their 3D geometry must match. Use the same optional `patient_id` for repeated scans of a patient so they stay in the same split group; otherwise `case_id` is used.

By default, provide at least two organs with five distinct cases per organ in each modality. Anatomical evaluation needs two support cases per organ and modality, plus case-disjoint query cases across all organs and modalities; the minimum counts alone do not guarantee a feasible split. Each organ mask must contain at least 100 voxels. The rows above illustrate the format, not a complete cohort.

Copy the configuration and set `manifest`, `checkpoint_root` (or `checkpoint_paths`), and `device` for your system:

```bash
cp configs/custom-organs.yaml configs/my-organs.yaml
# Edit configs/my-organs.yaml, then:
reform3d check --config configs/my-organs.yaml
reform3d run --config configs/my-organs.yaml
```

The example selects Med3DINO-REL, CLS features and seeds 42, 123 and 456. `check` validates inputs and constructs evaluation plans before feature extraction.

### Existing JSON manifests

Built-in datasets generate manifests automatically. Existing organ JSON uses a `samples` list with unique `sample_id`, paths, organ label/name, modality, and `source_case_id` or `patient_id`. Acquisition JSON uses `volumes` with paths, spacing, anisotropy bins, and stable case identities; it requires one source scan per patient and semantic eligibility from the complete cohort.

Paths resolve beside the JSON. Spacing/bins must match headers; acquisition modality must match configuration. Default source bins are CT: 0 and MR: 1. Keep input contents unchanged during a run.

## External encoders

Adapters use their own preprocessing and readouts; factory availability does not imply YAML-runner support. For SAM-Med3D, MiM, and GL-MAE:

```bash
python -m pip install -c requirements.txt -e '.[external]'
python -m reform3d.external_methods.sam_med3d.controlled_perturbation --help
```

Other adapters may require upstream code or separate environments. See the [frozen-feature example](../README.md#use-frozen-features-downstream) for direct loading.
