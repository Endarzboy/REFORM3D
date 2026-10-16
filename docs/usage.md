# Usage

Start from an example YAML. Paths are absolute or relative to that YAML; environment variables in paths are expanded.

| Setting | Meaning |
|---|---|
| `dataset` | Dataset adapter or `custom_organs` |
| `data_root` / `manifest` | Prepare from data, or use a JSON manifest/organ CSV |
| `checkpoint_root` | Root containing the registered weight layout |
| `checkpoint_paths` | Optional mapping from registered name to an explicit weight file |
| `checkpoints`, `feature_types` | Candidates and readouts to evaluate |
| `workspace`, `run_name` | Artifact location and immutable run identity |
| `seeds` | Evaluation seeds, each isolated under `seeds/seed_<N>/` |
| `device`, `batch_size`, `workers` | Resource settings |
| `modules` | Optional subset: `acquisition`, `cross_modal`, `anatomical` |

`demo` and `screening` use the selected cohort. `paper` defaults to seeds 42/123/456 for that surface; it does not run every paper dataset or extension. The acquisition module evaluates matched-case controlled spacing. Optional native-spacing, uncertainty, and support-size analyses have dedicated module entry points; they are separate from the three-module YAML workflow.

`check` validates inputs and writes the resolved manifest without inference. `run --dry-run` does the same. A normal run verifies SHA-256 receipts before reusing completed features and metrics. Changed configuration, weights, data fingerprint, or software requires a new `run_name`.

Artifacts are grouped by dataset and module; `run_name` keeps experiments separate. All encoder features live in `seeds/seed_<N>/<checkpoint>--<feature>/features/`, alongside their identity metadata. Acquisition uses one `controlled_embedding_bundle.npz` for both evaluation reuse and analysis; organ modules use `organ_features.npz`. The `cache/<dataset>/` directory holds preprocessed inputs only. Compatible runs, seeds and organ modules reuse sealed features through filesystem hard links: each encoder folder contains an ordinary file, without a second copy of the arrays. Reuse verifies the checkpoint, input geometry, readout, software and execution settings. Do not edit generated artifacts in place.

Incomplete extraction, missing required metrics, incompatible weights, and invalid geometry fail the run. Concurrent writes to the same run or shared artifacts are rejected. Inspect `run.json` and `logs/`; the report identifies incomplete results. Requested unsupported modules fail explicitly.

Open `report.html` locally. It compares encoders separately for CLS and average-pooled patches. Multilayer tables remain in the HTML but are hidden from the page; their values are included in CSV and JSON. Sort a table by the metric relevant to your use: semantic-transfer balanced accuracy or CKA (higher), or representation drift (lower). Values show means and sample SD across evaluation seeds. Missing evaluations are identified and excluded from ranking claims. CSV and JSON retain full numeric precision.

The acquisition report states the source cohort, spacing variants, and eligible semantic labels. A high stability score alone does not establish useful semantics. Use representative development data for selection, keep the final test set separate, and confirm choices on your downstream task. Report ranks describe individual properties, without a composite score or statistical significance claim.

```text
workspace/
  manifests/<dataset>/          # validated inputs, fingerprints, scan metadata
  cache/<dataset>/
    acquisition/<identity>/     # controlled-spacing geometry
    organs/<identity>/          # shared organ crops
  runs/<dataset>/<module>/<run_name>/
    report.html
    scores.csv
    profile.json
    run.json
    logs/
    seeds/seed_<N>/<checkpoint>--<feature>/
      <module>.json
      features/
        controlled_embedding_bundle.npz  # acquisition; organ_features.npz for organ modules
        identity.json
      figures/
```

When a run evaluates both organ modules, each module has its own report, logs and seed folders. The combined report and run receipt are at `runs/<dataset>/_runs/<run_name>/`; each module's `run.json` links to that receipt.

Refresh a report from saved results without loading checkpoints or running inference:

```bash
reform3d report --run-dir workspace/runs/kits23/acquisition/my-run
```

For optional case-bootstrap uncertainty from saved acquisition features:

```bash
run_dir=workspace/runs/kits23/acquisition/my-run
python -m reform3d.modules.acquisition_robustness.controlled_uncertainty_analysis \
  --metric-jsons "$run_dir/seeds/seed_42/med3dino_rel_96--cls/perturbation_robustness.json" \
  --bootstrap-resamples 1000 \
  --output-json "$run_dir/uncertainty.json" --output-csv "$run_dir/uncertainty.csv"
```

This CPU analysis reuses the saved feature bundle and manifest; it does not rerun encoder inference. Its intervals describe case resampling within one evaluation, separately from variation across evaluation seeds. Runtime grows with the cohort size and resample count.
