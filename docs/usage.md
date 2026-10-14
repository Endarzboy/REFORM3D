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

`demo` and `screening` use the selected cohort. `paper` defaults to seeds 42/123/456 for that surface; it does not run every paper dataset or extension. The public acquisition command evaluates controlled spacing; native-spacing and supplementary analyses retain their module entry points.

`check` validates inputs and writes the resolved manifest without inference. `run --dry-run` does the same. A normal run verifies SHA-256 receipts before reusing completed features and metrics. Changed configuration, weights, data fingerprint, or software requires a new `run_name`.

`cache/` stores geometry by input/protocol identity. `features/` additionally separates checkpoint hashes, readouts, precision, batch size, and GPU backend. They can be reused across metric seeds; random sampling remains in the seeded results. Do not edit generated artifacts in place.

Incomplete extraction, missing required metrics, incompatible weights, and invalid geometry fail the run. Concurrent writes to the same run or shared artifacts are rejected. Inspect `run.json` and `logs/`; the report identifies incomplete results. Requested unsupported modules fail explicitly.

Use representative development data for selection. Keep the final test set separate. Reports give per-property means and evaluation-seed SD, without a composite ranking.

```text
workspace/
  manifests/          # validated inputs and fingerprints
  cache/              # reusable geometry
  features/           # reusable checkpoint embeddings
  runs/<run_name>/
    report.html
    scores.csv
    profile.json
    run.json
    logs/
    seeds/seed_<N>/<checkpoint>--<feature>/<module>.json
```
