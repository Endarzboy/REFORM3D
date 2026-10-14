# Reproducibility

Evaluate the three REFORM-3D modules on KiTS23 and MMWHS using the example configurations.

| Dataset | Module | Primary metrics |
|---|---|---|
| KiTS23 | Controlled spacing sensitivity | Embedding drift, CKA, matched semantic transfer |
| MMWHS | Cross-modal alignment | mAP, Top-1, Top-5 |
| MMWHS | Anatomical generalization | Centroid and probe balanced accuracy, adaptation gain |

Install with the versions in `requirements.txt`, configure dataset and checkpoint paths, and run the examples:

```bash
reform3d check --config configs/kits23-demo.yaml
reform3d run --config configs/kits23-demo.yaml
reform3d check --config configs/mmwhs-demo.yaml
reform3d run --config configs/mmwhs-demo.yaml
```

The examples use REL/SA-96, CLS features, and seed 42. For repeated-seed evaluation, set `seeds: [42, 123, 456]`. Configure additional checkpoints and readouts explicitly; external encoders use their dedicated adapters. See [Encoders](encoders.md).

The [experiment definitions](../reproduction/experiments.json) describe reference cohorts, fold assignments, and retrieval pools: 69 controlled KiTS23 cases from a 489-case cohort, and 40 MMWHS source cases with 280 organ samples. The runner reads YAML configurations; it does not load this JSON as execution settings. Preserve case order, semantic-label eligibility, spacing variants, support/query assignments, checkpoint identity, and feature readout when comparing runs.

Record Python/library versions, precision, extraction batch size, device/backend, and numerical thread settings. Give each changed experiment a new run name; completed outputs are checked before reuse. [Usage](usage.md) describes the directory layout and cache rules.

Encoders remain frozen during evaluation. Spacing perturbations measure resampling sensitivity; low-shot probes fit labeled readouts.
