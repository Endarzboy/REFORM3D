# Reproducibility

This release supports validation of the three REFORM-3D modules on KiTS23 and MMWHS. Published main and supplementary table results are not distributed, and full-paper table regeneration is outside the release scope.

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

The examples use REL/SA-96, CLS features, and seed 42. For repeated-seed evaluation, set `seeds: [42, 123, 456]`. The completed validation panel covered 15 checkpoint/readout combinations per seed across the three dataset/module routes; external encoders use their dedicated adapters. See [Encoders](encoders.md).

Preserve case order, semantic-label eligibility, spacing variants, support/query assignments, checkpoint identity, and feature readout. KiTS23 uses 69 controlled cases selected from the full 489-case cohort. MMWHS uses 40 source cases and 280 organ samples. [Experiment definitions](../reproduction/experiments.json) record the verified protocol assignments; the runner does not enforce this file yet.

Record Python/library versions, precision, extraction batch size, device/backend, and numerical thread settings. Give each changed experiment a new run name; completed outputs are checked before reuse. [Usage](usage.md) describes the directory layout and cache rules.

All 135 panel evaluations completed. Historical numerical parity remains unresolved, so completion does not establish identical published scores. The GPU panel used temporary installed-wheel environments; the separately validated local environment has not repeated that full GPU panel. See [Validation](../validation/README.md).

These are frozen-encoder evaluations. Spacing perturbations test resampling sensitivity; low-shot probes fit labeled readouts. Their results do not establish pretraining repeatability or guarantee downstream performance.
