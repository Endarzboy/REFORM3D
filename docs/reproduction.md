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

The YAML runner canonicalizes records by stable identity and prepares one evaluation plan per module and seed before feature extraction. All encoders and readouts consume the same applicable plan:

| Module | Fixed selections |
|---|---|
| Acquisition | Source cases, semantic-label eligibility, and patient-disjoint folds shared across spacing variants |
| Cross-modal | Retrieval queries/galleries, positive/negative comparison pools, LISI cohorts, and bootstrap ordering/seeds |
| Anatomical | Case-disjoint support/query examples shared across target-organ evaluations |

Plans live beside the dataset manifest and are verified when reused. Reordering manifest records or feature rows does not redraw selections. Unchanged inputs and protocol regenerate the same plans after rebuilding the workspace. Missing samples, altered plans, or incompatible artifacts fail explicitly.

Use the same input contents, checkpoint files, code, dependency versions, precision, batch size, and hardware/backend when reproducing numerical results. The runner enables strict deterministic algorithms, fixes numerical thread counts and inference seeds, and records these settings with input and plan hashes. Unsupported nondeterministic operations fail instead of silently continuing. Different evaluation seeds intentionally select different folds or examples; compare matching seeds across runs.

Deterministic settings can reduce throughput. Exact floating-point equality across different hardware or library versions is not guaranteed; validate numerical tolerances when changing environments. Give each changed experiment a new run name. Completed outputs are checked before reuse. [Usage](usage.md) describes the directory layout and cache rules.

These automatic plans cover the three-module YAML workflow. Optional uncertainty analyses and external-encoder adapters retain their documented protocols and entry points.

Encoders remain frozen during evaluation. Spacing perturbations measure resampling sensitivity; low-shot probes fit labeled readouts.

This workflow covers the modules and datasets described above. It does not establish numerical equivalence with the full manuscript tables or reproduce encoder pretraining and downstream training.
