# Validation

The standalone migration completed 135 evaluations with seeds 42, 123, and 456:

| Dataset | Module | Evaluations |
|---|---|---:|
| KiTS23 | Controlled spacing sensitivity | 45 |
| MMWHS | Cross-modal alignment | 45 |
| MMWHS | Anatomical generalization | 45 |

Each route covered 15 checkpoint/readout combinations: ISO/REL/SA-96 and native-112 3DINO with CLS, average-pool, and multilayer features; SAM-Med3D, MiM, and GL-MAE with native readouts.

All 489 KiTS23 image/mask pairs passed geometry checks. Their image shapes, spacings, and bins match the recorded manifest. MMWHS provides 40 source cases and 280 organ samples. These checks do not prove historical voxel identity because the original raw inputs are unavailable.

The public checkout passes **140 regression tests** with three optional skips, without historical reference tables. A separate local environment passed dependency checks, external CPU forwards, and module replays. The GPU panel used temporary installed-wheel environments and has not been repeated in that local environment. [summary.json](summary.json) records verification status without distributing historical metric tables.

Historical score parity remains unresolved. Recorded split and pool definitions are not yet enforced by the runner. Validation is limited to these two datasets; it does not certify every published table, pretraining run, or downstream experiment.

Detailed logs and historical comparisons remain local. Run the public tests with `python -m pytest -q`; see [Repository verification](../docs/audit.md).
