# Repository verification

Validation covers the standalone package, input preparation, cache reuse, and the three evaluation modules on KiTS23 and MMWHS.

| Check | Required behavior |
|---|---|
| Inputs | Reject missing files, duplicate identities, unreadable annotations, and inconsistent spacing or image/mask geometry |
| Checkpoints | Require complete backbone weights with compatible shapes |
| Features | Reject missing, duplicate, nonfinite, or incompatible embeddings |
| Caches | Reuse only when input, protocol, checkpoint, and numerical settings agree |
| Completed runs | Verify feature and metric checksums before reuse; preserve results during validation |
| Imports | Run independently of the original Med3DINO checkout |

The tests exercise these failure cases, CPU feature extraction, external adapter interfaces, readout behavior, and seeded aggregation. Install with `python -m pip install -c requirements.txt '.[test]'`, then run `python -m pytest -q`. Optional artifact-dependent tests may be skipped when their inputs are unavailable.

[Validation](../validation/README.md) records the completed two-dataset scope and remaining limitations. Internal manuscript audits, historical table comparisons, and detailed migration logs are kept outside the release.
