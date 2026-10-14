# Integrity checks

REFORM-3D checks inputs and saved outputs before evaluation or reuse.

| Check | Required behavior |
|---|---|
| Inputs | Reject missing files, duplicate identities, unreadable annotations, and inconsistent spacing or image/mask geometry |
| Checkpoints | Require complete backbone weights with compatible shapes |
| Features | Reject missing, duplicate, nonfinite, or incompatible embeddings |
| Caches | Reuse only when input, protocol, checkpoint, and numerical settings agree |
| Completed runs | Verify feature and metric checksums before reuse; preserve results during validation |
| Imports | Run independently of the original Med3DINO checkout |

See [Testing](../validation/README.md) for the test suite and [Usage](usage.md) for cache reuse and failure handling.
