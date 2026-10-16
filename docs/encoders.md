# Encoders

## Med3DINO checkpoints

The public YAML runner supports registered Med3DINO and 3DINO checkpoints. Store weight files outside Git and set explicit paths in your YAML.

| Registry key | Backbone | Input tensor | Patch grid | Parameters | Download |
|---|---|---|---|---:|---|
| `med3dino_iso_96` | ViT-L/16 | `B × 1 × 96³` | `6³` | 306,780,160 | Hugging Face link pending |
| `med3dino_rel_96` | ViT-L/16 | `B × 1 × 96³` | `6³` | 306,780,160 | Hugging Face link pending |
| `med3dino_sa_96` | ViT-L/16 | `B × 1 × 96³` | `6³` | 306,780,160 | Hugging Face link pending |
| `3dino` | ViT-L/16 | `B × 1 × 112³` | `7³` | 306,910,208 | Obtain upstream 3DINO weights |

Counts describe the implemented inference backbone, including learned positions and mask tokens; pretraining projection heads are excluded. All four have 24 transformer blocks, 16 heads, and a 1,024-dimensional embedding. Positional embeddings account for the input-dependent parameter difference. Input tensors do not specify physical FOV.

Published Med3DINO checkpoint destinations will be linked here when available. Weight access and redistribution terms remain separate from source-code licensing; see [source notices](LICENSES/README.md).

Expected filenames when using `checkpoint_root`:

```text
checkpoints/
  med3dino_iso/c96_iso_teacher_checkpoint.pth
  med3dino_rel/c96_rel_teacher_checkpoint.pth
  med3dino_sa/c96_sa_teacher_checkpoint.pth
  3dinov2/3dinov2_teacher_checkpoint.pth
```

Alternatively use any local filenames with explicit mappings:

```yaml
checkpoints: [med3dino_iso_96, med3dino_rel_96, med3dino_sa_96]
checkpoint_paths:
  med3dino_iso_96: /path/to/iso_teacher.pth
  med3dino_rel_96: /path/to/rel_teacher.pth
  med3dino_sa_96: /path/to/sa_teacher.pth
feature_types: [cls, avg_pool]
```

Required backbone weights must load completely; known unused pretraining heads are allowed. Renaming a checkpoint does not make an incompatible architecture compatible.

## Use frozen features downstream

`reform3d.encoders.factory.create_feature_extractor` loads the registered architecture, sets evaluation mode, and freezes parameters. Its `extract` method accepts preprocessed tensors, not raw scans.

```python
import torch
from reform3d.encoders.factory import create_feature_extractor

encoder = create_feature_extractor(
    "med3dino_sa_96",
    checkpoint_path="/path/to/c96_sa_teacher_checkpoint.pth",
    device="cuda:0",
)
# batch: [B, 1, 96, 96, 96], prepared by your documented preprocessing pipeline.
with torch.inference_mode():
    features = encoder.extract(batch, normalize=True)
    global_features = features["cls"]       # [B, 1024]
    pooled_features = features["avg_pool"]  # [B, 1024]
```

`normalize=True` L2-normalizes returned embeddings; it does not normalize raw-image intensities. `multilayer` concatenates the last four CLS vectors into 4,096 dimensions. `avg_pool` is the runtime name for global average pooling (GAP) over patch tokens.

Use [the configured evaluation pipeline](usage.md) for benchmark comparisons. For patch-feature PCA and reference-feature similarity, the [visualization notebooks](visualization.md) provide complete image loading and a separate, explicitly documented viewing transform.

## External encoders

External encoders retain their dedicated `reform3d.external_methods` entry points, preprocessing, and native features. The generic factory provides lazy access to adapters; it does not make every external method a registered YAML-runner checkpoint.

For bundled SAM-Med3D, MiM, and GL-MAE dependencies, install:

```bash
python -m pip install -c requirements.txt '.[external]'
```

Other adapters may need upstream code or a separate environment. OpenMind requires its compatible upstream source and environment. Missing backends must be configured explicitly; no adapter searches the old research checkout.

YAML runs record checkpoint hashes in `run.json`. For external encoders, use the dedicated adapter's `--help` for its required inputs and options. Source attribution and license records are in [Source notices](LICENSES/README.md).
