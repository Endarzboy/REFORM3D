# CT/MR feature visualization

Compare Med3DINO-ISO, REL and SA with the same input tensor for each case. The full-volume notebook shows reference-feature similarity on uncropped scans. The input preview and PCA notebooks use shared 96³ crops.

```bash
python -m pip install -c requirements.txt '.[notebooks]'
cp notebooks/visualization_config.example.json notebooks/visualization_config.json
jupyter lab
```

Set paths in `visualization_config.json`. It contains shared `checkpoints` and `device` settings, a `pca` section for crop preview/PCA, and a `similarity` section for complete scans and reference points. Each section requires its own output directory. A section can override `device` when needed. Masks are optional and supply reference contours.

| Notebook | Purpose |
| --- | --- |
| [00 · CT/MR inputs](../notebooks/00_case_selection.ipynb) | Preview and freeze the shared crops used for PCA. |
| [01 · Full-volume similarity](../notebooks/01_attention.ipynb) | Compare CT/MR reference-feature overlays, including one additional example per modality. |
| [02 · PCA](../notebooks/02_pca.ipynb) | Extract or reload crop features, then compare patch-only and CLS-conditioned PCA. |

Relative paths resolve beside the JSON. Open notebooks from this checkout or `notebooks/`; elsewhere, set `REFORM3D_REPO_ROOT`. `REFORM3D_VIZ_CONFIG` selects an alternative configuration file. Restart the kernel after switching checkouts. All notebook helpers are in [visualization_utils.py](../notebooks/visualization_utils.py): `setup(workflow="pca")` selects crop settings and `setup(workflow="similarity")` selects complete-scan settings. Only the selected workflow's inputs are required.

## Full-volume reference-feature similarity

The entire source field of view is retained in an isotropic model grid. Its longest axis has at most 224 samples; symmetric padding makes every dimension divisible by 16. Physical proportions and the tensor are identical across encoders. This shared inference geometry differs from their individual pretraining augmentations.

Background images retain the complete native axial frame: 512² pixels for these KiTS23 CT scans and 240² for these standardized BraTS MR scans. The coarser 3D feature map is sampled onto the same physical slice. Interpolation smooths the overlay without adding feature detail.

### Reference and slice controls

Give each case a unique `id` and a `reference_ras_mm` point: **right, anterior, superior in millimetres**. Select the point from the source image before inspecting encoder responses. Its containing 16³ model patch is the shared reference across ISO, REL and SA.

```python
import matplotlib.pyplot as plt
from visualization_utils import setup, plot_full_volume_similarity
from reform3d.full_volume_features import load_or_extract_full_volume_features

config = setup(workflow="similarity")
arrays, receipt = load_or_extract_full_volume_features(config)
plot_full_volume_similarity(
    config, arrays, receipt, case_id="MR", num_slices=5,
)
plt.show()
```

Use `case_id="CT"`, `"MR-additional"` or `"CT-additional"` for the other examples. Change the configured reference point or pass `reference_ras_mm=(x, y, z)` directly to the plotter. Neither choice requires feature extraction again.

`num_slices` accepts **1–5**. Automatic levels surround the reference, spaced by physical patch depth. `slice_indices=[...]` instead selects up to five distinct axial indices in the canonical source scan. Slice selection uses neither annotations nor feature responses. White markers identify the reference point; dashed boxes show its physical patch footprint only within its axial slab. Cyan contours show the supplied ground truth.

### Reading the overlay

For each encoder, L2-normalize its `[patches, 1024]` feature matrix and take the dot product with the reference feature. Reshape the cosine values to the 3D patch grid and map patch centers into source coordinates. The reference patch has similarity 1 by definition; responses beyond that patch show matches elsewhere in the scan.

All encoders share the fixed cosine range **[−1, 1]**. Positive cosine values control overlay opacity; values are never scaled independently per encoder. Set `opacity=0.65` for maximum opacity, `interpolation="nearest"` to inspect the patch grid, or `show_contours=False` to hide contours. Annotations and foreground masks never alter similarity values. These maps illustrate feature similarity, not segmentation accuracy.

The physical feature resolution depends on scan extent and model grid. A sharper background does not imply finer localization. CT can use `display_window=[-160, 240]` for soft-tissue contrast; MR defaults to recorded source intensity percentile bounds. These settings affect only the background.

The additional CT, KiTS23 `case_00001`, has approximately 0.799 mm native in-plane spacing, compared with 0.977 mm for `case_00005`. Additional BraTS FLAIR `BraTS-GLI-00003-000` uses the same standardized 1 mm, 240 × 240 × 155 grid as the first MR case. Both additional cases were selected from source images before viewing their encoder features.

## Shared crops for input preview and PCA

Notebook 00 samples one seeded, foreground-informed physical crop per case and resamples it to 96³. ISO, REL and SA receive identical tensors, with no checkpoint-specific augmentation. Within the `pca` section, a case's `crop` can override the common field of view:

```json
{
  "seed": 42,
  "crop": {"fov_mm": 192.0},
  "cases": [
    {"modality": "CT", "image": "/path/to/ct.nii.gz"},
    {"modality": "MR", "image": "/path/to/mr.nii.gz",
     "crop": {"fov_mm": 160.0}}
  ]
}
```

This fragment shows the contents of the `pca` section; retain its output directory and the shared checkpoints/device from the example JSON. Choose the field of view during input preview and keep it fixed before extracting features. An optional `center_ras_mm` sets a physical anatomical center instead of sampling one. If that center was curated from annotations, describe the illustration as annotation-informed; the helper itself never derives a center from the mask.

Intensities are clipped at the source scan's 0.05th/99.95th percentiles and scaled to [−1, 1]. Image foreground is estimated on a grid of at most 128 samples per axis: threshold at 2.5% of the clipped intensity range, retain the largest connected component, and fill holes. Candidate crop centers favor foreground coverage and avoid source padding where possible.

Previewing freezes inputs under `output_dir/inputs/`. Changing source inputs, seed, field of view or center requires a new output directory. Receipts record the tensor identities, geometry, foreground fraction and source coverage.

Crop/PCA displays accept **1–5 axial slices**. Automatic levels span regions with at least 50% of peak image-foreground area; a single slice uses the maximum area. These levels are shared by preview and PCA and are independent of annotations and encoder responses. Full-volume similarity uses its separate, reference-centered source levels.

## PCA

Each case/checkpoint supplies `[216, 1024]` patch features from the 6³ grid. `mode="patch"` L2-normalizes them and fits three components on patches containing at least 25% image foreground from the **whole 3D crop**. Component scores form a `[6, 6, 6, 3]` RGB volume, scaled using foreground 1st/99th-percentile bounds. The same basis and scaling apply to every displayed slice of that case/checkpoint.

PCA is fitted independently for each encoder. Colors show feature variation within a panel; matching colors across encoders do not imply matching anatomy. The [DINOv3 PCA tutorial](https://github.com/facebookresearch/dinov3/blob/main/notebooks/pca.ipynb) provides the foreground patch-feature PCA inspiration.

`mode="combined"` uses `(1 − w) × unit_patch + w × unit_CLS`, normalizes each patch vector, then fits PCA. `cls_weight=w` accepts `0 ≤ w < 1`. This exploratory fusion adds global context to spatial features. Adding the same CLS vector without renormalization would disappear under PCA centering; CLS and GAP alone have no spatial PCA map. PCA colors do not establish organ boundaries or segmentation performance.

## Extraction and saved outputs

Notebook 01 extracts one full-volume feature set per case/checkpoint on its first run: **twelve float32 forwards** for four cases and three encoders, sequentially at batch size one. The maximum input axis is 224 and the patch-count limit is 2744. Later runs reuse those features when reference points, slices, opacity, contours or display windows change. A new source, checkpoint or model geometry requires a separate output directory; incomplete or incompatible caches fail explicitly.

Notebook 02 extracts crop features on its first run and reloads compatible features thereafter. Crop extraction uses two float32 forwards per case/checkpoint, sequentially at batch size one. Its cache stays separate from full-volume features.

Receipts record source/checkpoint hashes, input tensors, geometry and software. Full-volume PNG/PDF previews, galleries, native cosine-map NPZ files and display receipts are saved under `output_dir/full_volume_similarity/`; crop/PCA outputs use the base configuration's output directory. Display receipts record queries, patch footprints, physical slice levels and color settings.

Notebooks are shared without executed scan images. Obtain scans and checkpoints from their providers and follow their terms.
