# CT/MR PCA visualization

The self-contained [PCA notebook](../notebooks/visualize_PCA.ipynb) compares Med3DINO-ISO, REL and SA on CT and MR axial slices. It includes image loading, preprocessing, feature extraction, PCA and plotting in one runnable cell. REFORM-3D supplies the frozen encoder loader.

```bash
python -m pip install -c requirements.txt -e '.[notebooks]'
jupyter lab
```

Run from the repository root or `notebooks/`. Edit `cases`, `checkpoint_dir` and `device` at the top of the cell. Supply your own images and checkpoint files; example scans are not bundled. Mask paths can be `None`.

## Inputs and crops

All encoders receive the same prepared tensor within each modality. These shared inference inputs differ from the individual encoders' pretraining augmentations.

CT is resampled to **96³**. By default, its physical crop fits the supplied mask foreground plus `ct_margin_mm` on each side; `ct_fov_mm` is the minimum cube size. These mask-based crops are annotation-informed illustrations. Set `ct_foreground="image"` to use the largest image-body component instead. Missing or empty masks also fall back to image foreground. Short source scans are padded at the low intensity bound where the cube extends beyond the scan.

CT image foreground is estimated on a grid of at most 128 samples per axis: threshold at 2.5% of the source intensity range between the 0.05th and 99.95th percentiles, retain the largest connected component, and fill holes.

MR retains the complete source field of view. Antialiased isotropic resampling and symmetric padding produce dimensions divisible by 16; `mr_max_axis` controls the resolution, with a default of 224. Both modalities are clipped at the source 0.05th/99.95th intensity percentiles and scaled to [−1, 1].

## PCA

CT supplies a 6×6×6 grid of 216 patch features. MR supplies `[N, 1024]` features, where `N` is the product of its 3D patch-grid dimensions. Features are L2-normalized; three PCA components are fitted on patches containing at least 25% image foreground. CT uses the body foreground described above; MR uses normalized intensities above −0.95. Masks can select the CT crop, but PCA fitting within it uses image foreground rather than mask labels.

Component scores form a 3D RGB volume, scaled using foreground 1st/99th-percentile bounds. One basis and scaling apply to every displayed slice of that case/checkpoint. Colors are fitted independently per encoder: matching colors across panels do not imply matching anatomy. PCA illustrates feature variation and does not establish organ boundaries or segmentation accuracy.

RGB values are sampled from patch centers onto physical axial planes and limited to image foreground. Interpolation smooths the display without adding feature detail. The [DINOv3 PCA tutorial](https://github.com/facebookresearch/dinov3/blob/main/notebooks/pca.ipynb) provides the foreground patch-feature PCA inspiration.

## Display controls

- `num_slices=1` shows the CT crop's central plane. MR selects the closest-RAS source axial slice with the largest annotated ROI area and displays its physical z-coordinate. Missing or empty masks use the full-volume image-foreground center. Choose up to five slices for surrounding views.
- Multiple CT levels span slices with at least 50% of peak image-foreground area. MR levels surround the automatically selected physical plane, clipped to the source bounds. Selection is shared by all encoders, uses no feature responses, and leaves the full-volume PCA fit unchanged. Annotation-based slice selection is an illustrative use of ground truth.
- `overlay_opacity` controls the RGB blend. Supplied ground-truth masks appear as cyan contours.

CT displays its crop; MR displays the complete native frame. Compare encoders within each modality because their spatial coverage differs.

## Features and saved outputs

Each cell run performs six float32 encoder forwards, sequentially at batch size one: two modalities and three checkpoints. Features pass directly to PCA in memory. There is no cache dictionary, feature archive, or cache directory; rerunning the cell extracts the features again.

To save a PCA figure, assign the result of `plot_pca` to `fig` and call `fig.savefig("pca.png", dpi=200, bbox_inches="tight")`.

Obtain scans and checkpoints from their providers and follow their terms.
