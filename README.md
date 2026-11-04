# REFORM-3D

**A Representation-Centric Evaluation Framework for 3D Medical Vision Foundation Models**

[![Project](https://img.shields.io/badge/Project-Page-176b5b?logo=githubpages&logoColor=white)](https://endarzboy.github.io/reform3d-website/)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-64748b?logo=arxiv&logoColor=white)](#publication)
[![Weights](https://img.shields.io/badge/Weights-Hugging_Face-64748b?logo=huggingface)](docs/encoders.md#med3dino-checkpoints)
[![Notebooks](https://img.shields.io/badge/Examples-Jupyter-f37626?logo=jupyter&logoColor=white)](notebooks/visualize_PCA.ipynb)

REFORM-3D helps researchers compare **frozen 3D medical encoders before downstream adaptation**. It evaluates how representations respond to voxel spacing, align anatomy across CT and MR, and support organ recognition from a few labeled examples.

![Three complementary frozen-encoder evaluation modules](docs/assets/reform3d_arch.jpg)

| Module | Question | Primary metrics |
|---|---|---|
| Acquisition robustness | How do features change under controlled spacing changes? | Embedding drift, centered kernel alignment (CKA), matched semantic transfer |
| Cross-modal alignment | Can an organ in CT retrieve the same anatomy in MR, and vice versa? | Bidirectional retrieval mAP, Top-1, Top-5 |
| Anatomical generalization | How well do a few labeled examples support recognition on separate cases? | Centroid and logistic-probe balanced accuracy, adaptation gain |

The YAML workflow supports registered **Med3DINO and 3DINO** checkpoints. External encoders use dedicated adapters. The examples below use KiTS23 and MMWHS; [custom datasets](docs/data.md#custom-datasets) can be supplied through manifests. You can also [load frozen features directly](docs/encoders.md#use-frozen-features-downstream) for your own downstream pipeline.

Encoders remain frozen; low-shot probes learn from labeled support cases. See the [project website](https://endarzboy.github.io/reform3d-website/#findings) for selected manuscript findings.

## Installation

Use **Python 3.11 on Linux**. The supplied constraints target **PyTorch 2.4.0 with CUDA 12.1**:

```bash
git clone https://github.com/Endarzboy/REFORM3D.git
cd REFORM3D
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements.txt -e .
```

This installs the checkout in editable mode with the versions in `requirements.txt`. GPU evaluation requires a compatible NVIDIA driver.

For the bundled SAM-Med3D, MiM, and GL-MAE adapters, install `'.[external]'` instead of `.`. Other adapters may require upstream code or a separate environment; see [External encoders](docs/encoders.md#external-encoders).

## Checkpoints

| Encoder | YAML key | Input tensor | Backbone parameters | Weights |
|---|---|---|---:|---|
| Med3DINO-ISO | `med3dino_iso_96` | `B × 1 × 96³` | 306.78M | Hugging Face link pending |
| Med3DINO-REL | `med3dino_rel_96` | `B × 1 × 96³` | 306.78M | Hugging Face link pending |
| Med3DINO-SA | `med3dino_sa_96` | `B × 1 × 96³` | 306.78M | Hugging Face link pending |
| 3DINO | `3dino` | `B × 1 × 112³` | 306.91M | Obtain upstream 3DINO weights |

All four use ViT-L/16 backbones with 1,024-dimensional CLS and average-pooled patch features. ISO, REL, and SA refer to isotropic normalization, voxel-relative sampling, and spacing-aware sampling during Med3DINO pretraining.

Counts include learned positional embeddings and mask tokens, excluding pretraining projection heads. The small 96³/112³ difference comes from the stored positional grid; resizing inputs during inference does not add parameters.

Set local weights with `checkpoint_paths` or follow the [checkpoint directory layout](docs/encoders.md#med3dino-checkpoints). Med3DINO download links will be added here when published.

## Data preparation

Obtain the annotated datasets from their providers:

- **KiTS23 CT:** [official download instructions](https://github.com/neheller/kits23#data-download) and [provider imaging files](https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging).
- **MMWHS CT/MR:** [official access page and agreement](https://zmiclab.github.io/zxh/0/mmwhs/), with the provider's [download folder](https://mega.nz/folder/UNMF2YYI#1cqJVzo4p_wESv9P_pc8uA).

Keep data and weights outside the repository. REFORM-3D accepts these layouts:

```text
KiTS23/
├── images/
│   └── case_00000_ct.nii.gz
└── masks/
    └── case_00000_mask.nii.gz

mm_whs/
├── ct_train/
│   ├── ct_train_1001_image.nii.gz
│   └── ct_train_1001_label.nii.gz
└── mr_train/
    ├── mr_train_1001_image.nii.gz
    └── mr_train_1001_label.nii.gz
```

The [data guide](docs/data.md) explains how to convert the KiTS23 per-case layout without resampling, use MMWHS `.nii` files, and prepare a complete evaluation cohort. The tree above illustrates file naming.

## Evaluation

Start from [mmwhs-demo.yaml](configs/mmwhs-demo.yaml) for CT–MR alignment and anatomical generalization, or [kits23-demo.yaml](configs/kits23-demo.yaml) for acquisition robustness. For example, save the following as `configs/my-mmwhs.yaml` and replace the data and weight paths:

```yaml
dataset: mmwhs_ct_mr
data_root: /path/to/mm_whs
checkpoint_paths:
  med3dino_rel_96: /path/to/c96_rel_teacher_checkpoint.pth
  med3dino_sa_96: /path/to/c96_sa_teacher_checkpoint.pth
checkpoints: [med3dino_rel_96, med3dino_sa_96]
workspace: ../workspace
run_name: my-mmwhs-evaluation
preset: demo
seeds: [42]
feature_types: [cls]
device: cuda:0
batch_size: 1
```

Relative paths resolve beside the YAML file. The `demo` preset evaluates the full selected cohort.

Validate the inputs, then run:

```bash
reform3d check --config configs/my-mmwhs.yaml
reform3d run --config configs/my-mmwhs.yaml
```

`check` validates inputs and prepares the manifest without encoder inference. `run` extracts features, evaluates the selected modules, and writes reports. To compare both readouts, set `feature_types: [cls, avg_pool]`. For repeated evaluation, set `seeds: [42, 123, 456]`.

Use a new `run_name` when configuration, inputs, weights, or software change. See [Usage](docs/usage.md) for configuration options and cache reuse.

Evaluation plans fix case selections for each module and seed, shared by all encoders and readouts. Input checksums and deterministic execution settings are recorded with each run. See [Reproducibility](docs/reproduction.md).

## Results and reports

Open `workspace/runs/mmwhs_ct_mr/_runs/my-mmwhs-evaluation/report.html` for the combined report. Each module also has its own report and encoder outputs:

```text
workspace/
├── manifests/mmwhs_ct_mr/
├── cache/mmwhs_ct_mr/              # preprocessed inputs
└── runs/mmwhs_ct_mr/
    ├── _runs/my-mmwhs-evaluation/  # combined report and run receipt
    ├── cross_modal/my-mmwhs-evaluation/
    │   ├── report.html
    │   ├── scores.csv
    │   ├── logs/
    │   └── seeds/seed_42/<encoder>--cls/
    │       ├── cross_modal.json
    │       └── features/
    └── anatomical/my-mmwhs-evaluation/
```

For KiTS23, open `workspace/runs/kits23/acquisition/<run_name>/report.html`. Reports provide sortable encoder comparisons for each readout and summarize variation across evaluation seeds. See the [output guide](docs/usage.md) for saved features, metrics, and cohort details.

Rebuild a report from saved results without running inference:

```bash
reform3d report --run-dir workspace/runs/mmwhs_ct_mr/_runs/my-mmwhs-evaluation
```

## PCA visualization

Compare ISO, REL, and SA features on **CT and MR axial slices** using identical inputs for each encoder. The examples use KiTS23 CT and BraTS2023 FLAIR MR scans.

Use the self-contained [PCA notebook](notebooks/visualize_PCA.ipynb) to compare CT crop PCA and full-volume MR PCA. Image paths and settings are defined in its single runnable cell.

Install the notebook tools:

```bash
python -m pip install -c requirements.txt -e '.[notebooks]'
```

Open the PCA notebook and edit its image paths, optional mask paths, checkpoint directory, and device, then run its cell. It reads the scans directly and needs no visualization configuration file. CT crops follow each case's foreground. Features are extracted directly on each cell run; no feature cache is created. Example scans are not bundled. Launch notebooks with `jupyter lab`.

Choose one to five axial slices and add optional ground-truth contours. MR selects the slice with the largest annotated ROI area, falling back to the image-foreground center. PCA shows feature variation rather than segmentation predictions. PCA colors are fitted independently per encoder. See the [visualization guide](docs/visualization.md) for preprocessing, feature extraction, and display controls.

## Documentation

- [Data](docs/data.md): downloads, layouts, manifests, and custom datasets.
- [Encoders](docs/encoders.md): checkpoint specifications and frozen-feature loading.
- [Usage](docs/usage.md): configuration, caching, and reports.
- [Visualization](docs/visualization.md): CT/MR PCA preprocessing, feature extraction, and display controls.
- [Reproducibility](docs/reproduction.md): KiTS23/MMWHS evaluation protocols.
- [Integrity checks](docs/usage.md#integrity-checks): required input, cache, and output checks.

## Publication

Paper and arXiv links will be added here when available.

## Pretraining

The Med3DINO pretraining repository will be linked here when released. REFORM-3D includes the frozen encoder implementation needed for evaluation and feature extraction.

## License

See [LICENSE](LICENSE) for this repository and [Source notices](docs/LICENSES/README.md) for bundled components. Dataset and weight terms remain those of their respective providers.
