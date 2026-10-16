# Data

Keep data and weights outside the repository. Edit their paths in an example YAML; `reform3d check` prepares and validates the manifest automatically.

## Download annotated data

- **KiTS23:** follow the [official repository's download instructions](https://github.com/neheller/kits23#data-download). Its `kits23_download_data` command retrieves imaging from the [provider's Hugging Face dataset](https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging) into the per-case layout. Use the KiTS23 case list rather than all cases hosted in that cross-challenge imaging collection. Keep the corresponding segmentation files and case identifiers.
- **MMWHS:** use the [official access page](https://zmiclab.github.io/zxh/0/mmwhs/), its [registration agreement](https://zmiclab.github.io/zxh/0/mmwhs/res/Data_Access_Agreement_for_Participants.pdf), and the provider's [download folder](https://mega.nz/folder/UNMF2YYI#1cqJVzo4p_wESv9P_pc8uA). Follow the organizer's registration and non-redistribution conditions. Use the annotated CT and MR training data.

## Accepted layouts

```text
mm_whs/
  ct_train/
    ct_train_1001_image.nii.gz
    ct_train_1001_label.nii.gz
  mr_train/
    mr_train_1001_image.nii.gz
    mr_train_1001_label.nii.gz
KiTS23/
  images/case_00000_ct.nii.gz
  masks/case_00000_mask.nii.gz
```

MMWHS `.nii` and `.nii.gz` files are supported. Keep image/label stem pairs together; CT and MR source-case IDs do not imply paired scans of the same patient.

The KiTS23 source layout is typically:

```text
kits23/dataset/
  case_00000/
    imaging.nii.gz
    segmentation.nii.gz
```

Create a separate named-link layout without changing source voxels or labels. Edit the two paths and run this once on Linux:

```python
from pathlib import Path

source = Path("/path/to/kits23/dataset").resolve()
target = Path("/path/to/KiTS23").resolve()
cases = sorted(source.glob("case_*"))
if not cases:
    raise FileNotFoundError("No KiTS23 case directories found")
pairs = []
for case in cases:
    for src_name, folder, suffix in [
        ("imaging.nii.gz", "images", "ct"),
        ("segmentation.nii.gz", "masks", "mask"),
    ]:
        src = case / src_name
        dst = target / folder / f"{case.name}_{suffix}.nii.gz"
        if not src.is_file():
            raise FileNotFoundError(src)
        if dst.exists() or dst.is_symlink():
            raise FileExistsError(dst)
        pairs.append((src, dst))
for src, dst in pairs:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src)
```

If your data already use `images/*_ct.nii.gz` and `masks/*_mask.nii.gz`, no conversion is needed. The example creates symbolic links; retain the source data at their original locations. Do not relabel kidneys, tumors, or cysts to make a case pass validation.

## Validate before inference

Copy the matching example YAML and set `data_root` to the parent shown above, along with checkpoint paths and a new `run_name`:

```bash
reform3d check --config configs/kits23-demo.yaml
reform3d check --config configs/mmwhs-demo.yaml
```

The command prepares the manifest and checks the required input geometry and cohort eligibility without encoder inference. These examples require more than one source case; a single image/mask pair only illustrates naming.

MMWHS provides CT–MR retrieval and low-shot evaluation. KiTS23 provides controlled spacing evaluation; keep the full manifest because semantic-target eligibility uses the complete cohort. Source-bin selection happens afterward.

## Custom datasets

For another annotated CT–MR dataset, use `configs/custom-organs.yaml` and a CSV with one row per organ:

```csv
case_id,patient_id,modality,image,mask,organ,label_id
ct01,p01,ct,images/ct01.nii.gz,masks/ct01.nii.gz,kidney,1
mr01,p02,mr,images/mr01.nii.gz,masks/mr01.nii.gz,kidney,1
```

Paths are relative to the CSV. `patient_id` is optional; when supplied, it groups repeated scans for support/query separation. Use stable identities across modalities when the same person appears in both. The example rows illustrate the format, not a sufficient evaluation cohort.

Defaults require at least two organs, each present in at least five distinct cases per required modality. Image and mask geometry must match. Each organ needs enough cases for separate labeled support and query sets.

Existing JSON manifests are also accepted through `manifest`. Controlled evaluation needs volume spacing, anisotropy bins, and eligible semantic targets; organ evaluation needs masks, label IDs, organ names, modalities, and case identities. Declared spacing must match the image header and anisotropy bin. Acquisition modality must agree across configuration and manifest; the default source bin is 0 for CT and 1 for MR. Missing required annotations, duplicate scans/organ records, and incomplete cohorts fail explicitly.

Dataset names and paths do not determine pretraining exposure. Optional exposure annotations supplied in a JSON manifest are retained as user metadata; they do not affect cohort selection or scores.

Input fingerprints use resolved paths, file sizes, and modification times; weights use SHA-256. Do not modify data or generated caches in place. Dataset and weight access terms remain those of their providers.
