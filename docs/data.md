# Data

Keep data and weights outside the repository. Edit their paths in an example YAML; `reform3d check` prepares and validates the manifest automatically.

Built-in layouts:

```text
mm_whs/
  ct_train/ct_train_1001_{image,label}.nii
  mr_train/mr_train_1001_{image,label}.nii
KiTS23/
  images/case_00000_ct.nii.gz
  masks/case_00000_mask.nii.gz
```

MMWHS provides CT–MR retrieval and low-shot evaluation. KiTS23 provides controlled spacing evaluation; keep the full manifest because semantic-target eligibility uses the complete cohort. Source-bin selection happens afterward.

For another annotated CT–MR dataset, use `configs/custom-organs.yaml` and a CSV with one row per organ:

```csv
case_id,patient_id,modality,image,mask,organ,label_id
ct01,p01,ct,images/ct01.nii.gz,masks/ct01.nii.gz,kidney,1
mr01,p02,mr,images/mr01.nii.gz,masks/mr01.nii.gz,kidney,1
```

Paths are relative to the CSV. `patient_id` is optional; when supplied, it groups repeated scans for support/query separation. Use stable identities across modalities when the same person appears in both. The example rows illustrate the format, not a sufficient evaluation cohort.

Defaults require at least two organs, each present in at least five distinct cases per required modality. Image and mask geometry must match. Each organ needs enough cases for separate labeled support and query sets.

Existing JSON manifests are also accepted through `manifest`. Controlled evaluation needs volume spacing, anisotropy bins, and eligible semantic targets; organ evaluation needs masks, label IDs, organ names, modalities, and case identities. Declared spacing must match the image header and anisotropy bin. Acquisition modality must agree across configuration and manifest; the default source bin is 0 for CT and 1 for MR. Missing required annotations, duplicate scans/organ records, and incomplete cohorts fail explicitly.

Input fingerprints use resolved paths, file sizes, and modification times; weights use SHA-256. Do not modify data or generated caches in place. Dataset and weight access terms remain those of their providers.
