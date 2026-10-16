"""Shared crop features and PCA, separate from benchmark preprocessing/caches.

Inputs are annotated NIfTI/NRRD volumes and registered Med3DINO weights.
Outputs stay in a user-selected directory. The legacy viewing crop is centered
on annotations; notebooks share one crop sampled from image foreground or
fixed at an explicit physical center chosen before feature extraction.
Annotations never train a readout or define PCA foreground.
"""
from __future__ import annotations

import gc
import hashlib
import json
import inspect
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import scipy
import sklearn
from scipy import ndimage
from sklearn.decomposition import PCA

METHOD = "physical-roi-attention-pca-v1"
MODELS = ("med3dino_iso_96", "med3dino_rel_96", "med3dino_sa_96")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(path):
    path = Path(path).expanduser().resolve(strict=True)
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def load_config(path):
    """Read JSON; resolve paths relative to it. Never search a legacy checkout."""
    path = Path(path).expanduser().resolve(strict=True)
    cfg = json.loads(path.read_text())
    def resolve(value):
        p = Path(value).expanduser()
        return str((p if p.is_absolute() else path.parent / p).resolve())
    if set(cfg["checkpoints"]) != set(MODELS):
        raise ValueError(f"Provide exactly these three checkpoint keys: {MODELS}")
    if len(cfg["cases"]) != 2 or {c["modality"] for c in cfg["cases"]} != {"CT", "MR"}:
        raise ValueError("Provide one CT and one MR case")
    for case in cfg["cases"]:
        case["image"] = resolve(case["image"])
        case["mask"] = resolve(case["mask"]) if case.get("mask") else None
    cfg["checkpoints"] = {k: resolve(v) for k, v in cfg["checkpoints"].items()}
    cfg["output_dir"] = resolve(cfg["output_dir"])
    cfg.setdefault("device", "cuda:0")
    return cfg


def prepare_case(image, mask, *, padding_mm=24.0, min_fov_mm=80.0, size=96):
    """Canonical RAS physical cubic crop around annotated anatomy, sampled to 96³.

    Common deterministic viewing transform for all checkpoints; this is NOT a
    reproduction of the benchmark's foreground/slice-axis preprocessing.
    """
    if size < 16 or size % 16:
        raise ValueError("Size must be a positive multiple of the patch size (16)")
    if not np.isfinite([padding_mm, min_fov_mm]).all() or padding_mm < 0 or min_fov_mm <= 0:
        raise ValueError("Crop parameters must be finite; padding >= 0 and FOV > 0")
    image, mask = Path(image), Path(mask)
    im, seg = sitk.ReadImage(str(image)), sitk.ReadImage(str(mask))
    if im.GetDimension() != 3 or seg.GetDimension() != 3:
        raise ValueError("Expected scalar 3D image and mask")
    if im.GetNumberOfComponentsPerPixel() != 1 or seg.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("Vector images are unsupported")
    for getter in ("GetSize", "GetSpacing", "GetOrigin", "GetDirection"):
        if not np.allclose(getattr(im, getter)(), getattr(seg, getter)(), rtol=0, atol=1e-5):
            raise ValueError(f"Image/mask geometry mismatch: {getter}")
    spacing = np.asarray(im.GetSpacing())
    if not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("Invalid voxel spacing")
    affine = np.eye(4)
    affine[:3, :3] = np.asarray(im.GetDirection()).reshape(3, 3) @ np.diag(spacing)
    affine[:3, 3] = im.GetOrigin()
    affine = np.diag([-1., -1., 1., 1.]) @ affine  # SimpleITK LPS -> NIfTI RAS
    raw = sitk.GetArrayFromImage(im).transpose(2, 1, 0).astype(np.float32)
    labels = sitk.GetArrayFromImage(seg).transpose(2, 1, 0)
    if not np.isfinite(raw).all() or not np.isfinite(labels).all() or (labels < 0).any():
        raise ValueError("Nonfinite image/mask values or negative mask labels")
    canonical = nib.as_closest_canonical(nib.Nifti1Image(raw, affine))
    canonical_seg = nib.as_closest_canonical(nib.Nifti1Image((labels > 0).astype(np.uint8), affine))
    raw = np.asarray(canonical.dataobj)
    labels = np.asarray(canonical_seg.dataobj)
    bounds = ndimage.find_objects(labels, max_label=1)[0]
    if bounds is None:
        raise ValueError("The annotation mask is empty")
    spacing = np.asarray(canonical.header.get_zooms()[:3])
    # Oblique grids are reoriented, not de-obliqued: sample along their voxel axes.
    center = np.array([(axis.start + axis.stop - 1) / 2 for axis in bounds])
    extent = np.array([axis.stop - axis.start for axis in bounds]) * spacing
    fov = float(max(extent.max() + 2 * padding_mm, min_fov_mm))
    offsets = (np.arange(size) + 0.5) * fov / size - fov / 2
    coords = np.meshgrid(*(center[a] + offsets / spacing[a] for a in range(3)), indexing="ij")
    low, high = np.percentile(raw, [0.05, 99.95])
    if not high > low:
        raise ValueError("Constant/degenerate intensity distribution")
    crop = ndimage.map_coordinates(raw, coords, order=1, mode="constant", cval=float(low))
    crop = (np.clip(crop, low, high) - low) / (high - low) * 2 - 1
    cropped_mask = ndimage.map_coordinates(labels, coords, order=0, mode="constant", cval=0) > 0
    if not cropped_mask.any():
        raise ValueError("Annotation disappeared after resampling; enlarge resolution or choose another case")
    z = int(cropped_mask.sum(axis=(0, 1)).argmax())
    return {"volume": crop.astype(np.float32), "mask": cropped_mask,
            "foreground": crop > -0.95, "slice": z,
            "metadata": {"input": _fingerprint(image), "mask": _fingerprint(mask),
                         "canonical_shape": list(raw.shape), "canonical_spacing_mm": spacing.tolist(),
                         "crop_center_voxel": center.tolist(), "fov_mm": fov, "size": size,
                         "intensity_percentiles": [float(low), float(high)],
                         "axis_convention": "closest RAS; native obliquity retained",
                         "slice_rule": "maximum annotated area in canonical third-axis plane"}}


def _shared_foreground(raw, low, high):
    """A bounded, image-derived body/brain mask for sampling and display."""
    stride = np.maximum(1, np.ceil(np.asarray(raw.shape) / 128).astype(int))
    sampled = raw[tuple(slice(None, None, int(s)) for s in stride)]
    threshold = float(low + .025 * (high - low))
    labels, count = ndimage.label(sampled > threshold)
    if not count:
        raise ValueError("No image foreground was found")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    foreground = ndimage.binary_fill_holes(labels == sizes.argmax())
    return foreground, stride, threshold


def _shared_coordinates(center_ras, offsets, source_inverse):
    coordinates = np.asarray(np.meshgrid(*(c + offsets for c in center_ras), indexing="ij"))
    return np.einsum("ij,j...->i...", source_inverse[:3, :3], coordinates) + source_inverse[:3, 3, None, None, None]


def prepare_shared_case(image, mask=None, *, seed=42, fov_mm=192.0, size=96, center_ras_mm=None):
    """Sample one image-foreground crop on an isotropic, axis-aligned RAS grid.

    An explicit RAS center fixes the crop without clamping it to the source;
    choose it from anatomy before inspecting features. Otherwise use seeded
    image-foreground sampling. The optional annotation follows the same grid.
    The physical FOV describes voxel edges, so voxel spacing is ``fov_mm / size``.
    """
    if type(size) is not int or size < 16 or size % 16:
        raise ValueError("Size must be a positive multiple of the patch size (16)")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer from 0 to 4294967295")
    if not np.isscalar(fov_mm) or not np.isfinite(fov_mm) or fov_mm <= 0:
        raise ValueError("Physical FOV must be finite and positive")
    if center_ras_mm is not None:
        try:
            center_ras_mm = np.asarray(center_ras_mm, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("RAS center must contain three finite coordinates in millimetres") from exc
        if center_ras_mm.shape != (3,) or not np.isfinite(center_ras_mm).all():
            raise ValueError("RAS center must contain three finite coordinates in millimetres")
    image = Path(image)
    im = sitk.ReadImage(str(image))
    if im.GetDimension() != 3 or im.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("Expected a scalar 3D image")
    spacing = np.asarray(im.GetSpacing(), dtype=float)
    direction = np.asarray(im.GetDirection(), dtype=float).reshape(3, 3)
    origin = np.asarray(im.GetOrigin(), dtype=float)
    if not np.isfinite(spacing).all() or (spacing <= 0).any() or not np.isfinite(direction).all() or not np.isfinite(origin).all():
        raise ValueError("Image geometry must be finite with positive spacing")
    source_affine = np.eye(4)
    source_affine[:3, :3] = direction @ np.diag(spacing)
    source_affine[:3, 3] = origin
    source_affine = np.diag([-1., -1., 1., 1.]) @ source_affine
    if abs(np.linalg.det(source_affine[:3, :3])) < 1e-10:
        raise ValueError("Image direction must define an invertible physical grid")
    inverse = np.linalg.inv(source_affine)
    raw = sitk.GetArrayFromImage(im).transpose(2, 1, 0).astype(np.float32)
    if not np.isfinite(raw).all():
        raise ValueError("Image contains nonfinite values")
    if min(raw.shape) < 2:
        raise ValueError("Expected at least two voxels on every source axis")
    labels = None
    if mask is not None:
        mask = Path(mask)
        seg = sitk.ReadImage(str(mask))
        if seg.GetDimension() != 3 or seg.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError("Expected a scalar 3D mask")
        for getter in ("GetSize", "GetSpacing", "GetOrigin", "GetDirection"):
            if not np.allclose(getattr(im, getter)(), getattr(seg, getter)(), rtol=0, atol=1e-5):
                raise ValueError(f"Image/mask geometry mismatch: {getter}")
        labels = sitk.GetArrayFromImage(seg).transpose(2, 1, 0)
        if not np.isfinite(labels).all() or (labels < 0).any():
            raise ValueError("Mask contains nonfinite values or negative labels")
    low, high = np.percentile(raw, [.05, 99.95])
    if not high > low:
        raise ValueError("Constant/degenerate intensity distribution")
    foreground, stride, threshold = _shared_foreground(raw, low, high)
    shape = np.asarray(raw.shape)
    if center_ras_mm is None:
        candidates = np.argwhere(foreground)
        rng = np.random.default_rng(seed)
        candidates = candidates[rng.choice(len(candidates), size=min(32, len(candidates)), replace=False)] * stride
        half_span = float(fov_mm) * (size - 1) / (2 * size)
        # Bounding the RAS cube in source coordinates also handles oblique images.
        source_half_span = np.abs(inverse[:3, :3]).sum(1) * half_span
        feasible = source_half_span <= (shape - 1) / 2
        candidates = np.where(feasible, np.maximum(source_half_span, np.minimum(candidates, shape - 1 - source_half_span)), (shape - 1) / 2)
        preview_offsets = np.linspace(-half_span, half_span, 16)
        best = None
        for center in candidates:
            center_ras = source_affine[:3, :3] @ center + source_affine[:3, 3]
            coords = _shared_coordinates(center_ras, preview_offsets, inverse)
            valid = np.all((coords >= 0) & (coords <= (shape - 1)[:, None, None, None]), axis=0)
            sampled_fg = ndimage.map_coordinates(foreground.astype(np.uint8), coords / stride[:, None, None, None], order=0, mode="constant", cval=0)
            coverage = (float(sampled_fg.mean()), float(valid.mean()))
            if best is None or coverage > best[0]:
                best = (coverage, center_ras)
            if coverage[0] >= .15 and coverage[1] >= .85:
                best = (coverage, center_ras)
                break
        center_ras = best[1]
        center_selection = "seeded image-foreground sampling"
        sampling = "seeded foreground centers; prefer >=15% foreground and >=85% source coverage; otherwise best foreground coverage"
    else:
        center_ras = center_ras_mm
        center_selection = "explicit RAS center"
        sampling = "fixed physical center; no random sampling or source-boundary clamping; pad outside source"
    offsets = (np.arange(size) + .5) * float(fov_mm) / size - float(fov_mm) / 2
    coords = _shared_coordinates(center_ras, offsets, inverse)
    crop = ndimage.map_coordinates(raw, coords, order=1, mode="constant", cval=float(low))
    crop = np.ascontiguousarray((np.clip(crop, low, high) - low) / (high - low) * 2 - 1, dtype=np.float32)
    output_fg = ndimage.map_coordinates(foreground.astype(np.uint8), coords / stride[:, None, None, None], order=0, mode="constant", cval=0).astype(bool)
    if not output_fg.any():
        raise ValueError("Image foreground disappeared from the shared crop; reduce fov_mm or choose another case")
    output_mask = (ndimage.map_coordinates((labels > 0).astype(np.uint8), coords, order=0, mode="constant", cval=0).astype(bool)
                   if labels is not None else np.zeros(crop.shape, dtype=bool))
    affine = np.eye(4)
    affine[:3, :3] *= float(fov_mm) / size
    affine[:3, 3] = center_ras + offsets[0]
    valid = np.all((coords >= 0) & (coords <= (shape - 1)[:, None, None, None]), axis=0)
    return {"volume": crop, "mask": output_mask, "foreground": output_fg, "slice": size // 2,
            "metadata": {"input": _fingerprint(image), "mask": _fingerprint(mask) if mask is not None else None,
                         "method": "ct-mr-shared-crop-v1", "seed": seed, "size": size, "fov_mm": float(fov_mm),
                         "affine_ras": affine.tolist(), "crop_center_ras_mm": center_ras.tolist(),
                         "source_affine_ras": source_affine.tolist(), "source_shape": shape.tolist(),
                         "input_sha256": hashlib.sha256(crop.tobytes()).hexdigest(),
                         "normalization": {"method": "clip global 0.05/99.95 percentiles; scale to [-1, 1]", "bounds": [float(low), float(high)]},
                         "foreground_definition": {"method": "intensity above low + 0.025*(high-low); largest component; fill holes", "threshold": threshold, "source_grid_stride": stride.tolist()},
                         "center_selection": center_selection, "sampling": sampling,
                         "foreground_fraction": float(output_fg.mean()), "source_coverage_fraction": float(valid.mean()),
                         "axis_convention": "axis-aligned RAS; isotropic voxel-center affine", "slice_rule": "shared physical axial levels"}}


def _validate_shared_views(views, n_cases):
    """Reject a shared-input comparison containing different image tensors."""
    for i in range(n_cases):
        reference = views[0, i]
        for m in range(len(MODELS)):
            case = views[m, i]
            if case["volume"].dtype != np.float32:
                raise ValueError("Shared inputs must be float32 tensors")
            if case["volume"].ndim != 3 or any(case[key].shape != case["volume"].shape for key in ("mask", "foreground")):
                raise ValueError("Shared input images, masks and foreground must have matching 3D shapes")
            digest = hashlib.sha256(np.ascontiguousarray(case["volume"]).tobytes()).hexdigest()
            if digest != case["metadata"].get("input_sha256"):
                raise ValueError("Shared input tensor does not match its recorded hash")
            if any(not np.array_equal(case[key], reference[key]) for key in ("volume", "mask", "foreground")):
                raise ValueError("Shared inputs must be identical across all three encoders")
            affine = np.asarray(case["metadata"].get("affine_ras"))
            if affine.shape != (4, 4) or not np.isfinite(affine).all() or not np.array_equal(affine, reference["metadata"].get("affine_ras")):
                raise ValueError("Shared inputs must use the same valid RAS affine")


SHARED_CROP_FUNCTIONS = (_shared_foreground, _shared_coordinates, prepare_shared_case)


def shared_crop_signature():
    """Identify the shared crop independently of plotting implementation changes."""
    return {"method": "ct-mr-shared-crop-v1", "default_fov_mm": 192.0, "default_size": 96,
            "center_selection": "seeded image foreground or explicit RAS millimetres without clamping",
            "normalization_percentiles": [.05, 99.95], "foreground_range_fraction": .025,
            "foreground_grid_max_size": 128, "max_center_candidates": 32,
            "minimum_foreground_fraction": .15, "preferred_source_coverage_fraction": .85,
            "source_sha256": hashlib.sha256("".join(inspect.getsource(f) for f in SHARED_CROP_FUNCTIONS).encode()).hexdigest()}


def spatial_maps(model, volume):
    """Last normalized patch tokens and per-head final-layer CLS-to-patch mass."""
    shape = tuple(volume.shape)
    if len(shape) != 5 or shape[:2] != (1, 1) or any(n % model.patch_size for n in shape[2:]):
        raise ValueError("Expected [1,1,X,Y,Z], spatial sizes divisible by patch size")
    with torch.inference_mode():
        tokens = model.forward_features(volume)["x_norm_patchtokens"].float()
        attention = model.get_self_attention(volume, reshape=True).float()
    grid = tuple(n // model.patch_size for n in shape[2:])
    if tokens.shape[1] != np.prod(grid) or tuple(attention.shape[2:]) != grid:
        raise ValueError("Token/attention grid does not match the input")
    mass = attention.sum(dim=(2, 3, 4), keepdim=True)
    if not torch.isfinite(tokens).all() or not torch.isfinite(attention).all() or (mass <= 0).any() or (attention < 0).any():
        raise ValueError("Invalid model features or attention")
    attention = attention / mass  # conditional on attention directed to patch tokens
    return tokens[0].cpu().numpy(), attention[0].cpu().numpy()


def attention_summaries(attention, grid):
    """Conditional CLS and mean patch-query attention over patch keys, per head."""
    count = int(np.prod(grid))
    if attention.ndim != 4 or attention.shape[0] != 1 or attention.shape[-2:] != (count + 1, count + 1):
        raise ValueError("Full attention must match one CLS token and the spatial patch grid")
    if not torch.isfinite(attention).all() or (attention < 0).any():
        raise ValueError("Invalid attention weights")
    directed = attention[..., 1:]
    mass = directed.sum(-1, keepdim=True)
    if (mass <= 0).any():
        raise ValueError("Every query needs positive patch-directed attention mass")
    directed = directed / mass
    cls = directed[0, :, 0]
    patch = directed[0, :, 1:].mean(1)
    return cls.reshape(-1, *grid), patch.reshape(-1, *grid)


def spatial_features(model, volume):
    """Two forwards collect global/patch features and both final-layer attention views."""
    shape = tuple(volume.shape)
    if len(shape) != 5 or shape[:2] != (1, 1) or any(n % model.patch_size for n in shape[2:]):
        raise ValueError("Expected [1,1,X,Y,Z], spatial sizes divisible by patch size")
    with torch.inference_mode():
        features = model.forward_features(volume)
        full_attention = model.get_self_attention(volume, reshape=False).float()
    grid = tuple(n // model.patch_size for n in shape[2:])
    tokens, cls = features["x_norm_patchtokens"].float(), features["x_norm_clstoken"].float()
    if tokens.shape[:2] != (1, int(np.prod(grid))) or cls.shape != (1, tokens.shape[-1]):
        raise ValueError("Global/patch features do not match the input grid")
    if not torch.isfinite(tokens).all() or not torch.isfinite(cls).all():
        raise ValueError("Nonfinite model features")
    cls_attention, patch_attention = attention_summaries(full_attention, grid)
    return {"tokens": tokens[0].cpu().numpy(), "cls": cls[0].cpu().numpy(),
            "gap": tokens[0].mean(0).cpu().numpy(), "attention": cls_attention.cpu().numpy(),
            "patch_attention": patch_attention.cpu().numpy()}


def _signature(cfg, preprocessing=None):
    signature = {"method": METHOD, "cases": [{**c, "image_stat": _fingerprint(c["image"]),
             "mask_stat": _fingerprint(c["mask"]) if c.get("mask") else None} for c in cfg["cases"]],
            "checkpoints": {k: _fingerprint(v) for k, v in cfg["checkpoints"].items()},
            "device": cfg["device"], "torch": torch.__version__, "numpy": np.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "simpleitk": sitk.Version_VersionString(), "extraction_sha256": hashlib.sha256("".join(inspect.getsource(f) for f in (prepare_case, *SHARED_CROP_FUNCTIONS, _validate_shared_views, spatial_maps, attention_summaries, spatial_features, collect_maps)).encode()).hexdigest(),
            "encoder_source_sha256": {name: _sha256(Path(__file__).parent / name) for name in (
                "encoders/factory.py", "encoders/checkpoints.py",
                "encoders/med3dino/vision_transformer.py", "encoders/med3dino/layers/attention.py",
                "encoders/med3dino/layers/block.py", "encoders/med3dino/layers/patch_embed3d.py",
                "modules/acquisition_robustness/feature_extractor.py")}}
    if preprocessing is not None:
        signature["preprocessing"] = preprocessing
    return signature


def collect_maps(cfg, *, prepare_views=None, preprocessing=None):
    """Two forwards per case/checkpoint pair, sequentially at batch size one.

    Optional prepare_views(cfg) supplies a separate input for each (model, case).
    Its preprocessing signature isolates geometry and seeds in the local cache.
    These files never join benchmark caches.
    """
    out = Path(cfg["output_dir"])
    n_cases = len(cfg["cases"])
    if (prepare_views is None) != (preprocessing is None):
        raise ValueError("Custom views require a preprocessing signature")
    signature = _signature(cfg, preprocessing)
    receipt_path = out / "receipt.json"
    if receipt_path.exists():
        if json.loads(receipt_path.read_text())["signature"] != signature:
            raise ValueError("Visualization inputs changed; choose a new output_dir")
        return load_maps(cfg, preprocessing=preprocessing)
    shared = preprocessing is not None and preprocessing.get("method") == "ct-mr-shared-crop-v1"
    only_frozen_inputs = shared and out.is_dir() and {p.name for p in out.iterdir()} == {"inputs"} and (out / "inputs").is_dir()
    if out.exists() and any(out.iterdir()) and not only_frozen_inputs:
        raise ValueError("Incomplete/unrecognized output directory; choose a new output_dir")
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Select an accessible GPU or explicitly choose device='cpu'")
    if prepare_views is None:
        cases = [prepare_case(c["image"], c["mask"], **c.get("crop", {})) for c in cfg["cases"]]
        views = {(m, i): case for m in range(3) for i, case in enumerate(cases)}
    else:
        views = prepare_views(cfg)
    if set(views) != {(m, i) for m in range(len(MODELS)) for i in range(n_cases)}:
        raise ValueError("Provide one view for each case/checkpoint pair")
    if any(c["volume"].shape != (96, 96, 96) for c in views.values()):
        raise ValueError("These registered checkpoint examples require 96³ inputs")
    if shared:
        _validate_shared_views(views, n_cases)
    from reform3d.encoders.factory import create_feature_extractor
    out.mkdir(parents=True, exist_ok=True)
    arrays, metadata = {}, {}
    for (m, i), case in views.items():
        suffix = f"{m}_{i}" if preprocessing is not None else str(i)
        for key in ("volume", "mask", "foreground"):
            arrays[f"{key}_{suffix}"] = case[key]
        metadata[suffix] = {**case["metadata"], "slice": case["slice"]}
    counts, weights = {}, {}
    for m, name in enumerate(MODELS):
        weights[name] = _sha256(cfg["checkpoints"][name])
        extractor = create_feature_extractor(name, checkpoint_path=cfg["checkpoints"][name], device=str(device))
        model = extractor.model
        counts[name] = sum(p.numel() for p in model.parameters())
        for i in range(n_cases):
            case = views[m, i]
            print(f"Extracting {name} / {cfg['cases'][i]['modality']}", flush=True)
            x = torch.from_numpy(case["volume"])[None, None].to(device)
            # Float32 is deliberate: these bounded examples prioritize reproducible maps.
            for key, value in spatial_features(model, x).items():
                arrays[f"{key}_{m}_{i}"] = value
        del x, model, extractor
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    np.savez_compressed(out / "maps.npz", **arrays)
    receipt = {"signature": signature,
               "cases": metadata if preprocessing is not None else [metadata[str(i)] for i in range(n_cases)],
               "feature_schema": "cls-patch-attention-v1",
               "checkpoint_sha256": weights,
               "backbone_parameters": counts, "maps_sha256": _sha256(out / "maps.npz")}
    if shared:
        receipt["shared_input_sha256"] = {str(i): views[0, i]["metadata"]["input_sha256"] for i in range(n_cases)}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return arrays, receipt


def load_maps(cfg, *, preprocessing=None):
    """Verify local completed maps before a second notebook reuses them."""
    out = Path(cfg["output_dir"])
    if not (out / "receipt.json").is_file() or not (out / "maps.npz").is_file():
        raise FileNotFoundError("Run extract_features(config) in notebook 02 to prepare the PCA cache")
    receipt = json.loads((out / "receipt.json").read_text())
    if receipt["signature"] != _signature(cfg, preprocessing) or receipt["maps_sha256"] != _sha256(out / "maps.npz"):
        raise ValueError("Visualization inputs/results changed; choose a new output_dir")
    with np.load(out / "maps.npz", allow_pickle=False) as data:
        arrays = dict(data)
    fields = ("tokens", "attention", "cls", "gap", "patch_attention") if receipt.get("feature_schema") == "cls-patch-attention-v1" else ("tokens", "attention")
    for m in range(len(MODELS)):
        for i in range(len(cfg["cases"])):
            suffix = f"{m}_{i}" if preprocessing is not None else str(i)
            for key in (*[f"{field}_{m}_{i}" for field in fields], f"volume_{suffix}", f"mask_{suffix}", f"foreground_{suffix}"):
                if key not in arrays:
                    raise ValueError(f"Incomplete maps: {key}")
    if preprocessing is not None and preprocessing.get("method") == "ct-mr-shared-crop-v1":
        try:
            views = {(m, i): {**{key: arrays[f"{key}_{m}_{i}"] for key in ("volume", "mask", "foreground")},
                             "metadata": receipt["cases"][f"{m}_{i}"]}
                     for m in range(len(MODELS)) for i in range(len(cfg["cases"]))}
        except KeyError as error:
            raise ValueError("Incomplete shared input metadata") from error
        _validate_shared_views(views, len(cfg["cases"]))
        if receipt.get("shared_input_sha256") != {str(i): views[0, i]["metadata"]["input_sha256"] for i in range(len(cfg["cases"]))}:
            raise ValueError("Shared input receipt does not match its tensors")
    return arrays, receipt


def pca_colors(tokens, foregrounds, grids):
    """One PCA per checkpoint, fitted jointly across the provided cases.

    Colors are meaningful within that fitted basis, not between checkpoints.
    Foreground comes from intensity, never from the annotation mask.
    """
    if not (len(tokens) == len(foregrounds) == len(grids)) or not tokens:
        raise ValueError("Provide matching nonempty token, foreground and grid lists")
    normalized, selectors = [], []
    for values, fg, grid in zip(tokens, foregrounds, grids):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or len(values) != np.prod(grid) or not np.isfinite(values).all():
            raise ValueError("Invalid patch features/grid")
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if (norms <= 0).any():
            raise ValueError("Zero patch-feature norm")
        normalized.append(values / norms)
        # Fraction of intensity foreground inside each native patch.
        fg = np.asarray(fg, dtype=np.float32)
        if fg.ndim != 3 or len(grid) != 3 or not np.isfinite(fg).all() or any(g <= 0 or n % g for n, g in zip(fg.shape, grid)):
            raise ValueError("Foreground shape is incompatible with the patch grid")
        patch = tuple(n // g for n, g in zip(fg.shape, grid))
        fraction = fg.reshape(grid[0], patch[0], grid[1], patch[1], grid[2], patch[2]).mean(axis=(1, 3, 5))
        selectors.append(fraction.reshape(-1) >= 0.25)
    selected = np.concatenate([x[sel] for x, sel in zip(normalized, selectors)])
    if selected.shape[0] < 4 or selected.shape[1] < 3:
        raise ValueError("At least four foreground tokens and three feature dimensions are required")
    pca = PCA(n_components=3, svd_solver="randomized", random_state=0)
    scores = pca.fit_transform(selected)
    low, high = np.percentile(scores, [1, 99], axis=0)
    if np.any(high - low < 1e-8):
        raise ValueError("Degenerate PCA components")
    colors = [np.clip((pca.transform(x) - low) / (high - low), 0, 1).reshape(*grid, 3)
              for x, grid in zip(normalized, grids)]
    return colors, {"components": pca.components_, "mean": pca.mean_,
                    "explained_variance_ratio": pca.explained_variance_ratio_,
                    "low": low, "high": high}
