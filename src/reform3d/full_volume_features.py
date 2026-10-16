"""Full-volume patch features for qualitative reference-similarity examples.

These caches are independent of the cropped attention/PCA examples and benchmark
features. Each checkpoint sees the same frozen, uncropped volume for each case.
"""
from __future__ import annotations

from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile

import nibabel
import numpy as np
import scipy
import SimpleITK
import torch

from reform3d import full_volume

MODELS = ("med3dino_iso_96", "med3dino_rel_96", "med3dino_sa_96")
SCHEMA = "full-volume-patchfeatures-v1"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(path):
    path = Path(path).expanduser().resolve(strict=True)
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": _sha256(path)}


def _options(cfg):
    cases = cfg.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Provide at least one full-volume case")
    ids = []
    for case in cases:
        if (not isinstance(case, dict) or not isinstance(case.get("id"), str)
                or not case["id"].strip() or case.get("modality") not in {"CT", "MR"}
                or not case.get("image")):
            raise ValueError("Every case needs a unique id, CT/MR modality and image")
        ids.append(case["id"])
    if len(set(ids)) != len(ids):
        raise ValueError("Case ids must be unique")
    options = {"max_axis": cfg.get("max_axis", 224), "patch_size": 16,
               "max_patches": cfg.get("max_patches", 2744)}
    if (type(options["max_axis"]) is not int or not 16 <= options["max_axis"] <= 224
            or options["max_axis"] % 16 or type(options["max_patches"]) is not int
            or not 1 <= options["max_patches"] <= 2744):
        raise ValueError("max_axis must be a multiple of 16 up to 224; max_patches must be 1–2744")
    return options


def _input_signature(cfg):
    options = _options(cfg)
    return {"schema": "full-volume-inputs-v1", "options": options,
            "geometry_source_sha256": _sha256(full_volume.__file__),
            "cases": [{"id": c["id"], "modality": c["modality"],
                       "image": _fingerprint(c["image"]),
                       "mask": _fingerprint(c["mask"]) if c.get("mask") else None}
                      for c in cfg["cases"]],
            "software": {"numpy": np.__version__, "scipy": scipy.__version__,
                         "nibabel": nibabel.__version__,
                         "simpleitk": SimpleITK.Version_VersionString()}}


def _feature_signature(cfg, input_receipt):
    if set(cfg.get("checkpoints", {})) != set(MODELS):
        raise ValueError(f"Provide exactly these checkpoint keys: {MODELS}")
    try:
        device = str(torch.device(cfg.get("device", "cuda:0")))
    except (TypeError, RuntimeError) as error:
        raise ValueError("device must be a valid torch device") from error
    if torch.device(device).type not in {"cuda", "cpu"}:
        raise ValueError("Full-volume examples support CUDA or CPU")
    fraction = cfg.get("memory_fraction", 1.)
    if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
            or not np.isfinite(fraction) or not 0 < fraction <= 1):
        raise ValueError("memory_fraction must be greater than 0 and at most 1")
    root = Path(__file__).parent
    files = [Path(__file__), root / "encoders/factory.py", root / "encoders/checkpoints.py",
             root / "core/checkpoints.py", root / "core/paths.py",
             root / "modules/acquisition_robustness/feature_extractor.py",
             root / "modules/acquisition_robustness/config.py"]
    files.extend(sorted((root / "encoders/med3dino").rglob("*.py")))
    return {"schema": SCHEMA, "inputs": input_receipt["signature"],
            "inputs_sha256": input_receipt["sha256"],
            "checkpoints": {name: _fingerprint(cfg["checkpoints"][name]) for name in MODELS},
            "device": device, "dtype": "float32", "memory_fraction": float(fraction),
            "software": {"torch": torch.__version__, "cuda": torch.version.cuda,
                         **input_receipt["signature"]["software"]},
            "encoder_source_sha256": {str(p.relative_to(root)): _sha256(p) for p in files}}


def _read_json(path):
    try:
        result = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"Unreadable full-volume receipt: {path}") from error
    if not isinstance(result, dict):
        raise ValueError(f"Invalid full-volume receipt: {path}")
    return result


def _pair_state(archive, receipt):
    if archive.exists() != receipt.exists():
        raise ValueError(f"Incomplete full-volume cache in {archive.parent}; choose a new output_dir")
    return archive.is_file() and receipt.is_file()


@contextmanager
def _lock(folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    try:
        with path.open("x") as stream:
            stream.write(str(os.getpid()))
    except FileExistsError as error:
        raise ValueError(f"Full-volume cache is locked: {path}") from error
    try:
        yield
    finally:
        path.unlink()


def _atomic_new(path, value, *, arrays=False):
    """Publish a completed file; existing cache files are never overwritten."""
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".full-volume-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            if arrays:
                np.savez_compressed(stream, **value)
            else:
                stream.write((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        if path.exists():
            raise ValueError(f"Refusing to overwrite full-volume cache: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_archive(path):
    try:
        with np.load(path, allow_pickle=False) as archive:
            return dict(archive)
    except (OSError, ValueError, EOFError) as error:
        raise ValueError(f"Unreadable full-volume archive: {path}") from error


def _validate_inputs(arrays, metadata, n_cases):
    expected = {f"{key}_{i}" for i in range(n_cases) for key in ("volume", "mask", "foreground")}
    if set(arrays) != expected or not isinstance(metadata, list) or len(metadata) != n_cases:
        raise ValueError("Incomplete full-volume inputs")
    for i, case in enumerate(metadata):
        image, mask, foreground = (arrays[f"{key}_{i}"] for key in ("volume", "mask", "foreground"))
        if (image.ndim != 3 or image.dtype != np.float32 or not np.isfinite(image).all()
                or min(image.shape) < 16 or any(n % 16 for n in image.shape)
                or mask.shape != image.shape or foreground.shape != image.shape
                or mask.dtype != np.bool_ or foreground.dtype != np.bool_):
            raise ValueError("Invalid full-volume input arrays")
        try:
            affine = np.asarray(case["affine_ras"], dtype=float)
            grid = case["grid_shape"]
            digest = case["input_sha256"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Incomplete full-volume geometry") from error
        if (list(grid) != [n // 16 for n in image.shape] or affine.shape != (4, 4)
                or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12
                or hashlib.sha256(image.tobytes()).hexdigest() != digest):
            raise ValueError("Full-volume input geometry/hash mismatch")


def _load_inputs(cfg, signature):
    folder = Path(cfg["output_dir"]) / "inputs"
    archive, receipt_path = folder / "inputs.npz", folder / "receipt.json"
    if not _pair_state(archive, receipt_path):
        raise FileNotFoundError("Prepare full-volume inputs first")
    receipt = _read_json(receipt_path)
    if receipt.get("signature") != signature or receipt.get("sha256") != _sha256(archive):
        raise ValueError("Full-volume inputs changed; choose a new output_dir")
    arrays = _load_archive(archive)
    _validate_inputs(arrays, receipt.get("cases"), len(cfg["cases"]))
    return arrays, receipt


def prepare_full_volume_inputs(cfg):
    """Freeze one complete, bounded-resolution image per case without cropping."""
    signature = _input_signature(cfg)
    out = Path(cfg["output_dir"])
    folder = out / "inputs"
    archive, receipt_path = folder / "inputs.npz", folder / "receipt.json"
    if _pair_state(archive, receipt_path):
        return _load_inputs(cfg, signature)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Unrecognized/incomplete full-volume output_dir; choose a new directory")
    with _lock(out, ".inputs.lock"):
        arrays, metadata = {}, []
        for i, case in enumerate(cfg["cases"]):
            prepared = full_volume.prepare_full_volume(case["image"], case.get("mask"), **_options(cfg))
            for key in ("volume", "mask", "foreground"):
                arrays[f"{key}_{i}"] = prepared[key]
            metadata.append({**prepared["metadata"], "case_id": case["id"], "modality": case["modality"]})
        _validate_inputs(arrays, metadata, len(cfg["cases"]))
        folder.mkdir()
        _atomic_new(archive, arrays, arrays=True)
        receipt = {"signature": signature, "cases": metadata, "sha256": _sha256(archive)}
        _atomic_new(receipt_path, receipt)
    return arrays, receipt


def _validate_tokens(arrays, metadata, widths=None):
    expected = {f"{key}_{i}" for i in range(len(metadata)) for key in ("volume", "mask", "foreground")}
    expected.update(f"tokens_{m}_{i}" for m in range(len(MODELS)) for i in range(len(metadata)))
    if set(arrays) != expected:
        raise ValueError("Incomplete full-volume features")
    _validate_inputs({k: v for k, v in arrays.items() if not k.startswith("tokens_")}, metadata, len(metadata))
    for m, name in enumerate(MODELS):
        width = None if widths is None else widths.get(name)
        for i, case in enumerate(metadata):
            tokens = arrays[f"tokens_{m}_{i}"]
            if (tokens.ndim != 2 or tokens.dtype != np.float32
                    or tokens.shape[0] != int(np.prod(case["grid_shape"])) or tokens.shape[1] == 0
                    or not np.isfinite(tokens).all() or (np.max(np.abs(tokens), axis=1) == 0).any()
                    or (width is not None and tokens.shape[1] != width)):
                raise ValueError(f"Invalid full-volume patch features: {name}, case {i}")
            width = tokens.shape[1]


def _load_features(cfg, inputs, input_receipt, signature):
    out = Path(cfg["output_dir"])
    archive, receipt_path = out / "maps.npz", out / "receipt.json"
    if not _pair_state(archive, receipt_path):
        raise FileNotFoundError("Extract full-volume patch features first")
    receipt = _read_json(receipt_path)
    if (receipt.get("signature") != signature or receipt.get("feature_schema") != SCHEMA
            or receipt.get("maps_sha256") != _sha256(archive)
            or receipt.get("cases") != input_receipt["cases"]
            or receipt.get("checkpoint_sha256") != {k: v["sha256"] for k, v in signature["checkpoints"].items()}):
        raise ValueError("Full-volume features changed; choose a new output_dir")
    arrays = _load_archive(archive)
    _validate_tokens(arrays, receipt["cases"], receipt.get("embedding_dimensions"))
    if any(not np.array_equal(arrays[key], value) for key, value in inputs.items()):
        raise ValueError("Full-volume features do not match frozen inputs")
    return arrays, receipt


def load_full_volume_features(cfg):
    """Load a verified full-volume cache without preparation or model inference."""
    inputs, input_receipt = _load_inputs(cfg, _input_signature(cfg))
    return _load_features(cfg, inputs, input_receipt, _feature_signature(cfg, input_receipt))


def load_or_extract_full_volume_features(cfg):
    """One float32 features-only forward per case/checkpoint, sequentially."""
    inputs, input_receipt = prepare_full_volume_inputs(cfg)
    signature = _feature_signature(cfg, input_receipt)
    out = Path(cfg["output_dir"])
    archive, receipt_path = out / "maps.npz", out / "receipt.json"
    if _pair_state(archive, receipt_path):
        return _load_features(cfg, inputs, input_receipt, signature)
    if {path.name for path in out.iterdir()} != {"inputs"}:
        raise ValueError("Unrecognized/incomplete full-volume features; choose a new output_dir")
    device = torch.device(signature["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select an accessible GPU or explicitly choose device='cpu'")
    from reform3d.encoders.factory import create_feature_extractor

    with _lock(out, ".features.lock"):
        arrays, counts, widths = dict(inputs), {}, {}
        if device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(signature["memory_fraction"], device=device)
        for m, name in enumerate(MODELS):
            extractor = create_feature_extractor(name, checkpoint_path=cfg["checkpoints"][name], device=str(device))
            model = extractor.model
            if model.patch_size != 16:
                raise ValueError("Full-volume examples require patch size 16")
            counts[name] = sum(parameter.numel() for parameter in model.parameters())
            widths[name] = int(model.embed_dim)
            try:
                for i, case in enumerate(cfg["cases"]):
                    print(f"Extracting full volume: {name} / {case['id']}", flush=True)
                    x = torch.from_numpy(inputs[f"volume_{i}"])[None, None].to(device=device, dtype=torch.float32)
                    with torch.inference_mode():
                        features = model.forward_features(x)["x_norm_patchtokens"]
                    if features.ndim != 3 or features.shape[0] != 1:
                        raise ValueError("Expected one batch of spatial patch features")
                    arrays[f"tokens_{m}_{i}"] = features[0].detach().float().cpu().numpy().copy()
                    del x, features
            finally:
                del model, extractor
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        _validate_tokens(arrays, input_receipt["cases"], widths)
        _atomic_new(archive, arrays, arrays=True)
        receipt = {"signature": signature, "feature_schema": SCHEMA, "cases": input_receipt["cases"],
                   "maps_sha256": _sha256(archive), "inputs_sha256": input_receipt["sha256"],
                   "checkpoint_sha256": {k: v["sha256"] for k, v in signature["checkpoints"].items()},
                   "embedding_dimensions": widths, "backbone_parameters": counts,
                   "forward_count": len(MODELS) * len(cfg["cases"]), "attention_extracted": False}
        _atomic_new(receipt_path, receipt)
    return arrays, receipt
