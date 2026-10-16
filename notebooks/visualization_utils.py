"""Configuration, shared crops, PCA and full-volume CT/MR similarity displays."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
try:
    from reform3d import visualization as viz
except ImportError as error:
    raise RuntimeError("Restart the notebook kernel to use this checkout's reform3d package") from error

from reform3d.feature_similarity import reference_similarity

POLICIES = ("ISO", "REL", "SA")
METHOD = "ct-mr-shared-crop-v1"
DEFAULT_FOV_MM = 192.0


def setup(workflow="pca", config_path=None):
    """Read one workflow from the shared JSON; paths resolve beside that file."""
    if workflow not in {"pca", "similarity"}:
        raise ValueError("workflow must be pca or similarity")
    if not Path(sys.modules["reform3d"].__file__).resolve().is_relative_to(SRC):
        raise RuntimeError("Restart the notebook kernel to use this checkout's reform3d package")
    path = Path(config_path or os.environ.get("REFORM3D_VIZ_CONFIG", ROOT / "notebooks/visualization_config.json"))
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("Copy notebooks/visualization_config.example.json to visualization_config.json and edit its paths")
    document = json.loads(path.read_text())
    if not isinstance(document, dict) or not isinstance(document.get(workflow), dict):
        raise ValueError(f"Configuration requires a {workflow} section")
    cfg = {"checkpoints": document.get("checkpoints", {}),
           "device": document.get("device", "cuda:0"), **document[workflow]}
    if not cfg.get("output_dir"):
        raise ValueError(f"{workflow} requires its own explicit output_dir")

    def resolve(value):
        candidate = Path(value).expanduser()
        return str((candidate if candidate.is_absolute() else path.parent / candidate).resolve())

    outputs = [resolve(document[key]["output_dir"]) for key in ("pca", "similarity")
               if isinstance(document.get(key), dict) and document[key].get("output_dir")]
    if len(outputs) == 2 and outputs[0] == outputs[1]:
        raise ValueError("pca and similarity require separate output directories")
    cases = cfg.get("cases")
    if not isinstance(cases, list) or not cases or any(
            not isinstance(case, dict) or case.get("modality") not in {"CT", "MR"}
            or not case.get("image") for case in cases):
        raise ValueError("Cases require CT/MR modality and an image path")
    if workflow == "pca":
        if len(cases) != 2 or {case["modality"] for case in cases} != {"CT", "MR"}:
            raise ValueError("PCA requires one CT and one MR case")
        _crop_options(cfg)
    else:
        ids = [case.get("id") for case in cases]
        if any(not isinstance(case_id, str) or not case_id.strip() for case_id in ids):
            raise ValueError("Similarity cases require a unique id and CT/MR modality")
        if len(set(ids)) != len(ids):
            raise ValueError("Similarity case ids must be unique")
    if not isinstance(cfg["checkpoints"], dict) or set(cfg["checkpoints"]) != set(viz.MODELS):
        raise ValueError(f"Provide exactly these checkpoint keys: {viz.MODELS}")
    for case in cases:
        case["image"] = resolve(case["image"])
        case["mask"] = resolve(case["mask"]) if case.get("mask") else None
    missing = [f"{c['modality']} {key}: {c[key]}" for c in cfg["cases"]
               for key in ("image", "mask") if c.get(key) and not Path(c[key]).is_file()]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))
    cfg["checkpoints"] = {name: resolve(filename) for name, filename in cfg["checkpoints"].items()}
    for name, filename in cfg["checkpoints"].items():
        if not Path(filename).is_file():
            raise FileNotFoundError(f"Missing {name} checkpoint: {filename}")
    cfg["output_dir"] = resolve(cfg["output_dir"])
    return cfg


# Shared crops and PCA

def _crop_options(cfg):
    seed = cfg.get("seed", 42)
    if type(seed) is not int or not 0 <= seed < 2**32 - len(cfg["cases"]):
        raise ValueError("seed must be a nonnegative 32-bit integer with room for each case")
    options = []
    for case in cfg["cases"]:
        crop = {"fov_mm": DEFAULT_FOV_MM, **cfg.get("crop", {}), **case.get("crop", {})}
        if (set(crop) - {"fov_mm", "center_ras_mm"} or isinstance(crop["fov_mm"], bool)
                or not isinstance(crop["fov_mm"], (int, float))
                or not np.isfinite(crop["fov_mm"]) or crop["fov_mm"] <= 0):
            raise ValueError("crop accepts a finite, positive fov_mm and optional center_ras_mm")
        option = {"fov_mm": float(crop["fov_mm"]), "size": 96}
        if "center_ras_mm" in crop:
            center = crop["center_ras_mm"]
            if (not isinstance(center, (list, tuple)) or len(center) != 3
                    or any(isinstance(x, bool) or not isinstance(x, (int, float))
                           or not np.isfinite(x) for x in center)):
                raise ValueError("center_ras_mm must contain three finite RAS coordinates in millimetres")
            option["center_ras_mm"] = [float(x) for x in center]
        options.append(option)
    return seed, options


def _preprocessing(cfg):
    seed, options = _crop_options(cfg)
    return {"method": METHOD, "seed": seed, "crops": options,
            "crop_source": viz.shared_crop_signature(),
            "archive_source_sha256": hashlib.sha256("".join(inspect.getsource(f) for f in
                (_crop_options, _input_signature, prepare_inputs)).encode()).hexdigest()}


def _input_signature(cfg):
    return {"preprocessing": _preprocessing(cfg),
            "cases": [{"modality": c["modality"], "image": viz._fingerprint(c["image"]),
                       "mask": viz._fingerprint(c["mask"]) if c.get("mask") else None} for c in cfg["cases"]],
            "numpy": np.__version__, "scipy": viz.scipy.__version__,
            "simpleitk": viz.sitk.Version_VersionString()}


def prepare_inputs(cfg):
    """Freeze one physical crop per case; every checkpoint reuses the saved bytes."""
    folder = Path(cfg["output_dir"]) / "inputs"
    receipt_path, archive = folder / "receipt.json", folder / "inputs.npz"
    signature = _input_signature(cfg)
    if receipt_path.is_file() and archive.is_file():
        receipt = json.loads(receipt_path.read_text())
        if receipt["signature"] != signature or receipt["sha256"] != viz._sha256(archive):
            raise ValueError("Frozen inputs changed; choose a new output_dir")
        with np.load(archive, allow_pickle=False) as data:
            cases = [{key: data[f"{key}_{i}"] for key in ("volume", "mask", "foreground")}
                     for i in range(len(cfg["cases"]))]
        for i, case in enumerate(cases):
            case.update(metadata=receipt["cases"][i], slice=case["volume"].shape[2] // 2)
            if hashlib.sha256(case["volume"].tobytes()).hexdigest() != case["metadata"]["input_sha256"]:
                raise ValueError("Frozen input hash does not match its receipt")
    else:
        if folder.exists() and any(folder.iterdir()):
            raise ValueError("Incomplete frozen inputs; choose a new output_dir")
        out = Path(cfg["output_dir"])
        if out.exists() and any(out.iterdir()):
            raise ValueError("Existing visualization output; choose a new output_dir for shared inputs")
        seed, options = _crop_options(cfg)
        cases = [viz.prepare_shared_case(c["image"], c.get("mask"), seed=seed + i, **options[i])
                 for i, c in enumerate(cfg["cases"])]
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(archive, **{f"{key}_{i}": c[key] for i, c in enumerate(cases)
                                      for key in ("volume", "mask", "foreground")})
        receipt = {"signature": signature, "sha256": viz._sha256(archive),
                   "cases": [c["metadata"] for c in cases]}
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return {(m, i): case for i, case in enumerate(cases) for m in range(len(POLICIES))}


def extract_features(cfg):
    views = prepare_inputs(cfg)
    return viz.collect_maps(cfg, prepare_views=lambda config: views, preprocessing=_preprocessing(cfg))


def load_features(cfg):
    prepare_inputs(cfg)
    return viz.load_maps(cfg, preprocessing=_preprocessing(cfg))


def _axial_geometry(shape, metadata):
    affine = np.asarray(metadata["affine_ras"], dtype=float)
    if affine.shape != (4, 4) or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
        raise ValueError("A finite, invertible RAS affine is required")
    matrix, origin = affine[:3, :3], affine[:3, 3]
    corners = np.indices((2, 2, 2)).reshape(3, -1) * (np.asarray(shape) - 1)[:, None]
    world = matrix @ corners + origin[:, None]
    return matrix, origin, world.min(1), world.max(1)


def _axial_coordinates(shape, metadata, z_mm=None):
    matrix, origin, low, high = _axial_geometry(shape, metadata)
    z = (low[2] + high[2]) / 2 if z_mm is None else float(z_mm)
    if not np.isfinite(z) or not low[2] - 1e-6 <= z <= high[2] + 1e-6:
        raise ValueError("Axial level lies outside this crop")
    nx, ny = shape[:2]
    x, y = np.meshgrid(np.linspace(high[0], low[0], nx), np.linspace(low[1], high[1], ny))
    plane = np.stack([x, y, np.full_like(x, z)])
    coords = np.linalg.solve(matrix, (plane - origin[:, None, None]).reshape(3, -1)).reshape(3, ny, nx)
    for axis, n in enumerate(shape):
        coords[axis][np.abs(coords[axis]) < 1e-7] = 0
        coords[axis][np.abs(coords[axis] - (n - 1)) < 1e-7] = n - 1
    dx, dy = (high[:2] - low[:2]) / np.maximum(np.array([nx, ny]) - 1, 1)
    return coords, [high[0] + dx / 2, low[0] - dx / 2, low[1] - dy / 2, high[1] + dy / 2]


def _axial_plane(array, metadata, *, order=1, background=0., z_mm=None):
    coords, extent = _axial_coordinates(array.shape[:3], metadata, z_mm)
    if array.ndim == 4:
        plane = np.stack([ndimage.map_coordinates(array[..., c], coords, order=order,
                          mode="constant", cval=background) for c in range(array.shape[-1])], axis=-1)
    else:
        plane = ndimage.map_coordinates(array.astype(np.float32), coords, order=order,
                                       mode="constant", cval=background)
    return plane, extent


def _draw(ax, volume, mask, metadata, title=None, *, z_mm=None):
    plane, extent = _axial_plane(volume, metadata, background=-1., z_mm=z_mm)
    labels, _ = _axial_plane(mask, metadata, order=0, z_mm=z_mm)
    ax.imshow(plane, extent=extent, cmap="gray", vmin=-1, vmax=1, origin="lower")
    if labels.any() and not labels.all():
        ax.contour(labels, extent=extent, origin="lower", levels=[0.5], colors=["#5cf3ca"], linewidths=0.65)
    if title:
        ax.set_title(title, fontsize=12, fontweight="medium")
    ax.set_xticks([]); ax.set_yticks([])
    return extent, labels


def select_axial_slices(cfg, arrays, receipt, num_slices=5):
    """Choose shared levels from input foreground, independently of labels/features."""
    if type(num_slices) is not int or not 1 <= num_slices <= 5:
        raise ValueError("num_slices must be an integer from 1 to 5")
    selected = []
    for i, case in enumerate(cfg["cases"]):
        foreground = arrays[f"foreground_0_{i}"]
        metadata = receipt["cases"][f"0_{i}"]
        matrix, origin, low, high = _axial_geometry(foreground.shape, metadata)
        if not np.allclose(matrix, np.diag(np.diag(matrix))) or (np.diag(matrix) <= 0).any():
            raise ValueError("Shared inputs must use an axis-aligned RAS grid")
        for m in (1, 2):
            if not np.array_equal(arrays[f"volume_0_{i}"], arrays[f"volume_{m}_{i}"]) or not np.array_equal(
                    foreground, arrays[f"foreground_{m}_{i}"]) or not np.array_equal(
                    metadata["affine_ras"], receipt["cases"][f"{m}_{i}"]["affine_ras"]):
                raise ValueError("All encoders must share identical inputs and geometry")
        coverage = foreground.sum(axis=(0, 1))
        if not coverage.any():
            raise ValueError(f"{case['modality']} has no anatomy in the input foreground")
        # Avoid almost-empty end slices, then spread views across the useful slab.
        informative = np.flatnonzero(coverage >= .5 * coverage.max())
        if num_slices == 1:
            chosen = np.array([int(np.argmax(coverage))])
        else:
            count = min(num_slices, len(informative))
            positions = np.rint(np.linspace(0, len(informative) - 1, count)).astype(int)
            chosen = informative[positions]
        selected.append({"modality": case["modality"], "indices": chosen.tolist(),
                         "z_mm": (origin[2] + matrix[2, 2] * chosen).tolist(),
                         "rule": "image foreground coverage >= 50% of peak; evenly spaced levels",
                         "foreground_area_mm2": (coverage[chosen] * matrix[0, 0] * matrix[1, 1]).tolist(),
                         "shared_z_range_mm": [float(low[2]), float(high[2])]})
    return selected


def plot_inputs(cfg, views, num_slices=5):
    """Preview one shared input per case, before any encoder is loaded."""
    import matplotlib.pyplot as plt
    arrays = {f"{key}_{m}_{i}": view[key] for (m, i), view in views.items()
              for key in ("volume", "mask", "foreground")}
    receipt = {"cases": {f"{m}_{i}": view["metadata"] for (m, i), view in views.items()}}
    selection = select_axial_slices(cfg, arrays, receipt, num_slices)
    rows = max(len(s["z_mm"]) for s in selection)
    fig, axes = plt.subplots(rows, len(selection), figsize=(7 * len(selection), 3 * rows),
                             squeeze=False, layout="constrained")
    for i, chosen in enumerate(selection):
        view = views[0, i]
        for j, ax in enumerate(axes[:, i]):
            if j >= len(chosen["z_mm"]):
                ax.set_visible(False); continue
            z = chosen["z_mm"][j]
            _draw(ax, view["volume"], view["mask"], view["metadata"],
                  f"{chosen['modality']} · shared 96³ crop" if j == 0 else None, z_mm=z)
            ax.set_ylabel(f"{j + 1}/{len(chosen['z_mm'])} · z = {z:.1f} mm", fontsize=10)
    fig.suptitle("Shared inputs · axial · reference contours in cyan", fontsize=15)
    return fig


def contextual_tokens(patches, cls, cls_weight=.5):
    """Exploratory normalized fusion; a constant CLS shift alone cancels in PCA."""
    if not 0 <= cls_weight < 1:
        raise ValueError("PCA cls_weight must be in [0, 1); CLS alone has no spatial variation")
    patches, cls = np.asarray(patches), np.asarray(cls)
    if patches.ndim != 2 or cls.shape != (patches.shape[1],):
        raise ValueError("CLS and patch feature dimensions must match")
    norms = np.linalg.norm(patches, axis=1, keepdims=True)
    cls_norm = np.linalg.norm(cls)
    if not np.isfinite(patches).all() or not np.isfinite(cls).all() or (norms <= 0).any() or cls_norm <= 0:
        raise ValueError("Fusion needs finite, nonzero CLS and patch features")
    combined = (1 - cls_weight) * patches / norms + cls_weight * cls / cls_norm
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    if (norms < 1e-8).any():
        raise ValueError("CLS and patch features cancel at this fusion weight")
    return combined / norms


def _resize_map(raw, shape, interpolation):
    x = torch.as_tensor(raw, dtype=torch.float32)
    rgb = x.ndim == 4
    x = x.permute(3, 0, 1, 2)[None] if rgb else x[None, None]
    kwargs = {"align_corners": False} if interpolation == "trilinear" else {}
    x = F.interpolate(x, size=shape, mode=interpolation, **kwargs)[0]
    return x.permute(1, 2, 3, 0).numpy() if rgb else x[0].numpy()


def plot_features(cfg, arrays, receipt=None, *, kind="pca", mode="patch", cls_weight=.5,
                  num_slices=5, interpolation="trilinear"):
    """Plot per-encoder 3D PCA on shared CT/MR axial slices."""
    import matplotlib.pyplot as plt
    if kind != "pca":
        raise ValueError("Use plot_full_volume_similarity for reference-feature overlays")
    if interpolation not in {"trilinear", "nearest"}:
        raise ValueError("interpolation must be trilinear or nearest")
    if mode not in {"patch", "combined"}:
        raise ValueError("PCA mode must be patch or combined")
    if receipt is None:
        cached, receipt = load_features(cfg)
        if arrays.keys() != cached.keys() or any(not np.array_equal(arrays[k], cached[k]) for k in arrays):
            raise ValueError("Arrays do not match this cache; rerun extract_features or load_features")
    selection = select_axial_slices(cfg, arrays, receipt, num_slices)
    maps, bases = {}, {}
    for m in range(len(POLICIES)):
        for i in range(len(cfg["cases"])):
            suffix = f"{m}_{i}"
            grid = arrays[f"attention_{suffix}"].shape[1:]
            tokens = arrays[f"tokens_{suffix}"]
            if mode == "combined":
                tokens = contextual_tokens(tokens, arrays[f"cls_{suffix}"], cls_weight)
            colors, basis = viz.pca_colors([tokens], [arrays[f"foreground_{suffix}"]], [grid])
            maps[m, i] = _resize_map(colors[0], arrays[f"volume_{suffix}"].shape, interpolation)
            bases.update({f"{suffix}_{key}": value for key, value in basis.items()})
    label = "patch features" if mode == "patch" else f"normalized CLS/patch fusion ({cls_weight:.0%} CLS)"
    suffix = f"pca_{mode}_{interpolation}_slices{num_slices}"
    if mode == "combined":
        suffix += f"_cls{cls_weight:g}"
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    figures = []
    for i, chosen in enumerate(selection):
        rows = len(chosen["z_mm"])
        fig, axes = plt.subplots(rows, 4, figsize=(13, 3 * rows), squeeze=False, layout="constrained")
        for j, z in enumerate(chosen["z_mm"]):
            view_suffix = f"0_{i}"
            image, mask = arrays[f"volume_{view_suffix}"], arrays[f"mask_{view_suffix}"]
            metadata = receipt["cases"][view_suffix]
            _draw(axes[j, 0], image, mask, metadata, "Shared input" if j == 0 else None, z_mm=z)
            axes[j, 0].set_ylabel(f"{j + 1}/{rows} · z = {z:.1f} mm", fontsize=10)
            fg, _ = _axial_plane(arrays[f"foreground_{view_suffix}"], metadata, order=0, z_mm=z)
            for m, policy in enumerate(POLICIES):
                ax = axes[j, m + 1]
                extent, labels = _draw(ax, image, mask, metadata, policy if j == 0 else None, z_mm=z)
                plane, _ = _axial_plane(maps[m, i], metadata, order=0 if interpolation == "nearest" else 1, z_mm=z)
                ax.imshow(plane, alpha=fg.astype(float) * .8, extent=extent, origin="lower")
                if labels.any() and not labels.all():
                    ax.contour(labels, extent=extent, origin="lower", levels=[.5], colors=["#5cf3ca"], linewidths=.65)
        fig.suptitle(f"{chosen['modality']} · {label} · axial", fontsize=15)
        for extension in ("png", "pdf"):
            fig.savefig(out / f"{suffix}_{chosen['modality']}.{extension}", dpi=150, bbox_inches="tight")
        figures.append(fig)
    np.savez_compressed(out / f"{suffix}_bases.npz", **bases)
    (out / f"{suffix}_display.json").write_text(json.dumps({
        "maps_sha256": receipt["maps_sha256"], "plane": "shared physical axial levels",
        "num_slices": num_slices, "slice_selection": selection, "kind": kind, "mode": mode,
        "cls_weight": cls_weight, "interpolation": interpolation,
        "input_sha256": [receipt["cases"][f"0_{i}"]["input_sha256"] for i in range(len(selection))],
        "plot_source_sha256": viz._sha256(Path(__file__)),
    }, indent=2) + "\n")
    return figures


# Full-volume reference similarity

def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def _plane_coordinates(source, z_mm):
    """Use the entire source voxel-cell bounds and native in-plane pixel count."""
    low = np.asarray(source["metadata"]["source_bounds_ras_mm"]["low"])
    high = np.asarray(source["metadata"]["source_bounds_ras_mm"]["high"])
    nx, ny = source["volume"].shape[:2]
    dx, dy = (high[:2] - low[:2]) / [nx, ny]
    x, y = np.meshgrid(np.linspace(high[0] - dx / 2, low[0] + dx / 2, nx),
                       np.linspace(low[1] + dy / 2, high[1] - dy / 2, ny))
    world = np.stack([x, y, np.full_like(x, z_mm)])
    return world, [float(high[0]), float(low[0]), float(low[1]), float(high[1])]


def _sample_plane(volume, affine, world, *, order=1, mode="constant", background=0.):
    inverse = np.linalg.inv(np.asarray(affine, dtype=float))
    coords = (inverse[:3, :3] @ world.reshape(3, -1) + inverse[:3, 3, None])
    coords = coords.reshape(world.shape)
    for axis, size in enumerate(volume.shape):
        coords[axis][np.isclose(coords[axis], 0., atol=1e-7, rtol=0)] = 0.
        coords[axis][np.isclose(coords[axis], size - 1, atol=1e-7, rtol=0)] = size - 1
    return ndimage.map_coordinates(volume, coords, order=order, mode=mode,
                                   cval=background, prefilter=False)


def _slice_levels(source, reference_ras, patch_affine, num_slices, slice_indices):
    if type(num_slices) is not int or not 1 <= num_slices <= 5:
        raise ValueError("num_slices must be an integer from 1 to 5")
    affine = np.asarray(source["affine_ras"])
    shape = np.asarray(source["volume"].shape)
    if slice_indices is None:
        center = np.linalg.solve(affine, np.r_[reference_ras, 1.])[:3]
        step = abs(float(patch_affine[2, 2] / affine[2, 2]))
        offsets = np.zeros(1) if num_slices == 1 else np.linspace(-2, 2, num_slices) * step
        indices = np.clip(np.rint(center[2] + offsets).astype(int), 0, shape[2] - 1)
        indices = list(dict.fromkeys(indices.tolist()))
    else:
        if (not isinstance(slice_indices, (tuple, list, np.ndarray))
                or not 1 <= len(slice_indices) <= 5
                or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer))
                       or not 0 <= x < shape[2] for x in slice_indices)
                or len(set(slice_indices)) != len(slice_indices)):
            raise ValueError("slice_indices must contain one to five distinct source axial indices")
        indices = list(map(int, slice_indices))
    xy = (shape[:2] - 1) / 2
    levels = [float((affine @ np.r_[xy, z, 1.])[2]) for z in indices]
    return indices, levels


def _mark_reference(ax, reference_ras, bounds, z_mm):
    from matplotlib.patches import Rectangle
    import matplotlib.patheffects as effects

    low, high = np.asarray(bounds["low"]), np.asarray(bounds["high"])
    stroke = [effects.withStroke(linewidth=2.5, foreground="#202020")]
    if low[2] <= z_mm < high[2]:
        box = Rectangle(low[:2], *(high[:2] - low[:2]), fill=False, edgecolor="white",
                        linestyle="--", linewidth=1., zorder=6)
        box.set_path_effects(stroke)
        ax.add_patch(box)
    if np.isclose(reference_ras[2], z_mm, atol=1e-6, rtol=0):
        marker, = ax.plot(*reference_ras[:2], "+", color="white", markersize=12,
                          markeredgewidth=1.8, zorder=7)
        marker.set_path_effects(stroke)


def plot_full_volume_similarity(cfg, arrays, receipt, *, case_id, reference_ras_mm=None,
                                num_slices=5, slice_indices=None, opacity=.65,
                                show_contours=True, interpolation="trilinear"):
    """Compare ISO/REL/SA features on uncropped native CT/MR frames.

    One RAS point identifies the same physical patch in all three encoders.
    Similarity is sampled directly from the native feature grid onto each
    original axial frame; contours only annotate the resulting overlays.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from reform3d.full_volume import read_source

    if interpolation not in {"trilinear", "nearest"}:
        raise ValueError("interpolation must be trilinear or nearest")
    if (isinstance(opacity, (bool, np.bool_)) or not isinstance(opacity, (int, float))
            or not np.isfinite(opacity) or not 0 <= opacity <= 1):
        raise ValueError("opacity must be between 0 and 1")
    matches = [i for i, case in enumerate(cfg["cases"]) if case.get("id") == case_id]
    if len(matches) != 1:
        raise ValueError("case_id must identify exactly one configured case")
    i = matches[0]
    case, metadata = cfg["cases"][i], receipt["cases"][i]
    if metadata.get("case_id") != case_id or metadata.get("modality") != case["modality"]:
        raise ValueError("Feature receipt does not match the requested case")
    point = case.get("reference_ras_mm") if reference_ras_mm is None else reference_ras_mm
    if np.shape(point) != (3,) or not np.isfinite(np.asarray(point, dtype=float)).all():
        raise ValueError("reference_ras_mm must contain three finite RAS coordinates")
    point = np.asarray(point, dtype=float)
    model_affine = np.asarray(metadata["affine_ras"], dtype=float)
    model_volume = arrays[f"volume_{i}"]
    if hashlib.sha256(model_volume.tobytes()).hexdigest() != metadata["input_sha256"]:
        raise ValueError("Input array does not match the full-volume feature receipt")
    model_voxel = np.linalg.solve(model_affine, np.r_[point, 1.])[:3]
    if (model_voxel < -.5).any() or (model_voxel >= np.array(model_volume.shape) - .5).any():
        raise ValueError("Reference point lies outside the full-volume model input")
    grid = np.asarray(metadata["grid_shape"], dtype=int)
    if tuple(model_volume.shape) != tuple(grid * 16):
        raise ValueError("The native feature grid must match the 16-voxel model patches")
    patch = np.floor((model_voxel + .5) / 16).astype(int)
    patch = np.minimum(patch, grid - 1)
    patch_affine = model_affine.copy()
    patch_affine[:3, :3] *= 16
    patch_affine[:3, 3] = (model_affine @ np.array([7.5, 7.5, 7.5, 1.]))[:3]
    bounds_voxel = np.array([patch * 16 - .5, (patch + 1) * 16 - .5])
    corners = np.array(np.meshgrid(*zip(*bounds_voxel), indexing="ij")).reshape(3, -1)
    world_corners = model_affine[:3, :3] @ corners + model_affine[:3, 3, None]
    patch_bounds = {"low": world_corners.min(1).tolist(), "high": world_corners.max(1).tolist()}
    native = {policy: reference_similarity(arrays[f"tokens_{m}_{i}"], tuple(grid), tuple(patch))
              for m, policy in enumerate(POLICIES)}

    source = read_source(case["image"], case.get("mask"))
    source_metadata = source["metadata"]
    expected = receipt["signature"]["inputs"]["cases"][i]
    source_records = {}
    for role in ("image", "mask"):
        actual, saved = source_metadata.get(role), expected.get(role)
        if ((actual is None) != (saved is None)
                or (actual and any(actual[key] != saved[key] for key in ("path", "bytes", "mtime_ns")))
                or (actual and _digest(actual["path"]) != saved["sha256"])):
            raise ValueError(f"Source {role} changed since full-volume feature extraction")
        source_records[role] = None if actual is None else {**actual, "sha256": saved["sha256"]}
    source_low = np.asarray(source_metadata["source_bounds_ras_mm"]["low"])
    source_high = np.asarray(source_metadata["source_bounds_ras_mm"]["high"])
    if (point < source_low).any() or (point >= source_high).any():
        raise ValueError("Reference point lies outside the original image")
    selected, levels = _slice_levels(source, point, patch_affine, num_slices, slice_indices)
    window = case.get("display_window", [metadata["normalization"][key] for key in ("low", "high")])
    if np.shape(window) != (2,):
        raise ValueError("display_window must contain a finite, increasing low/high pair")
    low, high = map(float, window)
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError("The full-volume normalization bounds must be finite and ordered")
    order = 0 if interpolation == "nearest" else 1

    def draw(ax, z, title=None, similarity=None):
        world, extent = _plane_coordinates(source, z)
        raw = _sample_plane(source["volume"], source["affine_ras"], world, background=low)
        gray = 2 * np.clip((raw - low) / (high - low), 0, 1) - 1
        ax.imshow(gray, cmap="gray", vmin=-1, vmax=1, extent=extent, origin="lower", aspect="equal")
        if similarity is not None:
            # Nearest boundary extension covers half-patch edge cells without
            # inventing a dark frame; source pixels remain inside model bounds.
            plane = _sample_plane(similarity, patch_affine, world, order=order, mode="nearest")
            ax.imshow(plane, cmap="coolwarm", vmin=-1, vmax=1, extent=extent,
                      origin="lower", aspect="equal", alpha=float(opacity) * np.clip(plane, 0, 1))
        if show_contours:
            labels = _sample_plane(source["mask"], source["affine_ras"], world, order=0)
            if labels.any() and not labels.all():
                ax.contour(labels, extent=extent, origin="lower", levels=[.5], colors=["#5cf3ca"],
                           linewidths=.65, zorder=5)
        _mark_reference(ax, point, patch_bounds, z)
        if title:
            ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])

    preview, ax = plt.subplots(figsize=(6., 5.5), layout="constrained")
    draw(ax, point[2], f"{case_id} · full source reference")
    preview.suptitle(f"RAS ({', '.join(f'{x:.1f}' for x in point)}) mm → patch {tuple(patch)}", fontsize=11)
    ax.set_xlabel("White cross: reference point · dashed box: physical patch footprint", fontsize=9)
    gallery, axes = plt.subplots(len(levels), 4, figsize=(12., 2.9 * len(levels)),
                                 squeeze=False, layout="constrained")
    for row, (index, z) in enumerate(zip(selected, levels)):
        draw(axes[row, 0], z, f"{case['modality']} full source" if row == 0 else None)
        axes[row, 0].set_ylabel(f"Source slice {index}\nz = {z:.1f} mm", fontsize=9)
        for column, policy in enumerate(POLICIES, 1):
            draw(axes[row, column], z, policy if row == 0 else None, native[policy])
    contour_note = " · cyan: ground truth" if show_contours else ""
    gallery.suptitle(f"{case_id} · full-volume reference-feature similarity\n"
                     f"Uncropped source frames · positive cosine overlaid · fixed scale{contour_note}", fontsize=12)
    gallery.colorbar(ScalarMappable(norm=Normalize(-1, 1), cmap="coolwarm"), ax=axes,
                     shrink=.55, label="Cosine similarity to the reference patch")

    record = {"case_id": case_id, "case_index": i, "modality": case["modality"],
              "maps_sha256": receipt["maps_sha256"], "checkpoint_sha256": receipt["checkpoint_sha256"],
              "input_sha256": metadata["input_sha256"], "source_image": source_records["image"],
              "source_mask": source_records["mask"], "source_shape": list(source["volume"].shape),
              "source_affine_ras": np.asarray(source["affine_ras"]).tolist(),
              "source_bounds_ras_mm": source_metadata["source_bounds_ras_mm"],
              "reference_ras_mm": point.tolist(), "reference_model_voxel": model_voxel.tolist(),
              "reference_patch": patch.tolist(), "reference_patch_bounds_ras_mm": patch_bounds,
              "reference_source": "manual image coordinate", "native_grid": grid.tolist(),
              "patch_affine_ras": patch_affine.tolist(), "source_slice_indices": selected,
              "slice_z_mm": levels, "native_plane_shape": [source["volume"].shape[1], source["volume"].shape[0]],
              "num_slices": num_slices, "normalization": metadata["normalization"],
              "display_window": [low, high],
              "method": "cosine similarity to one L2-normalized reference patch feature",
              "sampling": "native 3D cosine grid directly onto full source physical axial planes",
              "color_limits": [-1., 1.], "opacity": float(opacity), "interpolation": interpolation,
              "alpha": "opacity times positive cosine similarity, clipped to [0,1]",
              "show_contours": bool(show_contours), "reference_self_match": "Always 1; not evidence of localization",
              "plot_source_sha256": _digest(__file__),
              "similarity_source_sha256": _digest(inspect.getfile(reference_similarity))}
    token = hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest()[:12]
    output = Path(cfg["output_dir"]) / "full_volume_similarity"
    output.mkdir(parents=True, exist_ok=True)
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in case_id) + "_" + token
    files = {}
    for name, figure in (("reference", preview), ("gallery", gallery)):
        for extension in ("png", "pdf"):
            path = output / f"{stem}_{name}.{extension}"
            figure.savefig(path, dpi=150, bbox_inches="tight")
            files[f"{name}_{extension}"] = str(path.relative_to(cfg["output_dir"]))
    path = output / f"{stem}_maps.npz"
    np.savez_compressed(path, **native)
    files["maps"] = str(path.relative_to(cfg["output_dir"]))
    record["files"] = files
    (output / f"{stem}_display.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return [preview, gallery]
