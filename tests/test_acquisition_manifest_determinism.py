"""Raw-data preparation must keep the same identity across Python processes."""
import json
import os
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
import yaml


def test_raw_acquisition_manifest_identity_is_stable_across_hash_seeds(tmp_path):
    data = tmp_path / "data"
    images, masks = data / "images", data / "masks"
    images.mkdir(parents=True)
    masks.mkdir()
    for index in reversed(range(12)):
        affine = np.diag([.8 + index * .003, .8, .8, 1.])
        volume = np.arange(216, dtype=np.float32).reshape(6, 6, 6) + index
        nib.save(nib.Nifti1Image(volume, affine), images / f"case_{index:05d}_ct.nii.gz")
        nib.save(nib.Nifti1Image(np.ones(volume.shape, np.uint8), affine),
                 masks / f"case_{index:05d}_mask.nii.gz")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "dataset": "kits23", "data_root": str(data), "workspace": str(tmp_path / "workspace"),
        "checkpoints": ["med3dino_rel_96"], "device": "cpu", "seeds": [42],
    }))
    script = """
import contextlib, io, json, logging, sys
from reform3d.settings import configure_environment, load_settings
cfg = load_settings(sys.argv[1])
configure_environment(cfg)
from reform3d.prepare import prepare_manifest
logging.disable(logging.CRITICAL)
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    path, payload, identity = prepare_manifest(cfg)
print(json.dumps({'identity': identity, 'payload': payload}))
"""
    results = []
    source = Path(__file__).resolve().parents[1] / "src"
    for hash_seed in (17, 83):
        env = {**os.environ, "PYTHONHASHSEED": str(hash_seed), "PYTHONPATH": str(source),
               "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
        result = subprocess.run([sys.executable, "-c", script, str(config)], cwd=tmp_path,
                                env=env, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        results.append(json.loads(result.stdout))
    assert results[0]["identity"] == results[1]["identity"]
    assert results[0]["payload"] == results[1]["payload"]
    paths = [row["file_path"] for row in results[0]["payload"]["volumes"]]
    assert len(paths) == 12 and paths == sorted(set(paths))
