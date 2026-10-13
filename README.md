# REFORM-3D

Evaluate frozen 3D medical encoders for controlled spacing sensitivity, CT–MR anatomical retrieval, and case-disjoint low-shot organ recognition.

## Install

Use Python 3.11 (validated) and a PyTorch build compatible with your CUDA installation.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements.txt .
```

`requirements.txt` records dependency versions for the validated Python 3.11/Linux environment. The `-c` option applies these versions to the dependencies selected by the package. For SAM-Med3D, MiM, and GL-MAE, use `python -m pip install -c requirements.txt '.[external]'`; see [Encoders](docs/encoders.md) for other adapters.

## Run

Edit the data and checkpoint paths in `configs/mmwhs-demo.yaml`, then:

```bash
reform3d check --config configs/mmwhs-demo.yaml
reform3d run --config configs/mmwhs-demo.yaml
```

The command prepares the manifest, validates image/mask geometry, shares compatible crops and embeddings, evaluates every requested checkpoint/seed, and writes:

```text
workspace/runs/mmwhs-demo/report.html
workspace/runs/mmwhs-demo/scores.csv
workspace/runs/mmwhs-demo/run.json
```

MMWHS runs retrieval and low-shot transfer. `configs/kits23-demo.yaml` runs controlled spacing evaluation. Missing modules are explicitly listed in the report.

## Interpret

Compare each property separately. Low embedding drift alone does not establish useful anatomy. These measurements support a task-specific shortlist; they do not guarantee a downstream winner. Low-shot probes fit labeled readouts while the encoder remains frozen.

## Documentation

- [Data](docs/data.md): datasets and manifests.
- [Usage](docs/usage.md): configuration, reuse, and failure handling.
- [Encoders](docs/encoders.md): weights, native features, and optional adapters.
- [Reproducibility](docs/reproduction.md): three-module evaluation on KiTS23 and MMWHS.
- [Validation](validation/README.md): completed two-dataset checks and limitations.
- [Repository verification](docs/audit.md): input, cache, and output integrity checks.

The runtime is bundled under `src/reform3d/`; the original Med3DINO checkout is not required. Source attribution and license records are in `docs/LICENSES/`.
