# Encoders

The public YAML runner supports registered Med3DINO and 3DINO checkpoints with `cls`, `avg_pool`, or `multilayer` features. Registered names include `med3dino_{iso,rel,sa}_96` and `3dino`. Use `checkpoint_paths` to supply an explicit file while retaining its declared architecture and crop size.

```yaml
checkpoints: [med3dino_rel_96]
checkpoint_paths:
  med3dino_rel_96: /path/to/teacher.pth
```

A different architecture needs an explicit adapter; renaming its checkpoint does not make it compatible. Required backbone weights must load completely. Known unused pretraining heads are allowed.

For the bundled SAM-Med3D, MiM, and GL-MAE adapters, install `python -m pip install -c requirements.txt '.[external]'` in the repository environment. This installs TorchVision and timm; other adapters may require separate dependencies or environments.

Optional published external encoders retain their dedicated `reform3d.external_methods` entry points and native preprocessing/feature definitions. They are not silently routed through the Med3DINO preprocessing pipeline. Runtime imports alone do not establish successful inference; check validation coverage and upstream requirements.

`reform3d.encoders.factory.create_feature_extractor` provides lazy access to bundled/optional adapters. See its function signature for model names and paths. OpenMind requires its compatible upstream source and environment. Configure missing backends explicitly; no adapter searches the original research checkout.

Checkpoint identity, source hashes, licenses, and provenance coverage are recorded in `reproduction/`. Weight binaries are stored outside Git.
