# Testing

Install the test dependencies and run the suite from the repository root:

```bash
python -m pip install -c requirements.txt '.[test]'
python -m pytest -q
```

The suite covers input geometry, checkpoint loading, feature completeness, cache reuse, seeded evaluation, and encoder adapter interfaces. Some tests require optional dependencies or evaluation artifacts.

See [Integrity checks](../docs/audit.md) for runtime requirements and [Reproducibility](../docs/reproduction.md) for dataset evaluation.
