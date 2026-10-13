"""Seed aggregation rejects incomplete or nonfinite metric records."""

import pytest

from reform3d.core import benchmark_seed_tables as seeded


def test_seed_summary_rejects_duplicate_metric_rows():
    row = {
        "status": "seeded_report", "checkpoint": "med3dino_rel_96",
        "feature_type": "cls", "metric_family": "controlled_drift",
        "metric_name": "compact_non_reference_mean_drift",
    }
    with pytest.raises(RuntimeError, match="Duplicate seeded metric"):
        seeded._summary_index([row, row.copy()])


@pytest.mark.parametrize("failure", ["nonfinite", "extra_seed", "missing_source"])
def test_seed_summary_rejects_invalid_metric_contract(tmp_path, monkeypatch, failure):
    checkpoint = seeded.CHECKPOINTS[0]
    row = {
        "status": "seeded_report", "checkpoint": checkpoint,
        "feature_type": "cls", "metric_family": "controlled_drift",
        "metric_name": "compact_non_reference_mean_drift", "n": 3,
        "missing_seeds": [], "mean": 0.2, "std": 0.0,
        "ci95_low": 0.2, "ci95_high": 0.2,
        "values_by_seed": {str(seed): 0.2 for seed in seeded.SEEDS},
        "source_files_by_seed": {str(seed): "metric.json" for seed in seeded.SEEDS},
    }
    if failure == "nonfinite":
        row["mean"] = float("nan")
    elif failure == "extra_seed":
        row["values_by_seed"]["789"] = 0.2
    else:
        row["source_files_by_seed"].pop("42")
    monkeypatch.setattr(seeded, "CHECKPOINTS", (checkpoint,))
    monkeypatch.setattr(seeded, "FEATURE_TYPES", ("cls",))
    monkeypatch.setattr(seeded, "collect_repeated_seed_summary", lambda **kwargs: {"rows": [row]})
    spec = dict(seeded.TABLE_SPECS["table3"])
    spec.update(datasets=("kits23",), metrics=(spec["metrics"][0],), expected_rows=1)
    with pytest.raises(RuntimeError):
        seeded._compact_rows(tmp_path, "table3", spec)

