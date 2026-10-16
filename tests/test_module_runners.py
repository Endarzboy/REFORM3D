"""Dedicated runners retain supported stages and preserve source results."""
from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from reform3d.modules.anatomical_generalization import run_anatomical_generalization as anatomical
from reform3d.modules.cross_modal_alignment import aggregate_checkpoint_results as aggregate
from reform3d.modules.cross_modal_alignment import run_cross_modal_alignment as cross_modal


def _capture_commands(monkeypatch, module):
    commands = []

    def capture(command, dry_run):
        assert dry_run
        commands.append(command)

    monkeypatch.setattr(module, "_run", capture)
    return commands


@pytest.mark.parametrize("dataset", ["mmwhs_ct_mr", "chaos_ct_mr"])
def test_cross_modal_dry_run_keeps_supported_stages(monkeypatch, dataset):
    commands = _capture_commands(monkeypatch, cross_modal)
    cross_modal.main([
        "--dataset", dataset, "--feature-types", "cls", "avg_pool", "--dry-run",
    ])
    modules = [command[3] for command in commands]
    prefix = "reform3d.modules.cross_modal_alignment."
    expected = {
        prefix + "build_manifests",
        prefix + "feature_extractor",
        prefix + "evaluation_pipeline",
        prefix + "aggregate_checkpoint_results",
    }
    if dataset == "chaos_ct_mr":
        expected.update({prefix + "single_organ_alignment", prefix + "single_organ_retrieval_uncertainty"})
    else:
        expected.add(prefix + "retrieval_uncertainty_analysis")
    assert set(modules) == expected
    assert all(importlib.util.find_spec(name) is not None for name in modules)
    assert modules.count(prefix + "feature_extractor") == 2
    assert modules.count(prefix + "evaluation_pipeline") == 2


def test_anatomical_preparation_uses_valid_cross_modal_options(monkeypatch):
    commands = _capture_commands(monkeypatch, anatomical)
    anatomical.main([
        "--dataset", "mmwhs_ct_mr", "--prepare-cross-modal-features", "--dry-run",
    ])
    preparation, *remaining = commands
    assert preparation[3] == "reform3d.modules.cross_modal_alignment.run_cross_modal_alignment"
    assert {command[3].rsplit(".", 1)[1] for command in remaining} == {
        "build_manifest", "evaluation_pipeline", "statistical_rigor_summary",
    }
    child_commands = _capture_commands(monkeypatch, cross_modal)
    cross_modal.main([*preparation[4:], "--dry-run"])
    assert {command[3].rsplit(".", 1)[1] for command in child_commands} == {
        "build_manifests", "feature_extractor", "evaluation_pipeline",
    }


@pytest.mark.parametrize("feature_type", ["cls", "avg_pool"])
def test_aggregation_ranks_checkpoints_without_moving_source_results(tmp_path, monkeypatch, feature_type):
    root_metrics = tmp_path / "primary_metrics.json"
    root_metrics.write_text('{"existing_root_result": true}\n')
    source_bytes = {root_metrics: root_metrics.read_bytes()}
    for checkpoint, score in [("med3dino_rel_96", 0.4), ("med3dino_iso_96", 0.8)]:
        directory = tmp_path / checkpoint
        if feature_type != "cls":
            directory /= feature_type
        directory.mkdir(parents=True)
        metrics = directory / "primary_metrics.json"
        metrics.write_text(json.dumps({
            "checkpoint_name": checkpoint,
            "balanced_cross_modal_retrieval": {
                "status": "ok", "bidirectional_mean": {"map": score, "top@1": score},
            },
        }))
        source_bytes[metrics] = metrics.read_bytes()

    monkeypatch.setattr(sys, "argv", [
        "aggregate", "--results-dir", str(tmp_path), "--feature-type", feature_type,
    ])
    aggregate.main()

    summary_path = aggregate.get_checkpoint_comparison_json_path(tmp_path, feature_type)
    summary = json.loads(summary_path.read_text())
    assert [row["checkpoint_name"] for row in summary["rows"]] == ["med3dino_iso_96", "med3dino_rel_96"]
    assert [row["retrieval_map"] for row in summary["rows"]] == [0.8, 0.4]
    assert summary["best_by_retrieval_map"]["checkpoint_name"] == "med3dino_iso_96"
    assert all(path.read_bytes() == original for path, original in source_bytes.items())
    assert not (tmp_path / "primary_metrics.previous_root.json").exists()


def test_empty_aggregation_preserves_existing_root_result(tmp_path, monkeypatch):
    metrics = tmp_path / "primary_metrics.json"
    metrics.write_bytes(b'{"existing_root_result": true}\n')
    original = metrics.read_bytes()
    monkeypatch.setattr(sys, "argv", ["aggregate", "--results-dir", str(tmp_path)])

    with pytest.raises(FileNotFoundError, match="No per-checkpoint"):
        aggregate.main()

    assert metrics.read_bytes() == original
    assert list(tmp_path.iterdir()) == [metrics]
