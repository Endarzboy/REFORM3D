"""Orchestrator for the unified readout-sensitivity analysis.

CPU-only. Reuses completed manifests and frozen-feature artifacts; never runs
feature extraction, crop generation, encoder inference, or CUDA. The logistic
C=1 result of every cell is gated against a freshly recomputed current-production
baseline (an independent code path); any mismatch stops the run.
"""

from __future__ import annotations

# Guarantee no CUDA is ever initialised, before any torch-importing module loads.
import os as _os

_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from . import anatomical, config, semantic, statistics, tables
from .io_utils import read_json, write_csv, write_json, write_text

logger = logging.getLogger("readout_sensitivity")

# The independent production path uses n_jobs=4 while this outer-process sweep
# keeps each classifier single-threaded. Allow only solver-level variation below
# 0.01 percentage points while still rejecting substantive reconstruction drift.
GATE_TOLERANCE = 1e-4

SEMANTIC_MODULE = "semantic_transfer"
ANATOMICAL_MODULE = "low_shot_transfer"


# --------------------------------------------------------------------------- #
# Readout enumeration helpers
# --------------------------------------------------------------------------- #
def readout_keys(logistic_c_values: Sequence[float]) -> List[str]:
    return [f"logistic_C{c:g}" for c in logistic_c_values] + ["nearest_centroid"]


def parse_readout_key(key: str) -> Tuple[str, float | None]:
    if key == "nearest_centroid":
        return "nearest_centroid", None
    return "logistic", float(key.replace("logistic_C", ""))


# --------------------------------------------------------------------------- #
# Cell evaluation + reproduction gate
# --------------------------------------------------------------------------- #
def _semantic_cell_path(dataset: str, checkpoint: str, feature_type: str, seed: int) -> Path:
    return config.SEMANTIC_ROOT / dataset / checkpoint / feature_type / f"seed_{seed}.json"


def _anatomical_cell_path(dataset: str, checkpoint: str, feature_type: str, seed: int) -> Path:
    return config.ANATOMICAL_ROOT / dataset / checkpoint / feature_type / f"seed_{seed}.json"


def evaluate_semantic_one(task: Tuple[str, str, str, int, Tuple[float, ...]]) -> Dict[str, Any]:
    """Evaluate + gate + persist one semantic cell (module-level for pool workers)."""
    dataset, checkpoint, feature_type, seed, logistic_c_values = task
    cell = semantic.evaluate_semantic_cell(
        dataset, checkpoint, feature_type, seed, logistic_c_values=logistic_c_values
    )
    baseline = semantic.production_semantic_baseline(dataset, checkpoint, feature_type, seed)
    mine_c1 = cell["readouts"]["logistic_C1"]["six_direction_mean"]
    diff = abs(mine_c1 - baseline)
    if diff > GATE_TOLERANCE:
        raise RuntimeError(
            "Semantic C=1 reproduction mismatch for "
            f"{dataset}/{checkpoint}/{feature_type}/seed_{seed}: "
            f"readout_sensitivity={mine_c1:.12f} production_baseline={baseline:.12f} "
            f"|diff|={diff:.2e} > {GATE_TOLERANCE:.0e}"
        )
    _verify_semantic_reconstruction(cell)
    cell["reproduction"] = {
        "production_logistic_c1": float(baseline),
        "readout_sensitivity_logistic_c1": float(mine_c1),
        "abs_diff": float(diff),
        "tolerance": GATE_TOLERANCE,
        "baseline_source": "fresh_current_production_compute_matched_semantic_transfer",
    }
    write_json(_semantic_cell_path(dataset, checkpoint, feature_type, seed), cell)
    return cell


def evaluate_anatomical_one(task: Tuple[str, str, str, int, Tuple[float, ...]]) -> Dict[str, Any]:
    """Evaluate + gate + persist one anatomical cell (module-level for pool workers)."""
    dataset, checkpoint, feature_type, seed, logistic_c_values = task
    cell = anatomical.evaluate_anatomical_cell(
        dataset, checkpoint, feature_type, seed, logistic_c_values=logistic_c_values
    )
    baseline = anatomical.production_anatomical_baseline(dataset, checkpoint, feature_type, seed)
    macro_c1 = _anatomical_macro(cell, "logistic_C1")
    macro_centroid = _anatomical_macro(cell, "nearest_centroid")
    diff_c1 = abs(macro_c1 - baseline["logistic_C1"])
    diff_cen = abs(macro_centroid - baseline["nearest_centroid"])
    target_diffs = []
    for organ, expected in baseline["target_query_recall_by_organ"].items():
        observed = cell["per_organ"][organ]["target_query_recall_by_readout"]
        target_diffs.extend(
            [
                abs(float(observed["logistic_C1"]) - float(expected["logistic_C1"])),
                abs(float(observed["nearest_centroid"]) - float(expected["nearest_centroid"])),
            ]
        )
    max_target_diff = max(target_diffs, default=0.0)
    if diff_c1 > GATE_TOLERANCE or diff_cen > GATE_TOLERANCE or max_target_diff > GATE_TOLERANCE:
        raise RuntimeError(
            "Anatomical reproduction mismatch for "
            f"{dataset}/{checkpoint}/{feature_type}/seed_{seed}: "
            f"C1 |diff|={diff_c1:.2e}, centroid |diff|={diff_cen:.2e}, "
            f"target recall max |diff|={max_target_diff:.2e} (tol {GATE_TOLERANCE:.0e})"
        )
    cell["reproduction"] = {
        "production_logistic_c1": baseline["logistic_C1"],
        "readout_sensitivity_logistic_c1": float(macro_c1),
        "production_nearest_centroid": baseline["nearest_centroid"],
        "readout_sensitivity_nearest_centroid": float(macro_centroid),
        "abs_diff_logistic_c1": float(diff_c1),
        "abs_diff_nearest_centroid": float(diff_cen),
        "max_abs_diff_target_query_recall": float(max_target_diff),
        "tolerance": GATE_TOLERANCE,
        "baseline_source": "fresh_current_production_evaluate_low_shot_target_transfer",
    }
    write_json(_anatomical_cell_path(dataset, checkpoint, feature_type, seed), cell)
    return cell


def _cell_key(cell: Mapping[str, Any]) -> Tuple[str, str, str, int]:
    return (cell["dataset"], cell["checkpoint"], cell["feature_type"], int(cell["seed"]))


def _evaluate_cells(
    worker,
    datasets: Sequence[str],
    checkpoints: Sequence[str],
    feature_types: Sequence[str],
    seeds: Sequence[int],
    logistic_c_values: Sequence[float],
    workers: int,
    label: str,
    cell_path_fn,
    resume: bool,
) -> Dict[Tuple[str, str, str, int], Dict[str, Any]]:
    tasks = [
        (dataset, checkpoint, feature_type, seed, tuple(logistic_c_values))
        for dataset in datasets
        for checkpoint in checkpoints
        for feature_type in feature_types
        for seed in seeds
    ]
    cells: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
    pending = []
    expected_readouts = set(readout_keys(logistic_c_values))
    for task in tasks:
        path = cell_path_fn(task[0], task[1], task[2], task[3])
        cell = _load_completed_cell(path, task, label, expected_readouts) if resume else None
        if cell is None:
            pending.append(task)
        else:
            cells[_cell_key(cell)] = cell
    logger.info(
        "%s: %d/%d validated cells reused; %d pending",
        label,
        len(cells),
        len(tasks),
        len(pending),
    )
    if not pending:
        return cells
    if workers <= 1:
        for task in pending:
            logger.info("%s cell %s/%s/%s/seed_%s", label, task[0], task[1], task[2], task[3])
            cell = worker(task)
            cells[_cell_key(cell)] = cell
            logger.info("%s completed %s/%s/%s/seed_%s", label, task[0], task[1], task[2], task[3])
        return cells
    workers = min(int(workers), len(pending))
    logger.info("%s: evaluating %d pending cells with %d workers", label, len(pending), workers)
    # Use 'spawn' to avoid fork+threads deadlocks (numexpr/joblib/BLAS threads are
    # already live in the parent). Spawned workers re-import config, so parallel
    # execution targets the real result root; monkeypatched output roots (tests)
    # must use --workers 1.
    import multiprocessing as _mp

    with ProcessPoolExecutor(max_workers=workers, mp_context=_mp.get_context("spawn")) as executor:
        futures = {executor.submit(worker, task): task for task in pending}
        for future in as_completed(futures):
            task = futures[future]
            try:
                cell = future.result()
            except BaseException:
                logger.exception(
                    "%s failed %s/%s/%s/seed_%s",
                    label,
                    task[0],
                    task[1],
                    task[2],
                    task[3],
                )
                for outstanding in futures:
                    outstanding.cancel()
                raise
            cells[_cell_key(cell)] = cell
            logger.info("%s completed %s/%s/%s/seed_%s", label, task[0], task[1], task[2], task[3])
    return cells


def _load_completed_cell(
    path: Path,
    task: Tuple[str, str, str, int, Tuple[float, ...]],
    label: str,
    expected_readouts: set[str],
) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    dataset, checkpoint, feature_type, seed, _ = task
    try:
        cell = read_json(path)
        if (
            cell.get("dataset") != dataset
            or cell.get("checkpoint") != checkpoint
            or cell.get("feature_type") != feature_type
            or int(cell.get("seed")) != int(seed)
        ):
            return None
        reproduction = cell["reproduction"]
        if label == "semantic":
            if not expected_readouts.issubset(cell["readouts"]):
                return None
            if float(reproduction["abs_diff"]) > float(reproduction["tolerance"]):
                return None
            _verify_semantic_reconstruction(cell)
        else:
            if int(cell.get("schema_version", 0)) < 2:
                return None
            if not expected_readouts.issubset(cell["balanced_accuracy_by_readout"]):
                return None
            if float(reproduction["abs_diff_logistic_c1"]) > float(reproduction["tolerance"]):
                return None
            if float(reproduction["abs_diff_nearest_centroid"]) > float(reproduction["tolerance"]):
                return None
            if float(reproduction["max_abs_diff_target_query_recall"]) > float(reproduction["tolerance"]):
                return None
            for payload in cell["per_organ"].values():
                if payload.get("status") == "ok" and not expected_readouts.issubset(
                    payload["target_query_recall_by_readout"]
                ):
                    return None
        return cell
    except (KeyError, TypeError, ValueError, OSError):
        logger.warning("Ignoring incomplete or stale %s cell: %s", label, path)
        return None


def _verify_semantic_reconstruction(cell: Mapping[str, Any]) -> None:
    """Confirm each readout's six-direction mean == mean of its stored per-transfer values."""
    for readout_key, payload in cell["readouts"].items():
        values = [entry["value"] for entry in payload["per_transfer"].values()]
        recon = float(np.mean(values)) if values else 0.0
        if abs(recon - payload["six_direction_mean"]) > 1e-12:
            raise RuntimeError(
                f"Semantic reconstruction failed for {readout_key}: "
                f"stored={payload['six_direction_mean']:.12f} reconstructed={recon:.12f}"
            )


def _anatomical_macro(cell: Mapping[str, Any], readout_key: str) -> float:
    return float(cell["balanced_accuracy_by_readout"][readout_key])


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _semantic_headline(cell: Mapping[str, Any], readout_key: str) -> float:
    return float(cell["readouts"][readout_key]["six_direction_mean"])


def _semantic_transfer_values(cell: Mapping[str, Any], readout_key: str) -> Dict[str, float]:
    return {
        key: float(entry["value"])
        for key, entry in cell["readouts"][readout_key]["per_transfer"].items()
    }


def aggregate_module(
    cells: Mapping[Tuple[str, str, str, int], Mapping[str, Any]],
    datasets: Sequence[str],
    checkpoints: Sequence[str],
    feature_types: Sequence[str],
    seeds: Sequence[int],
    readout_key_list: Sequence[str],
    headline_fn,
) -> Dict[str, Any]:
    """Build per-(dataset, feature, readout) checkpoint summaries and orderings."""
    entries: List[Dict[str, Any]] = []
    for dataset in datasets:
        for feature_type in feature_types:
            for readout_key in readout_key_list:
                readout, logistic_c = parse_readout_key(readout_key)
                per_checkpoint: Dict[str, Any] = {}
                mean_by_ckpt: Dict[str, float] = {}
                for checkpoint in checkpoints:
                    seed_values = []
                    for seed in seeds:
                        cell = cells[(dataset, checkpoint, feature_type, seed)]
                        seed_values.append(headline_fn(cell, readout_key))
                    summary = statistics.mean_std(seed_values)
                    per_checkpoint[checkpoint] = summary
                    mean_by_ckpt[checkpoint] = summary["mean"]
                ordering = statistics.checkpoint_ordering(mean_by_ckpt)
                entries.append(
                    {
                        "dataset": dataset,
                        "feature_type": feature_type,
                        "readout": readout,
                        "logistic_c": logistic_c,
                        "readout_key": readout_key,
                        "per_checkpoint": per_checkpoint,
                        "ordering": ordering,
                    }
                )
    return {"entries": entries}


# --------------------------------------------------------------------------- #
# Paired statistics
# --------------------------------------------------------------------------- #
def _semantic_paired_rows(
    cells, datasets, feature_types, seeds, readout_key_list, resamples, bootstrap_seed
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        for feature_type in feature_types:
            for readout_key in readout_key_list:
                readout, logistic_c = parse_readout_key(readout_key)
                for name, high_ckpt, low_ckpt in statistics.paired_comparison_pairs():
                    if any((dataset, ckpt, feature_type, seed) not in cells
                           for ckpt in (high_ckpt, low_ckpt) for seed in seeds):
                        continue
                    diffs_by_seed: Dict[int, np.ndarray] = {}
                    for seed in seeds:
                        high = _semantic_transfer_values(cells[(dataset, high_ckpt, feature_type, seed)], readout_key)
                        low = _semantic_transfer_values(cells[(dataset, low_ckpt, feature_type, seed)], readout_key)
                        keys = sorted(set(high) & set(low))
                        diffs_by_seed[seed] = np.asarray([high[k] - low[k] for k in keys], dtype=np.float64)
                    summary = statistics.hierarchical_paired_bootstrap(
                        diffs_by_seed, resamples=resamples, seed=bootstrap_seed
                    )
                    rows.append(_paired_row(SEMANTIC_MODULE, dataset, feature_type, readout, logistic_c, name, summary))
    return rows


def _anatomical_paired_rows(
    cells, datasets, feature_types, seeds, readout_key_list, resamples, bootstrap_seed
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        for feature_type in feature_types:
            for readout_key in readout_key_list:
                readout, logistic_c = parse_readout_key(readout_key)
                for name, high_ckpt, low_ckpt in statistics.paired_comparison_pairs():
                    if any((dataset, ckpt, feature_type, seed) not in cells
                           for ckpt in (high_ckpt, low_ckpt) for seed in seeds):
                        continue
                    # Complete-case organs: ok for both checkpoints across all seeds.
                    organ_sets = []
                    for seed in seeds:
                        high_cell = cells[(dataset, high_ckpt, feature_type, seed)]
                        low_cell = cells[(dataset, low_ckpt, feature_type, seed)]
                        ok = {
                            organ
                            for organ in high_cell["per_organ"]
                            if high_cell["per_organ"][organ].get("status") == "ok"
                            and low_cell["per_organ"].get(organ, {}).get("status") == "ok"
                        }
                        organ_sets.append(ok)
                    common = sorted(set.intersection(*organ_sets)) if organ_sets else []
                    matrix = np.asarray(
                        [
                            [
                                cells[(dataset, high_ckpt, feature_type, seed)]["per_organ"][organ][
                                    "target_query_recall_by_readout"
                                ][readout_key]
                                - cells[(dataset, low_ckpt, feature_type, seed)]["per_organ"][organ][
                                    "target_query_recall_by_readout"
                                ][readout_key]
                                for organ in common
                            ]
                            for seed in seeds
                        ],
                        dtype=np.float64,
                    )
                    summary = statistics.crossed_paired_bootstrap(
                        matrix, resamples=resamples, seed=bootstrap_seed
                    )
                    summary["n_common_organs"] = len(common)
                    rows.append(_paired_row(ANATOMICAL_MODULE, dataset, feature_type, readout, logistic_c, name, summary))
    return rows


def _paired_row(module, dataset, feature_type, readout, logistic_c, comparison, summary) -> Dict[str, Any]:
    return {
        "module": module,
        "dataset": dataset,
        "feature_type": feature_type,
        "readout": readout,
        "logistic_c": logistic_c,
        "comparison": comparison,
        "effect": summary["effect"],
        "ci_95": summary["ci_95"],
        "ci_bonferroni": summary["ci_bonferroni"],
        "excludes_zero_95": summary["excludes_zero_95"],
        "excludes_zero_bonferroni": summary["excludes_zero_bonferroni"],
        "direction": summary["direction"],
        "design": summary["design"],
        "n_bootstrap": summary["n_bootstrap"],
        "bonferroni_family_size": summary["bonferroni_family_size"],
        "n_common_organs": summary.get("n_common_organs"),
    }


# --------------------------------------------------------------------------- #
# Agreement
# --------------------------------------------------------------------------- #
def _agreement_for_module(module_summary, paired_rows, module, datasets, feature_types) -> Dict[str, Any]:
    per_surface = []
    n_match = 0
    logistic_orderings = set()
    centroid_orderings = set()
    for dataset in datasets:
        for feature_type in feature_types:
            ordering_by_readout = {}
            for entry in module_summary["entries"]:
                if entry["dataset"] == dataset and entry["feature_type"] == feature_type:
                    ordering_by_readout[entry["readout_key"]] = entry["ordering"]["ranking"]
            c1 = ordering_by_readout.get("logistic_C1", [])
            centroid = ordering_by_readout.get("nearest_centroid", [])
            match = c1 == centroid
            n_match += int(match)
            logistic_orderings.add(">".join(c1))
            centroid_orderings.add(">".join(centroid))

            sa_rel_effects = {}
            for row in paired_rows:
                if (
                    row["module"] == module
                    and row["dataset"] == dataset
                    and row["feature_type"] == feature_type
                    and row["comparison"] == "SA-REL"
                ):
                    label = "nearest_centroid" if row["readout"] == "nearest_centroid" else f"logistic_C{row['logistic_c']:g}"
                    sa_rel_effects[label] = row["effect"]
            direction = statistics.direction_agreement(sa_rel_effects)
            per_surface.append(
                {
                    "dataset": dataset,
                    "feature_type": feature_type,
                    "logistic_c1_ordering": ">".join(c1),
                    "nearest_centroid_ordering": ">".join(centroid),
                    "ordering_matches": match,
                    "sa_rel_direction_agreement": direction,
                }
            )
    n_surfaces = len(datasets) * len(feature_types)
    return {
        "ordering_agreement": {
            "n_surfaces": n_surfaces,
            "n_matching_surfaces": n_match,
            "logistic_c1_orderings": ",".join(sorted(logistic_orderings)),
            "nearest_centroid_orderings": ",".join(sorted(centroid_orderings)),
        },
        "sa_rel_direction_agreement_all_surfaces": all(
            item["sa_rel_direction_agreement"]["all_agree"] for item in per_surface
        ),
        "per_surface": per_surface,
    }


# --------------------------------------------------------------------------- #
# CSV rendering
# --------------------------------------------------------------------------- #
def _module_csv_rows(module_summary, module, seeds) -> List[Dict[str, Any]]:
    rows = []
    for entry in module_summary["entries"]:
        for checkpoint, summary in entry["per_checkpoint"].items():
            row = {
                "module": module,
                "dataset": entry["dataset"],
                "feature_type": entry["feature_type"],
                "readout": entry["readout"],
                "logistic_c": entry["logistic_c"],
                "checkpoint": checkpoint,
                "checkpoint_label": config.CHECKPOINT_LABELS[checkpoint],
                "headline_mean": summary["mean"],
                "headline_std": summary["std"],
                "ordering_string": entry["ordering"]["ordering_string"],
            }
            for seed, value in zip(seeds, summary["per_seed"]):
                row[f"seed_{seed}"] = value
            rows.append(row)
    return rows


def _paired_csv_rows(paired_rows) -> List[Dict[str, Any]]:
    rows = []
    for row in paired_rows:
        rows.append(
            {
                "module": row["module"],
                "dataset": row["dataset"],
                "feature_type": row["feature_type"],
                "readout": row["readout"],
                "logistic_c": row["logistic_c"],
                "comparison": row["comparison"],
                "effect": row["effect"],
                "ci95_low": row["ci_95"][0],
                "ci95_high": row["ci_95"][1],
                "ci_bonferroni_low": row["ci_bonferroni"][0],
                "ci_bonferroni_high": row["ci_bonferroni"][1],
                "excludes_zero_95": row["excludes_zero_95"],
                "excludes_zero_bonferroni": row["excludes_zero_bonferroni"],
                "direction": row["direction"],
                "design": row["design"],
                "n_bootstrap": row["n_bootstrap"],
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> None:
    if 1.0 not in tuple(float(c) for c in args.logistic_c_values):
        raise SystemExit("--logistic-c-values must include 1.0 for the reproduction gate")
    if set(args.readouts) != set(config.READOUTS):
        raise SystemExit(
            "This analysis requires both readouts: logistic and nearest_centroid"
        )
    if int(args.workers) < 1:
        raise SystemExit("--workers must be at least 1")
    config.LOG_ROOT.mkdir(parents=True, exist_ok=True)

    logistic_c_values = tuple(float(c) for c in args.logistic_c_values)
    keys = readout_keys(logistic_c_values)
    seeds = tuple(int(s) for s in args.seeds)
    checkpoints = tuple(args.checkpoints)
    feature_types = tuple(args.feature_types)
    started = time.time()

    semantic_summary = anatomical_summary = None
    semantic_paired: List[Dict[str, Any]] = []
    anatomical_paired: List[Dict[str, Any]] = []
    agreement: Dict[str, Any] = {"ordering_agreement": {}, "sa_rel_direction_agreement": {}}

    if args.semantic_datasets:
        logger.info("Evaluating semantic cells ...")
        semantic_cells = _evaluate_cells(
            evaluate_semantic_one, args.semantic_datasets, checkpoints, feature_types, seeds,
            logistic_c_values, args.workers, "semantic", _semantic_cell_path, args.resume,
        )
        semantic_summary = aggregate_module(
            semantic_cells, args.semantic_datasets, checkpoints, feature_types, seeds, keys, _semantic_headline
        )
        semantic_paired = _semantic_paired_rows(
            semantic_cells, args.semantic_datasets, feature_types, seeds, keys,
            args.bootstrap_resamples, args.bootstrap_seed,
        )
        write_json(config.SUMMARY_ROOT / "semantic_readout_sensitivity.json", semantic_summary)
        write_csv(
            config.SUMMARY_ROOT / "semantic_readout_sensitivity.csv",
            _module_csv_rows(semantic_summary, SEMANTIC_MODULE, seeds),
            fieldnames=["module", "dataset", "feature_type", "readout", "logistic_c", "checkpoint",
                        "checkpoint_label", "headline_mean", "headline_std", "ordering_string",
                        *[f"seed_{s}" for s in seeds]],
        )

    if args.anatomical_datasets:
        logger.info("Evaluating anatomical cells ...")
        anatomical_cells = _evaluate_cells(
            evaluate_anatomical_one, args.anatomical_datasets, checkpoints, feature_types, seeds,
            logistic_c_values, args.workers, "anatomical", _anatomical_cell_path, args.resume,
        )
        anatomical_summary = aggregate_module(
            anatomical_cells, args.anatomical_datasets, checkpoints, feature_types, seeds, keys,
            lambda cell, key: _anatomical_macro(cell, key),
        )
        anatomical_paired = _anatomical_paired_rows(
            anatomical_cells, args.anatomical_datasets, feature_types, seeds, keys,
            args.bootstrap_resamples, args.bootstrap_seed,
        )
        write_json(config.SUMMARY_ROOT / "anatomical_readout_sensitivity.json", anatomical_summary)
        write_csv(
            config.SUMMARY_ROOT / "anatomical_readout_sensitivity.csv",
            _module_csv_rows(anatomical_summary, ANATOMICAL_MODULE, seeds),
            fieldnames=["module", "dataset", "feature_type", "readout", "logistic_c", "checkpoint",
                        "checkpoint_label", "headline_mean", "headline_std", "ordering_string",
                        *[f"seed_{s}" for s in seeds]],
        )

    paired_rows = semantic_paired + anatomical_paired
    paired = {"rows": paired_rows}
    write_json(config.SUMMARY_ROOT / "paired_readout_statistics.json", paired)
    write_csv(
        config.SUMMARY_ROOT / "paired_readout_statistics.csv",
        _paired_csv_rows(paired_rows),
        fieldnames=["module", "dataset", "feature_type", "readout", "logistic_c", "comparison",
                    "effect", "ci95_low", "ci95_high", "ci_bonferroni_low", "ci_bonferroni_high",
                    "excludes_zero_95", "excludes_zero_bonferroni", "direction", "design", "n_bootstrap"],
    )

    if semantic_summary is not None:
        agreement["ordering_agreement"][SEMANTIC_MODULE] = _agreement_for_module(
            semantic_summary, paired_rows, SEMANTIC_MODULE, args.semantic_datasets, feature_types
        )["ordering_agreement"]
        agreement.setdefault("detail", {})[SEMANTIC_MODULE] = _agreement_for_module(
            semantic_summary, paired_rows, SEMANTIC_MODULE, args.semantic_datasets, feature_types
        )
    if anatomical_summary is not None:
        agreement["ordering_agreement"][ANATOMICAL_MODULE] = _agreement_for_module(
            anatomical_summary, paired_rows, ANATOMICAL_MODULE, args.anatomical_datasets, feature_types
        )["ordering_agreement"]
        agreement.setdefault("detail", {})[ANATOMICAL_MODULE] = _agreement_for_module(
            anatomical_summary, paired_rows, ANATOMICAL_MODULE, args.anatomical_datasets, feature_types
        )
    write_json(config.SUMMARY_ROOT / "readout_agreement.json", agreement)

    if semantic_summary is not None and anatomical_summary is not None:
        markdown = tables.render_markdown(semantic_summary, anatomical_summary, paired, agreement)
        write_text(config.SUMMARY_ROOT / "rebuttal_compact_tables.md", markdown)

    elapsed = time.time() - started
    logger.info("Readout-sensitivity analysis complete in %.1fs", elapsed)
    logger.info("Summary artifacts under %s", config.SUMMARY_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified readout-sensitivity analysis (CPU-only)")
    parser.add_argument("--semantic-datasets", nargs="*", default=list(config.SEMANTIC_DATASETS))
    parser.add_argument("--anatomical-datasets", nargs="*", default=list(config.ANATOMICAL_DATASETS))
    parser.add_argument("--checkpoints", nargs="+", default=list(config.CHECKPOINTS))
    parser.add_argument("--feature-types", nargs="+", default=list(config.FEATURE_TYPES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(config.SEEDS))
    parser.add_argument("--readouts", nargs="+", default=list(config.READOUTS))
    parser.add_argument("--logistic-c-values", nargs="+", type=float, default=list(config.LOGISTIC_C_VALUES))
    parser.add_argument("--bootstrap-resamples", type=int, default=config.DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=config.DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes for cell evaluation.")
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Recompute cells even when a validated artifact already exists.",
    )
    parser.set_defaults(resume=True)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()
