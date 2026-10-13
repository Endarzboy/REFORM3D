"""Validation tests for the readout-sensitivity analysis.

Fast synthetic tests run unconditionally. Real-data reproduction tests are
skipped automatically when the frozen artifacts are not present, so the suite is
runnable in any environment while still gating reproduction where data exists.
"""

from __future__ import annotations

import numpy as np
import pytest

from reform3d.modules.anatomical_generalization.few_shot_transfer_analysis import (
    _predict_nearest_centroid,
)
from reform3d.modules.readout_sensitivity import anatomical, config, semantic, statistics
from reform3d.modules.readout_sensitivity import readouts as R
from reform3d.modules.acquisition_robustness.perturbation_robustness_analysis import (
    _build_matched_split_indices,
)

GATE = 1e-9


# --------------------------------------------------------------------------- #
# 1. Binary nearest-centroid on a synthetic separable dataset
# --------------------------------------------------------------------------- #
def test_binary_nearest_centroid_separable():
    rng = np.random.default_rng(0)
    pos = rng.normal(5.0, 0.1, size=(20, 4))
    neg = rng.normal(-5.0, 0.1, size=(20, 4))
    train = np.vstack([neg, pos])
    labels = np.array([0] * 20 + [1] * 20)
    test = np.vstack([rng.normal(5.0, 0.1, size=(5, 4)), rng.normal(-5.0, 0.1, size=(5, 4))])
    expected = np.array([1] * 5 + [0] * 5)
    predictions = _predict_nearest_centroid(train, labels, test)
    assert np.array_equal(predictions, expected)


# --------------------------------------------------------------------------- #
# 2. Multiclass nearest-centroid compatibility with the anatomical helper
# --------------------------------------------------------------------------- #
def test_multiclass_centroid_matches_helper():
    rng = np.random.default_rng(1)
    centers = rng.normal(0, 10, size=(3, 6))
    train = np.repeat(centers, 5, axis=0) + rng.normal(0, 0.05, size=(15, 6))
    labels = np.repeat([0, 1, 2], 5)
    query = centers + rng.normal(0, 0.05, size=(3, 6))
    query_labels = np.array([0, 1, 2])
    ba = R.multiclass_readout_balanced_accuracy(
        train, labels, query, query_labels,
        readout=R.NEAREST_CENTROID, logistic_c=1.0, random_state=42, max_iter=2000,
    )
    manual = _predict_nearest_centroid(train, labels, query)
    from sklearn.metrics import balanced_accuracy_score
    assert abs(ba - balanced_accuracy_score(query_labels, manual)) < 1e-12
    assert ba == 1.0


# --------------------------------------------------------------------------- #
# 3. Standardization is fitted only on training/support samples
# --------------------------------------------------------------------------- #
def test_standardization_fit_on_train_only():
    # If the scaler leaked test statistics, shifting only the test block would
    # change train predictions. Fit-on-train keeps train predictions invariant.
    rng = np.random.default_rng(2)
    train_features = rng.normal(0, 1, size=(40, 5))
    labels = np.tile(np.array([[1, 0], [0, 1]]), (20, 1))[:40]
    test_a = rng.normal(0, 1, size=(20, 5))
    test_b = test_a + 100.0  # large shift in the test block only
    s_a, _ = R.transfer_once(train_features, labels, test_a, labels[:20], readout=R.LOGISTIC,
                             logistic_c=1.0, random_state=42, max_iter=200)
    s_b, _ = R.transfer_once(train_features, labels, test_b, labels[:20], readout=R.LOGISTIC,
                             logistic_c=1.0, random_state=42, max_iter=200)
    # The scaler is refit per call on the same train block, so train->test mapping
    # uses train statistics; a pure additive test shift must not be corrected by
    # test-derived centering (that would require fitting on test).
    assert s_a != s_b or np.allclose(test_a, test_b)


# --------------------------------------------------------------------------- #
# 4. Identical semantic folds across all readouts
# --------------------------------------------------------------------------- #
def test_semantic_folds_deterministic_across_readouts():
    labels = np.random.default_rng(3).integers(0, 2, size=(60, 4))
    folds_a = _build_matched_split_indices(labels, n_splits=5, random_state=42)
    folds_b = _build_matched_split_indices(labels, n_splits=5, random_state=42)
    assert len(folds_a) == len(folds_b) == 5
    for (tr_a, te_a), (tr_b, te_b) in zip(folds_a, folds_b):
        assert np.array_equal(tr_a, tr_b) and np.array_equal(te_a, te_b)


# --------------------------------------------------------------------------- #
# 5. Deterministic case-disjoint support/query construction
# --------------------------------------------------------------------------- #
def test_support_query_construction_deterministic():
    from reform3d.modules.anatomical_generalization.few_shot_transfer_analysis import (
        _build_case_disjoint_support_and_query_sets,
        _group_by_organ_and_modality,
    )
    records = []
    for organ in ("a", "b"):
        for modality in ("ct", "mr"):
            for case in range(6):
                records.append({
                    "sample_id": f"{organ}_{modality}_{case}",
                    "primary_organ": organ, "modality": modality,
                    "source_case_id": f"{modality}_{case}",
                })
    grouped = _group_by_organ_and_modality(records, ("ct", "mr"))
    out1 = _build_case_disjoint_support_and_query_sets(grouped, ["a", "b"], ("ct", "mr"), 2, None, np.random.default_rng(42))
    out2 = _build_case_disjoint_support_and_query_sets(grouped, ["a", "b"], ("ct", "mr"), 2, None, np.random.default_rng(42))
    ids1 = [r["sample_id"] for r in out1[0]], [r["sample_id"] for r in out1[1]]
    ids2 = [r["sample_id"] for r in out2[0]], [r["sample_id"] for r in out2[1]]
    assert ids1 == ids2
    # Support and query cases are disjoint.
    assert not (set(r["source_case_id"] for r in out1[0]) & set(r["source_case_id"] for r in out1[1]))


# --------------------------------------------------------------------------- #
# 6/8/9/12. Semantic real-data reproduction, 6 directed transfers, ids, reconstruction
# --------------------------------------------------------------------------- #
def _semantic_available():
    return config.controlled_bundle_path("abdomenatlas", "med3dino_rel_96", "cls", 42).exists()


@pytest.mark.skipif(not _semantic_available(), reason="semantic bundle unavailable")
def test_semantic_c1_reproduces_production_and_invariants():
    ds, ckpt, feat, seed = "abdomenatlas", "med3dino_rel_96", "cls", 42
    cell = semantic.evaluate_semantic_cell(ds, ckpt, feat, seed)
    baseline = semantic.production_semantic_baseline(ds, ckpt, feat, seed)
    mine = cell["readouts"]["logistic_C1"]["six_direction_mean"]
    assert abs(mine - baseline) < GATE
    # Exactly six directed transfers per readout.
    for payload in cell["readouts"].values():
        assert payload["n_directed_transfers"] == 6
        assert len(payload["per_transfer"]) == 6
        # Reconstruction: six-direction mean == mean of stored per-transfer values.
        recon = float(np.mean([e["value"] for e in payload["per_transfer"].values()]))
        assert abs(recon - payload["six_direction_mean"]) < 1e-12
    # No duplicate sample ids; all finite scores.
    assert len(cell["sample_ids"]) == len(set(cell["sample_ids"]))
    for payload in cell["readouts"].values():
        assert np.isfinite(payload["six_direction_mean"])


@pytest.mark.skipif(
    not config.controlled_bundle_path("kits23", "med3dino_rel_96", "cls", 42).exists(),
    reason="KiTS23 semantic bundle unavailable",
)
def test_kits23_semantic_case_ids_align_across_layouts():
    bundle = semantic.load_controlled_bundle(
        config.controlled_bundle_path("kits23", "med3dino_rel_96", "cls", 42)
    )
    labels, positions, _ = semantic.align_semantic_labels("kits23", 42, bundle["case_ids"])
    assert labels.shape[0] == len(bundle["case_ids"])
    assert np.array_equal(positions, np.arange(len(bundle["case_ids"])))


# --------------------------------------------------------------------------- #
# 7. Anatomical real-data reproduction (logistic C=1 + centroid)
# --------------------------------------------------------------------------- #
def _anatomical_available():
    return config.anatomical_embeddings_path("mmwhs_ct_mr", "med3dino_rel_96", "cls", 42).exists()


@pytest.mark.skipif(not _anatomical_available(), reason="anatomical embeddings unavailable")
def test_anatomical_reproduces_production():
    ds, ckpt, feat, seed = "mmwhs_ct_mr", "med3dino_rel_96", "cls", 42
    cell = anatomical.evaluate_anatomical_cell(ds, ckpt, feat, seed)
    baseline = anatomical.production_anatomical_baseline(ds, ckpt, feat, seed)
    ok = [p for p in cell["per_organ"].values() if p["status"] == "ok"]
    macro_c1 = float(cell["balanced_accuracy_by_readout"]["logistic_C1"])
    macro_cen = float(cell["balanced_accuracy_by_readout"]["nearest_centroid"])
    assert abs(macro_c1 - baseline["logistic_C1"]) < GATE
    assert abs(macro_cen - baseline["nearest_centroid"]) < GATE
    assert set(cell["per_organ"]) == set(baseline["target_query_recall_by_organ"])
    for payload in ok:
        expected = baseline["target_query_recall_by_organ"][payload["target_organ"]]
        assert abs(payload["target_query_recall_by_readout"]["logistic_C1"] - expected["logistic_C1"]) < GATE
        assert abs(
            payload["target_query_recall_by_readout"]["nearest_centroid"] - expected["nearest_centroid"]
        ) < GATE
        for value in payload["target_query_recall_by_readout"].values():
            assert np.isfinite(value)
    assert len(cell["support_sample_ids"]) == len(set(cell["support_sample_ids"]))
    assert not (set(cell["support_sample_ids"]) & set(cell["query_sample_ids"]))
    assert len(
        {payload["target_query_recall_by_readout"]["logistic_C1"] for payload in ok}
    ) > 1
    for key, value in cell["balanced_accuracy_by_readout"].items():
        reconstructed = float(np.mean([p["target_query_recall_by_readout"][key] for p in ok]))
        assert abs(value - reconstructed) < GATE


# --------------------------------------------------------------------------- #
# 13. Bootstrap determinism (repeated execution -> identical outputs)
# --------------------------------------------------------------------------- #
def test_bootstrap_deterministic():
    diffs = {42: np.array([0.1, -0.2, 0.05, 0.0, 0.3, -0.1]),
             123: np.array([0.2, -0.1, 0.15, 0.1, 0.2, -0.05]),
             456: np.array([0.0, 0.1, -0.1, 0.2, 0.1, 0.0])}
    a = statistics.hierarchical_paired_bootstrap(diffs, resamples=5000, seed=2026)
    b = statistics.hierarchical_paired_bootstrap(diffs, resamples=5000, seed=2026)
    assert a["effect"] == b["effect"]
    assert a["ci_95"] == b["ci_95"] and a["ci_bonferroni"] == b["ci_bonferroni"]
    # Bonferroni interval is at least as wide as the unadjusted 95% interval.
    assert a["ci_bonferroni"][0] <= a["ci_95"][0] and a["ci_bonferroni"][1] >= a["ci_95"][1]

    matrix = np.array([[0.1, -0.1, 0.2], [0.0, 0.1, 0.05], [0.2, -0.05, 0.1]])
    c = statistics.crossed_paired_bootstrap(matrix, resamples=5000, seed=2026)
    d = statistics.crossed_paired_bootstrap(matrix, resamples=5000, seed=2026)
    assert c["ci_95"] == d["ci_95"]


# --------------------------------------------------------------------------- #
# 14. No CUDA initialisation
# --------------------------------------------------------------------------- #
def test_no_cuda_initialised():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-c", "import torch; from reform3d.modules.readout_sensitivity import readouts; assert not torch.cuda.is_initialized()"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
