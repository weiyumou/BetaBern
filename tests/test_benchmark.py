"""Tests for the static-ability measurement benchmark harness."""
import numpy as np
import pandas as pd

from betabern.benchmark import datasets, metrics, stats
from betabern.benchmark.runner import run_benchmark

DATA_PATH = "data/simulated_student_log.csv"


# ===========================================================================
# Metrics
# ===========================================================================

def test_prediction_metrics_perfect_and_chance():
    y = np.array([0, 1, 1, 0, 1, 0, 1, 0], dtype=float)
    near_perfect = np.where(y > 0.5, 0.999, 0.001)
    m = metrics.prediction_metrics(near_perfect, y)
    assert m["acc"] == 1.0 and m["auc"] == 1.0
    assert m["brier"] < 1e-3 and m["nll"] < 1e-2 and m["ece"] < 1e-2

    chance = np.full_like(y, 0.5)
    c = metrics.prediction_metrics(chance, y)
    assert abs(c["nll"] - np.log(2)) < 1e-6 and abs(c["brier"] - 0.25) < 1e-9


def test_theta_and_prob_recovery():
    t = np.array([0.1, 0.5, 0.9, -0.3, 1.2, -1.0])
    assert metrics.theta_recovery(t, t)["theta_spearman"] > 0.99       # identical -> rank 1
    assert metrics.theta_recovery(t, -t)["theta_spearman"] < -0.99     # reversed -> rank -1

    p = np.array([0.2, 0.4, 0.6, 0.8])
    pr = metrics.prob_recovery(p, p)
    assert pr["prob_mae"] < 1e-9 and pr["prob_pearson"] > 0.99


# ===========================================================================
# Dataset resolution
# ===========================================================================

def test_resolve_simulate_has_truth():
    df, has_truth = datasets.resolve({"simulate": {
        "num_students": 10, "num_items": 8, "n_skills": 1,
        "min_interactions": 5, "max_interactions": 8, "irt_model": "2PL", "random_state": 0}})
    assert has_truth and {"user_theta", "p_true_correct"}.issubset(df.columns)


def test_resolve_csv_without_full_truth():
    # The on-disk simulated CSV carries p_true_correct but not user_theta -> no recovery.
    _, has_truth = datasets.resolve({"path": DATA_PATH})
    assert has_truth is False


# ===========================================================================
# Paired statistical tests
# ===========================================================================

def _runs_df(nll_by_model: dict, auc_by_model: dict) -> pd.DataFrame:
    rows = []
    for model in nll_by_model:
        for fold, (nll, auc) in enumerate(zip(nll_by_model[model], auc_by_model[model])):
            rows.append({"dataset": "d", "model": model, "split": "within_random",
                         "fold": fold, "nll": nll, "auc": auc})
    # a recovery row per model (one fold) -> must be ignored by paired_tests
    for model in nll_by_model:
        rows.append({"dataset": "d", "model": model, "split": "recovery", "fold": 0,
                     "theta_spearman": 0.7})
    return pd.DataFrame(rows)


def test_paired_tests_directions_and_skips_recovery():
    runs = _runs_df(
        nll_by_model={"quad_logistic_irt": [0.50, 0.52, 0.48], "other": [0.55, 0.58, 0.50]},   # other worse NLL
        auc_by_model={"quad_logistic_irt": [0.70, 0.72, 0.68], "other": [0.60, 0.61, 0.59]},   # other worse AUC
    )
    paired = stats.paired_tests(runs, baseline="quad_logistic_irt")

    assert (paired["split"] != "recovery").all()                 # recovery (1 fold) excluded
    assert set(paired["model"]) == {"other"}                     # baseline not compared to itself

    nll = paired[(paired.model == "other") & (paired.metric == "nll")].iloc[0]
    assert nll["baseline"] == "quad_logistic_irt" and nll["n_folds"] == 3
    assert nll["delta_mean"] > 0 and nll["folds_better"] == 0    # higher (worse) NLL every fold

    auc = paired[(paired.model == "other") & (paired.metric == "auc")].iloc[0]
    assert auc["delta_mean"] < 0 and auc["folds_better"] == 0    # lower (worse) AUC every fold
    assert np.isfinite(auc["p_ttest"])


def test_paired_tests_empty_without_folds():
    # A single fold per model -> nothing to pair -> empty frame (and no crash).
    runs = _runs_df({"quad_logistic_irt": [0.5], "other": [0.6]}, {"quad_logistic_irt": [0.7], "other": [0.6]})
    assert stats.paired_tests(runs, baseline="quad_logistic_irt").empty


# ===========================================================================
# End-to-end sweep
# ===========================================================================

def test_run_benchmark_smoke():
    config = {
        "datasets": [{"name": "sim", "simulate": {
            "num_students": 40, "num_items": 12, "n_skills": 1,
            "min_interactions": 8, "max_interactions": 12, "irt_model": "2PL", "random_state": 0}}],
        "models": ["quad_logistic_irt", "quad_bern_free"],
        "splits": ["within_random"],
        "n_folds": 2,
        "hyperparams": {"degree": 4, "num_nodes": 12, "irt_model": "2PL", "item_weight_rank": 1},
        "trainer": {"max_epochs": 2, "patience": 2, "accelerator": "cpu", "batch_size": 32},
    }
    runs, summary, predictions = run_benchmark(config)

    pred = runs[runs["split"] == "within_random"]
    rec = runs[runs["split"] == "recovery"]
    assert len(pred) == 4 and len(rec) == 2                 # 2 models x 2 folds prediction; 1 recovery each
    assert set(pred["fold"]) == {0, 1}                      # both folds present
    assert np.isfinite(pred["nll"]).all() and np.isfinite(pred["auc"]).all()
    assert (pred["brier"].between(0, 1)).all()
    assert "theta_spearman" in rec.columns and "prob_mae" in rec.columns
    assert not summary.empty

    # Row-level predictions are the source of every statistic above.
    assert set(predictions.columns) >= {"dataset", "model", "split", "fold",
                                        "user_id", "skill_name", "item_id", "y", "p_correct", "ability"}
    assert ((predictions["p_correct"] >= 0) & (predictions["p_correct"] <= 1)).all()
    # Ids are decoded back to the dataset's original identifiers (here, the simulator's labels).
    assert predictions["user_id"].astype(str).str.startswith("user-").all()
    assert predictions["skill_name"].astype(str).str.startswith("skill-").all()
    # Recovery rows carry the joined ground truth; prediction rows do not.
    rec_rows = predictions[predictions["split"] == "recovery"]
    assert rec_rows["user_theta"].notna().all() and rec_rows["p_true_correct"].notna().all()
    # Held-out prediction metrics recomputed from the frame match the reported runs (single source).
    g = predictions[(predictions["model"] == "quad_logistic_irt") & (predictions["split"] == "within_random")
                    & (predictions["fold"] == 0)]
    expect = metrics.prediction_metrics(g["p_correct"].to_numpy(), g["y"].to_numpy())["nll"]
    got = pred[(pred["model"] == "quad_logistic_irt") & (pred["fold"] == 0)]["nll"].iloc[0]
    assert abs(expect - got) < 1e-9
