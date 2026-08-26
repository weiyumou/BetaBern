"""Sequential measurement-efficiency curve + posterior-coverage read-outs (benchmark/measurement.py)."""
import math

import torch

from betabern.benchmark.measurement import coverage_stats, measurement_curve
from betabern.benchmark.simfit import (
    complete_response_matrix,
    fit_mml,
    padded_batch_from_log,
    set_oracle_items,
)
from betabern.bernstein.estimator.quad import build_quad_bernstein_estimator_irt
from betabern.core.data.simulate_irt import generate_log_data
from betabern.irt.estimator.quad_irt import build_quad_irt_estimator


def test_measurement_curve_shape_and_contraction():
    """The curve has one row per prefix length, starts at the prior SD, and uncertainty contracts as
    items arrive; the one-step predictive columns are finite (and NaN only at the final, no-target step)."""
    skill, item, ans, _theta = complete_response_matrix(num_students=12, num_items=10, seed=1)
    model = build_quad_irt_estimator(num_users=12, num_skills=1, num_items=10, num_nodes=40, irt_model="2PL")

    curve = measurement_curve(model, skill, item, ans, n_orders=6, seed=0)

    assert list(curve["k_items"]) == list(range(11))             # k = 0..S
    assert math.isclose(curve["post_sd"].iloc[0], 1.0, abs_tol=0.05)  # k=0 is the N(0,1) prior
    assert curve["post_sd"].iloc[-1] < curve["post_sd"].iloc[0]  # uncertainty shrinks with data
    assert (curve["post_sd"].to_numpy() > 0).all()
    assert math.isnan(curve["preq_nll"].iloc[-1]) and curve["preq_nll"].iloc[:-1].notna().all()
    assert curve["preq_acc"].iloc[:-1].between(0.0, 1.0).all()


def test_measurement_curve_bridged_on_eta_scale():
    """A [0,1] Bernstein-bridged model's posterior SD is reported on the ℝ (η) scale, so its prior SD is
    ≈ 1 (Beta(4,4) std 0.167 × 6) — comparable to a Normal model's, not the raw [0,1] value."""
    skill, item, ans, _theta = complete_response_matrix(num_students=12, num_items=10, seed=4)
    model = build_quad_bernstein_estimator_irt(degree_n=40, num_users=12, num_skills=1, num_items=10,
                                               num_nodes=40, irt_model="2PL", prior=(4.0, 4.0))
    curve = measurement_curve(model, skill, item, ans, n_orders=4, seed=0)
    assert math.isclose(curve["post_sd"].iloc[0], 1.0, abs_tol=0.05)  # η-scaled prior SD, not 0.167
    assert curve["post_sd"].iloc[-1] < curve["post_sd"].iloc[0]


def test_coverage_stats_keys_and_nesting():
    """coverage_stats returns recovery + per-level coverage; nested intervals give non-decreasing coverage
    (50% ⊆ 80% ⊆ 95%), independent of fit quality, and all values are valid fractions / correlations."""
    skill, item, ans, _theta = complete_response_matrix(num_students=40, num_items=12, seed=2)
    mask = torch.ones_like(item, dtype=torch.bool)
    true_theta = torch.randn(40)
    model = build_quad_irt_estimator(num_users=40, num_skills=1, num_items=12, num_nodes=40, irt_model="2PL")

    out = coverage_stats(model, skill, item, ans, mask, true_theta, levels=(0.5, 0.8, 0.95))

    assert set(out) == {"theta_corr", "theta_rmse", "cov_50", "cov_80", "cov_95"}
    assert -1.0 <= out["theta_corr"] <= 1.0 and out["theta_rmse"] >= 0.0
    assert all(0.0 <= out[k] <= 1.0 for k in ("cov_50", "cov_80", "cov_95"))
    assert out["cov_50"] <= out["cov_80"] <= out["cov_95"]


def test_coverage_is_calibrated_on_wellspecified_sim():
    """End-to-end: fit a 2PL logistic estimator by MML on 2PL data drawn with θ ~ N(0, 1) (its own prior).
    The closed-form posterior should then recover θ and the 95% credible interval should have ~nominal
    coverage — the headline calibration claim, on a well-specified model."""
    log_df, _items = generate_log_data(num_students=120, num_items=30, min_interactions=28, max_interactions=30,
                                        n_skills=1, drift_scale=0.0, irt_model="2PL", random_state=0)
    skill, item, ans, mask, true_theta = padded_batch_from_log(log_df)
    model = build_quad_irt_estimator(num_users=int(skill.size(0)), num_skills=1,
                                     num_items=int(item.max()) + 1, num_nodes=60, irt_model="2PL")
    fit_mml(model, skill, item, ans, mask, epochs=250, lr=0.05)

    out = coverage_stats(model, skill, item, ans, mask, true_theta)

    assert out["theta_corr"] > 0.7                 # θ is recovered
    assert 0.86 <= out["cov_95"] <= 1.0            # 95% interval ≈ nominal (loose for finite sample)
    assert out["cov_50"] <= out["cov_80"] <= out["cov_95"]


def test_oracle_items_inject_truth_and_restore_central_coverage():
    """set_oracle_items loads the true a/b into the IRTBase (no fitting); with items known exactly the
    closed-form posterior is calibrated at *every* level — including the central 50%, which item-estimation
    error makes under-cover in the fitted case. Isolates θ-posterior calibration from item error."""
    log_df, item_df = generate_log_data(num_students=300, num_items=40, min_interactions=40, max_interactions=40,
                                        n_skills=1, drift_scale=0.0, irt_model="2PL", random_state=1)
    skill, item, ans, mask, true_theta = padded_batch_from_log(log_df)
    a_true, b_true = (torch.tensor(item_df[c].to_numpy()) for c in ("a_irt", "b_irt"))
    n_items = len(item_df)
    model = build_quad_irt_estimator(num_users=int(skill.size(0)), num_skills=1, num_items=n_items,
                                     num_nodes=60, irt_model="2PL", item_weight_rank=-1)
    set_oracle_items(model, a_true, b_true)

    ids = torch.arange(n_items).view(-1, 1)
    p = model.irf.irt_base.get_irt_params(ids, torch.zeros_like(ids))  # injected params equal the truth
    assert torch.allclose(p.a.flatten(), a_true.float(), atol=1e-4)
    assert torch.allclose(p.b.flatten(), b_true.float(), atol=1e-4)

    out = coverage_stats(model, skill, item, ans, mask, true_theta)
    assert out["theta_corr"] > 0.85
    # With items known and the cell-edge credible interval, the closed-form posterior is calibrated at every
    # level (the fitted-items run under-covers centrally only from item-estimation error).
    assert 0.43 <= out["cov_50"] <= 0.60
    assert 0.73 <= out["cov_80"] <= 0.88
    assert 0.90 <= out["cov_95"] <= 1.0


def test_coverage_maps_bridged_model_to_eta_scale():
    """A [0, 1] Bernstein-bridged model is mapped through the link before comparison, so its EAP correlates
    with the simulator's ℝ-scale θ (a sanity check that the scale handling is wired, not identity)."""
    log_df, _items = generate_log_data(num_students=80, num_items=24, min_interactions=22, max_interactions=24,
                                        n_skills=1, drift_scale=0.0, irt_model="2PL", random_state=3)
    skill, item, ans, mask, true_theta = padded_batch_from_log(log_df)
    model = build_quad_bernstein_estimator_irt(degree_n=40, num_users=int(skill.size(0)), num_skills=1,
                                               num_items=int(item.max()) + 1, num_nodes=40, irt_model="2PL",
                                               prior=(4.0, 4.0))
    fit_mml(model, skill, item, ans, mask, epochs=200, lr=0.05)

    out = coverage_stats(model, skill, item, ans, mask, true_theta)
    assert out["theta_corr"] > 0.6                 # EAP mapped 6θ-3 tracks true θ on ℝ
    assert 0.80 <= out["cov_95"] <= 1.0
