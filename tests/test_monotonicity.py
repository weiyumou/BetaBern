"""Tests for the Bayes-factor monotonicity diagnostic.

The headline check is a calibration identity that needs no simulation: under an **exchangeable**
likelihood every ordering of the weights is equally likely, so the monotone cone carries exactly
its prior share and the Bayes factor must be 1. That pins the estimator's normalisation, including
the ``1/(n+1)!`` cone volume, without appealing to any fitted model.
"""
import math

import numpy as np
import pytest
import torch

from betabern.benchmark.simfit import fit_mml
from betabern.bernstein.estimator.exact import build_exact_bernstein_estimator_bounded
from betabern.bernstein.monotonicity import (
    _stepping_stone,
    item_evidence_operator,
    log_prior_cone,
    monotonicity_bayes_factor,
)

FAST = dict(n_rungs=16, n_draws=120, burn=20, n_replicates=2)


def _fit(P, seed=0, degree=6, epochs=150):
    """Fit the bounded (non-monotone-capable) estimator to responses drawn from ``P`` (B, I)."""
    rng = np.random.default_rng(seed)
    X = (rng.random(P.shape) < P).astype(int)
    n_resp, n_items = X.shape
    torch.manual_seed(0)
    m = build_exact_bernstein_estimator_bounded(degree_n=degree, num_users=1, num_skills=1,
                                                num_items=n_items, item_weight_rank=-1,
                                                prior=(1.0, 1.0))
    batch = (torch.zeros(n_resp, n_items, dtype=torch.long),
             torch.arange(n_items).expand(n_resp, n_items).contiguous(),
             torch.tensor(X), torch.ones(n_resp, n_items, dtype=torch.bool))
    fit_mml(m, *batch, epochs=epochs, lr=0.05)
    return m, batch


def _bank(n_resp=700, n_items=6, seed=0):
    """A bank of monotone 2PL-ish items; caller overwrites individual columns."""
    th = np.random.default_rng(seed).beta(2.0, 2.0, n_resp)
    return th, np.stack([1 / (1 + np.exp(-6 * (th - 0.5)))] * n_items, axis=1)


def test_log_prior_cone_is_reciprocal_factorial():
    for n_w in (2, 5, 9):
        assert log_prior_cone(n_w) == pytest.approx(-math.log(math.factorial(n_w)))


def test_exchangeable_likelihood_gives_unit_bayes_factor():
    """Calibration identity: a symmetric likelihood must leave the cone at its prior share."""
    n_w = 6
    A = np.ones((40, n_w))          # S = sum_k w_k  -> exchangeable in the weights
    sgn = np.ones(40)
    diffs = []
    for r in range(3):
        zf = _stepping_stone(A, sgn, n_w, -1, np.random.default_rng(r),
                             n_rungs=24, n_draws=250, burn=30)
        zm = _stepping_stone(A, sgn, n_w, n_w - 1, np.random.default_rng(50 + r),
                             n_rungs=24, n_draws=250, burn=30)
        diffs.append((zf - zm) / math.log(10))
    assert abs(float(np.mean(diffs))) < 0.5   # exact answer is 0; allow the Monte Carlo floor


def test_evidence_operator_reproduces_the_exact_marginal():
    """The linear reduction in one item's weights must agree with the estimator's own marginal."""
    _, P = _bank(n_resp=300, n_items=5)
    m, (skill, item, ans, mask) = _fit(P, degree=5, epochs=40)
    j = 2
    log_a, y = item_evidence_operator(m, skill, item, ans, mask, item_pos=j)

    w = torch.rand(m.irf.n + 1)
    log_wt = torch.where(torch.as_tensor(y).bool().unsqueeze(-1), w.log(), torch.log1p(-w))
    linear = torch.logsumexp(log_wt + torch.as_tensor(log_a), dim=-1).sum()

    with torch.no_grad():                       # write w into the model and ask for the truth
        u = torch.logit(w.clamp(1e-6, 1 - 1e-6))
        m.irf.item_weights.weight[j] = u - m.irf.skill_weights.weight[0]
        exact = m.log_marginal_evidence(skill, item, ans, mask).sum()
    # the operator drops a per-respondent constant (the other items' evidence); compare differences
    w2 = torch.rand(m.irf.n + 1)
    log_wt2 = torch.where(torch.as_tensor(y).bool().unsqueeze(-1), w2.log(), torch.log1p(-w2))
    linear2 = torch.logsumexp(log_wt2 + torch.as_tensor(log_a), dim=-1).sum()
    with torch.no_grad():
        u2 = torch.logit(w2.clamp(1e-6, 1 - 1e-6))
        m.irf.item_weights.weight[j] = u2 - m.irf.skill_weights.weight[0]
        exact2 = m.log_marginal_evidence(skill, item, ans, mask).sum()
    assert (linear - linear2).item() == pytest.approx((exact - exact2).item(), abs=1e-2)


def test_detects_a_deep_valley_and_clears_monotone_items():
    th, P = _bank(n_resp=700, n_items=6)
    P[:, 0] = np.where((th > 0.3) & (th < 0.7), 0.20, 0.85)     # a deep non-monotone valley
    m, batch = _fit(P)

    valley = monotonicity_bayes_factor(m, *batch, item_pos=0, seed=1, **FAST)
    assert valley.log10_bf_free_vs_mono > 2.0, valley
    assert "non-monotone" in valley.verdict

    for j in (3, 4):                                            # untouched monotone items
        res = monotonicity_bayes_factor(m, *batch, item_pos=j, seed=1, **FAST)
        assert res.log10_bf_free_vs_mono < 0.5, res


def test_result_fields_and_rejects_bad_item():
    _, P = _bank(n_resp=400, n_items=4)
    m, batch = _fit(P, degree=5, epochs=60)
    res = monotonicity_bayes_factor(m, *batch, item_pos=1, seed=0, **FAST)
    assert res.n_respondents == 400
    assert res.degree == m.irf.n
    assert np.isfinite([res.log10_bf_free_vs_mono, res.log_evidence_free, res.log_evidence_mono]).all()
    assert str(res).startswith("item 1:")

    with pytest.raises(ValueError):
        monotonicity_bayes_factor(m, *batch, item_pos=99, seed=0, **FAST)
