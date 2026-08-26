"""Tests for the "Bernstein slot": pluggable monotone weight-generators on a conjugate Bernstein IRF.

Covers the spline-g (`SplineBernsteinIRF`) and CDF-g (`CDFBernsteinIRF`) generators — monotone/bounded
shapes, the degree-vs-parameter decoupling, and the headline property that the slot **retains** the exact
Beta-Bernstein conjugate posterior (matched to brute force) that the direct local I-spline gives up.
"""
import numpy as np
import torch
import torch.distributions as D
from helpers import make_batch as _make_batch

from betabern.benchmark import registry
from betabern.benchmark.runner import run_benchmark
from betabern.bernstein.bernstein_irf import CDFBernsteinIRF, SplineBernsteinIRF
from betabern.bernstein.estimator.exact import ExactBernsteinEstimator, _log_terms
from betabern.core.model.estimator import QuadratureEstimator
from betabern.core.model.irf import ISplineIRF
from betabern.core.model.posterior import AbilityPosterior
from betabern.core.model.prior import BetaJacobiPrior, BetaPrior


def _randomize(module, std=1.0):
    torch.manual_seed(0)
    for p in module.parameters():
        with torch.no_grad():
            p.normal_(std=std)
    return module


# ===========================================================================
# Generator IRFs: monotone, bounded, normalized
# ===========================================================================

def _slot_irfs():
    return {
        "spline-g": SplineBernsteinIRF(degree_n=20, num_skills=3, num_items=6,
                                       gen_degree=6, gen_order=2, item_weight_rank=2),
        "cdf-g": CDFBernsteinIRF(degree_n=20, num_skills=3, num_items=6, item_weight_rank=2),
    }


def test_slot_generators_monotone_bounded_normalized():
    for irf in _slot_irfs().values():
        _randomize(irf)
        items = torch.arange(6).unsqueeze(0)
        skill = torch.zeros(1, dtype=torch.long)
        grid = torch.linspace(0, 1, 60)
        log_irf = irf.log_irf(grid.unsqueeze(0), items, skill)  # (1, 6, 2, 60)
        assert log_irf.shape == (1, 6, 2, 60)
        P = log_irf[0, :, 1, :].exp()  # P(correct), (6, 60)
        assert (P >= 0).all() and (P <= 1).all()
        assert (P[:, 1:] - P[:, :-1] >= -1e-5).all()  # monotone non-decreasing in theta
        assert torch.allclose(log_irf.exp().sum(dim=2), torch.ones(1, 6, 60), atol=1e-5)  # classes sum to 1


# ===========================================================================
# Efficiency: per-item parameters are decoupled from the Bernstein degree
# ===========================================================================

def test_spline_g_param_count_independent_of_degree():
    def n_params(degree_n):
        irf = SplineBernsteinIRF(degree_n=degree_n, num_skills=2, num_items=8,
                                 gen_degree=6, gen_order=2, item_weight_rank=2)
        return sum(p.numel() for p in irf.parameters() if p.requires_grad)

    assert n_params(10) == n_params(40)  # degree (representation) is decoupled from params (capacity)


# ===========================================================================
# Tractability retained: exact == quad, and the closed-form posterior is correct
# ===========================================================================

def test_slot_exact_equals_quad_marginal():
    irf = _randomize(SplineBernsteinIRF(degree_n=8, num_skills=1, num_items=6,
                                        gen_degree=5, gen_order=3, item_weight_rank=2))
    quad = QuadratureEstimator(prior=BetaJacobiPrior(1.0, 1.0, 60), irf=irf)
    exact = ExactBernsteinEstimator(BetaPrior(), irf=irf)
    item_ids, skill_ids, answers, mask = _make_batch(4, 5, 6, 1)
    mq = quad.log_marginal_evidence(skill_ids, item_ids, answers, mask)
    me = exact.log_marginal_evidence(skill_ids, item_ids, answers, mask)
    assert torch.allclose(mq, me, atol=1e-4), (mq, me)


def test_slot_closed_form_posterior_matches_brute_force():
    """The slot is a global Bernstein polynomial, so the conjugate Beta-mixture posterior is exact —
    its mean AND variance match a brute-force grid posterior (the property the direct I-spline loses)."""
    S = 5
    irf = _randomize(SplineBernsteinIRF(degree_n=8, num_skills=1, num_items=S,
                                        gen_degree=5, gen_order=3, item_weight_rank=2))
    exact = ExactBernsteinEstimator(BetaPrior(), irf=irf)
    items = torch.arange(S).unsqueeze(0)
    skill = torch.zeros(1, dtype=torch.long)
    answers = torch.randint(0, 2, (1, S))
    mask = torch.ones(1, S, dtype=torch.bool)

    # Closed-form posterior: mixture of Beta(alpha + k, beta + D - k) weighted by softmax(_log_terms).
    alpha, beta = exact.get_prior(skill)
    log_c, Dg = exact._fold(items, answers, mask, skill)
    pi = torch.log_softmax(_log_terms(alpha, beta, log_c, Dg), dim=-1)[0].exp().double()
    k = torch.arange(Dg + 1, dtype=torch.double)
    a, b = float(alpha[0]), float(beta[0])
    ca, cb = a + k, b + (Dg - k)
    cf_mean = float((pi * ca / (ca + cb)).sum())
    cf_e2 = float((pi * ca * (ca + 1) / ((ca + cb) * (ca + cb + 1))).sum())
    cf_var = cf_e2 - cf_mean ** 2

    # Brute-force posterior: prior * likelihood on a fine grid.
    grid = torch.linspace(1e-6, 1 - 1e-6, 200_000)
    Pc = irf.evaluate_irf(grid, items[0], skill.expand(S))  # (S, G)
    ll = torch.where(answers[0].bool().unsqueeze(-1), Pc, 1 - Pc).clamp_min(1e-12).log().sum(0)
    post = (ll + D.Beta(torch.tensor(a), torch.tensor(b)).log_prob(grid)).exp()
    post = post / torch.trapezoid(post, grid)
    bf_mean = float(torch.trapezoid(post * grid, grid))
    bf_var = float(torch.trapezoid(post * grid * grid, grid)) - bf_mean ** 2

    assert abs(cf_mean - bf_mean) < 1e-4, (cf_mean, bf_mean)
    assert abs(cf_var - bf_var) < 1e-4, (cf_var, bf_var)
    # posterior_stats() exposes the same posterior mean.
    pm_mean = exact.posterior_stats(skill, items, answers, mask, level=None).mean
    assert abs(float(pm_mean[0]) - bf_mean) < 1e-4


def test_posterior_stats_matches_brute_force():
    """The single-call posterior summary (mean/std/credible interval) matches a brute-force grid posterior."""
    S = 6
    irf = _randomize(SplineBernsteinIRF(degree_n=8, num_skills=1, num_items=S,
                                        gen_degree=5, gen_order=3, item_weight_rank=2))
    exact = ExactBernsteinEstimator(BetaPrior(), irf=irf)
    items = torch.arange(S).unsqueeze(0)
    skill = torch.zeros(1, dtype=torch.long)
    answers = torch.randint(0, 2, (1, S))
    mask = torch.ones(1, S, dtype=torch.bool)

    post = exact.posterior_stats(skill, items, answers, mask, level=0.90)
    assert isinstance(post, AbilityPosterior) and post.level == 0.90
    # level=None gives the same mean without computing the credible interval.
    assert torch.allclose(post.mean, exact.posterior_stats(skill, items, answers, mask, level=None).mean, atol=1e-5)

    grid = torch.linspace(1e-6, 1 - 1e-6, 200_000)
    Pc = irf.evaluate_irf(grid, items[0], skill.expand(S))
    ll = torch.where(answers[0].bool().unsqueeze(-1), Pc, 1 - Pc).clamp_min(1e-12).log().sum(0)
    pp = (ll + D.Beta(torch.tensor(1.0), torch.tensor(1.0)).log_prob(grid)).exp()
    pp = pp / torch.trapezoid(pp, grid)
    bf_mean = float(torch.trapezoid(pp * grid, grid))
    bf_std = (float(torch.trapezoid(pp * grid * grid, grid)) - bf_mean ** 2) ** 0.5
    assert abs(float(post.mean[0]) - bf_mean) < 1e-4
    assert abs(float(post.std[0]) - bf_std) < 1e-4

    lo, hi = float(post.ci_low[0]), float(post.ci_high[0])
    assert 0.0 <= lo < float(post.mean[0]) < hi <= 1.0
    mass = float(pp[(grid >= lo) & (grid <= hi)].sum() * (grid[1] - grid[0]))  # coverage ~ level
    assert abs(mass - 0.90) < 0.02


def test_quad_posterior_stats_matches_eap_and_exact():
    """The quadrature-node posterior: mean == eap, agrees with the exact posterior at moderate degree,
    and exposes the discrete node support (no continuous density)."""
    S = 6
    irf = _randomize(SplineBernsteinIRF(degree_n=8, num_skills=1, num_items=S,
                                        gen_degree=5, gen_order=3, item_weight_rank=2))
    quad = QuadratureEstimator(prior=BetaJacobiPrior(1.0, 1.0, 80), irf=irf)
    exact = ExactBernsteinEstimator(BetaPrior(), irf=irf)
    items = torch.arange(S).unsqueeze(0)
    skill = torch.zeros(1, dtype=torch.long)
    answers = torch.randint(0, 2, (1, S))
    mask = torch.ones(1, S, dtype=torch.bool)

    qp = quad.posterior_stats(skill, items, answers, mask, level=0.90)
    assert isinstance(qp, AbilityPosterior) and qp.mixture is None
    assert qp.nodes.shape == (80,) and qp.log_weights.shape == (1, 80)
    assert torch.allclose(qp.mean, quad.posterior_stats(skill, items, answers, mask, level=None).mean, atol=1e-6)

    ep = exact.posterior_stats(skill, items, answers, mask, level=0.90)  # agrees with exact
    assert abs(float(qp.mean[0]) - float(ep.mean[0])) < 1e-3
    assert abs(float(qp.std[0]) - float(ep.std[0])) < 1e-3
    assert abs(float(qp.ci_low[0]) - float(ep.ci_low[0])) < 0.02
    assert abs(float(qp.ci_high[0]) - float(ep.ci_high[0])) < 0.02
    assert 0.0 <= float(qp.ci_low[0]) < float(qp.mean[0]) < float(qp.ci_high[0]) <= 1.0

    # Both posteriors expose a continuous log-density: exact is the Beta mixture, quadrature is a
    # reconstruction from the discrete node posterior.
    grid = torch.linspace(0.001, 0.999, 1000)
    q_logp = qp.log_pdf(grid)  # (1, T)
    assert q_logp.shape == (1, grid.numel())
    dens = q_logp.exp()[0]
    assert abs(float(torch.trapezoid(dens, grid)) - 1.0) < 0.05  # reconstructed density ~ normalized
    assert abs(float(torch.trapezoid(dens * grid, grid)) - float(qp.mean[0])) < 0.02  # consistent w/ EAP
    assert torch.isfinite(ep.log_pdf(grid)).all()  # exact Beta-mixture log-density
    assert torch.isinf(qp.log_pdf(torch.tensor([2.0]))).all()  # outside the node support -> -inf


def test_slot_has_conjugacy_direct_ispline_does_not():
    slot = SplineBernsteinIRF(degree_n=10, num_skills=1, num_items=6,
                              gen_degree=5, gen_order=2, item_weight_rank=2)
    direct = ISplineIRF(degree_n=10, num_skills=1, num_items=6, item_weight_rank=2, order=2)
    assert hasattr(slot, "conjugate_update")  # slot: global Bernstein -> conjugate
    assert not hasattr(direct, "conjugate_update")  # direct local spline -> no Beta conjugacy


# ===========================================================================
# Registry + benchmark integration
# ===========================================================================

def test_slot_models_registered_and_run():
    assert {"quad_spline_bern_irt", "quad_bern_cdfg"}.issubset(registry.MODELS)
    config = {
        "datasets": [{"name": "sim", "simulate": {
            "num_students": 40, "num_items": 12, "n_skills": 1,
            "min_interactions": 8, "max_interactions": 12, "irt_model": "2PL", "random_state": 0}}],
        "models": ["quad_logistic_irt", "quad_spline_bern_irt", "quad_bern_cdfg"],
        "splits": ["within_random"],
        "n_folds": 2,
        "hyperparams": {"degree": 12, "num_nodes": 24, "irt_model": "2PL",
                        "item_weight_rank": 2, "gen_degree": 5, "gen_order": 2},
        "trainer": {"max_epochs": 2, "patience": 2, "accelerator": "cpu", "batch_size": 32},
    }
    runs, summary, predictions = run_benchmark(config)
    for key in ("quad_spline_bern_irt", "quad_bern_cdfg"):
        rows = runs[(runs["model"] == key) & (runs["split"] == "within_random")]
        assert len(rows) == 2
        assert np.isfinite(rows["nll"]).all() and np.isfinite(rows["auc"]).all()
        assert (rows["brier"].between(0, 1)).all()
