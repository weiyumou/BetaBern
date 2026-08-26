"""Tests for the normal-ogive IRT family: direct (Gauss-Hermite) and Bernstein-bridged (conjugate).

The ogive link is the second classic IRF (alongside logistic). It threads through the ``irt_model`` code (``2PO``) on
``IRTBase`` (``log_ndtr`` instead of ``logsigmoid``); the Bernstein bridge samples the ogive curve into the
global Bernstein basis, so it retains the exact Beta-Bernstein conjugate marginal/posterior.
"""
import numpy as np
import pytest
import torch
import torch.distributions as D
from helpers import make_batch as _make_batch

from betabern.benchmark import registry
from betabern.benchmark.runner import run_benchmark
from betabern.bernstein.bernstein_irf import make_logistic_bernstein_irf
from betabern.bernstein.estimator.exact import ExactBernsteinEstimator
from betabern.core.model.estimator import QuadratureEstimator
from betabern.core.model.prior import BetaJacobiPrior, BetaPrior
from betabern.irt.estimator.quad_irt import build_quad_irt_estimator
from betabern.irt.model import IRTBase


def _randomize(module, std=0.4):
    torch.manual_seed(0)
    for p in module.parameters():
        if p.requires_grad:
            with torch.no_grad():
                p.normal_(std=std)
    return module


# ===========================================================================
# IRF: valid, monotone, normalized — and genuinely distinct from logistic
# ===========================================================================

def test_ogive_irf_valid_monotone_and_distinct_from_logistic():
    torch.manual_seed(0)
    base = IRTBase(num_users=5, num_skills=2, num_items=6, irt_model="2PO",
                   enable_item_params=True)
    for p in base.parameters():
        with torch.no_grad():
            p.normal_(std=0.5)

    items = torch.arange(6).unsqueeze(0)  # (1, 6)
    skill = torch.zeros(1, dtype=torch.long)
    grid = torch.linspace(-6, 6, 80).unsqueeze(0)  # (1, 80)
    log_irf = base.log_irt(grid, items, skill)  # (1, 6, 2, 80)

    assert log_irf.shape == (1, 6, 2, 80)
    P = log_irf[0, :, 1, :].exp()  # P(correct)
    assert (P >= 0).all() and (P <= 1).all()
    assert (P[:, 1:] - P[:, :-1] >= -1e-6).all()  # monotone non-decreasing in theta
    assert torch.allclose(log_irf.exp().sum(dim=2), torch.ones(1, 6, 80), atol=1e-5)  # classes sum to 1

    # Same parameters, different link -> a different curve (ogive is steeper than logistic).
    base_logit = IRTBase(num_users=5, num_skills=2, num_items=6, irt_model="2PL",
                         enable_item_params=True)
    base_logit.load_state_dict(base.state_dict())  # the model code is not a parameter, so it stays logistic
    P_logit = base_logit.log_irt(grid, items, skill)[0, :, 1, :].exp()
    assert not torch.allclose(P, P_logit, atol=1e-2)


def test_bad_irt_model_raises():
    with pytest.raises(ValueError):
        IRTBase(num_users=2, num_skills=1, num_items=2, irt_model="2PX")


def test_link_key_mismatch_is_blocked():
    """Each IRT registry key asserts the config's irt_model link matches: an ogive key needs '2PO',
    a logistic key needs '2PL'. A mismatch fails loudly instead of silently running the wrong model."""

    class _Stats:
        num_users, num_skills, num_items = 5, 2, 8

    class _DM:
        stats = _Stats()

    hp = {"degree": 6, "num_nodes": 12, "item_weight_rank": 1, "lr": 0.02}
    with pytest.raises(ValueError):  # ogive key, logistic code
        registry.MODELS["quad_ogive_irt"](_DM(), {**hp, "irt_model": "2PL"})
    with pytest.raises(ValueError):  # logistic key, ogive code
        registry.MODELS["quad_logistic_bern_irt"](_DM(), {**hp, "irt_model": "2PO"})
    registry.MODELS["quad_ogive_irt"](_DM(), {**hp, "irt_model": "2PO"})  # matched -> builds fine


# ===========================================================================
# Direct ogive estimator (ℝ + Normal, Gauss-Hermite)
# ===========================================================================

def test_ogive_marginal_evidence_matches_numerical_integral():
    """log_marginal_evidence matches a fine brute-force integral of Normal prior * ogive likelihood."""
    B, S, num_items, num_skills = 4, 3, 10, 3
    model = _randomize(build_quad_irt_estimator(
        num_users=5, num_skills=num_skills, num_items=num_items,
        num_nodes=60, irt_model="2PO", item_weight_rank=1))
    item_ids, skill_ids, answers, mask = _make_batch(B, S, num_items, num_skills)

    log_evidence = model.log_marginal_evidence(skill_ids, item_ids, answers, mask)

    mu, sigma = model.prior.get_params(B)
    grid = torch.linspace(-12, 12, 6000)
    grid_b = grid.unsqueeze(0).expand(B, -1)
    log_lik = model.irf.log_irf(grid_b, item_ids, skill_ids)  # (B, S, 2, G)
    idx = answers[:, :, None, None].expand(-1, -1, 1, grid.size(0))
    obs = log_lik.gather(2, idx).squeeze(2).sum(dim=1)  # (B, G)
    log_prior = D.Normal(mu.unsqueeze(-1), sigma.unsqueeze(-1)).log_prob(grid)
    ref = torch.log(torch.trapezoid((log_prior + obs).exp(), grid, dim=-1))

    assert torch.allclose(log_evidence, ref, atol=2e-2), (log_evidence, ref)


def test_ogive_estimator_trains():
    model = build_quad_irt_estimator(num_users=5, num_skills=2, num_items=8,
                                     num_nodes=21, irt_model="2PO", item_weight_rank=1)
    item_ids, skill_ids, answers, mask = _make_batch(4, 3, 8, 2)
    loss = -model.log_marginal_evidence(skill_ids, item_ids, answers, mask).sum()
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


# ===========================================================================
# Bernstein-bridged ogive ([0, 1] + Beta) — keeps exact conjugacy
# ===========================================================================

def test_ogive_bernstein_exact_equals_quad_and_matches_integral():
    """The ogive curve sampled into the global Bernstein basis is still a Bernstein polynomial, so the
    exact (conjugate) marginal equals high-Q Gauss-Jacobi and a brute-force Beta integral."""
    num_skills, S, num_items, degree = 2, 4, 8, 5
    irf = _randomize(make_logistic_bernstein_irf(
        degree_n=degree, num_users=5, num_skills=num_skills, num_items=num_items,
        irt_model="2PO", item_weight_rank=1))
    quad = QuadratureEstimator(prior=BetaJacobiPrior(1.0, 1.0, 80), irf=irf)
    exact = ExactBernsteinEstimator(BetaPrior(), irf=irf)
    item_ids, skill_ids, answers, mask = _make_batch(4, S, num_items, num_skills)

    mq = quad.log_marginal_evidence(skill_ids, item_ids, answers, mask)
    me = exact.log_marginal_evidence(skill_ids, item_ids, answers, mask)
    assert torch.allclose(mq, me, atol=1e-3), (mq, me)  # ogive bridge retains Beta-Bernstein conjugacy

    alpha, beta = exact.get_prior(skill_ids)
    grid = torch.linspace(1e-4, 1 - 1e-4, 8000)
    grid_b = grid.unsqueeze(0).expand(4, -1)
    log_lik = irf.log_irf(grid_b, item_ids, skill_ids)
    idx = answers[:, :, None, None].expand(-1, -1, 1, grid.size(0))
    obs = log_lik.gather(2, idx).squeeze(2).sum(dim=1)
    log_prior = D.Beta(alpha.unsqueeze(-1), beta.unsqueeze(-1)).log_prob(grid)
    ref = torch.log(torch.trapezoid((log_prior + obs).exp(), grid, dim=-1))
    assert torch.allclose(me, ref, atol=1e-2), (me, ref)


# ===========================================================================
# Registry + benchmark integration
# ===========================================================================

def test_ogive_models_registered_and_run():
    assert "quad_ogive_irt" in registry.MODELS and "quad_ogive_bern_irt" in registry.MODELS
    config = {
        "datasets": [{"name": "sim", "simulate": {
            "num_students": 40, "num_items": 12, "n_skills": 1,
            "min_interactions": 8, "max_interactions": 12, "irt_model": "2PL", "random_state": 0}}],
        "models": ["quad_ogive_irt", "quad_ogive_bern_irt"],
        "splits": ["within_random"],
        "n_folds": 2,
        "hyperparams": {"degree": 6, "num_nodes": 24, "irt_model": "2PO", "item_weight_rank": 1},
        "trainer": {"max_epochs": 2, "patience": 2, "accelerator": "cpu", "batch_size": 32},
    }
    runs, summary, predictions = run_benchmark(config)
    for m in ("quad_ogive_irt", "quad_ogive_bern_irt"):
        r = runs[(runs["model"] == m) & (runs["split"] == "within_random")]
        assert len(r) == 2
        assert np.isfinite(r["nll"]).all() and np.isfinite(r["auc"]).all()
        assert (r["brier"].between(0, 1)).all()
