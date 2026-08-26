"""Tests for the locally-supported I-spline monotone IRF and its quadrature estimator."""
import numpy as np
import torch
import torch.distributions as D
from helpers import make_batch as _make_batch

from betabern.benchmark import registry
from betabern.benchmark.runner import run_benchmark
from betabern.bernstein.estimator.quad import build_quad_ispline_estimator_free
from betabern.core.model.irf import ISplineIRF
from betabern.core.util import log1mexp

# ===========================================================================
# log1mexp util
# ===========================================================================

def test_log1mexp_matches_reference():
    x = torch.tensor([-1e-4, -0.01, -0.5, -1.0, -5.0, -20.0], dtype=torch.double)
    assert torch.allclose(log1mexp(x), torch.log1p(-torch.exp(x)), atol=1e-9)


# ===========================================================================
# I-spline basis
# ===========================================================================

def test_ispline_basis_properties():
    theta = torch.linspace(0, 1, 41)
    for order in (2, 3, 4):
        irf = ISplineIRF(degree_n=8, num_skills=2, num_items=5, item_weight_rank=2, order=order)
        log_b = irf.log_basis(theta)  # (41, m + 1)
        assert log_b.shape == (41, irf.m + 1)
        assert torch.isfinite(log_b).all()
        assert torch.allclose(log_b[:, 0], torch.zeros(41))  # intercept column = log 1 = 0
        basis = log_b[:, 1:].exp()  # (41, m) the basis-splines
        assert torch.allclose(basis[0], torch.zeros(irf.m), atol=1e-6)  # I_i(0) = 0
        assert torch.allclose(basis[-1], torch.ones(irf.m), atol=1e-6)  # I_i(1) = 1
        assert (basis[1:] - basis[:-1] >= -1e-6).all()  # monotone non-decreasing in theta


# ===========================================================================
# I-spline IRF
# ===========================================================================

def test_ispline_irf_monotone_normalized_and_shaped():
    torch.manual_seed(0)
    irf = ISplineIRF(degree_n=10, num_skills=3, num_items=6, item_weight_rank=2, order=3)
    for p in irf.parameters():  # non-trivial per-item curves
        with torch.no_grad():
            p.normal_()

    items, skills = torch.arange(6), torch.zeros(6, dtype=torch.long)
    grid = torch.linspace(0, 1, 60)
    P = irf.evaluate_irf(grid, items, skills)  # (6, 60) = P(correct)
    assert P.shape == (6, 60)
    assert (P >= 0).all() and (P <= 1).all()
    assert (P[:, 1:] - P[:, :-1] >= -1e-6).all()  # monotone increasing in theta

    # P(incorrect) + P(correct) = 1, and log_irf has the (B, S, 2, Q) layout.
    log_irf = irf.log_irf(grid.unsqueeze(0), items.unsqueeze(0), torch.zeros(1, dtype=torch.long))
    assert log_irf.shape == (1, 6, 2, 60)
    assert torch.allclose(log_irf.exp().sum(dim=2), torch.ones(1, 6, 60), atol=1e-5)


# ===========================================================================
# Quadrature estimator
# ===========================================================================

def test_ispline_marginal_evidence_matches_numerical_integral():
    """log_marginal_evidence matches a fine brute-force integral of prior * likelihood."""
    B, S, num_items, num_skills = 4, 3, 8, 2
    model = build_quad_ispline_estimator_free(
        degree_n=8, num_users=5, num_skills=num_skills, num_items=num_items,
        num_nodes=120, item_weight_rank=2, order=3,
    )
    torch.manual_seed(0)
    for p in model.parameters():
        with torch.no_grad():
            p.normal_(std=0.5)
    item_ids, skill_ids, answers, mask = _make_batch(B, S, num_items, num_skills)

    log_evidence = model.log_marginal_evidence(skill_ids, item_ids, answers, mask)

    alpha, beta = model.prior.get_params(skill_ids.shape[0])
    grid = torch.linspace(1e-4, 1 - 1e-4, 8000)
    grid_b = grid.unsqueeze(0).expand(B, -1)
    log_lik = model.irf.log_irf(grid_b, item_ids, skill_ids)  # (B, S, 2, G)
    idx = answers[:, :, None, None].expand(-1, -1, 1, grid.size(0))
    obs = log_lik.gather(2, idx).squeeze(2).sum(dim=1)  # (B, G)
    log_prior = D.Beta(alpha.unsqueeze(-1), beta.unsqueeze(-1)).log_prob(grid)
    ref = torch.log(torch.trapezoid((log_prior + obs).exp(), grid, dim=-1))

    assert torch.allclose(log_evidence, ref, atol=2e-2), (log_evidence, ref)


def test_ispline_estimator_trains():
    model = build_quad_ispline_estimator_free(
        degree_n=6, num_users=5, num_skills=2, num_items=8, num_nodes=40, item_weight_rank=2, order=3)
    item_ids, skill_ids, answers, mask = _make_batch(4, 3, 8, 2)
    loss = -model.log_marginal_evidence(skill_ids, item_ids, answers, mask).sum()
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


# ===========================================================================
# Registry + benchmark integration
# ===========================================================================

def test_quad_ispline_free_registered_and_runs():
    assert "quad_ispline_free" in registry.MODELS
    config = {
        "datasets": [{"name": "sim", "simulate": {
            "num_students": 40, "num_items": 12, "n_skills": 1,
            "min_interactions": 8, "max_interactions": 12, "irt_model": "2PL", "random_state": 0}}],
        "models": ["quad_logistic_irt", "quad_ispline_free"],
        "splits": ["within_random"],
        "n_folds": 2,
        "hyperparams": {"degree": 6, "num_nodes": 24, "irt_model": "2PL",
                        "item_weight_rank": 2, "ispline_order": 3},
        "trainer": {"max_epochs": 2, "patience": 2, "accelerator": "cpu", "batch_size": 32},
    }
    runs, summary, predictions = run_benchmark(config)
    isp = runs[(runs["model"] == "quad_ispline_free") & (runs["split"] == "within_random")]
    assert len(isp) == 2
    assert np.isfinite(isp["nll"]).all() and np.isfinite(isp["auc"]).all()
    assert (isp["brier"].between(0, 1)).all()
