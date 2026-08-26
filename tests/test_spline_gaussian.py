"""Tests for the spline-Gaussian model: a free monotone I-spline IRF on an R window + a Normal prior.

The R analog of the [0, 1] free I-spline (``test_ispline.py``): the same ``ISplineIRF`` machinery placed on
a bounded window of R with constant-saturating tails, paired with ``NormalHermitePrior`` and Gauss-Hermite
quadrature. Fit by quadrature only (the "fast" path; the exact conjugate counterpart is not implemented)."""
import numpy as np
import torch
import torch.distributions as D
from helpers import make_batch as _make_batch

from betabern.benchmark import registry
from betabern.benchmark.runner import run_benchmark
from betabern.core.model.irf import ISplineIRF
from betabern.irt.estimator.quad_irt import build_quad_spline_irt_estimator

# ===========================================================================
# I-spline basis on an R window: saturating tails
# ===========================================================================

def test_spline_gaussian_basis_saturates_outside_window():
    """On an R window the I-splines stay monotone 0 -> 1 inside and saturate (0 below, 1 above)."""
    lo, hi = -5.0, 5.0
    irf = ISplineIRF(degree_n=8, num_skills=2, num_items=5, item_weight_rank=2, order=3, domain=(lo, hi))
    assert irf.domain == (lo, hi)

    inside = torch.linspace(lo, hi, 41)
    log_b = irf.log_basis(inside)  # (41, m + 1)
    assert log_b.shape == (41, irf.m + 1)
    assert torch.isfinite(log_b).all()
    assert torch.allclose(log_b[:, 0], torch.zeros(41))  # intercept column = log 1 = 0
    basis = log_b[:, 1:].exp()  # (41, m)
    assert torch.allclose(basis[0], torch.zeros(irf.m), atol=1e-6)  # I_i(lo) = 0
    assert torch.allclose(basis[-1], torch.ones(irf.m), atol=1e-6)  # I_i(hi) = 1
    assert (basis[1:] - basis[:-1] >= -1e-6).all()  # monotone non-decreasing

    # Far outside the window (where stray Gauss-Hermite nodes land) the basis is fully saturated.
    far = torch.tensor([-50.0, -hi - 1e-3, lo - 1e-3])
    assert torch.allclose(irf.log_basis(far)[:, 1:].exp(), torch.zeros(3, irf.m), atol=1e-6)  # all 0 below
    assert torch.allclose(irf.log_basis(torch.tensor([hi + 1e-3, 50.0]))[:, 1:].exp(),
                          torch.ones(2, irf.m), atol=1e-6)  # all 1 above


def test_ispline_default_domain_is_unit_interval():
    """The default domain is unchanged ([0, 1]) so the Beta-prior I-spline path is unaffected."""
    irf = ISplineIRF(degree_n=6, num_skills=1, num_items=3, order=2)
    assert irf.domain == (0.0, 1.0)
    basis = irf.log_basis(torch.tensor([0.0, 1.0]))[:, 1:].exp()
    assert torch.allclose(basis[0], torch.zeros(irf.m), atol=1e-6) and torch.allclose(basis[-1], torch.ones(irf.m), atol=1e-6)


# ===========================================================================
# IRF: monotone, normalized, saturating
# ===========================================================================

def test_spline_gaussian_irf_monotone_normalized_and_shaped():
    torch.manual_seed(0)
    model = build_quad_spline_irt_estimator(num_skills=3, num_items=6, degree_n=10, order=3,
                                            num_nodes=40, item_weight_rank=2)
    for p in model.parameters():
        with torch.no_grad():
            p.normal_()
    irf = model.irf

    items, skills = torch.arange(6), torch.zeros(6, dtype=torch.long)
    grid = torch.linspace(-9, 9, 80)  # spans well beyond the +/-5 window
    P = irf.evaluate_irf(grid, items, skills)  # (6, 80) = P(correct)
    assert P.shape == (6, 80)
    assert (P >= 0).all() and (P <= 1).all()
    assert (P[:, 1:] - P[:, :-1] >= -1e-6).all()  # monotone increasing in theta

    # Saturated beyond the window: P(+9) == P(+5), P(-9) == P(-5).
    edges = irf.evaluate_irf(torch.tensor([-5.0, 5.0]), items, skills)  # (6, 2)
    assert torch.allclose(P[:, 0], edges[:, 0], atol=1e-5) and torch.allclose(P[:, -1], edges[:, 1], atol=1e-5)

    # P(incorrect) + P(correct) = 1, and log_irf has the (B, S, 2, Q) layout.
    log_irf = irf.log_irf(grid.unsqueeze(0), items.unsqueeze(0), torch.zeros(1, dtype=torch.long))
    assert log_irf.shape == (1, 6, 2, 80)
    assert torch.allclose(log_irf.exp().sum(dim=2), torch.ones(1, 6, 80), atol=1e-5)


# ===========================================================================
# Quadrature estimator
# ===========================================================================

def test_spline_gaussian_marginal_evidence_matches_numerical_integral():
    """log_marginal_evidence matches a fine brute-force integral of the Normal prior * likelihood."""
    B, S, num_items, num_skills = 4, 3, 8, 2
    model = build_quad_spline_irt_estimator(num_skills=num_skills, num_items=num_items,
                                            degree_n=8, order=3, num_nodes=120, item_weight_rank=2)
    torch.manual_seed(0)
    for p in model.parameters():
        with torch.no_grad():
            p.normal_(std=0.6)
    item_ids, skill_ids, answers, mask = _make_batch(B, S, num_items, num_skills)

    log_evidence = model.log_marginal_evidence(skill_ids, item_ids, answers, mask)

    mu, sigma = model.prior.get_params(B)
    grid = torch.linspace(-12, 12, 8000)
    grid_b = grid.unsqueeze(0).expand(B, -1)
    log_lik = model.irf.log_irf(grid_b, item_ids, skill_ids)  # (B, S, 2, G)
    idx = answers[:, :, None, None].expand(-1, -1, 1, grid.size(0))
    obs = log_lik.gather(2, idx).squeeze(2).sum(dim=1)  # (B, G)
    log_prior = D.Normal(mu.unsqueeze(-1), sigma.unsqueeze(-1)).log_prob(grid)
    ref = torch.log(torch.trapezoid((log_prior + obs).exp(), grid, dim=-1))

    assert torch.allclose(log_evidence, ref, atol=2e-2), (log_evidence, ref)


def test_spline_gaussian_eap_with_no_data_equals_prior_mean():
    """With nothing observed the posterior equals the prior, so the EAP is the fixed prior mean."""
    model = build_quad_spline_irt_estimator(num_skills=3, num_items=8, degree_n=8, num_nodes=60,
                                            item_weight_rank=2, prior_mean=0.4, prior_std=1.2)
    item_ids, skill_ids, answers, mask = _make_batch(5, 4, 8, 3)
    eap = model.posterior_stats(skill_ids, item_ids, answers, torch.zeros_like(mask), level=None).mean
    assert torch.allclose(eap, torch.full((5,), 0.4), atol=1e-3)


def test_spline_gaussian_estimator_trains():
    model = build_quad_spline_irt_estimator(num_skills=2, num_items=8, degree_n=6, num_nodes=40,
                                            item_weight_rank=2, order=3)
    item_ids, skill_ids, answers, mask = _make_batch(4, 3, 8, 2)
    loss = -model.log_marginal_evidence(skill_ids, item_ids, answers, mask).sum()
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    assert not any(p.requires_grad for p in model.prior.parameters())  # prior is fixed


# ===========================================================================
# Registry + benchmark integration
# ===========================================================================

def test_quad_spline_gauss_registered_and_runs():
    assert "quad_spline_irt" in registry.MODELS
    config = {
        "datasets": [{"name": "sim", "simulate": {
            "num_students": 40, "num_items": 12, "n_skills": 1,
            "min_interactions": 8, "max_interactions": 12, "irt_model": "2PL", "random_state": 0}}],
        "models": ["quad_logistic_irt", "quad_spline_irt"],
        "splits": ["within_random"],
        "n_folds": 2,
        "hyperparams": {"degree": 6, "num_nodes": 24, "irt_model": "2PL",
                        "item_weight_rank": 2, "ispline_order": 3},
        "trainer": {"max_epochs": 2, "patience": 2, "accelerator": "cpu", "batch_size": 32},
    }
    runs, summary, predictions = run_benchmark(config)
    sg = runs[(runs["model"] == "quad_spline_irt") & (runs["split"] == "within_random")]
    assert len(sg) == 2
    assert np.isfinite(sg["nll"]).all() and np.isfinite(sg["auc"]).all()
    assert (sg["brier"].between(0, 1)).all()
