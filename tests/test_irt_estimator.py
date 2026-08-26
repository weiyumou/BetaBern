"""
Test suite for the IRT Gauss-Hermite MML estimator.

Validates the standardized Gauss-Hermite quadrature, the Normal-prior marginal-evidence
computation (against a brute-force numerical integral), and the estimator's shapes / gradients.
"""

import torch
import torch.distributions as D
import torch.nn as nn
from helpers import make_batch as _make_batch

from betabern.core.model.estimator import BayesianEstimator
from betabern.core.model.prior import gauss_hermite_standard
from betabern.core.util import count_free_params, raw_trainable_params
from betabern.irt.estimator.quad_irt import build_quad_irt_estimator


def test_count_free_params_protocol():
    """The dispatcher: raw fallback for plain modules, delegation + softmax correction otherwise."""
    # No override -> raw trainable count.
    lin = nn.Linear(4, 3)
    assert count_free_params(lin) == raw_trainable_params(lin) == sum(p.numel() for p in lin.parameters())

    # 2PL IRT: endpoints are frozen, so free == raw (the estimator delegates to its IRF).
    est2 = build_quad_irt_estimator(num_users=5, num_skills=3, num_items=10, num_nodes=16, irt_model="2PL",
                                    item_weight_rank=1)
    assert count_free_params(est2) == est2.irf.num_free_params() == raw_trainable_params(est2)

    # 4PL IRT: the [c, s, m] softmax over-counts -> a correction is applied (free < raw).
    est4 = build_quad_irt_estimator(num_users=5, num_skills=3, num_items=10, num_nodes=16, irt_model="4PL",
                                    item_weight_rank=1)
    assert count_free_params(est4) < raw_trainable_params(est4)

    # A softmax-simplex IRF (spline-g slot): free == raw - num_skills (per-skill shift redundancy).
    from betabern.bernstein.estimator.quad import build_quad_bernstein_estimator_spline_g
    sg = build_quad_bernstein_estimator_spline_g(degree_n=20, num_users=5, num_skills=3, num_items=10,
                                                 num_nodes=16, gen_degree=6, gen_order=2, item_weight_rank=1)
    assert count_free_params(sg) == raw_trainable_params(sg) - 3  # num_skills = 3


# ===========================================================================
# Quadrature
# ===========================================================================

def test_gauss_hermite_standard_moments():
    """Standardized log-weights normalize (sum exp = 1) and reproduce the e^{-x^2}/sqrt(pi) moments."""
    x, log_v = gauss_hermite_standard(20)
    v = log_v.exp()
    assert x.shape == (20,) and log_v.shape == (20,)
    assert abs(v.sum().item() - 1.0) < 1e-9
    assert abs((v * x).sum().item() - 0.0) < 1e-9  # odd moment
    assert abs((v * x ** 2).sum().item() - 0.5) < 1e-9  # E[x^2] = 1/2


def test_gauss_hermite_log_weights_high_order():
    """At high order the log-weights stay finite and exact where the old float32+clamp(EPS=1e-8) path
    lost the tails: the smallest log-weight sits far below log(1e-8) yet the moments still hold."""
    x, log_v = gauss_hermite_standard(128)
    v = log_v.exp()
    assert torch.isfinite(log_v).all()
    assert abs(v.sum().item() - 1.0) < 1e-9
    assert abs((v * x ** 2).sum().item() - 0.5) < 1e-9  # E[x^2] = 1/2 reproduced at order 128
    assert log_v.min().item() < -18.4  # tail far below log(EPS); previously clamped


def test_prior_quadrature_matches_normal_moments():
    """prior_quadrature reproduces the fixed Normal prior's moments E[theta]=mean, Var=std^2."""
    model = build_quad_irt_estimator(num_users=5, num_skills=3, num_items=10, num_nodes=40,
                                irt_model="2PL", prior_mean=0.7, prior_std=1.3)
    skill_ids = torch.arange(3)
    theta_q, log_v_q = model.prior.quadrature(skill_ids.size(0))
    v_q = log_v_q.exp()
    assert theta_q.shape == (3, 40) and log_v_q.shape == (3, 40)
    assert torch.allclose(v_q.sum(dim=-1), torch.ones(3), atol=1e-5)

    mean = (v_q * theta_q).sum(dim=-1)
    var = (v_q * (theta_q - mean[:, None]) ** 2).sum(dim=-1)
    assert torch.allclose(mean, torch.full((3,), 0.7), atol=1e-5)
    assert torch.allclose(var, torch.full((3,), 1.3 ** 2), atol=1e-4)


def test_is_bayesian_estimator():
    """The IRT quadrature estimator satisfies the method-agnostic BayesianEstimator contract."""
    model = build_quad_irt_estimator(num_users=5, num_skills=3, num_items=10, num_nodes=21)
    assert isinstance(model, BayesianEstimator)


# ===========================================================================
# Estimator
# ===========================================================================

def test_marginal_evidence_matches_numerical_integral():
    """log_marginal_evidence matches a fine brute-force integral of Normal prior * likelihood."""
    B, S, num_items, num_skills = 4, 3, 10, 3
    model = build_quad_irt_estimator(num_users=5, num_skills=num_skills, num_items=num_items,
                                num_nodes=40, irt_model="2PL")
    item_ids, skill_ids, answers, mask = _make_batch(B, S, num_items, num_skills)

    log_evidence = model.log_marginal_evidence(skill_ids, item_ids, answers, mask)

    mu, sigma = model.prior.get_params(skill_ids.shape[0])  # (B,)
    grid = torch.linspace(-12, 12, 6000)  # (G,) wide enough for the standard Normal prior
    grid_b = grid.unsqueeze(0).expand(B, -1)  # (B, G)

    log_lik = model.irf.log_irf(grid_b, item_ids, skill_ids)  # (B, S, 2, G)
    idx = answers[:, :, None, None].expand(-1, -1, 1, grid.size(0))
    obs = log_lik.gather(2, idx).squeeze(2).sum(dim=1)  # (B, G)

    log_prior = D.Normal(mu.unsqueeze(-1), sigma.unsqueeze(-1)).log_prob(grid)  # (B, G)
    ref = torch.log(torch.trapz((log_prior + obs).exp(), grid, dim=-1))  # (B,)

    assert torch.allclose(log_evidence, ref, atol=2e-2), \
        f"quadrature vs numerical: {log_evidence} vs {ref}"


def test_predictive_log_probs_normalized_and_shaped():
    """Predictive log-probs are a valid distribution over the 2 classes."""
    B, S, num_items, num_skills = 4, 3, 10, 3
    model = build_quad_irt_estimator(num_users=5, num_skills=num_skills, num_items=num_items,
                                num_nodes=21, irt_model="3PL")
    item_ids, skill_ids, answers, mask = _make_batch(B, S, num_items, num_skills)
    target = {"item_ids": item_ids, "skill_ids": skill_ids, "answers": answers, "mask": mask}

    for history in (None, target):
        pred = model.predictive_log_probs(history, target)
        assert pred.shape == (B, S, 2)
        assert torch.allclose(torch.logsumexp(pred, dim=-1), torch.zeros(B, S), atol=1e-5)
        assert (pred <= 1e-5).all()


def test_eap_with_no_data_equals_prior_mean():
    """With no observed responses the posterior equals the prior, so EAP == the fixed prior mean."""
    B, S, num_items, num_skills = 6, 4, 12, 4
    model = build_quad_irt_estimator(num_users=5, num_skills=num_skills, num_items=num_items,
                                num_nodes=40, irt_model="2PL", prior_mean=0.5)

    item_ids, skill_ids, answers, mask = _make_batch(B, S, num_items, num_skills)
    empty_mask = torch.zeros_like(mask)  # observe nothing
    eap = model.posterior_stats(skill_ids, item_ids, answers, empty_mask, level=None).mean

    assert torch.allclose(eap, torch.full((B,), 0.5), atol=1e-4)


def test_posterior_stats_mean_and_variance():
    """posterior_stats(level=None) returns the lean (mean, variance); no data -> prior variance, no CI."""
    B, S, num_items, num_skills = 6, 8, 12, 3
    model = build_quad_irt_estimator(num_users=5, num_skills=num_skills, num_items=num_items,
                                     num_nodes=40, irt_model="2PL", prior_std=1.0)
    item_ids, skill_ids, answers, mask = _make_batch(B, S, num_items, num_skills)

    post = model.posterior_stats(skill_ids, item_ids, answers, mask, level=None)
    assert post.mean.shape == (B,) and post.variance.shape == (B,)
    assert post.ci_low is None and post.ci_high is None  # level=None skips the interval
    assert (post.variance >= 0).all()

    # No observations -> posterior == prior -> variance == prior variance (std^2 = 1).
    var0 = model.posterior_stats(skill_ids, item_ids, answers, torch.zeros_like(mask), level=None).variance
    assert torch.allclose(var0, torch.ones(B), atol=1e-3)


def test_gradients_flow_to_items_and_prior_is_fixed():
    """The MML loss produces finite item gradients; the Normal prior is fixed (not learnable)."""
    B, S, num_items, num_skills = 8, 5, 12, 3
    model = build_quad_irt_estimator(num_users=5, num_skills=num_skills, num_items=num_items,
                                num_nodes=21, irt_model="2PL")
    item_ids, skill_ids, answers, mask = _make_batch(B, S, num_items, num_skills)

    loss = -model.log_marginal_evidence(skill_ids, item_ids, answers, mask).mean()
    loss.backward()

    item_grads = [p.grad for n, p in model.irf.irt_base.named_parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in item_grads), "No IRT item param received gradient"

    param_names = {n for n, _ in model.named_parameters()}
    assert not any("prior" in n for n in param_names)
    assert model.prior.mean.requires_grad is False and model.prior.std.requires_grad is False


def test_padding_is_ignored():
    """Masked-out (padded) positions do not affect the marginal evidence."""
    B, S, num_items, num_skills = 3, 4, 10, 2
    model = build_quad_irt_estimator(num_users=5, num_skills=num_skills, num_items=num_items,
                                num_nodes=21, irt_model="2PL")
    item_ids, skill_ids, answers, mask = _make_batch(B, S, num_items, num_skills)

    base = model.log_marginal_evidence(skill_ids, item_ids, answers, mask)

    pad_item = torch.cat([item_ids, torch.zeros(B, 1, dtype=torch.long)], dim=1)
    pad_ans = torch.cat([answers, torch.full((B, 1), -1, dtype=torch.long)], dim=1)
    pad_mask = torch.cat([mask, torch.zeros(B, 1, dtype=torch.bool)], dim=1)
    padded = model.log_marginal_evidence(skill_ids, pad_item, pad_ans, pad_mask)

    assert torch.allclose(base, padded, atol=1e-5)


def test_prequential_sums_to_marginal_evidence():
    """The static estimator run sequentially: summing the per-step prequential predictives recovers the
    batch marginal log-evidence (chain rule). Holds for both the node-eval (logistic) and basis-cache
    (Bernstein) paths, and the running posterior SD shrinks as responses accrue."""
    from betabern.bernstein.estimator.quad import build_quad_bernstein_estimator_free
    B, S, num_items, num_skills = 5, 6, 10, 2
    item_ids, skill_ids, answers, mask = _make_batch(B, S, num_items, num_skills)
    models = [
        build_quad_irt_estimator(num_users=5, num_skills=num_skills, num_items=num_items,
                                 num_nodes=40, irt_model="2PL"),  # node-eval path
        build_quad_bernstein_estimator_free(degree_n=6, num_users=5, num_skills=num_skills,
                                            num_items=num_items, num_nodes=40),  # basis-cache path
    ]
    for model in models:
        pred, mean, std = model.prequential(skill_ids, item_ids, answers, mask)
        assert pred.shape == (B, S, 2) and mean.shape == (B, S + 1) and std.shape == (B, S + 1)

        # Chain rule: sum_t log p(y_t | y_<t) over valid steps == batch log p(Y).
        obs = pred.gather(2, answers.clamp(min=0)[:, :, None]).squeeze(-1) * mask  # (B, S)
        preq_evidence = obs.sum(dim=1)
        batch_evidence = model.log_marginal_evidence(skill_ids, item_ids, answers, mask)
        assert torch.allclose(preq_evidence, batch_evidence, atol=1e-4), (preq_evidence, batch_evidence)

        # Measurement efficiency: on average the posterior SD shrinks from prior (k=0) to all items seen.
        assert std[:, -1].mean() < std[:, 0].mean()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
    print("\n✅ All IRT estimator tests passed!")
