"""
Test suite for the exact Bernstein-basis MML estimator.

Validates the Bernstein-product primitive, the cancellation-free marginal evidence (against a
brute-force integral and against high-Q Gauss-Jacobi), the posterior-predictive / EAP, and shows
the payoff: exact for all N where small-Q quadrature deviates.
"""
import torch
import torch.distributions as D
from helpers import make_batch as _make_batch

from betabern.bernstein.bernstein_irf import LogisticBernsteinIRF
from betabern.bernstein.estimator.exact import (
    ExactBernsteinEstimator,
    build_exact_bernstein_estimator_free,
    build_exact_bernstein_estimator_irt,
    log_bernstein_product,
)
from betabern.core.model.estimator import BayesianEstimator, QuadratureEstimator
from betabern.core.model.prior import BetaJacobiPrior, BetaPrior
from betabern.core.util import log_binom_coeff
from betabern.irt.model import IRTBase


def _bernstein_eval(coeffs: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Evaluate sum_k coeffs_k b_{k,D}(theta) at theta in (0, 1)."""
    deg = coeffs.shape[0] - 1
    k = torch.arange(deg + 1, dtype=torch.float64)
    log_c = log_binom_coeff(torch.tensor(float(deg)), k)
    th = theta.double()[:, None]
    basis = (log_c[None, :] + k[None, :] * torch.log(th) + (deg - k)[None, :] * torch.log1p(-th)).exp()
    return basis @ coeffs.double()


# ===========================================================================
# Bernstein product primitive
# ===========================================================================

def test_log_bernstein_product_matches_direct_multiplication():
    """log_bernstein_product reproduces the pointwise product of two Bernstein polynomials."""
    g = torch.Generator().manual_seed(0)
    d, m = 3, 4
    a = 0.1 + 0.8 * torch.rand(d + 1, generator=g, dtype=torch.float64)
    f = 0.1 + 0.8 * torch.rand(m + 1, generator=g, dtype=torch.float64)

    log_prod = log_bernstein_product(a.log()[None, :], d, f.log()[None, :], m)[0]
    prod_coeffs = log_prod.exp()
    assert prod_coeffs.shape == (d + m + 1,)

    theta = torch.linspace(0.02, 0.98, 50)
    lhs = _bernstein_eval(prod_coeffs, theta)
    rhs = _bernstein_eval(a, theta) * _bernstein_eval(f, theta)
    assert torch.allclose(lhs, rhs, atol=1e-9)


# ===========================================================================
# Estimator
# ===========================================================================

def test_is_bayesian_estimator():
    model = build_exact_bernstein_estimator_irt(degree_n=4, num_users=5, num_skills=3, num_items=10)
    assert isinstance(model, BayesianEstimator)


def test_marginal_evidence_matches_numerical_integral():
    """Exact log-evidence matches a fine brute-force integral of Beta * prod p_i."""
    B, S, num_items, num_skills, degree = 4, 4, 10, 3, 4
    model = build_exact_bernstein_estimator_irt(
        degree_n=degree, num_users=5, num_skills=num_skills, num_items=num_items,
        irt_model="2PL", item_weight_rank=1,
    )
    item_ids, skill_ids, answers, mask = _make_batch(B, S, num_items, num_skills)

    log_evidence = model.log_marginal_evidence(skill_ids, item_ids, answers, mask)

    alpha, beta = model.get_prior(skill_ids)
    grid = torch.linspace(1e-4, 1 - 1e-4, 6000)
    grid_b = grid.unsqueeze(0).expand(B, -1)
    log_lik = model.irf.log_irf(grid_b, item_ids, skill_ids)  # (B, S, 2, G)
    idx = answers[:, :, None, None].expand(-1, -1, 1, grid.size(0))
    obs = log_lik.gather(2, idx).squeeze(2).sum(dim=1)  # (B, G)
    log_prior = D.Beta(alpha.unsqueeze(-1), beta.unsqueeze(-1)).log_prob(grid)
    ref = torch.log(torch.trapz((log_prior + obs).exp(), grid, dim=-1))

    assert torch.allclose(log_evidence, ref, atol=1e-2), f"{log_evidence} vs {ref}"


def test_matches_high_q_gauss_jacobi():
    """Exact == Gauss-Jacobi at high Q (same IRF + prior) on short sequences."""
    num_skills = 3
    irt_base = IRTBase(num_users=5, num_skills=num_skills, num_items=10, irt_model="2PL", enable_item_params=True)
    irf = LogisticBernsteinIRF(degree_n=4, irt_base=irt_base)
    exact = ExactBernsteinEstimator(BetaPrior(), irf)
    gj = QuadratureEstimator(prior=BetaJacobiPrior(1.0, 1.0, 60), irf=irf)  # shares the same irf; same prior

    item_ids, skill_ids, answers, mask = _make_batch(B=5, S=3, num_items=10, num_skills=num_skills)
    e = exact.log_marginal_evidence(skill_ids, item_ids, answers, mask)
    q = gj.log_marginal_evidence(skill_ids, item_ids, answers, mask)
    assert torch.allclose(e, q, atol=1e-3), f"exact {e} vs GJ {q}"


def test_exact_beats_small_q_quadrature_on_long_sequences():
    """On long sequences, exact matches the integral while small-Q Gauss-Jacobi deviates."""
    num_skills, S, degree = 2, 24, 5
    irt_base = IRTBase(num_users=5, num_skills=num_skills, num_items=15, irt_model="2PL", enable_item_params=True)
    irf = LogisticBernsteinIRF(degree_n=degree, irt_base=irt_base)
    exact = ExactBernsteinEstimator(BetaPrior(), irf)
    gj_small = QuadratureEstimator(prior=BetaJacobiPrior(1.0, 1.0, 8), irf=irf)  # Q=8 << (S*m+1)/2 = 60.5

    item_ids, skill_ids, answers, mask = _make_batch(B=4, S=S, num_items=15, num_skills=num_skills)

    # Brute-force reference.
    alpha, beta = exact.get_prior(skill_ids)
    grid = torch.linspace(1e-4, 1 - 1e-4, 8000)
    grid_b = grid.unsqueeze(0).expand(item_ids.size(0), -1)
    log_lik = irf.log_irf(grid_b, item_ids, skill_ids)
    idx = answers[:, :, None, None].expand(-1, -1, 1, grid.size(0))
    obs = log_lik.gather(2, idx).squeeze(2).sum(dim=1)
    log_prior = D.Beta(alpha.unsqueeze(-1), beta.unsqueeze(-1)).log_prob(grid)
    ref = torch.log(torch.trapz((log_prior + obs).exp(), grid, dim=-1))

    e = exact.log_marginal_evidence(skill_ids, item_ids, answers, mask)
    q = gj_small.log_marginal_evidence(skill_ids, item_ids, answers, mask)

    exact_err = (e - ref).abs().max()
    gj_err = (q - ref).abs().max()
    assert exact_err < 1e-2, f"exact deviated from integral: {exact_err}"
    assert gj_err > exact_err, f"expected small-Q quadrature to be worse: gj={gj_err}, exact={exact_err}"


def test_uniform_prior_collapses_to_coefficient_average():
    """With the default uniform Beta(1,1) prior, log p(Y) == logsumexp(log C_k) − log(D+1)."""
    B, S, num_items, num_skills = 4, 4, 10, 3
    model = build_exact_bernstein_estimator_free(degree_n=4, num_users=5, num_skills=num_skills,
                                                 num_items=num_items)  # default prior=(1,1)
    item_ids, skill_ids, answers, mask = _make_batch(B, S, num_items, num_skills)

    log_c, deg = model._fold(item_ids, answers, mask, skill_ids)
    collapsed = torch.logsumexp(log_c, dim=-1) - torch.log(torch.tensor(float(deg + 1)))
    evidence = model.log_marginal_evidence(skill_ids, item_ids, answers, mask)
    assert torch.allclose(evidence, collapsed.to(evidence.dtype), atol=1e-5)


def test_predictive_log_probs_normalized_and_shaped():
    B, S, num_items, num_skills = 4, 3, 10, 3
    model = build_exact_bernstein_estimator_free(degree_n=5, num_users=5, num_skills=num_skills, num_items=num_items)
    item_ids, skill_ids, answers, mask = _make_batch(B, S, num_items, num_skills)
    target = {"item_ids": item_ids, "skill_ids": skill_ids, "answers": answers, "mask": mask}

    for history in (None, target):
        pred = model.predictive_log_probs(history, target)
        assert pred.shape == (B, S, 2)
        assert torch.allclose(torch.logsumexp(pred, dim=-1), torch.zeros(B, S), atol=1e-5)
        assert (pred <= 1e-5).all()


def test_eap_in_unit_interval():
    B, S, num_items, num_skills = 6, 4, 12, 3
    model = build_exact_bernstein_estimator_irt(degree_n=4, num_users=5, num_skills=num_skills,
                                                num_items=num_items, irt_model="2PL", item_weight_rank=1)
    item_ids, skill_ids, answers, mask = _make_batch(B, S, num_items, num_skills)
    eap = model.posterior_stats(skill_ids, item_ids, answers, mask, level=None).mean
    assert eap.shape == (B,)
    assert (eap >= 0).all() and (eap <= 1).all()


def test_gradients_flow_to_items_and_prior_is_fixed():
    B, S, num_items, num_skills = 8, 5, 12, 3
    model = build_exact_bernstein_estimator_irt(degree_n=4, num_users=5, num_skills=num_skills,
                                                num_items=num_items, irt_model="2PL", item_weight_rank=1)
    item_ids, skill_ids, answers, mask = _make_batch(B, S, num_items, num_skills)

    loss = -model.log_marginal_evidence(skill_ids, item_ids, answers, mask).mean()
    loss.backward()

    missing = [n for n, p in model.irf.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"No gradient for IRF params: {missing}"

    param_names = {n for n, _ in model.named_parameters()}
    assert not any("prior" in n or "mu_raw" in n or "nu_raw" in n for n in param_names)
    assert model.prior.alpha.requires_grad is False


def test_padding_is_ignored():
    B, S, num_items, num_skills = 3, 4, 10, 2
    model = build_exact_bernstein_estimator_free(degree_n=4, num_users=5, num_skills=num_skills, num_items=num_items)
    item_ids, skill_ids, answers, mask = _make_batch(B, S, num_items, num_skills)

    base = model.log_marginal_evidence(skill_ids, item_ids, answers, mask)

    pad_item = torch.cat([item_ids, torch.zeros(B, 1, dtype=torch.long)], dim=1)
    pad_ans = torch.cat([answers, torch.full((B, 1), -1, dtype=torch.long)], dim=1)
    pad_mask = torch.cat([mask, torch.zeros(B, 1, dtype=torch.bool)], dim=1)
    padded = model.log_marginal_evidence(skill_ids, pad_item, pad_ans, pad_mask)

    assert torch.allclose(base, padded, atol=1e-4)


def test_prequential_sums_to_marginal_evidence():
    """Exact prequential: per-step predictives sum to the batch marginal log-evidence (chain rule), the
    returned shapes match the contract, and posterior SD shrinks as responses accrue."""
    B, S, num_items, num_skills = 5, 6, 10, 2
    model = build_exact_bernstein_estimator_irt(degree_n=5, num_users=5, num_skills=num_skills,
                                                num_items=num_items, irt_model="2PL", item_weight_rank=1)
    item_ids, skill_ids, answers, mask = _make_batch(B, S, num_items, num_skills)

    pred, mean, std = model.prequential(skill_ids, item_ids, answers, mask)
    assert pred.shape == (B, S, 2) and mean.shape == (B, S + 1) and std.shape == (B, S + 1)

    obs = pred.gather(2, answers.clamp(min=0)[:, :, None]).squeeze(-1) * mask  # (B, S)
    preq_evidence = obs.sum(dim=1)
    batch_evidence = model.log_marginal_evidence(skill_ids, item_ids, answers, mask)
    assert torch.allclose(preq_evidence, batch_evidence, atol=1e-4), (preq_evidence, batch_evidence)

    assert std[:, -1].mean() < std[:, 0].mean()  # measurement efficiency: uncertainty shrinks


def test_prequential_matches_predictive_log_probs_prefixwise():
    """Each prequential step equals predictive_log_probs conditioned on the explicit prefix history."""
    B, S, num_items, num_skills = 4, 5, 10, 2
    model = build_exact_bernstein_estimator_free(degree_n=6, num_users=5, num_skills=num_skills, num_items=num_items)
    item_ids, skill_ids, answers, mask = _make_batch(B, S, num_items, num_skills)

    pred, _, _ = model.prequential(skill_ids, item_ids, answers, mask)
    for t in range(S):
        history = None if t == 0 else {"item_ids": item_ids[:, :t], "answers": answers[:, :t],
                                       "mask": mask[:, :t], "skill_ids": skill_ids}
        target = {"item_ids": item_ids[:, t:t + 1], "skill_ids": skill_ids,
                  "answers": answers[:, t:t + 1], "mask": mask[:, t:t + 1]}
        expected = model.predictive_log_probs(history, target)[:, 0, :]  # (B, 2)
        assert torch.allclose(pred[:, t, :], expected, atol=1e-5), (t, pred[:, t, :], expected)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
    print("\n✅ All exact estimator tests passed!")
