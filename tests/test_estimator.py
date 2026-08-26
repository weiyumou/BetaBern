"""
Test suite for the Beta-Bernstein Gauss-Jacobi MML estimator.

Validates the batched Gauss-Jacobi quadrature, the marginal-evidence computation
(against a brute-force numerical integral), and the estimator's shapes / gradients.
"""
import torch
import torch.distributions as D
from helpers import make_batch as _make_batch

from betabern.bernstein.estimator.quad import (
    build_quad_bernstein_estimator_free,
    build_quad_bernstein_estimator_irt,
)
from betabern.core.model.prior import gauss_jacobi_nodes

# ===========================================================================
# Quadrature
# ===========================================================================

def test_quadrature_weights_and_nodes():
    """Weights sum to 1 and nodes lie in [0, 1] for a fixed Beta prior."""
    Q = 16
    for alpha, beta in [(2.0, 3.0), (0.7, 1.5), (5.0, 2.0), (1.0, 1.0)]:
        theta_q, log_v_q = gauss_jacobi_nodes(alpha, beta, Q)
        v_q = log_v_q.exp()
        assert theta_q.shape == (Q,) and log_v_q.shape == (Q,)
        assert (theta_q >= 0).all() and (theta_q <= 1).all()
        assert torch.allclose(v_q.sum(), torch.tensor(1.0, dtype=v_q.dtype), atol=1e-9)


def test_quadrature_matches_closed_form_moments():
    """E[theta^k] from quadrature matches the closed-form Beta moment."""
    for alpha, beta in [(2.0, 3.0), (0.7, 1.5), (5.0, 2.0)]:
        theta_q, log_v_q = gauss_jacobi_nodes(alpha, beta, 12)
        v_q = log_v_q.exp()
        for k in [1, 2, 3, 5]:
            quad = (v_q * theta_q ** k).sum()
            closed = 1.0
            for r in range(k):
                closed *= (alpha + r) / (alpha + beta + r)
            assert torch.allclose(quad, torch.tensor(closed, dtype=quad.dtype), atol=1e-9), \
                f"moment k={k} mismatch for Beta({alpha},{beta})"


# ===========================================================================
# Estimator
# ===========================================================================

def test_marginal_evidence_matches_numerical_integral():
    """log_marginal_evidence matches a fine brute-force integral of prior * likelihood."""
    B, S, num_items, num_skills, degree = 4, 3, 10, 3, 4
    model = build_quad_bernstein_estimator_irt(
        degree_n=degree, num_users=5, num_skills=num_skills, num_items=num_items,
        num_nodes=40, irt_model="2PL", item_weight_rank=1,
    )
    item_ids, skill_ids, answers, mask = _make_batch(B, S, num_items, num_skills)

    log_evidence = model.log_marginal_evidence(skill_ids, item_ids, answers, mask)

    # Brute-force reference integral over a dense theta grid.
    alpha, beta = model.prior.get_params(skill_ids.shape[0])  # (B,)
    grid = torch.linspace(1e-4, 1 - 1e-4, 4000)  # (G,)
    grid_b = grid.unsqueeze(0).expand(B, -1)  # (B, G)

    log_lik = model.irf.log_irf(grid_b, item_ids, skill_ids)  # (B, S, 2, G)
    idx = answers[:, :, None, None].expand(-1, -1, 1, grid.size(0))
    obs = log_lik.gather(2, idx).squeeze(2).sum(dim=1)  # (B, G)

    log_prior = D.Beta(alpha.unsqueeze(-1), beta.unsqueeze(-1)).log_prob(grid)  # (B, G)
    integrand = (log_prior + obs).exp()  # (B, G)
    ref = torch.log(torch.trapz(integrand, grid, dim=-1))  # (B,)

    assert torch.allclose(log_evidence, ref, atol=2e-2), \
        f"quadrature vs numerical: {log_evidence} vs {ref}"


def test_predictive_log_probs_normalized_and_shaped():
    """Predictive log-probs are a valid distribution over the 2 classes."""
    B, S, num_items, num_skills = 4, 3, 10, 3
    model = build_quad_bernstein_estimator_free(
        degree_n=5, num_users=5, num_skills=num_skills, num_items=num_items, num_nodes=24,
    )
    item_ids, skill_ids, answers, mask = _make_batch(B, S, num_items, num_skills)
    target = {"item_ids": item_ids, "skill_ids": skill_ids, "answers": answers, "mask": mask}

    # Prior predictive (no history) and posterior predictive (with history).
    for history in (None, target):
        pred = model.predictive_log_probs(history, target)
        assert pred.shape == (B, S, 2)
        assert torch.allclose(torch.logsumexp(pred, dim=-1), torch.zeros(B, S), atol=1e-5)
        assert (pred <= 1e-5).all()


def test_eap_in_unit_interval():
    """Expected ability lies in [0, 1]."""
    B, S, num_items, num_skills = 6, 4, 12, 3
    model = build_quad_bernstein_estimator_irt(
        degree_n=4, num_users=5, num_skills=num_skills, num_items=num_items,
        num_nodes=30, irt_model="2PL", item_weight_rank=1,
    )
    item_ids, skill_ids, answers, mask = _make_batch(B, S, num_items, num_skills)
    eap = model.posterior_stats(skill_ids, item_ids, answers, mask, level=None).mean
    assert eap.shape == (B,)
    assert (eap >= 0).all() and (eap <= 1).all()


def test_gradients_flow_to_items_and_prior_is_fixed():
    """The MML loss produces finite item gradients; the prior is a fixed buffer (not learnable)."""
    B, S, num_items, num_skills = 8, 5, 12, 3
    model = build_quad_bernstein_estimator_irt(
        degree_n=4, num_users=5, num_skills=num_skills, num_items=num_items,
        num_nodes=30, irt_model="2PL", item_weight_rank=1,
    )
    item_ids, skill_ids, answers, mask = _make_batch(B, S, num_items, num_skills)

    loss = -model.log_marginal_evidence(skill_ids, item_ids, answers, mask).mean()
    loss.backward()

    missing = [n for n, p in model.irf.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"No gradient for IRF params: {missing}"

    # No learnable prior parameters anywhere in the model.
    param_names = {n for n, _ in model.named_parameters()}
    assert not any("prior" in n or "mu_raw" in n or "nu_raw" in n for n in param_names)
    assert model.prior.alpha.requires_grad is False


def test_padding_is_ignored():
    """Masked-out (padded) positions do not affect the marginal evidence."""
    B, S, num_items, num_skills = 3, 4, 10, 2
    model = build_quad_bernstein_estimator_free(
        degree_n=4, num_users=5, num_skills=num_skills, num_items=num_items, num_nodes=24,
    )
    item_ids, skill_ids, answers, mask = _make_batch(B, S, num_items, num_skills)

    base = model.log_marginal_evidence(skill_ids, item_ids, answers, mask)

    # Append a padded column; mask=False, answers=-1, arbitrary item -> evidence unchanged.
    pad_item = torch.cat([item_ids, torch.zeros(B, 1, dtype=torch.long)], dim=1)
    pad_ans = torch.cat([answers, torch.full((B, 1), -1, dtype=torch.long)], dim=1)
    pad_mask = torch.cat([mask, torch.zeros(B, 1, dtype=torch.bool)], dim=1)
    padded = model.log_marginal_evidence(skill_ids, pad_item, pad_ans, pad_mask)

    assert torch.allclose(base, padded, atol=1e-5)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
    print("\n✅ All estimator tests passed!")
