"""Dependency-free fitting + simulated-data assembly for standalone measurement analyses
(:mod:`betabern.benchmark.measurement`, the paper's AI-mediated simulation study).

The main benchmark fits models via Lightning with early stopping (``runner.py``). The sequential
measurement-efficiency curve and the posterior-coverage study don't need that machinery — they just need
fitted item parameters and tensors to read out — so these helpers give a plain full-batch MML fit and
assemble the simulator's output into the ``(skill, item, answers, mask)`` batch the estimators consume.
"""
import numpy as np
import pandas as pd
import torch

from betabern.core.data.simulate_irt import (
    create_items,
    create_students,
    get_ground_truth_prob,
)


def fit_mml(model, skill_ids: torch.Tensor, item_ids: torch.Tensor, answers: torch.Tensor,
            mask: torch.Tensor, *, epochs: int = 300, lr: float = 0.05):
    """Full-batch marginal-maximum-likelihood fit of a ``BayesianEstimator``'s item parameters (the fixed
    quadrature prior has no free parameters). Mutates and returns ``model``. Adds the IRF's optional
    ``weight_penalty`` (e.g. the bounded model's soft-monotonicity prior) to the loss, matching the Lightning
    task's objective."""
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    irf = getattr(model, "irf", None)  # optional IRF-level regularizer (soft-monotonicity prior on the weights)
    has_pen = irf is not None and hasattr(irf, "weight_penalty")
    for _ in range(epochs):
        opt.zero_grad()
        loss = -model.log_marginal_evidence(skill_ids, item_ids, answers, mask).mean()
        if has_pen:
            loss = loss + irf.weight_penalty()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def set_oracle_items(model, a_true: torch.Tensor, b_true: torch.Tensor):
    """Inject the simulator's *true* 2PL item parameters into a parametric (logistic-direct or
    Bernstein-bridged) estimator's ``IRTBase`` — no fitting — so a coverage study isolates the θ-posterior
    calibration from item-estimation error. Both IRF types expose ``.irf.irt_base``; since
    ``a = softplus(item_a_raw + skill_a_raw)`` and ``b = item_b + skill_b``, we load all of a/b into the
    per-item terms and zero the per-skill terms. Requires item params enabled (``item_weight_rank`` set).
    """
    base = model.irf.irt_base
    a = a_true.to(torch.float32)
    inv_softplus_a = a + torch.log(-torch.expm1(-a))  # tensor-safe inverse softplus: softplus(this) = a
    base.skill_a_raw.weight.zero_()
    base.skill_b.weight.zero_()
    base.item_a_raw.weight.copy_(inv_softplus_a.view(-1, 1))
    base.item_b.weight.copy_(b_true.to(torch.float32).view(-1, 1))
    return model


def _suffix_int(s: pd.Series) -> np.ndarray:
    """The trailing integer of the simulator's ``'user-3'`` / ``'item-7'`` ids (contiguous, 0-based)."""
    return s.str.rsplit("-", n=1).str[-1].astype(int).to_numpy()


def padded_batch_from_log(log_df: pd.DataFrame) -> tuple[torch.Tensor, ...]:
    """Assemble a single-skill interaction log into a padded ``(skill, item, answers, mask)`` batch plus
    per-student ``true_theta`` (constant per student for a static trait). Ids are the simulator's native
    contiguous indices, so a model built with the matching ``num_users`` / ``num_items`` lines up.

    :return: ``(skill, item, answers, mask, true_theta)`` — first four ``(B, S)``, last ``(B,)``.
    """
    u, it = _suffix_int(log_df["user_id"]), _suffix_int(log_df["item_id"])
    y, th = log_df["is_correct"].to_numpy(), log_df["user_theta"].to_numpy()
    B, S = int(u.max()) + 1, int(np.bincount(u).max())
    item = torch.zeros(B, S, dtype=torch.long)
    answers = torch.zeros(B, S, dtype=torch.long)
    mask = torch.zeros(B, S, dtype=torch.bool)
    true_theta = torch.zeros(B)
    fill = np.zeros(B, dtype=int)
    for k in range(len(u)):
        b, j = u[k], fill[u[k]]
        item[b, j], answers[b, j], mask[b, j], true_theta[b] = it[k], y[k], True, th[k]
        fill[b] += 1
    return torch.zeros(B, S, dtype=torch.long), item, answers, mask, true_theta


def complete_response_matrix(num_students: int, num_items: int, *, irt_model: str = "2PL",
                             seed: int = 0) -> tuple[torch.Tensor, ...]:
    """A fully-observed ``students x items`` response matrix with known abilities — every student answers
    every item exactly once. Feeds the sequential curve, which permutes each complete response vector.

    :return: ``(skill, item, answers, true_theta)`` — first three ``(B, I)``, last ``(B,)``.
    """
    ss, si, sd = np.random.SeedSequence(seed).spawn(3)
    theta = np.stack(create_students(num_students, 1, random_state=ss)["thetas"].to_numpy())[:, 0]  # (B,)
    items = create_items(num_items, 1, irt_model=irt_model, random_state=si)
    a, b, c, s = items[["a_irt", "b_irt", "c_guess", "s_slip"]].to_numpy().T
    p = get_ground_truth_prob(theta[:, None], a[None, :], b[None, :], c[None, :], s[None, :])  # (B, I)
    y = np.random.default_rng(sd).binomial(1, p)
    B, n_items = y.shape
    return (torch.zeros(B, n_items, dtype=torch.long), torch.arange(n_items).expand(B, n_items).contiguous(),
            torch.tensor(y, dtype=torch.long), torch.tensor(theta, dtype=torch.float32))
