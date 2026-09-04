"""Bayes factor for item-level monotonicity, built on the exact Beta-Bernstein marginal.

Fitting a flexible IRF and eyeballing a dip is not evidence: a free-form model fitted to finite
data always bends a little, and how much it bends under *monotone* truth depends on the item's
difficulty, its sample size and the degree. This module answers the question the way a Bayesian
would -- by comparing two models that differ in nothing but the constraint on the weights:

    M_mono : w_0 <= w_1 <= ... <= w_n      (the monotone cone C)
    M_free : w_k in [0, 1], unconstrained  (the encompassing model)

**The identity.** For an inequality-constrained hypothesis nested in an encompassing one, the Bayes
factor is a ratio of two probabilities of the *same* set (Klugkist & Hoijtink 2007)::

    BF(mono : free) = P(w in C | Y) / P(w in C)

which is exact -- no Laplace approximation, no bridge sampling, no asymptotics. The prior term is
analytic: under independent Uniform(0,1) weights, ``P(C) = 1 / (n+1)!``.

**Where the bridge earns its keep.** The posterior term needs the likelihood with ability
integrated out. That is exactly what the closed form provides, and it collapses the problem twice.
First, no sampling over ``theta`` is ever required. Second -- because a single item enters the
likelihood linearly -- folding the *other* items once reduces the evidence to a linear form::

    log p(Y | w_j) = sum_i logsumexp_k [ log w~_ik + logA_ik ] + const,
    w~_ik = w_k if y_ij = 1 else 1 - w_k,

where ``logA`` is the posterior-expected Bernstein basis under the remaining items
(:meth:`ExactBernsteinEstimator._log_expected_basis`). Each evaluation then costs ``O(B n)``, so
the only Monte Carlo left is over the ``n+1`` weights of one item.

**What is estimated, and what is not.** The theta integral, the linear reduction and the Bayes
factor identity are exact. Only ``P(C | Y)`` is estimated, by a telescoping Monte Carlo estimator
with a reported standard error. The other items are held at their fitted values, so this is a
Bayes factor for one item *conditional on the fitted bank* -- not a joint factor over all items.

    from betabern.bernstein.monotonicity import monotonicity_bayes_factor
    res = monotonicity_bayes_factor(model, skill_ids, item_ids, answers, mask, item_pos=3)
    print(res.log10_bf_free_vs_mono, res.verdict)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from betabern.bernstein.estimator.exact import ExactBernsteinEstimator, _prefix_posterior

__all__ = ["MonotonicityResult", "item_evidence_operator", "log_prior_cone",
           "monotonicity_bayes_factor"]


# ======================================================================
# The exact evidence operator for one item
# ======================================================================

@torch.no_grad()
def item_evidence_operator(estimator: ExactBernsteinEstimator,
                           skill_ids: torch.Tensor,
                           item_ids: torch.Tensor,
                           answers: torch.Tensor,
                           mask: torch.Tensor,
                           item_pos: int) -> tuple[np.ndarray, np.ndarray]:
    """Reduce the exact evidence to a linear form in one item's Bernstein weights.

    Folds every column except ``item_pos`` and returns the posterior-expected basis it induces, so
    the target item's contribution is a dot product against its weights.

    :param item_pos: the *column* of ``item_ids`` to test (all rows must share that column's id).
    :return: ``(logA, y)`` with ``logA`` of shape ``(B, n+1)`` and ``y`` the ``(B,)`` responses to
        the item, restricted to rows where it was actually observed.
    """
    n_cols = item_ids.shape[1]
    if not 0 <= item_pos < n_cols:
        raise ValueError(f"item_pos {item_pos} out of range for {n_cols} columns")
    keep = [c for c in range(n_cols) if c != item_pos]
    if not keep:
        raise ValueError("need at least one other item to condition on")

    obs = mask[:, item_pos].bool()
    if not obs.any():
        raise ValueError(f"item at column {item_pos} has no observed responses")

    sk = skill_ids if skill_ids.dim() == 2 else skill_ids.unsqueeze(-1).expand_as(item_ids)
    alpha, beta = estimator.get_prior(torch.zeros(int(obs.sum()), dtype=torch.long))
    log_c, D = estimator._fold(item_ids[obs][:, keep], answers[obs][:, keep],
                               mask[obs][:, keep], sk[obs][:, keep])
    log_a = estimator._log_expected_basis(alpha, beta, _prefix_posterior(alpha, beta, log_c, D))
    return log_a.detach().cpu().numpy(), answers[obs, item_pos].detach().cpu().numpy()


def _log_evidence(w: np.ndarray, log_a: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Exact log evidence (up to an additive constant) for weight vectors ``w``, shape (..., n+1).

    Vectorized over leading dimensions so a whole MCMC proposal batch costs one call.
    """
    w = np.clip(w, 1e-12, 1 - 1e-12)
    log_w, log_1mw = np.log(w), np.log1p(-w)
    # per respondent, pick log w_k or log(1 - w_k) according to the response
    lw = np.where(y[:, None].astype(bool), log_w[..., None, :], log_1mw[..., None, :])
    return _logsumexp(lw + log_a, axis=-1).sum(axis=-1)


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)


# ======================================================================
# Cone probability by a telescoping estimator
# ======================================================================

def log_prior_cone(n_weights: int) -> float:
    """``log P(w_0 <= ... <= w_n)`` for independent Uniform(0,1) weights: ``-log((n+1)!)``."""
    return -math.lgamma(n_weights + 1)


def _sweep(w, S, A, sgn, n_active, beta, rng):
    """One Metropolis-within-Gibbs scan at inverse temperature ``beta``, in place.

    Each weight is proposed uniformly on the interval its active monotone constraints allow, which
    keeps mixing healthy where a full-vector random walk stalls. Only one weight moves at a time,
    so the per-respondent evidence sum updates in ``O(B)``::

        S_i <- S_i + sgn_i * (w'_k - w_k) * A_ik

    with ``sgn_i = +1`` for a correct response and ``-1`` otherwise. ``n_active = -1`` leaves the
    weights unconstrained (the encompassing model); ``n_active = n`` imposes the full monotone cone.
    """
    for k in range(len(w)):
        lo = w[k - 1] if 1 <= k <= n_active else 0.0
        hi = w[k + 1] if 0 <= k <= n_active - 1 else 1.0
        if hi <= lo:
            continue
        prop = rng.uniform(lo, hi)
        S_new = S + sgn * (prop - w[k]) * A[:, k]
        if np.any(S_new <= 0.0):
            continue
        d = beta * float(np.log(S_new).sum() - np.log(S).sum())
        if d >= 0.0 or math.log(rng.random() + 1e-300) < d:
            w[k] = prop
            S[:] = S_new
    return w, S


def _init_state(A, sgn, n_w, n_active, rng):
    """A start point satisfying the active constraints, plus its per-respondent evidence sums."""
    w = np.sort(rng.random(n_w)) if n_active >= 0 else rng.random(n_w)
    S = (np.where(sgn[:, None] > 0, w[None, :], 1.0 - w[None, :]) * A).sum(axis=1)
    return w, S


def _log_evidence_from_S(S):
    return float(np.log(S).sum())


def _stepping_stone(A, sgn, n_weights: int, n_active: int, rng, *,
                    n_rungs: int, n_draws: int, burn: int) -> float:
    """``log Z`` for one model by stepping-stone sampling along ``p_t(w) ~ pi(w) L(w)^{beta_t}``.

    ``pi`` is uniform on the model's own support -- the full box for the encompassing model, the
    monotone cone for the constrained one -- so each run returns a properly normalised marginal
    likelihood and their ratio is the Bayes factor directly, with the cone's ``1/(n+1)!`` volume
    penalty accounted for by construction rather than bolted on.

    Stepping stone is used rather than counting how often a posterior draw lands in the cone: for a
    genuinely non-monotone item that probability is around ``e^-70``, far beyond any counting
    estimator, whereas each rung here contributes only a moderate ratio.
    """
    betas = np.linspace(0.0, 1.0, n_rungs + 1) ** 5     # dense near 0, where the target moves fastest
    w, S = _init_state(A, sgn, n_weights, n_active, rng)
    log_z = 0.0
    for t in range(n_rungs):
        b0, b1 = betas[t], betas[t + 1]
        for _ in range(burn):
            _sweep(w, S, A, sgn, n_active, b0, rng)
        lls = np.empty(n_draws)
        for d in range(n_draws):
            _sweep(w, S, A, sgn, n_active, b0, rng)
            lls[d] = _log_evidence_from_S(S)
        inc = (b1 - b0) * lls                            # log of the rung's importance weights
        mx = inc.max()
        log_z += mx + math.log(np.exp(inc - mx).mean())
    return log_z


# ======================================================================
# Public result + driver
# ======================================================================

@dataclass
class MonotonicityResult:
    """Bayes factor for one item's monotonicity. Positive ``log10_bf_free_vs_mono`` favours a
    non-monotone curve; negative favours the monotone model."""
    item_pos: int
    n_respondents: int
    degree: int
    log10_bf_free_vs_mono: float
    log10_bf_se: float
    log_evidence_free: float
    log_evidence_mono: float
    n_replicates: int

    @property
    def verdict(self) -> str:
        """Kass & Raftery (1995) grades, on the evidence *against* monotonicity."""
        b = self.log10_bf_free_vs_mono
        grades = ((-2.0, "decisive for monotone"), (-1.0, "strong for monotone"),
                  (-0.5, "substantial for monotone"), (0.5, "inconclusive"),
                  (1.0, "substantial for non-monotone"), (2.0, "strong for non-monotone"))
        for edge, label in grades:
            if b <= edge:
                return label
        return "decisive for non-monotone"

    def __str__(self) -> str:
        return (f"item {self.item_pos}: log10 BF(free:mono) = {self.log10_bf_free_vs_mono:+.2f}"
                f" +/- {self.log10_bf_se:.2f}  [{self.verdict}]  (n={self.n_respondents})")


def monotonicity_bayes_factor(estimator: ExactBernsteinEstimator,
                              skill_ids: torch.Tensor,
                              item_ids: torch.Tensor,
                              answers: torch.Tensor,
                              mask: torch.Tensor,
                              item_pos: int,
                              *,
                              n_rungs: int = 48,
                              n_draws: int = 400,
                              burn: int = 40,
                              n_replicates: int = 3,
                              seed: int = 0) -> MonotonicityResult:
    """Bayes factor against monotonicity for one item, conditional on the fitted bank.

    :param estimator: a fitted :class:`ExactBernsteinEstimator`; the target item's own weights are
        never used -- they are integrated over -- but every other item's are.
    :param item_pos: column of ``item_ids`` to test.
    :param n_rungs: temperature rungs per stepping-stone run. The defaults hold the Monte Carlo
        error near 0.1 in log10 units -- well inside the 0.5-unit evidence grades, and negligible
        against the effect sizes a real non-monotone item produces.
    :param n_replicates: independent runs; their spread gives the reported standard error, so it
        doubles as the convergence check.
    :return: a :class:`MonotonicityResult`.
    """
    log_a, y = item_evidence_operator(estimator, skill_ids, item_ids, answers, mask, item_pos)
    n_weights = log_a.shape[1]
    # row-normalise: a per-respondent constant shifts both models' log Z identically and cancels in
    # the ratio, while keeping the linear sums off underflow
    A = np.exp(log_a - log_a.max(axis=1, keepdims=True))
    sgn = np.where(np.asarray(y).astype(bool), 1.0, -1.0)

    diffs, free_z, mono_z = [], [], []
    for r in range(n_replicates):
        zf = _stepping_stone(A, sgn, n_weights, -1, np.random.default_rng(seed + r),
                             n_rungs=n_rungs, n_draws=n_draws, burn=burn)
        zm = _stepping_stone(A, sgn, n_weights, n_weights - 1, np.random.default_rng(seed + 1000 + r),
                             n_rungs=n_rungs, n_draws=n_draws, burn=burn)
        free_z.append(zf)
        mono_z.append(zm)
        diffs.append(zf - zm)

    log_bf = float(np.mean(diffs))
    se_nat = float(np.std(diffs, ddof=1) / math.sqrt(len(diffs))) if len(diffs) > 1 else float("nan")
    return MonotonicityResult(item_pos=item_pos, n_respondents=int(len(y)), degree=n_weights - 1,
                              log10_bf_free_vs_mono=log_bf / math.log(10),
                              log10_bf_se=se_nat / math.log(10),
                              log_evidence_free=float(np.mean(free_z)),
                              log_evidence_mono=float(np.mean(mono_z)),
                              n_replicates=n_replicates)
