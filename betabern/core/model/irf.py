"""Item-response function (IRF) contract and the locally-supported I-spline IRF.

``MonotoneIRF`` is the interface a :class:`~betabern.core.model.estimator.QuadratureEstimator`
depends on: a log item-response evaluation at ability nodes (:meth:`MonotoneIRF.log_irf`), plus an
*optional* cacheable basis (:meth:`MonotoneIRF.log_basis` / :meth:`MonotoneIRF.log_irf_from_basis`) for
the IRFs that are basis-decomposable. Lives in ``core`` so every estimator can be parameterized by an IRF.

- ``ISplineIRF`` (here): a freely-learned monotone IRF on a locally-supported I-spline basis.
- ``LogisticIRF`` (``betabern.irt.model``): the (1-4PL) logistic curve, *not* basis-decomposable.
- ``BernsteinIRF`` family (``betabern.bernstein.bernstein_irf``): the global Bernstein basis.

Response classes are ordered ``[incorrect, correct]`` along the class axis.
"""
import abc

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import BSpline

from betabern.core.util import LowRankEmbedding, log1mexp, raw_trainable_params

EPS = 1e-8


def ispline_basis_matrix(theta_np: np.ndarray, knots: np.ndarray, order: int, n_bases: int) -> np.ndarray:
    """I-spline values ``I_i(theta)``, shape ``(T, n_bases)``; each column monotone ``0 -> 1`` (local support).

    Each I-spline is the normalized antiderivative of an order-``order`` B-spline basis element on ``knots``
    (Ramsay 1988), normalized over the knot window ``[knots[0], knots[-1]]``. Shared by the direct I-spline
    IRF and the Bernstein spline weight-generator.

    Inputs outside the knot window are **saturated** (clamped to the window before evaluation): each column
    is ``0`` below ``knots[0]`` and ``1`` above ``knots[-1]``. This makes the basis valid on all of
    ``R`` — the constant-tail extrapolation an I-spline IRF needs to model a saturated probability when the
    latent scale is unbounded (Gauss-Hermite nodes can land far outside the window).
    """
    degree = order - 1
    lo, hi = float(knots[0]), float(knots[-1])
    # Clip in float64: a float32 theta clipped to `hi` can round to just *above* the float64 knot, which a
    # non-extrapolating BSpline maps to NaN. Double precision keeps the clipped node inside the knot span.
    th = np.clip(np.asarray(theta_np, dtype=np.float64), lo, hi)  # saturating tails: 0 below lo, 1 above hi
    cols = []
    for i in range(n_bases):  # one basis function per coefficient; integrate over the full [lo, hi] window
        coeff = np.zeros(n_bases)
        coeff[i] = 1.0
        anti = BSpline(knots, coeff, degree, extrapolate=False).antiderivative()
        a0, a1 = float(anti(lo)), float(anti(hi))
        cols.append(np.clip((anti(th) - a0) / (a1 - a0), 0.0, 1.0))
    return np.stack(cols, axis=-1)


class MonotoneIRF(nn.Module, abc.ABC):
    """Abstract monotone item-response function: ``P(correct | theta)`` non-decreasing in ``theta``.

    The required read-out is :meth:`log_irf` — the log response at a set of ability nodes. The latent
    scale is the model's own (``[0, 1]`` for Bernstein / I-spline, ``R`` for the logistic IRF). Response
    classes are ordered ``[incorrect, correct]`` along the class axis.

    Basis-decomposable IRFs (Bernstein, I-spline) additionally override :meth:`log_basis` (the node basis,
    constant at the estimator's fixed quadrature nodes) and :meth:`log_irf_from_basis`, letting the
    estimator precompute the basis once.
    """

    @abc.abstractmethod
    def log_irf(self, theta: torch.Tensor, item_ids: torch.Tensor, skill_ids: torch.Tensor) -> torch.Tensor:
        """Log item-response at ability nodes ``theta`` (B, Q) -> ``(B, S, C, Q)`` (C = 2, [incorrect, correct]).

        ``item_ids`` is ``(B, S)``; ``skill_ids`` is ``(B,)`` or ``(B, S)``.
        """

    def log_irf_from_basis(self, log_basis: torch.Tensor, item_ids: torch.Tensor,
                           skill_ids: torch.Tensor) -> torch.Tensor:
        """Log item-response from a precomputed :meth:`log_basis` -> ``(B, S, C, Q)``. Basis IRFs only."""
        raise NotImplementedError(f"{type(self).__name__} is not basis-decomposable (log_basis is None).")

    @torch.no_grad()
    def evaluate_irf(self, p: torch.Tensor, item_ids: torch.Tensor, skill_ids: torch.Tensor) -> torch.Tensor:
        """Evaluate ``P(correct)`` at shared points ``p`` (shape ``(num_points,)``) -> ``(B, num_points)``."""
        theta = p.unsqueeze(0).expand(item_ids.size(0), -1)  # (B, P)
        log_vals = self.log_irf(theta, item_ids.unsqueeze(-1), skill_ids)  # (B, 1, C, P)
        return log_vals[:, 0, 1, :].exp()  # (B, P) positive class


class ISplineIRF(MonotoneIRF):
    """Freely-learned monotone IRF on a *locally-supported* I-spline basis (Ramsay 1988).

    ``P(correct | theta) = beta_0 + sum_i alpha_i * I_i(theta)`` where the ``I_i`` are I-splines (each a
    monotone ``0 -> 1`` ramp with local support) and ``(beta_0, alpha_1, ..., alpha_m)`` are non-negative
    and sum to ``<= 1`` (a softmax with a slack class). That guarantees ``P in [0, 1]`` and monotone. Unlike
    Bernstein's global basis, each I-spline only moves a local theta-band, so sharp/threshold-like response
    shapes need far less degree (hence less variance) to represent. Quadrature-only (no Beta-Bernstein
    conjugacy), so there is no ``exact``/filter counterpart.

    The knots live on a bounded window ``domain = (lo, hi)``; outside it the I-splines saturate (constant
    tails), so the IRF is valid on the whole window's ambient scale:

    - ``domain = (0, 1)`` (default): the latent ``theta in [0, 1]``, paired with a ``Beta`` prior — the
      direct local counterpart of the Bernstein IRF (see ``ispline-irf.md``).
    - ``domain = (mean - L*std, mean + L*std)``: an ``R`` window, paired with a ``Normal`` prior and
      Gauss-Hermite quadrature — the **spline-Gaussian** measurement model (see ``spline-gaussian-irf.md``).
      Gauss-Hermite nodes beyond the window land on the saturated (constant) tails, the correct behavior
      for a probability that has flattened to its guess/slip asymptotes.

    Parameters
    ----------
    degree_n : int
        Number ``m`` of I-spline basis functions (the local-resolution knob; ~ Bernstein's ``degree + 1``).
    order : int
        Spline order (``order = polynomial degree + 1``); 2 = piecewise-linear, 3 = quadratic (default).
    domain : tuple[float, float]
        The knot window ``(lo, hi)`` on the latent scale; defaults to the unit interval ``(0.0, 1.0)``.
    """

    def __init__(self, degree_n: int, num_skills: int, num_items: int,
                 item_weight_rank: int | None = None, order: int = 3,
                 domain: tuple[float, float] = (0.0, 1.0)):
        super().__init__()

        if degree_n < order:
            raise ValueError(f"ISplineIRF needs degree_n (m={degree_n}) >= order ({order}).")
        if domain[0] >= domain[1]:
            raise ValueError(f"domain must be (lo, hi) with lo < hi, got {domain}.")
        self.m = degree_n
        self.order = order
        self.domain = (float(domain[0]), float(domain[1]))

        # Fixed knot vector on [lo, hi]: boundary knots with multiplicity `order`, uniform interior knots.
        lo, hi = self.domain
        interior = np.linspace(lo, hi, self.m - order + 2)[1:-1]
        self._knots = np.concatenate([np.full(order, lo), interior, np.full(order, hi)])  # len m + order

        # Per-item coefficients over (intercept + m I-splines + slack), pushed through a softmax simplex.
        self.skill_logits = nn.Embedding(num_skills, self.m + 2)
        self.item_logits = LowRankEmbedding(num_items, self.m + 2, rank=item_weight_rank)

    def num_free_params(self) -> int:
        """The per-skill softmax simplex (full embedding) is shift-invariant -> 1 redundant logit per
        skill; the low-rank per-item logits are counted as storage (a factor can't represent a shift)."""
        return raw_trainable_params(self) - self.skill_logits.num_embeddings

    def log_basis(self, theta: torch.Tensor) -> torch.Tensor:
        """Log I-spline basis at ``theta``, shape ``(*theta.shape, m + 1)`` (column 0 = intercept ``log 1``).

        The knots are fixed, so this is a constant of ``theta`` (no autograd through ``theta``); it is built
        once at the estimator's quadrature nodes and otherwise only used for plotting.
        """
        theta_np = theta.detach().reshape(-1).cpu().numpy()
        basis = ispline_basis_matrix(theta_np, self._knots, self.order, self.m)  # (T, m)
        basis = torch.as_tensor(basis, dtype=theta.dtype, device=theta.device).reshape(*theta.shape, self.m)
        intercept = theta.new_zeros((*theta.shape, 1))  # log(1) = 0
        return torch.cat([intercept, basis.clamp_min(EPS).log()], dim=-1)  # (*T, m + 1)

    def log_irf_from_basis(self,
                           log_basis: torch.Tensor,
                           item_ids: torch.Tensor,
                           skill_ids: torch.Tensor) -> torch.Tensor:
        """Log IRF of shape ``(B, S, 2, Q)`` from a precomputed I-spline basis ``(Q, m + 1)``."""
        if skill_ids.dim() == 1:
            skill_ids = skill_ids.unsqueeze(-1).expand_as(item_ids)  # (B, S)
        item_ids, skill_ids = item_ids.clamp(min=0), skill_ids.clamp(min=0)  # neutralize padding (-1)

        logits = self.item_logits(item_ids) + self.skill_logits(skill_ids)  # (*L, m + 2)
        log_coef = torch.log_softmax(logits, dim=-1)[..., :self.m + 1]  # drop the slack class; (B, S, m + 1)

        # log P(correct) = logsumexp_d [ log_coef_d + log_basis_{q,d} ] -> (B, S, Q).
        log_basis = log_basis if log_basis.dim() == 2 else log_basis.unsqueeze(1)  # (Q,D) or (B,1,Q,D)
        log_p_correct = torch.logsumexp(log_coef.unsqueeze(-2) + log_basis, dim=-1).clamp(max=-EPS)
        log_p_incorrect = log1mexp(log_p_correct)
        return torch.stack([log_p_incorrect, log_p_correct], dim=-2)  # (B, S, 2, Q)

    def log_irf(self, theta: torch.Tensor, item_ids: torch.Tensor, skill_ids: torch.Tensor) -> torch.Tensor:
        return self.log_irf_from_basis(self.log_basis(theta), item_ids, skill_ids)
