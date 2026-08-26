"""Posterior summary over a model's latent ability — the shared read-out of the estimator families.

Lives in ``core`` (not ``bernstein``) because it is the return type of the ``BayesianEstimator``
contract's :meth:`posterior_stats`. The latent scale is the model's own (``theta in [0, 1]`` for the
Bernstein estimators, ``theta in R`` for IRT).
"""
from dataclasses import dataclass

import torch
import torch.distributions as D

_EPS = 1e-12  # density floor before taking a log


@dataclass
class BetaMixture:
    comp_alphas: torch.Tensor  # Batched alpha for each mixture component (B, n + 1)
    comp_betas: torch.Tensor  # Batched beta for each mixture component   (B, n + 1)
    log_mix_weights: torch.Tensor  # Batched mixture weights in log-space  (B, n + 1)

    @property
    def dist(self) -> D.MixtureSameFamily:
        return D.MixtureSameFamily(D.Categorical(logits=self.log_mix_weights),
                                   D.Beta(self.comp_alphas, self.comp_betas))


@dataclass
class AbilityPosterior:
    """Summary of the posterior over a student's latent ability, batched over students (shape ``(B,)``).

    ``mean``/``variance``/``std`` are always populated (``std`` is the Bayesian measurement SE). The
    equal-tailed credible bounds are present only when computed with a ``level``; ``level=None`` leaves
    them ``None`` — the lean mean/variance path used in prediction. The full posterior is retained for
    densities / other quantiles, in one of two forms depending on the estimator:

    - exact (conjugate) estimator -> ``mixture`` is the closed-form Beta mixture (``log_pdf`` is exact);
    - quadrature estimator -> ``nodes`` (Q,) and ``log_weights`` (B, Q) give the discrete node posterior.
    """
    mean: torch.Tensor  # posterior mean (EAP)
    variance: torch.Tensor  # posterior variance
    std: torch.Tensor  # posterior standard deviation (the Bayesian measurement SE)
    ci_low: torch.Tensor | None = None  # lower equal-tailed credible bound at ``level`` (None if skipped)
    ci_high: torch.Tensor | None = None  # upper equal-tailed credible bound at ``level``
    level: float | None = None  # credible mass (e.g. 0.95), or None when no interval was computed
    mixture: BetaMixture | None = None  # exact posterior as a Beta mixture (continuous)
    nodes: torch.Tensor | None = None  # quadrature support theta_q (Q,)
    log_weights: torch.Tensor | None = None  # quadrature posterior log-weights (B, Q), normalized

    def log_pdf(self, theta: torch.Tensor) -> torch.Tensor:
        """Posterior log-density at query points ``theta`` (1-D, shape ``(T,)``) on the model's latent
        scale, per student -> shape ``(B, T)``.

        Exact estimators evaluate the closed-form Beta-mixture log-density. Quadrature estimators
        reconstruct a continuous density from the discrete node posterior by linearly interpolating the
        per-node density (posterior mass / trapezoidal node width); it is ``-inf`` outside the node support.
        """
        if self.mixture is not None:
            return self.mixture.dist.log_prob(theta[:, None]).transpose(0, 1)  # (T, B) -> (B, T)
        if self.nodes is None or self.log_weights is None:
            raise ValueError("AbilityPosterior carries no posterior representation to evaluate.")
        return _node_log_density(self.nodes, self.log_weights, theta)


def _node_log_density(nodes: torch.Tensor, log_weights: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Continuous log-density from a discrete node posterior, by linear interpolation.

    The node masses ``p_q`` (``exp(log_weights)``) are turned into per-node densities ``p_q / width_q``
    (trapezoidal cell widths), then linearly interpolated at ``theta``; ``-inf`` outside ``[min, max]``
    node. ``nodes`` (Q,), ``log_weights`` (B, Q), ``theta`` (T,) -> (B, T).
    """
    order = torch.argsort(nodes)
    x = nodes[order]  # (Q,) ascending
    p = log_weights[:, order].exp()  # (B, Q) posterior masses
    edges = torch.cat([x[:1], 0.5 * (x[:-1] + x[1:]), x[-1:]])  # (Q+1,) trapezoidal cell edges
    dens = p / (edges[1:] - edges[:-1]).clamp_min(_EPS)  # (B, Q) per-node density

    idx = torch.searchsorted(x, theta).clamp(1, x.numel() - 1)  # (T,) bracketing node
    x0, x1 = x[idx - 1], x[idx]  # (T,)
    d0, d1 = dens[:, idx - 1], dens[:, idx]  # (B, T)
    frac = ((theta - x0) / (x1 - x0).clamp_min(_EPS)).clamp(0.0, 1.0)  # (T,)
    log_d = (d0 + frac * (d1 - d0)).clamp_min(_EPS).log()  # (B, T)
    inside = (theta >= x[0]) & (theta <= x[-1])  # (T,)
    return torch.where(inside, log_d, log_d.new_full((), float("-inf")))


def node_credible_interval(theta_q: torch.Tensor, pi: torch.Tensor, level: float):
    """Equal-tailed credible bounds from a discrete node posterior.

    Each node's posterior mass is treated as filling its *cell* — the span to the midpoints of the
    neighbouring nodes — so the CDF is piecewise-linear across cell edges. This stays calibrated for the
    *non-uniformly* spaced quadrature nodes; placing all of a node's mass at the node itself (interpolating
    the inclusive cumulative sum) gives slightly-narrow, under-covering intervals. Bounds stay within
    ``[min node, max node]``.

    ``theta_q`` (B, Q) nodes, ``pi`` (B, Q) normalized posterior weights -> ``(ci_low, ci_high)`` each (B,).
    """
    th, order = torch.sort(theta_q, dim=-1)
    w = torch.gather(pi, -1, order)            # (B, Q) posterior mass per node, ascending in theta
    cum = torch.cumsum(w, dim=-1)              # CDF at each cell's right edge

    mids = 0.5 * (th[..., :-1] + th[..., 1:])  # (B, Q-1) interior cell boundaries (node midpoints)
    edges = torch.cat([th[..., :1], mids, th[..., -1:]], dim=-1)    # (B, Q+1), within [min, max] node
    cdf = torch.cat([torch.zeros_like(cum[..., :1]), cum], dim=-1)  # (B, Q+1), CDF at the cell edges, 0..1

    def quantile(q: float) -> torch.Tensor:
        B, E = edges.shape
        qt = edges.new_full((B, 1), q)
        idx = torch.searchsorted(cdf, qt).squeeze(-1).clamp(1, E - 1)  # (B,)
        rows = torch.arange(B, device=edges.device)
        c0, c1 = cdf[rows, idx - 1], cdf[rows, idx]
        e0, e1 = edges[rows, idx - 1], edges[rows, idx]
        frac = ((q - c0) / (c1 - c0).clamp_min(1e-12)).clamp(0.0, 1.0)
        return e0 + frac * (e1 - e0)

    tail = (1.0 - level) / 2.0
    return quantile(tail), quantile(1.0 - tail)
