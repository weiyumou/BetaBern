"""
Exact Beta-Bernstein MML estimator (cancellation-free, no quadrature).

The marginal integral ``int_0^1 Beta(theta; alpha, beta) prod_i p(y_i | theta) dtheta`` has an
exact closed form when the item likelihoods are kept in the **Bernstein basis**: the product of
Bernstein polynomials is again Bernstein with non-negative coefficients, and each basis function
integrates against the Beta prior to a positive Beta-function ratio. Everything is therefore a sum
of strictly positive terms, exact for any sequence length and numerically stable in log-space — no
Gauss-Jacobi quadrature and no eigendecomposition.

See ``QuadratureEstimator`` for the approximate Gauss-Jacobi counterpart.
"""
import functools

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import betainc

from betabern.bernstein.bernstein_irf import (
    BernsteinIRF,
    BoundedBernsteinIRF,
    FreeBernsteinIRF,
    SplineBernsteinIRF,
    make_logistic_bernstein_irf,
)
from betabern.core.model.estimator import BayesianEstimator
from betabern.core.model.posterior import AbilityPosterior, BetaMixture
from betabern.core.model.prior import BetaPrior
from betabern.core.util import gather_observed, log_beta_func, log_binom_coeff

# The Bernstein algebra runs in the model's native dtype (float32 by default). The log-space formulation
# keeps it cancellation-free, so float32 tracks the float64 result to ~1e-5; call ``model.double()`` for
# extra headroom at extreme degree ``D = S * n``.


@functools.lru_cache(maxsize=None)
def _log_comb(n: int, device, dtype) -> torch.Tensor:
    """The whole ``log_binom_coeff`` row ``log C(n, k)`` for k = 0, ..., n, shape (n + 1,), memoized.

    A thin wrapper over :func:`~betabern.core.util.log_binom_coeff` (which is elementwise in ``k``): it
    builds the ``k = 0..n`` index vector and caches the result. These rows are fixed degree-only constants
    reused on every fold step and every forward pass, so memoizing per (degree, device, dtype) keeps them
    off the hot path. The returned tensor is treated as read-only — never mutate it.
    """
    k = torch.arange(n + 1, device=device, dtype=dtype)
    return log_binom_coeff(torch.tensor(float(n), device=device, dtype=dtype), k)


def _log_terms(alpha: torch.Tensor, beta: torch.Tensor, log_c: torch.Tensor, D: int) -> torch.Tensor:
    """Unnormalized log posterior-mixture weights, shape (B, D+1).

    Term k = log C_k + log C(D, k) + log B(alpha + k, beta + D - k).
    """
    k = torch.arange(D + 1, device=log_c.device, dtype=log_c.dtype)
    a, b = alpha[:, None], beta[:, None]
    return log_c + _log_comb(D, log_c.device, log_c.dtype) + log_beta_func(a + k, b + (D - k))


def _prefix_posterior(alpha: torch.Tensor, beta: torch.Tensor,
                      log_c: torch.Tensor, D: int) -> BetaMixture:
    """The exact posterior over a response prefix as the closed-form Beta mixture
    ``sum_k pi_k Beta(alpha + k, beta + D - k)``, from the joint-likelihood coefficients ``log_c`` (degree
    ``D``). Callers read moments off :attr:`BetaMixture.dist`, or use the raw mixture components.
    """
    log_pi = torch.log_softmax(_log_terms(alpha, beta, log_c, D), dim=-1)  # (B, D+1)
    k = torch.arange(D + 1, device=log_c.device, dtype=log_c.dtype)
    comp_a = alpha[:, None] + k  # (B, D+1)
    comp_b = beta[:, None] + (D - k)
    return BetaMixture(comp_a, comp_b, log_pi)


def log_bernstein_product(log_c: torch.Tensor, degree_d: int, log_f: torch.Tensor, m: int) -> torch.Tensor:
    """Multiply two Bernstein polynomials in log-space (one fold step).

    Given the log Bernstein coefficients ``log_c`` (B, d+1) of a degree-``d`` polynomial and
    ``log_f`` (B, m+1) of a degree-``m`` factor, return the log coefficients (B, d+m+1) of the
    degree-(d+m) product. Uses the positive structure constant
    ``b_{i,d} b_{j,m} = [C(d,i) C(m,j) / C(d+m,i+j)] b_{i+j,d+m}``.

    Scaling each operand by its binomials turns the Bernstein product into a plain convolution
    ``out[k] = logsumexp_{i+j=k} (lc[i] + lf[j])``, evaluated in log-space. We pad ``lc`` by ``m`` on each
    side, slide a length-``(m+1)`` window (a strided view, no copy), add the reversed ``lf``, and reduce —
    one ``logsumexp`` over the window axis replaces the per-shift Python loop and its ``-inf`` scatter buffer.
    """
    device, dtype = log_c.device, log_c.dtype

    lc = log_c + _log_comb(degree_d, device, dtype)  # (B, d+1)
    lf = log_f + _log_comb(m, device, dtype)  # (B, m+1)

    lc_pad = F.pad(lc, (m, m), value=float("-inf"))  # (B, d+2m+1)
    windows = lc_pad.unfold(-1, m + 1, 1)  # (B, d+m+1, m+1): window k holds lc[k-m], ..., lc[k]
    out = torch.logsumexp(windows + lf.flip(-1)[:, None, :], dim=-1)  # (B, d+m+1)
    return out - _log_comb(degree_d + m, device, dtype)


class ExactBernsteinEstimator(BayesianEstimator):
    """Static Beta-Bernstein IRT model fit by *exact* marginal maximum likelihood.

    Parameters
    ----------
    prior : BetaPrior
        The fixed global Beta prior anchoring the latent metric, passed as an object (mirrors
        ``QuadratureEstimator``'s ``QuadraturePrior`` argument). Required — a caller wanting the uniform
        prior constructs ``BetaPrior()`` explicitly.
    irf : BernsteinIRF
        The Bernstein item-response parameterization (free or IRT-derived).
    """

    def __init__(self, prior: BetaPrior, irf: BernsteinIRF):
        super().__init__(prior, irf)

        # Fixed, global Beta prior (non-learnable) anchoring the latent metric. No quadrature nodes:
        # the exact method integrates in Bernstein-coefficient space, not at sample points.

    # ------------------------------------------------------------------
    # Bernstein algebra
    # ------------------------------------------------------------------

    def _observed_log_factors(self,
                              item_ids: torch.Tensor,
                              answers: torch.Tensor,
                              mask: torch.Tensor,
                              skill_ids: torch.Tensor) -> torch.Tensor:
        """Log Bernstein coefficients of each response's likelihood factor, shape (B, S, m+1).

        Padded positions become the constant-1 factor (log-coeffs = 0), i.e. exact degree elevation.
        """
        if skill_ids.dim() == 1:
            skill_ids = skill_ids.unsqueeze(-1).expand_as(item_ids)  # (B, S)

        item_ids, skill_ids = item_ids.clamp(min=0), skill_ids.clamp(min=0)  # neutralize padding (-1)
        log_w = self.irf.get_log_bernstein_weights(item_ids, skill_ids)  # (B, S, 2, m+1)
        return gather_observed(log_w, answers, mask)  # (B, S, m+1), padded -> 0 (the log-1 factor)

    def _fold(self,
              item_ids: torch.Tensor,
              answers: torch.Tensor,
              mask: torch.Tensor,
              skill_ids: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Fold all S item factors into the joint-likelihood Bernstein coefficients.

        Runs in binomial-scaled space, where the Bernstein product is a plain log-convolution: the per-step
        ``+ log C(degree)`` / ``- log C(degree + m)`` round-trips of :func:`log_bernstein_product` telescope
        away, so we scale (and pre-reverse) the factors once up front and de-scale the accumulator once at the
        end. Same arithmetic as ``S`` chained ``log_bernstein_product`` calls, minus the repeated bookkeeping.

        :return: log_C of shape (B, D+1) and the degree D = S * m.
        """
        factor = self._observed_log_factors(item_ids, answers, mask, skill_ids)  # (B, S, m+1)
        B, S, _ = factor.shape
        m = self.irf.n
        dev, dtype = factor.device, factor.dtype

        sf = (factor + _log_comb(m, dev, dtype)).flip(-1)  # (B, S, m+1): binomial-scaled, pre-reversed once
        s_c = factor.new_zeros((B, 1))  # scaled constant 1, degree 0 (C(0, 0) = 1 -> log 0)
        for s in range(S):
            windows = F.pad(s_c, (m, m), value=float("-inf")).unfold(-1, m + 1, 1)  # (B, deg+m+1, m+1)
            s_c = torch.logsumexp(windows + sf[:, s, :][:, None, :], dim=-1)  # convolve; stays scaled
        D = S * m
        return s_c - _log_comb(D, dev, dtype), D  # de-scale to Bernstein log-coeffs once

    # ------------------------------------------------------------------
    # Posterior read-outs (shared by the predictive / posterior / prequential paths)
    # ------------------------------------------------------------------

    def _log_expected_basis(self, alpha: torch.Tensor, beta: torch.Tensor, mix: BetaMixture) -> torch.Tensor:
        """Posterior-expected log Bernstein basis ``log E_post[b_{j,m}(theta)]`` for j = 0..m, shape (B, m+1).

        Per basis index ``j``, the log of ``sum_k pi_k E_Beta[b_{j,m}]`` under the prefix posterior ``mix``
        (the closed-form Beta mixture from :func:`_prefix_posterior`). Any next item's predictive is then a dot
        of its log Bernstein weights against this vector, so it is computed once per posterior and reused across
        target items and response classes. Callers pass ``mix`` in so the same build can also feed the moment
        read-out (mean/variance) without rebuilding it.

        The component term ``E_{Beta(a, b)}[b_{j,m}] = C(m, j) B(a + j, b + m - j) / B(a, b)`` has numerator
        ``B(alpha + k + j, beta + D - k + m - j) = B(alpha + i, beta + (D + m) - i)`` with ``i = k + j`` —
        it depends only on the *sum* ``k + j``, so there are ``D + m + 1`` distinct Beta-function values, not
        ``(D + 1)(m + 1)``. We evaluate those once, then the ``sum_k`` becomes a windowed log-correlation
        (``unfold``, as in :func:`log_bernstein_product`): basis ``j`` reads the length-``(D+1)`` window
        ``i = j .. j + D``. This trims the ``lgamma`` count by a factor ``~m`` versus the dense ``(B, D+1, m+1)``
        form, the bulk of the saving on the ``prequential`` per-step call.
        """
        m = self.irf.n
        dev, dtype = alpha.device, alpha.dtype
        comp_a, comp_b, log_pi = mix.comp_alphas, mix.comp_betas, mix.log_mix_weights  # (B, D+1)
        D = comp_a.shape[-1] - 1  # prefix mixture degree (= S * n folded so far)

        i = torch.arange(D + m + 1, device=dev, dtype=dtype)
        log_beta_num = log_beta_func(alpha[:, None] + i, beta[:, None] + (D + m - i))  # (B, D+m+1): per-i numerator
        g = log_pi - log_beta_func(comp_a, comp_b)  # (B, D+1): k-only factor (mixture weight / denominator)
        windows = log_beta_num.unfold(-1, D + 1, 1)  # (B, m+1, D+1): window j holds i = j .. j+D (= k+j, k=0..D)
        h = torch.logsumexp(windows + g[:, None, :], dim=-1)  # (B, m+1): sum_k over each window
        return _log_comb(m, dev, dtype) + h  # (B, m+1)

    # ------------------------------------------------------------------
    # BayesianEstimator contract
    # ------------------------------------------------------------------

    def log_marginal_evidence(self, skill_ids, item_ids, answers, mask):  # (B,)
        alpha, beta = self.get_prior(skill_ids)
        log_c, D = self._fold(item_ids, answers, mask, skill_ids)
        log_terms = _log_terms(alpha, beta, log_c, D)
        return torch.logsumexp(log_terms, dim=-1) - log_beta_func(alpha, beta)

    def predictive_log_probs(self, history: dict | None, target: dict):  # (B, S, 2)
        skill_ids = target["skill_ids"]
        alpha, beta = self.get_prior(skill_ids)
        B = skill_ids.shape[0]

        if history is not None and history["item_ids"].size(1) > 0:
            log_c, D = self._fold(history["item_ids"], history["answers"], history["mask"], history["skill_ids"])
        else:
            log_c, D = alpha.new_zeros((B, 1)), 0  # prior only

        # Reduce the (growing) mixture to the posterior-expected basis once, then dot each target item's
        # weights against it: log p(y = c) = logsumexp_j [ log_w[..., c, j] + log E_post[b_{j,m}] ].
        log_basis = self._log_expected_basis(alpha, beta, _prefix_posterior(alpha, beta, log_c, D))  # (B, m+1)

        # Target items, both classes: log Bernstein coeffs (B, St, 2, m+1). Padded targets are masked
        # out by the caller; clamp their -1 ids so the embedding lookup stays in range.
        t_skill = skill_ids.unsqueeze(-1).expand_as(target["item_ids"]) if skill_ids.dim() == 1 else skill_ids
        log_w = self.irf.get_log_bernstein_weights(target["item_ids"].clamp(min=0), t_skill.clamp(min=0))
        return torch.logsumexp(log_w + log_basis[:, None, None, :], dim=-1)  # (B, St, 2)

    # ------------------------------------------------------------------
    # Exact ability posterior
    # ------------------------------------------------------------------

    @torch.no_grad()
    def posterior_stats(self, skill_ids, item_ids, answers, mask, *,
                        level: float | None = 0.95) -> AbilityPosterior:
        """The exact posterior over ``theta`` as the closed-form Beta mixture ``sum_k pi_k Beta(alpha + k,
        beta + D - k)`` (``D = N * n``), plus summaries.

        Mean and variance are analytic mixture moments (always); when ``level`` is given, the equal-tailed
        credible interval inverts the mixture CDF ``F(theta) = sum_k pi_k I_theta(alpha + k, beta + D - k)``
        (regularized incomplete beta) by bisection — ``level=None`` skips that bisection (the lean path).
        With no responses it returns the prior. Batched over students.
        """
        alpha, beta = self.get_prior(skill_ids)
        log_c, D = self._fold(item_ids, answers, mask, skill_ids)

        # The closed-form posterior mixture exposes its own exact mean/variance.
        mixture = _prefix_posterior(alpha, beta, log_c, D)
        dist = mixture.dist
        mean, var = dist.mean, dist.variance
        if level is None:
            return AbilityPosterior(mean, var, var.sqrt(), mixture=mixture)
        ci_low, ci_high = self._credible_interval(
            mixture.log_mix_weights.exp(), mixture.comp_alphas, mixture.comp_betas, level)
        return AbilityPosterior(mean, var, var.sqrt(), ci_low, ci_high, level, mixture)

    @torch.no_grad()
    def prequential(self, skill_ids: torch.Tensor, item_ids: torch.Tensor, answers: torch.Tensor, mask: torch.Tensor):
        """Incremental exact fold. At each prefix the one-step predictive is the posterior-expected item
        likelihood (:meth:`_log_expected_basis`) and the running moments are the prefix mixture's. Because the
        exact posterior representation grows with the prefix (degree ``D = t * n``), this is an ``S``-step loop
        costing ``O(B * S^2 * n^2)`` — there is no constant-size ``cumsum`` shortcut as in the quadrature view.
        See :meth:`BayesianEstimator.prequential` for the contract.
        """
        alpha, beta = self.get_prior(skill_ids)
        skill_e = skill_ids.unsqueeze(-1).expand_as(item_ids) if skill_ids.dim() == 1 else skill_ids
        B, S = item_ids.shape
        m = self.irf.n

        factor = self._observed_log_factors(item_ids, answers, mask, skill_ids)  # (B, S, m+1), padded -> log-1
        log_w = self.irf.get_log_bernstein_weights(item_ids.clamp(min=0), skill_e.clamp(min=0))  # (B, S, 2, m+1)

        pred = log_w.new_zeros((B, S, 2))
        mean = alpha.new_zeros((B, S + 1))
        var = alpha.new_zeros((B, S + 1))

        log_c = alpha.new_zeros((B, 1))  # prior, degree 0 (posterior over the empty prefix)
        D = 0
        for t in range(S):
            mix = _prefix_posterior(alpha, beta, log_c, D)  # prefix y_{<t}; shared by its moments + predictive
            dist = mix.dist
            mean[:, t], var[:, t] = dist.mean, dist.variance
            log_basis = self._log_expected_basis(alpha, beta, mix)  # (B, m+1): conditions on y_{<t}
            pred[:, t, :] = torch.logsumexp(log_w[:, t, :, :] + log_basis[:, None, :], dim=-1)  # (B, 2)
            log_c = log_bernstein_product(log_c, D, factor[:, t, :], m)  # fold y_t -> posterior over y_{<=t}
            D += m
        dist = _prefix_posterior(alpha, beta, log_c, D).dist  # final prefix (all S folded): moments only
        mean[:, S], var[:, S] = dist.mean, dist.variance
        # Padded steps fold the log-1 factor (a no-op), so their moments carry the last value forward; only
        # the predictive needs masking (its target weights are from the clamped padding ids).
        return pred * mask.unsqueeze(-1), mean, var.clamp_min(0.0).sqrt()

    @staticmethod
    def _credible_interval(pi, comp_a, comp_b, level: float, iters: int = 60):
        """Equal-tailed credible bounds by bisecting the mixture CDF (scipy regularized incomplete beta)."""
        w = pi.detach().cpu().numpy()
        a = comp_a.detach().cpu().numpy()
        b = comp_b.detach().cpu().numpy()
        n = w.shape[0]

        def mixture_cdf(x: np.ndarray) -> np.ndarray:  # x: (B,) -> F(x): (B,)
            return (w * betainc(a, b, x[:, None])).sum(-1)

        def quantile(q: float) -> np.ndarray:
            lo, hi = np.zeros(n), np.ones(n)
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                below = mixture_cdf(mid) < q
                lo = np.where(below, mid, lo)
                hi = np.where(below, hi, mid)
            return 0.5 * (lo + hi)

        tail = (1.0 - level) / 2.0
        lo = torch.as_tensor(quantile(tail), dtype=comp_a.dtype, device=pi.device)
        hi = torch.as_tensor(quantile(1.0 - tail), dtype=comp_a.dtype, device=pi.device)
        return lo, hi


# ======================================================================
# Factories
# ======================================================================

def build_exact_bernstein_estimator_irt(degree_n: int,
                                        num_users: int,
                                        num_skills: int,
                                        num_items: int,
                                        irt_model: str = "2PL",
                                        irt_c_init: float = 0.1,
                                        irt_s_init: float = 0.1,
                                        item_weight_rank: int | None = None,
                                        prior: tuple[float, float] = (1.0, 1.0)) -> ExactBernsteinEstimator:
    """Build an exact estimator whose Bernstein weights are generated from an IRT curve."""
    irf = make_logistic_bernstein_irf(degree_n=degree_n, num_users=num_users, num_skills=num_skills,
                                      num_items=num_items, irt_model=irt_model, c_init=irt_c_init,
                                      s_init=irt_s_init, item_weight_rank=item_weight_rank)
    return ExactBernsteinEstimator(BetaPrior(prior[0], prior[1]), irf)


def build_exact_bernstein_estimator_free(degree_n: int,
                                         num_users: int,
                                         num_skills: int,
                                         num_items: int,
                                         c_init: float = 0.1,
                                         s_init: float = 0.1,
                                         item_weight_rank: int | None = None,
                                         prior: tuple[float, float] = (1.0, 1.0)) -> ExactBernsteinEstimator:
    """Build an exact estimator with freely-learned Bernstein weights (skill table initialized to a linear
    guess/slip ramp ``c_init -> 1 - s_init``)."""
    irf = FreeBernsteinIRF(degree_n=degree_n, num_skills=num_skills, num_items=num_items,
                           c_init=c_init, s_init=s_init, item_weight_rank=item_weight_rank)
    return ExactBernsteinEstimator(BetaPrior(prior[0], prior[1]), irf)


def build_exact_bernstein_estimator_bounded(degree_n: int,
                                            num_users: int,
                                            num_skills: int,
                                            num_items: int,
                                            c_init: float = 0.1,
                                            s_init: float = 0.1,
                                            item_weight_rank: int | None = None,
                                            prior: tuple[float, float] = (1.0, 1.0),
                                            mono_lambda: float = 0.0) -> ExactBernsteinEstimator:
    """Build an exact estimator with freely-learned **bounded (non-monotone)** Bernstein weights.

    Identical conjugate machinery to :func:`build_exact_bernstein_estimator_free`, but the IRF weights are
    unconstrained in ``(0, 1)`` rather than monotone-chained — so the closed-form Beta-mixture posterior holds
    even when the fitted IRF is non-monotone (e.g. an AI-mediated valley). Used by the non-monotone capability
    demonstration; not part of the default benchmark lineup. ``mono_lambda`` adds a soft-monotonicity prior on
    the per-node weights (a dial toward the hard monotone model).
    """
    irf = BoundedBernsteinIRF(degree_n=degree_n, num_skills=num_skills, num_items=num_items,
                              c_init=c_init, s_init=s_init, item_weight_rank=item_weight_rank,
                              mono_lambda=mono_lambda)
    return ExactBernsteinEstimator(BetaPrior(prior[0], prior[1]), irf)


def build_exact_bernstein_estimator_spline_g(degree_n: int,
                                             num_users: int,
                                             num_skills: int,
                                             num_items: int,
                                             gen_degree: int = 6,
                                             gen_order: int = 3,
                                             item_weight_rank: int | None = None,
                                             prior: tuple[float, float] = (1.0, 1.0)) -> ExactBernsteinEstimator:
    """Build an exact estimator whose Bernstein weights come from a monotone I-spline generator.

    The IRF is still a global Bernstein polynomial, so the exact (conjugate) marginal/posterior apply — this
    is what the "slot" buys: spline-flexible weights with the closed-form Beta-Bernstein machinery intact.
    """
    irf = SplineBernsteinIRF(degree_n=degree_n, num_skills=num_skills, num_items=num_items,
                             gen_degree=gen_degree, gen_order=gen_order,
                             item_weight_rank=item_weight_rank)
    return ExactBernsteinEstimator(BetaPrior(prior[0], prior[1]), irf)
