import abc

import torch
import torch.nn as nn

from betabern.core.model.irf import MonotoneIRF
from betabern.core.model.posterior import AbilityPosterior, node_credible_interval
from betabern.core.model.prior import FixedPrior, QuadraturePrior
from betabern.core.util import count_free_params, gather_observed


class BayesianEstimator(nn.Module, abc.ABC):
    """Static latent-ability model fit by marginal maximum likelihood.

    Integrates a latent ability ``theta`` out of a student's whole response vector. HOW the
    integral is solved (quadrature, closed form, sampling, variational, ...) is left to
    subclasses; this class only fixes the statistical contract the task layer depends on.
    """

    def __init__(self, prior: FixedPrior, irf: MonotoneIRF):
        super().__init__()
        self.prior = prior
        self.irf = irf

    def get_prior(self, skill_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the prior parameters broadcast to the batch, each shape ``(B,)``."""
        return self.prior.get_params(skill_ids.shape[0])

    def num_free_params(self) -> int:
        """All learnable parameters live in the IRF (the prior is fixed)."""
        return count_free_params(self.irf)

    @abc.abstractmethod
    def log_marginal_evidence(self,
                              skill_ids: torch.Tensor,
                              item_ids: torch.Tensor,
                              answers: torch.Tensor,
                              mask: torch.Tensor) -> torch.Tensor:
        """log p(Y | params) per student, shape (B,) — the MML training objective."""

    @abc.abstractmethod
    def predictive_log_probs(self, history: dict | None, target: dict) -> torch.Tensor:
        """Posterior-predictive log p(y_target = c | history), shape (B, S, 2)."""

    @abc.abstractmethod
    def posterior_stats(self,
                        skill_ids: torch.Tensor,
                        item_ids: torch.Tensor,
                        answers: torch.Tensor,
                        mask: torch.Tensor,
                        *,
                        level: float | None = 0.95) -> AbilityPosterior:
        """Posterior over each student's latent ability given ``Y``, as an :class:`AbilityPosterior`.

        ``mean`` (the EAP point estimate), ``variance`` and ``std`` (the Bayesian measurement SE) are always returned.
        An equal-tailed credible interval at ``level`` is added when given; passing ``level=None`` skips it."""

    @abc.abstractmethod
    def prequential(self,
                    skill_ids: torch.Tensor,
                    item_ids: torch.Tensor,
                    answers: torch.Tensor,
                    mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process each student's responses left-to-right — the static estimator run sequentially, with NO
        learning transition (this is online *measurement*, not knowledge tracing).

        At step ``t`` the prequential predictive ``log p(y_t | y_{<t})`` conditions on the posterior over the
        *earlier* responses only; summing those per-step predictives over a student recovers
        :meth:`log_marginal_evidence` exactly (the chain rule), so 'sequential' and 'whole-vector' are the
        same estimator in two orderings. Also returns the running ability posterior after each prefix — the
        basis for a sequential 'measurement-efficiency' curve (uncertainty shrinks as items arrive).

        :return:
            ``pred`` (B, S, 2): prequential ``log p(y_t = c | y_{<t})``; masked steps zeroed.
            ``mean`` (B, S+1), ``std`` (B, S+1): posterior ``E[theta]`` and SD after ``0, 1, ..., S``
            responses (index 0 is the prior; padded tail steps hold the last valid value).
        """


class QuadratureEstimator(BayesianEstimator):
    """Solves the marginal integral by quadrature: ``E_prior[f] = sum_q v_q f(theta_q)``.

    Parameterized by a fixed :class:`~betabern.core.model.prior.QuadraturePrior` (prior params + nodes)
    and a :class:`~betabern.core.model.irf.MonotoneIRF` (the per-node response). The marginal-evidence /
    posterior / predictive / EAP reductions are shared; the IRF decides everything else — including whether
    a constant node basis can be cached: if it is basis-decomposable, the basis at the fixed quadrature
    nodes is precomputed once; otherwise the IRF is evaluated at the nodes each call.
    """

    def __init__(self, prior: QuadraturePrior, irf: MonotoneIRF):
        super().__init__(prior, irf)

        if callable(getattr(irf, "log_basis", None)):
            theta_q, _ = prior.quadrature(1)  # the constant quadrature nodes, (1, Q)
            self.register_buffer("_log_basis", irf.log_basis(theta_q[0]))  # (Q, D) cache
        else:
            self.register_buffer("_log_basis", None)

    def _log_irf(self,
                 theta_q: torch.Tensor,
                 item_ids: torch.Tensor,
                 skill_ids: torch.Tensor) -> torch.Tensor:
        """Per-node response log-likelihood, shape (B, S, 2, Q), ordered [incorrect, correct]."""
        if self._log_basis is not None:  # basis-decomposable IRF -> reuse the cached node basis
            return self.irf.log_irf_from_basis(self.get_buffer("_log_basis"), item_ids, skill_ids)
        return self.irf.log_irf(theta_q, item_ids, skill_ids)

    def _sum_log_likelihood(self,
                            theta_q: torch.Tensor,
                            skill_ids: torch.Tensor,
                            item_ids: torch.Tensor,
                            answers: torch.Tensor,
                            mask: torch.Tensor) -> torch.Tensor:
        """Sum of observed-response log-likelihoods at each quadrature node, shape (B, Q)."""
        log_lik = self._log_irf(theta_q, item_ids, skill_ids)  # (B, S, 2, Q)
        return gather_observed(log_lik, answers, mask).sum(dim=1)  # (B, Q), padding zeroed

    def log_marginal_evidence(self,
                              skill_ids: torch.Tensor,
                              item_ids: torch.Tensor,
                              answers: torch.Tensor,
                              mask: torch.Tensor):  # (B,)
        theta_q, log_v_q = self.prior.quadrature(skill_ids.size(0))
        summed = self._sum_log_likelihood(theta_q, skill_ids, item_ids, answers, mask)
        return torch.logsumexp(log_v_q + summed, dim=-1)  # (B, )

    def predictive_log_probs(self, history: dict | None, target: dict):
        skill_ids = target["skill_ids"]
        theta_q, log_v_q = self.prior.quadrature(skill_ids.size(0))

        # Posterior log weights start at the prior log-weights.
        log_post = log_v_q
        if history is not None and history["item_ids"].size(1) > 0:
            log_post = log_post + self._sum_log_likelihood(
                theta_q, skill_ids, history["item_ids"], history["answers"], history["mask"])
        log_post = torch.log_softmax(log_post, dim=-1)  # (B, Q)

        target_log_lik = self._log_irf(theta_q, target["item_ids"], skill_ids)  # (B, S, 2, Q)
        return torch.logsumexp(log_post[:, None, None, :] + target_log_lik, dim=-1)

    @torch.no_grad()
    def posterior_stats(self, skill_ids: torch.Tensor, item_ids: torch.Tensor, answers: torch.Tensor,
                        mask: torch.Tensor, *, level: float | None = 0.95) -> AbilityPosterior:
        """The discrete posterior over the quadrature nodes. Mean/variance are the quadrature estimates
        ``E[theta] = sum_q p_q theta_q``, ``Var[theta] = sum_q p_q theta_q^2 - mean^2`` (the linear weighted
        sum supports unbounded ``theta``); the credible interval interpolates the discrete node CDF."""
        theta_q, log_v_q = self.prior.quadrature(skill_ids.size(0))
        summed = self._sum_log_likelihood(theta_q, skill_ids, item_ids, answers, mask)
        log_post = torch.log_softmax(log_v_q + summed, dim=-1)  # (B, Q), normalized
        pi = log_post.exp()
        mean = torch.sum(pi * theta_q, dim=-1)
        var = (torch.sum(pi * theta_q ** 2, dim=-1) - mean ** 2).clamp_min(0.0)
        ci_low, ci_high = (None, None) if level is None else node_credible_interval(theta_q, pi, level)
        return AbilityPosterior(mean=mean, variance=var, std=var.sqrt(), ci_low=ci_low, ci_high=ci_high,
                                level=level, nodes=theta_q[0].clone(), log_weights=log_post)

    @torch.no_grad()
    def prequential(self,
                    skill_ids: torch.Tensor,
                    item_ids: torch.Tensor,
                    answers: torch.Tensor,
                    mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized over the fixed quadrature nodes: one ``cumsum`` of the per-node response
        log-likelihoods yields every prefix posterior at once (the node support is constant, so all prefixes
        share a single ``(B, Q)`` representation), so the whole pass is ``O(B * S * Q)``. See
        :meth:`BayesianEstimator.prequential` for the contract.
        """
        theta_q, log_v_q = self.prior.quadrature(skill_ids.size(0))  # (B, Q)
        node = self._log_irf(theta_q, item_ids, skill_ids)  # (B, S, 2, Q)
        obs = gather_observed(node, answers, mask)  # (B, S, Q), padded steps -> 0
        cum = obs.cumsum(dim=1)  # (B, S, Q) inclusive prefix sums of node log-lik

        # Prequential predictive at step t from the posterior over y_{<t} (the exclusive prefix cum - obs).
        log_post_pre = torch.log_softmax(log_v_q[:, None, :] + (cum - obs), dim=-1)  # (B, S, Q)
        pred = torch.logsumexp(log_post_pre[:, :, None, :] + node, dim=-1)  # (B, S, 2)
        pred = pred * mask.unsqueeze(-1)  # masked steps -> 0

        # Running posterior moments after k = 0..S responses (k = 0 is the prior; states = inclusive cum).
        states = torch.cat([torch.zeros_like(cum[:, :1]), cum], dim=1)  # (B, S+1, Q)
        pi = torch.softmax(log_v_q[:, None, :] + states, dim=-1)  # (B, S+1, Q)
        mean = (pi * theta_q[:, None, :]).sum(dim=-1)  # (B, S+1)
        var = (pi * theta_q[:, None, :] ** 2).sum(dim=-1) - mean ** 2
        return pred, mean, var.clamp_min(0.0).sqrt()
