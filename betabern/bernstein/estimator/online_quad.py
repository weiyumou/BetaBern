"""
Online Beta-Bernstein quadrature estimator (static trait, processed sequentially).

A *third* model kind alongside the sequential conjugate filters (which carry a learning transition and
live in the separate *Filter* project) and the batch static estimators (``bernstein/estimator``, which
marginalize the latent trait in one shot). This processes responses one at a time like a filter but assumes a single
**static** latent trait — there is NO transition between steps. It therefore yields, at every step
``t``, the prequential predictive ``p(y_t | y_{<t})`` and a running EAP estimate of the trait.

It is built *around the filter base* (subclasses ``BayesianFilter``) and is a **standalone** module
that owns its IRF, prior, and fixed Gauss-Jacobi quadrature — the same construction as the batch
``QuadratureEstimator`` but run sequentially. The two are mathematically identical when given the
same IRF/prior: summing the per-step predictive log-likelihoods recovers the batch marginal
log-evidence (chain rule), and the final-step EAP equals the batch ``posterior_stats(...).mean``.
See ``tests/test_online.py``.

The Gauss-Jacobi flavor (fixed ``Q`` nodes) keeps the filter state a constant-size ``(B, Q)`` vector
``S_q = sum_{i<=t} log p(y_i | theta_q)`` — the per-node cumulative log-likelihood. That update is
purely additive (transition-free), i.e. an *associative scan*, so the whole sequence is computed by a
single vectorized cumulative sum instead of the sequential ``BayesianFilter`` loop — and the state
stays bounded, unlike the exact method's growing coefficient state.
"""
import torch

from betabern.bernstein.bernstein_irf import BernsteinIRF, FreeBernsteinIRF, make_logistic_bernstein_irf
from betabern.core.model.filter import BayesianFilter
from betabern.core.model.prior import BetaJacobiPrior, QuadraturePrior
from betabern.core.util import gather_observed


class OnlineQuadEstimator(BayesianFilter):
    """Sequential, static-trait Beta-Bernstein estimator over Gauss-Jacobi nodes.

    Standalone module: it owns the Bernstein IRF, the fixed Beta prior, and the constant Gauss-Jacobi
    nodes/weights/basis. The filter state is the per-node cumulative log-likelihood ``S_q`` of shape
    ``(B, Q)``; there is no learning transition. See the module docstring for the equivalence to the
    batch ``QuadratureEstimator``.

    Parameters
    ----------
    irf : BernsteinIRF
        The Bernstein item-response parameterization (free or IRT-derived).
    num_nodes : int
        Number of Gauss-Jacobi quadrature nodes ``Q``.
    prior : tuple[float, float]
        The fixed global Beta(alpha, beta) prior anchoring the latent metric.
    """

    def __init__(self,
                 irf: BernsteinIRF,
                 num_nodes: int = 30,
                 prior: tuple[float, float] = (1.0, 1.0)):
        super().__init__()

        self.irf = irf
        self.num_nodes = num_nodes
        # Fixed Beta prior + its constant Gauss-Jacobi nodes (the same prior the batch estimator uses).
        self.prior: QuadraturePrior = BetaJacobiPrior(prior[0], prior[1], num_nodes)
        # Constant Bernstein basis at the prior's nodes (precomputed once).
        self.register_buffer("log_basis", irf.log_basis(self.prior.get_buffer("theta_q")))  # (Q, n + 1)

    # ------------------------------------------------------------------
    # BayesianFilter contract
    # ------------------------------------------------------------------

    def get_init_state(self, skill_ids: torch.Tensor) -> torch.Tensor:
        """Initial state ``S_q = 0`` — the posterior over nodes then equals the prior weights ``v_q``."""
        B = skill_ids.shape[0]
        return self.prior.log_v_q.new_zeros((B, self.num_nodes))

    def step(self,
             state: torch.Tensor,
             user_t: torch.Tensor,
             skill_t: torch.Tensor,
             item_t: torch.Tensor,
             answer_t: torch.Tensor,
             delta_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Single online update — the recurrent (streaming) form of one timestep.

        This is the one-response-at-a-time interface for streaming / CAT serving (and the per-step hook
        the default ``BayesianFilter`` loop would call). Training and batched evaluation instead use the
        vectorized :meth:`forward`, the associative-scan equivalent of this loop; the two agree exactly.

        state: per-node cumulative log-likelihood ``S_q`` of shape ``(B, Q)``
        :return: prequential predictive ``log p(y_t | y_{<t})`` (B, 2) and the updated state (B, Q)
        """
        node = self.irf.log_irf_from_basis(
            self.get_buffer("log_basis"), item_t.unsqueeze(-1), skill_t).squeeze(1)  # (B, 2, Q)

        log_vq = self.prior.log_v_q
        log_post = torch.log_softmax(log_vq + state, dim=-1)  # (B, Q)
        log_probs = torch.logsumexp(log_post.unsqueeze(1) + node, dim=-1)  # (B, 2)

        ans = answer_t.clamp(min=0)  # active rows only, but be defensive
        obs = node.gather(1, ans[:, None, None].expand(-1, 1, node.size(-1))).squeeze(1)  # (B, Q)
        return log_probs, state + obs

    # ------------------------------------------------------------------
    # Vectorized prequential pass (associative scan over time)
    # ------------------------------------------------------------------
    # The additive, transition-free update makes the recurrence a prefix sum, so ``forward`` is
    # overridden with a single vectorized cumulative sum — the associative-scan equivalent of the
    # ``step`` loop and the fast path for training/eval.

    def forward(
            self,
            user_ids: torch.Tensor,
            skill_ids: torch.Tensor,
            item_ids: torch.Tensor,
            answers: torch.Tensor,
            time_deltas: torch.Tensor,
            mask: torch.Tensor,
            current_state: torch.Tensor | None = None,
    ):
        """Vectorized prequential pass, identical to a sequential left-to-right filter over the sequence.

        An exclusive cumulative sum over time gives the pre-step state at every position in one shot —
        ``O(B * T * Q * m)`` with no Python loop.

        :return:
        - ``all_log_probs`` (B, T, 2): prequential ``log p(y_t | y_{<t})`` with masked steps set to 0.
        - ``all_states`` (B, T + 1, Q): the ``S_q`` trajectory (index 0 is the initial state).
        - ``current_state``: continues from a prior history's accumulated state (stacked batches).
        """
        node = self.irf.log_irf_from_basis(self.get_buffer("log_basis"), item_ids, skill_ids)  # (B, T, 2, Q)

        if current_state is None:
            current_state = self.get_init_state(skill_ids)  # (B, Q)

        # Observed-class node log-lik per step; padded steps contribute 0 (state carried forward).
        obs = gather_observed(node, answers, mask)  # (B, T, Q)
        cum = obs.cumsum(dim=1)  # inclusive prefix sum over time, (B, T, Q)
        prefix = current_state.unsqueeze(1) + (cum - obs)  # pre-step (exclusive) S_q, (B, T, Q)

        log_vq = self.prior.log_v_q
        log_post = torch.log_softmax(log_vq + prefix, dim=-1)  # (B, T, Q)
        all_log_probs = torch.logsumexp(log_post.unsqueeze(2) + node, dim=-1)  # (B, T, 2)
        all_log_probs = all_log_probs * mask.unsqueeze(-1)  # match the loop: masked steps -> 0

        incl = current_state.unsqueeze(1) + cum  # post-step S_q, (B, T, Q)
        all_states = torch.cat([current_state.unsqueeze(1), incl], dim=1)  # (B, T + 1, Q)
        return all_log_probs, all_states

    # ------------------------------------------------------------------
    # Read-out
    # ------------------------------------------------------------------

    def eap_from_state(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Posterior EAP ``E[theta]`` and SD from a state ``S_q`` of any leading shape ``(*, Q)``."""
        log_vq = self.prior.log_v_q
        post = torch.softmax(log_vq + state, dim=-1)  # (*, Q)
        theta_q = self.prior.theta_q
        eap = (post * theta_q).sum(dim=-1)  # (*,)
        var = (post * (theta_q - eap.unsqueeze(-1)) ** 2).sum(dim=-1)
        return eap, var.clamp_min(0).sqrt()


# ======================================================================
# Factories
# ======================================================================

def build_online_quad_estimator_irt(degree_n: int,
                                    num_users: int,
                                    num_skills: int,
                                    num_items: int,
                                    num_nodes: int = 30,
                                    irt_model: str = "2PL",
                                    irt_c_init: float = 0.1,
                                    irt_s_init: float = 0.1,
                                    item_weight_rank: int | None = None,
                                    prior: tuple[float, float] = (1.0, 1.0)) -> OnlineQuadEstimator:
    """Build an online estimator whose Bernstein weights are generated from an IRT curve."""
    irf = make_logistic_bernstein_irf(degree_n=degree_n, num_users=num_users, num_skills=num_skills,
                                      num_items=num_items, irt_model=irt_model, c_init=irt_c_init,
                                      s_init=irt_s_init, item_weight_rank=item_weight_rank)
    return OnlineQuadEstimator(irf=irf, num_nodes=num_nodes, prior=prior)


def build_online_quad_estimator_free(degree_n: int,
                                     num_users: int,
                                     num_skills: int,
                                     num_items: int,
                                     num_nodes: int = 30,
                                     item_weight_rank: int | None = None,
                                     prior: tuple[float, float] = (1.0, 1.0)) -> OnlineQuadEstimator:
    """Build an online estimator with freely-learned Bernstein weights."""
    irf = FreeBernsteinIRF(degree_n=degree_n, num_skills=num_skills, num_items=num_items,
                           item_weight_rank=item_weight_rank)
    return OnlineQuadEstimator(irf=irf, num_nodes=num_nodes, prior=prior)
