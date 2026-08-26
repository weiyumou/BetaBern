"""
Static, ``R``-scale measurement estimators fit by Marginal Maximum Likelihood (MML).

A static counterpart to the sequential :class:`BayesianIRT` filter: each (student, skill) has a latent
ability ``theta in R`` with a fixed Normal(mean, std^2) prior, integrated out of the whole response vector
via Gauss-Hermite quadrature. Global item parameters are fit by maximizing the marginal log-evidence — the
classic Bock-Aitkin MML estimation, with the population fixed to anchor the latent metric. Each is just a
:class:`~betabern.core.model.estimator.QuadratureEstimator` over a
:class:`~betabern.core.model.prior.NormalHermitePrior` and a :class:`MonotoneIRF`:

- :func:`build_quad_irt_estimator`: a :class:`LogisticIRF` (the 1-4PL logistic curve) — the classic IRT.
- :func:`build_quad_spline_irt_estimator`: an :class:`ISplineIRF` on an ``R`` window with saturating
  tails — a *free monotone* IRF on the unbounded scale, the ``R`` analog of the ``[0, 1]`` free I-spline
  (see ``spline-gaussian-irf.md``).
"""
from betabern.core.model.estimator import QuadratureEstimator
from betabern.core.model.irf import ISplineIRF
from betabern.core.model.prior import NormalHermitePrior
from betabern.irt.model import make_logistic_irf


def build_quad_irt_estimator(num_users: int,
                             num_skills: int,
                             num_items: int,
                             num_nodes: int = 30,
                             irt_model: str = "2PL",
                             c_init: float = 0.1,
                             s_init: float = 0.1,
                             item_weight_rank: int | None = None,
                             prior_mean: float = 0.0,
                             prior_std: float = 1.0) -> QuadratureEstimator:
    """Build a static IRT MML estimator: a ``QuadratureEstimator`` over a fresh IRT IRF (``irt_model`` selects
    logistic ``2PL`` vs. normal-ogive ``2PO``) and a fixed ``Normal(mean, std^2)`` prior (``num_nodes`` GH nodes)."""
    irf = make_logistic_irf(num_users=num_users, num_skills=num_skills, num_items=num_items, irt_model=irt_model,
                            c_init=c_init, s_init=s_init, item_weight_rank=item_weight_rank)
    prior = NormalHermitePrior(prior_mean, prior_std, num_nodes)
    return QuadratureEstimator(prior=prior, irf=irf)


def build_quad_spline_irt_estimator(num_skills: int,
                                    num_items: int,
                                    degree_n: int = 8,
                                    order: int = 3,
                                    num_nodes: int = 40,
                                    item_weight_rank: int | None = None,
                                    prior_mean: float = 0.0,
                                    prior_std: float = 1.0,
                                    window_sd: float = 5.0) -> QuadratureEstimator:
    """Build the spline-Gaussian MML estimator: a ``QuadratureEstimator`` over a free monotone I-spline IRF
    on the ``R`` window ``[mean - window_sd*std, mean + window_sd*std]`` (constant-saturating tails) and a
    fixed ``Normal(mean, std^2)`` prior with ``num_nodes`` Gauss-Hermite nodes. The ``R`` counterpart of the
    ``[0, 1]`` free I-spline (``build_quad_ispline_estimator_free``); see ``spline-gaussian-irf.md``.

    The window brackets the prior mass: ``window_sd = 5`` covers ``+/- 5`` SD, beyond which the IRF is
    assumed saturated (Gauss-Hermite nodes that land there contribute through the constant tails)."""
    domain = (prior_mean - window_sd * prior_std, prior_mean + window_sd * prior_std)
    irf = ISplineIRF(degree_n=degree_n, num_skills=num_skills, num_items=num_items,
                     item_weight_rank=item_weight_rank, order=order, domain=domain)
    prior = NormalHermitePrior(prior_mean, prior_std, num_nodes)
    return QuadratureEstimator(prior=prior, irf=irf)
