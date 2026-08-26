"""
Beta-Bernstein MML estimator via Gauss-Jacobi quadrature (fixed prior).

These are just :class:`~betabern.core.model.estimator.QuadratureEstimator`\\s over a Bernstein /
I-spline :class:`~betabern.core.model.irf.MonotoneIRF` and a
:class:`~betabern.core.model.prior.BetaJacobiPrior` (the uniform ``Beta(1, 1)`` case is plain
Gauss-Legendre). The factories below pick the IRF; quadrature is exact when ``Q >= (N * degree + 1) / 2``,
and a moderate ``Q`` is an accurate, numerically stable approximation otherwise.
for the derivation and ``ExactBernsteinEstimator`` for the quadrature-free exact counterpart.
"""
from betabern.bernstein.bernstein_irf import (
    BoundedBernsteinIRF,
    CDFBernsteinIRF,
    FreeBernsteinIRF,
    SplineBernsteinIRF,
    make_logistic_bernstein_irf,
)
from betabern.core.model.estimator import QuadratureEstimator
from betabern.core.model.irf import ISplineIRF
from betabern.core.model.prior import BetaJacobiPrior

# ======================================================================
# Factories
# ======================================================================

def build_quad_bernstein_estimator_irt(degree_n: int,
                                       num_users: int,
                                       num_skills: int,
                                       num_items: int,
                                       num_nodes: int = 30,
                                       irt_model: str = "2PL",
                                       irt_c_init: float = 0.1,
                                       irt_s_init: float = 0.1,
                                       item_weight_rank: int | None = None,
                                       prior: tuple[float, float] = (1.0, 1.0),
                                       eta_half_width: float = 3.0) -> QuadratureEstimator:
    """Quadrature estimator whose Bernstein weights are sampled from an IRT curve (``irt_model``: ``2PL``/``2PO``)."""
    irf = make_logistic_bernstein_irf(degree_n=degree_n, num_users=num_users, num_skills=num_skills,
                                      num_items=num_items, irt_model=irt_model, c_init=irt_c_init,
                                      s_init=irt_s_init, item_weight_rank=item_weight_rank,
                                      eta_half_width=eta_half_width)
    return QuadratureEstimator(prior=BetaJacobiPrior(prior[0], prior[1], num_nodes), irf=irf)


def build_quad_bernstein_estimator_free(degree_n: int,
                                        num_users: int,
                                        num_skills: int,
                                        num_items: int,
                                        num_nodes: int = 30,
                                        c_init: float = 0.1,
                                        s_init: float = 0.1,
                                        item_weight_rank: int | None = None,
                                        prior: tuple[float, float] = (1.0, 1.0)) -> QuadratureEstimator:
    """Quadrature estimator with freely-learned Bernstein weights (skill table initialized to a linear
    guess/slip ramp ``c_init -> 1 - s_init``)."""
    irf = FreeBernsteinIRF(degree_n=degree_n,
                           num_skills=num_skills,
                           num_items=num_items,
                           c_init=c_init,
                           s_init=s_init,
                           item_weight_rank=item_weight_rank)
    return QuadratureEstimator(prior=BetaJacobiPrior(prior[0], prior[1], num_nodes), irf=irf)


def build_quad_bernstein_estimator_bounded(degree_n: int,
                                           num_users: int,
                                           num_skills: int,
                                           num_items: int,
                                           num_nodes: int = 30,
                                           c_init: float = 0.1,
                                           s_init: float = 0.1,
                                           item_weight_rank: int | None = None,
                                           prior: tuple[float, float] = (1.0, 1.0),
                                           mono_lambda: float = 0.0) -> QuadratureEstimator:
    """Quadrature estimator with freely-learned **bounded (non-monotone)** Bernstein weights.

    Identical to :func:`build_quad_bernstein_estimator_free` but with the monotonicity constraint dropped
    (each weight unconstrained in ``(0, 1)`` instead of monotone-chained), so it can fit a non-monotone IRF
    while keeping the conjugate Beta-Bernstein posterior. Used as the "monotonicity is optional" control on
    real (monotone) data, and as the non-monotone-capable model in the AI-mediated demonstration.
    ``mono_lambda`` adds a soft-monotonicity prior on the per-node weights (a dial toward the hard monotone
    model)."""
    irf = BoundedBernsteinIRF(degree_n=degree_n,
                              num_skills=num_skills,
                              num_items=num_items,
                              c_init=c_init,
                              s_init=s_init,
                              item_weight_rank=item_weight_rank,
                              mono_lambda=mono_lambda)
    return QuadratureEstimator(prior=BetaJacobiPrior(prior[0], prior[1], num_nodes), irf=irf)


def build_quad_ispline_estimator_free(degree_n: int,
                                      num_users: int,
                                      num_skills: int,
                                      num_items: int,
                                      num_nodes: int = 30,
                                      item_weight_rank: int | None = None,
                                      order: int = 3,
                                      prior: tuple[float, float] = (1.0, 1.0)) -> QuadratureEstimator:
    """Quadrature estimator with a freely-learned, locally-supported I-spline IRF."""
    irf = ISplineIRF(degree_n=degree_n, num_skills=num_skills, num_items=num_items,
                     item_weight_rank=item_weight_rank, order=order)
    return QuadratureEstimator(prior=BetaJacobiPrior(prior[0], prior[1], num_nodes), irf=irf)


def build_quad_bernstein_estimator_spline_g(degree_n: int,
                                            num_users: int,
                                            num_skills: int,
                                            num_items: int,
                                            num_nodes: int = 30,
                                            gen_degree: int = 6,
                                            gen_order: int = 3,
                                            item_weight_rank: int | None = None,
                                            prior: tuple[float, float] = (1.0, 1.0)) -> QuadratureEstimator:
    """Quadrature estimator whose Bernstein weights come from a monotone I-spline generator."""
    irf = SplineBernsteinIRF(degree_n=degree_n, num_skills=num_skills, num_items=num_items,
                             gen_degree=gen_degree, gen_order=gen_order,
                             item_weight_rank=item_weight_rank)
    return QuadratureEstimator(prior=BetaJacobiPrior(prior[0], prior[1], num_nodes), irf=irf)


def build_quad_bernstein_estimator_cdf_g(degree_n: int,
                                         num_users: int,
                                         num_skills: int,
                                         num_items: int,
                                         num_nodes: int = 30,
                                         item_weight_rank: int | None = None,
                                         prior: tuple[float, float] = (1.0, 1.0)) -> QuadratureEstimator:
    """Quadrature estimator whose Bernstein weights come from a Kumaraswamy-CDF generator."""
    irf = CDFBernsteinIRF(degree_n=degree_n,
                          num_skills=num_skills,
                          num_items=num_items,
                          item_weight_rank=item_weight_rank)
    return QuadratureEstimator(prior=BetaJacobiPrior(prior[0], prior[1], num_nodes), irf=irf)
