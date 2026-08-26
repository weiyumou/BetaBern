"""
Model registry for the benchmark: name -> builder returning ``(model, task)``.

The default lineup is the Core 4 (a JML baseline, the MML logistic baseline, the Bernstein control,
and the proposed free-Bernstein model), plus the exact (cancellation-free) MML counterparts of the two
Bernstein models. Add a model by adding one builder here. All builders take the datamodule (for dims)
and a hyperparameter dict.
"""
from betabern.bernstein.estimator.exact import (
    build_exact_bernstein_estimator_bounded,
    build_exact_bernstein_estimator_free,
    build_exact_bernstein_estimator_irt,
    build_exact_bernstein_estimator_spline_g,
)
from betabern.bernstein.estimator.quad import (
    build_quad_bernstein_estimator_bounded,
    build_quad_bernstein_estimator_cdf_g,
    build_quad_bernstein_estimator_free,
    build_quad_bernstein_estimator_irt,
    build_quad_bernstein_estimator_spline_g,
    build_quad_ispline_estimator_free,
)
from betabern.core.task import BayesianEstimatorTask
from betabern.irt.estimator.quad_irt import build_quad_irt_estimator, build_quad_spline_irt_estimator
from betabern.irt.estimator.static_irt import StaticIRT
from betabern.irt.task import StaticIRTTask

DEFAULT_HP = {
    "degree": 6, "num_nodes": 40, "irt_model": "2PL",
    "item_weight_rank": -1, "lr": 0.02, "theta_reg": 1.0,
    "ispline_order": 3,  # spline order for quad_ispline_free (2 = piecewise-linear, 3 = quadratic)
    "gen_degree": 6, "gen_order": 3,  # Bernstein-slot spline generator: resolution m_g and order
    "bern_prior": (1.0, 1.0),  # Beta(alpha, beta) prior of the [0,1] Bernstein-bridged models; (4,4) ~ N(0,1) under eta=6θ-3
    "bound_mono": 0.0,  # soft-monotonicity penalty for the bounded free model (dial toward the monotone model)
    "eta_half_width": 3.0,  # IRT-bridge link eta = 2h*θ - h, sampled on [-h,+h]; match the prior so Var(eta)=1 (h=3->Beta(4,4), h=4->Beta(7.5,7.5))
}


def _static_logistic_irt(dm, hp):
    model = StaticIRT(num_users=dm.stats.num_users, num_skills=dm.stats.num_skills, num_items=dm.stats.num_items,
                      irt_model=hp["irt_model"], item_weight_rank=hp["item_weight_rank"])
    task = StaticIRTTask(model, learning_rate=hp["lr"], theta_reg=hp["theta_reg"], eap_num_nodes=hp["num_nodes"])
    return model, task


def _require_link(model, link: str, hp: dict):
    """Guard the link/key match: a config must set an irt_model with the link this registry key promises
    (logistic='2PL', ogive='2PO'). Reads the link back off the built model (the single source of truth)."""
    got = model.irf.irt_base.link
    if got != link:
        raise ValueError(f"this registry key expects a {link} irt_model "
                         f"(logistic='2PL', ogive='2PO'); got '{hp['irt_model']}' ({got}) — set it in the config")


def _quad_irt(link: str):  # direct IRT; the link is set explicitly by the config's irt_model
    def build(dm, hp):
        model = build_quad_irt_estimator(num_users=dm.stats.num_users, num_skills=dm.stats.num_skills, num_items=dm.stats.num_items,
                                         num_nodes=hp["num_nodes"], irt_model=hp["irt_model"],
                                         item_weight_rank=hp["item_weight_rank"])
        _require_link(model, link, hp)
        return model, BayesianEstimatorTask(model, learning_rate=hp["lr"])
    return build


def _quad_spline_irt(dm, hp):
    model = build_quad_spline_irt_estimator(num_skills=dm.stats.num_skills, num_items=dm.stats.num_items,
                                                 degree_n=hp["degree"], order=hp.get("ispline_order", 3),
                                                 num_nodes=hp["num_nodes"], item_weight_rank=hp["item_weight_rank"])
    return model, BayesianEstimatorTask(model, learning_rate=hp["lr"])


def _quad_bern_irt(link: str):  # IRT curve sampled into the Bernstein basis; link set explicitly by the config
    def build(dm, hp):
        model = build_quad_bernstein_estimator_irt(degree_n=hp["degree"], num_users=dm.stats.num_users,
                                                   num_skills=dm.stats.num_skills, num_items=dm.stats.num_items,
                                                   num_nodes=hp["num_nodes"], irt_model=hp["irt_model"],
                                                   item_weight_rank=hp["item_weight_rank"],
                                                   prior=tuple(hp.get("bern_prior", (1.0, 1.0))),
                                                   eta_half_width=hp.get("eta_half_width", 3.0))
        _require_link(model, link, hp)
        return model, BayesianEstimatorTask(model, learning_rate=hp["lr"])
    return build


def _quad_bern_free(dm, hp):
    model = build_quad_bernstein_estimator_free(degree_n=hp["degree"], num_users=dm.stats.num_users,
                                                num_skills=dm.stats.num_skills, num_items=dm.stats.num_items,
                                                num_nodes=hp["num_nodes"], item_weight_rank=hp["item_weight_rank"],
                                                prior=tuple(hp.get("bern_prior", (1.0, 1.0))))
    return model, BayesianEstimatorTask(model, learning_rate=hp["lr"])


def _quad_bern_free_bounded(dm, hp):
    model = build_quad_bernstein_estimator_bounded(degree_n=hp["degree"], num_users=dm.stats.num_users,
                                                   num_skills=dm.stats.num_skills, num_items=dm.stats.num_items,
                                                   num_nodes=hp["num_nodes"], item_weight_rank=hp["item_weight_rank"],
                                                   prior=tuple(hp.get("bern_prior", (1.0, 1.0))),
                                                   mono_lambda=hp.get("bound_mono", 0.0))
    return model, BayesianEstimatorTask(model, learning_rate=hp["lr"])


def _quad_ispline_free(dm, hp):
    model = build_quad_ispline_estimator_free(degree_n=hp["degree"], num_users=dm.stats.num_users,
                                              num_skills=dm.stats.num_skills, num_items=dm.stats.num_items,
                                              num_nodes=hp["num_nodes"], item_weight_rank=hp["item_weight_rank"],
                                              order=hp.get("ispline_order", 3),
                                              prior=tuple(hp.get("bern_prior", (1.0, 1.0))))
    return model, BayesianEstimatorTask(model, learning_rate=hp["lr"])


def _quad_spline_bern_irt(dm, hp):
    model = build_quad_bernstein_estimator_spline_g(degree_n=hp["degree"], num_users=dm.stats.num_users,
                                                    num_skills=dm.stats.num_skills, num_items=dm.stats.num_items,
                                                    num_nodes=hp["num_nodes"], gen_degree=hp["gen_degree"],
                                                    gen_order=hp["gen_order"], item_weight_rank=hp["item_weight_rank"],
                                                    prior=tuple(hp.get("bern_prior", (1.0, 1.0))))
    return model, BayesianEstimatorTask(model, learning_rate=hp["lr"])


def _quad_bern_cdfg(dm, hp):
    model = build_quad_bernstein_estimator_cdf_g(degree_n=hp["degree"], num_users=dm.stats.num_users,
                                                 num_skills=dm.stats.num_skills, num_items=dm.stats.num_items,
                                                 num_nodes=hp["num_nodes"], item_weight_rank=hp["item_weight_rank"],
                                                 prior=tuple(hp.get("bern_prior", (1.0, 1.0))))
    return model, BayesianEstimatorTask(model, learning_rate=hp["lr"])


def _exact_bern_irt(link: str):  # IRT curve sampled into Bernstein (exact, cancellation-free); link set by the config
    def build(dm, hp):
        model = build_exact_bernstein_estimator_irt(degree_n=hp["degree"], num_users=dm.stats.num_users,
                                                    num_skills=dm.stats.num_skills, num_items=dm.stats.num_items,
                                                    irt_model=hp["irt_model"], item_weight_rank=hp["item_weight_rank"],
                                                    prior=tuple(hp.get("bern_prior", (1.0, 1.0))))
        _require_link(model, link, hp)
        return model, BayesianEstimatorTask(model, learning_rate=hp["lr"])

    return build


def _exact_spline_bern_irt(dm, hp):
    # Exact (cancellation-free) MML counterpart of quad_spline_bern_irt. The fold builds a degree-(N*degree)
    # Bernstein polynomial, so this is only feasible at *moderate* degree — use it to confirm exact == quad
    # (the conjugate posterior is faithful), not for high-degree training.
    model = build_exact_bernstein_estimator_spline_g(degree_n=hp["degree"], num_users=dm.stats.num_users,
                                                     num_skills=dm.stats.num_skills, num_items=dm.stats.num_items,
                                                     gen_degree=hp["gen_degree"], gen_order=hp["gen_order"],
                                                     item_weight_rank=hp["item_weight_rank"],
                                                     prior=tuple(hp.get("bern_prior", (1.0, 1.0))))
    return model, BayesianEstimatorTask(model, learning_rate=hp["lr"])


def _exact_bern_free(dm, hp):
    model = build_exact_bernstein_estimator_free(degree_n=hp["degree"], num_users=dm.stats.num_users,
                                                 num_skills=dm.stats.num_skills, num_items=dm.stats.num_items,
                                                 item_weight_rank=hp["item_weight_rank"],
                                                 prior=tuple(hp.get("bern_prior", (1.0, 1.0))))
    return model, BayesianEstimatorTask(model, learning_rate=hp["lr"])


def _exact_bern_free_bounded(dm, hp):
    model = build_exact_bernstein_estimator_bounded(degree_n=hp["degree"], num_users=dm.stats.num_users,
                                                    num_skills=dm.stats.num_skills, num_items=dm.stats.num_items,
                                                    item_weight_rank=hp["item_weight_rank"],
                                                    prior=tuple(hp.get("bern_prior", (1.0, 1.0))),
                                                    mono_lambda=hp.get("bound_mono", 0.0))
    return model, BayesianEstimatorTask(model, learning_rate=hp["lr"])


MODELS = {
    # IRT models read the link (logistic 2PL / normal-ogive 2PO) from the config's irt_model; set it
    # explicitly per model (e.g. an override irt_model="2PO" on the ogive entries). Each key asserts the
    # config's link matches the key's promise, so a mismatch fails loudly instead of running the wrong model.
    "static_logistic_irt": _static_logistic_irt,       # JML IRT baseline (EAP scoring)
    "quad_logistic_irt": _quad_irt("logistic"),        # MML IRT, ℝ + Normal, Gauss-Hermite  (config: irt_model="2PL")
    "quad_ogive_irt": _quad_irt("ogive"),              # same builder; config: irt_model="2PO"
    "quad_spline_irt": _quad_spline_irt,  # free monotone I-spline on R + Normal prior (Gauss-Hermite MML)
    "quad_logistic_bern_irt": _quad_bern_irt("logistic"),  # IRT curve sampled into Bernstein (conjugate); irt_model="2PL"
    "quad_ogive_bern_irt": _quad_bern_irt("ogive"),        # same builder; config: irt_model="2PO"
    "quad_bern_free": _quad_bern_free,  # proposed: free monotone Bernstein (Gauss-Jacobi MML)
    "quad_bern_free_bounded": _quad_bern_free_bounded,
    # free BOUNDED (non-monotone) Bernstein — "monotonicity is optional" control
    "quad_ispline_free": _quad_ispline_free,  # proposed: free monotone I-spline (local basis, Gauss-Jacobi MML)
    "quad_spline_bern_irt": _quad_spline_bern_irt,  # Bernstein slot, I-spline weight-generator (conjugate)
    "quad_bern_cdfg": _quad_bern_cdfg,        # Bernstein slot, Kumaraswamy-CDF weight-generator (conjugate)
    "exact_logistic_bern_irt": _exact_bern_irt("logistic"),
    # exact counterpart of quad_logistic_bern_irt; irt_model="2PL"
    "exact_ogive_bern_irt": _exact_bern_irt("ogive"),  # same builder; config: irt_model="2PO"
    "exact_spline_bern_irt": _exact_spline_bern_irt,  # exact MML counterpart of quad_spline_bern_irt (moderate degree)
    "exact_bern_free": _exact_bern_free,  # exact (cancellation-free) MML counterpart of quad_bern_free
    "exact_bern_free_bounded": _exact_bern_free_bounded,
    # exact counterpart of quad_bern_free_bounded (non-monotone control)
}


def build(key: str, dm, hp: dict):
    """Build ``(model, task)`` for a registry key with the given datamodule + hyperparameters."""
    if key not in MODELS:
        raise KeyError(f"unknown model '{key}'; available: {list(MODELS)}")
    return MODELS[key](dm, hp)
