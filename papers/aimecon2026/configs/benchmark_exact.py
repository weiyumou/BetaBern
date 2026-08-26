"""Exact closed-form IRT vs. quadrature vs. direct IRT, at a feasible Bernstein degree (real data).

Two claims at one matched, *feasible* degree (10):
  1. **Faithfulness** — the EXACT Beta-mixture estimator and its Gauss-Jacobi quadrature twin reach the same
     held-out NLL (quadrature is a controllably-exact evaluator of the closed form; cf. the ~1e-5
     fixed-parameter check).
  2. **Equivalence** — both are TOST-equivalent to the *direct* IRT (``R`` + Normal): the genuine closed-form
     estimator matches standard IRT at no practical accuracy cost, with feasible computation.

  | family   | direct (R + Normal)   | quadrature bridge          | exact (Beta mixture)        |
  |----------|-----------------------|----------------------------|-----------------------------|
  | logistic | ``quad_logistic_irt`` | ``quad_logistic_bern_irt`` | ``exact_logistic_bern_irt`` |
  | ogive    | ``quad_ogive_irt``    | ``quad_ogive_bern_irt``    | ``exact_ogive_bern_irt``    |
  | I-spline | ``quad_spline_irt``   | ``quad_spline_bern_irt``   | ``exact_spline_bern_irt``   |
  | free     | (none — see below)    | ``quad_bern_free``         | ``exact_bern_free``         |

The ``free`` row is a **generality ablation**: a fully nonparametric monotone IRF (every Bernstein weight
learned, no parametric curve, no generator). It has no direct-``R`` counterpart, so it is paired only as
exact == quad (faithfulness), demonstrating the closed form survives even a free-form monotone IRF.

Degree is fixed at 10 on *all* Bernstein sides — the regime where the exact fold (``O(B*S^2*m^2)``) is cheap
on every dataset. Any residual bridged−direct gap here is the degree-10 Bernstein *approximation* of the
curve (below the ~40 saturation point); it closes at higher degree, where the quadrature cost is *linear*
in degree (``O(B*S*Q*m)``) but the exact fold is not — so quadrature, validated identical to exact here, is
the scalable realization. The faithful-sampling (degree-100, saturated) quadrature-only run is this same config
with the exact twins dropped::

    betabern benchmark --config papers/aimecon2026/configs/benchmark_exact.py --degree 100 \\
        --models static_logistic_irt,quad_logistic_irt,quad_logistic_bern_irt,quad_ogive_irt,quad_ogive_bern_irt,quad_spline_irt,quad_spline_bern_irt

**Prior matching + link width.** The IRT bridge samples the curve on ``eta in [-h, +h]`` via the link
``eta = 2h*θ - h`` (``eta_half_width`` = h). The matched symmetric Beta prior keeps ``Var(eta) = 1``, the direct
models' ``N(0, 1)``: ``alpha = beta = (4h^2 - 4) / 8`` — i.e. ``h=3`` → ``Beta(4, 4)``, ``h=4`` → ``Beta(7.5, 7.5)``.
The default ``Beta(1, 1)`` is a *wider* prior (``Var(eta) = 3`` at ``h=3``) that under-regularizes and leaves a
small residual NLL gap against the direct models; matching the prior closes it. Change ``h`` and ``bern_prior``
together.

All six non-longitudinal IRW datasets are included (the exact fold is feasible at degree 10 even on the
larger ones). The exact models dominate runtime — run with ``--workers 6``.
"""
CONFIG = {
    "datasets": [  # all non-longitudinal (single-occasion) dichotomous educational tables
        {"name": "art", "irw": "art"},                              # Author Recognition (1402x50)
        {"name": "spelling", "irw": "spelling_assessment_study1"},  # spelling (673x109)
        {"name": "malawi_math", "irw": "gilbert_meta_39"},          # education RCT, math (6818x10)
        {"name": "online_rct", "irw": "gilbert_meta_1"},            # large education RCT (7797x30)
        {"name": "content_lit", "irw": "gilbert_meta_2"},           # content-literacy RCT (2174x20)
        {"name": "vocab", "irw": "gilbert_meta_11"},                # vocabulary knowledge (2588x24)
    ],
    "models": [
        "static_logistic_irt",  # JML logistic baseline

        "quad_logistic_irt",  # direct reference + paired-test baseline
        "quad_logistic_bern_irt",  # logistic bridge, quadrature
        "exact_logistic_bern_irt",  # logistic bridge, EXACT  (== quad_logistic_bern_irt?)

        {"key": "quad_ogive_irt", "overrides": {"irt_model": "2PO"}},  # 2PO (normal ogive)
        {"key": "quad_ogive_bern_irt", "overrides": {"irt_model": "2PO"}},  # 2PO curve sampled into Bernstein
        {"key": "exact_ogive_bern_irt", "overrides": {"irt_model": "2PO"}},  # 2PO curve sampled into Bernstein

        {"key": "quad_spline_irt", "overrides": {"degree": 10}},  # I-spline slot, quadrature
        "quad_spline_bern_irt",
        "exact_spline_bern_irt",  # I-spline slot, EXACT    (== quad_spline_bern_irt?)

        # Generality ablation: a fully FREE-FORM monotone IRF (every Bernstein weight learned, no parametric
        # curve and no generator). Has NO direct-R twin -> not an exact-vs-direct row; included to show the
        # closed form survives even a nonparametric monotone IRF. Paired only as exact == quad (faithfulness).
        "quad_bern_free",   # free-form monotone Bernstein, quadrature
        "exact_bern_free",  # free-form monotone Bernstein, EXACT  (== quad_bern_free?)
    ],
    "splits": ["within_random"],
    "n_folds": 10,
    "baseline": "quad_logistic_irt",
    "hyperparams": {
        "irt_model": "2PL",
        "degree": 10,  # matched LOW Bernstein degree so the exact fold is feasible on both sides
        "num_nodes": 80,  # Gauss-Jacobi nodes for the quadrature twins
        "item_weight_rank": 2,
        "ispline_order": 2,
        "gen_degree": 10, "gen_order": 2,  # slot I-spline generator (shared by quad + exact slot)
        "eta_half_width": 3.0,  # IRT bridge samples eta in [-3, +3] (link eta = 6θ-3); pairs with bern_prior below
        "bern_prior": (4.0, 4.0),  # matched to N(0,1) under eta=6θ-3: alpha=beta=(4h^2-4)/8 (same prior on both sides)
    },
    "trainer": {"max_epochs": 100, "patience": 10, "accelerator": "cpu", "batch_size": 256},
}
