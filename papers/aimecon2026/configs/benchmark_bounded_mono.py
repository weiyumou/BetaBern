"""Does the bounded model's held-out CV win survive the soft-monotonicity prior? (predict-vs-interpret test)

The bounded (non-monotone) free Bernstein model beats the monotone free model on held-out 10-fold CV across all
six datasets — genuine generalization. But its fitted ICCs are non-monotone in data-sparse regions (a boundary
spike on `art`) that the model-free empirical ICC does not support. This run asks the decisive question on the
*test* folds (not a single validation split): when we add a soft-monotonicity prior that pulls the curves
toward interpretable, near-monotone shapes, does the predictive edge persist or disappear?

Lineup (all conjugate Beta-Bernstein, identical except the constraint/prior), baseline = monotone free:

  | model         | weights                        | soft-mono λ |
  |---------------|--------------------------------|-------------|
  | quad_bern_free| free, monotone-chained         | —           |
  | bounded_l0    | free, bounded (non-monotone)   | λ = 0       |
  | bounded_l20   | free, bounded (non-monotone)   | λ = 20      |

Read paired.csv for bounded_l0 vs monotone (the original win) and bounded_l20 vs monotone (the win once the
curves are regularized toward monotone). Edge survives -> real predictive signal + clean ICCs; edge vanishes
-> the win was tied to the non-interpretable, non-monotone flexibility.

    betabern benchmark --config papers/aimecon2026/configs/benchmark_bounded_mono.py --workers 6
"""
CONFIG = {
    "datasets": [
        {"name": "art", "irw": "art"},
        {"name": "spelling", "irw": "spelling_assessment_study1"},
        {"name": "malawi_math", "irw": "gilbert_meta_39"},
        {"name": "online_rct", "irw": "gilbert_meta_1"},
        {"name": "content_lit", "irw": "gilbert_meta_2"},
        {"name": "vocab", "irw": "gilbert_meta_11"},
    ],
    "models": [
        "quad_bern_free",  # monotone free Bernstein (baseline)
        {"key": "quad_bern_free_bounded", "name": "bounded_l0", "overrides": {"bound_mono": 0.0}},
        {"key": "quad_bern_free_bounded", "name": "bounded_l20", "overrides": {"bound_mono": 20.0}},
    ],
    "splits": ["within_random"],
    "n_folds": 10,
    "baseline": "quad_bern_free",
    "hyperparams": {
        "irt_model": "2PL",
        "degree": 20,
        "num_nodes": 80,
        "item_weight_rank": 2,
        "eta_half_width": 3.0,
        "bern_prior": (4.0, 4.0),
    },
    "trainer": {"max_epochs": 100, "patience": 10, "accelerator": "cpu", "batch_size": 256},
}
