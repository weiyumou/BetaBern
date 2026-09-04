"""Non-monotone capability demonstration: fit IRFs to simulated AI-mediated assessment data.

Generative model (per item, ability theta on [0,1], with the IRT link eta = 6*theta - 3):

    p_S(theta) = sigmoid(a * (eta - b))               # student-alone success (a 2PL curve)
    r(theta)   = sigmoid(-(eta - tr) / sr)            # reliance: prob. the student defers to the AI
    P(theta)   = r(theta) * p_A + (1 - r(theta)) * p_S(theta)

Standard items set r == 0 (a plain monotone 2PL). AI-mediated items use a decreasing r and a competent AI
(p_A high), which carves a NON-monotone "valley": weak respondents ride the AI, mid-ability respondents dip.

Fairness: the student IRF is a genuine 2PL logistic (only the AI mixture bends the realized curve off-logistic),
and the true ability is drawn Normal(0,1) by default -- so the 4PL's Normal prior is *correctly specified* and
the Bernstein Beta(4,4) prior is the mildly misspecified one. The design does not favour the proposed model.

We fit three estimators by exact/Gauss-Hermite MML and compare:
    - 4PL direct IRT            (strongest MONOTONE parametric baseline; both asymptotes)
    - monotone free Bernstein   (strongest MONOTONE nonparametric; isolates monotonicity as the culprit)
    - bounded free Bernstein    (OURS: conjugate, non-monotone-capable; closed-form Beta-mixture posterior)

Deliverables: (1) IRF recovery (bounded traces the valley; the monotone models cannot), (2) held-out
marginal NLL, (3) the closed-form ability posterior under a recovered valley IRF is MULTIMODAL -- exactly
where a Gaussian SE would place its single peak the true posterior has a trough.

    python papers/aimecon2026/scripts/ai_mediated_fit.py --out figures/
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as sps
from scipy.stats import beta as beta_dist
from scipy.stats import norm as norm_dist

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for figstyle (sibling module)

import figstyle

figstyle.apply()

from betabern.benchmark.simfit import fit_mml
from betabern.bernstein.estimator.exact import (
    build_exact_bernstein_estimator_bounded,
    build_exact_bernstein_estimator_free,
)
from betabern.irt.estimator.quad_irt import build_quad_irt_estimator

C_TRUE = figstyle.OKABE_ITO["black"]
C_4PL = figstyle.OKABE_ITO["blue"]
C_MONO = figstyle.OKABE_ITO["green"]
C_BOUND = figstyle.OKABE_ITO["vermillion"]
H = 3.0  # link half-width: eta = 2H*theta - H


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def to_eta(theta01):
    """Bernstein ability ``theta in [0,1]`` -> the IRT scale, via the link ``eta = 2H*theta - H``."""
    return 2 * H * np.asarray(theta01, dtype=float) - H


def item_spec(kind="ai", *, a=1.6, b=0.0, p_ai=0.95, tr=0.0, sr=0.55, r_max=1.0):
    """One item of the generative model. ``kind='std'`` pins reliance to 0 (a plain 2PL)."""
    return dict(type=kind, a=a, b=b, p_ai=(p_ai if kind == "ai" else 0.0), tr=tr, sr=sr, r_max=r_max)


def hybrid_parts(eta, spec):
    """The AI-mediated generative model on the IRT scale ``eta in R`` -- the single definition.

    Returns ``(P, p_student, reliance)`` so callers can draw the mechanism (which ingredient bends the
    curve) as well as the realized IRF, without restating the model.

        p_S(eta) = sigmoid(a * (eta - b))                  # student alone (a 2PL curve)
        r(eta)   = r_max * sigmoid(-(eta - tr) / sr)       # reliance: P(defer to the AI); 0 for 'std' items
        P(eta)   = r * p_A + (1 - r) * p_S                 # reliance-weighted mixture
    """
    eta = np.asarray(eta, dtype=float)
    p_s = sig(spec["a"] * (eta - spec["b"]))
    if spec["type"] == "ai":
        r = spec.get("r_max", 1.0) * sig(-(eta - spec["tr"]) / spec["sr"])
        p_a = spec["p_ai"]
    else:
        r, p_a = np.zeros_like(eta), 0.0
    return r * p_a + (1 - r) * p_s, p_s, r


def hybrid_p(theta01, spec):
    """Realized ``P(correct)`` on the ``[0,1]`` Bernstein grid -- :func:`hybrid_parts` through the link."""
    return hybrid_parts(to_eta(theta01), spec)[0]


def make_items(p_ai=0.95, sr=0.55):
    """A small bank: 4 standard monotone 2PL items + 6 AI-mediated valley items.

    ``p_ai`` (AI competence) and ``sr`` (reliance scale; smaller = sharper ability-graded reliance) control
    how pronounced the AI-mediation valley is. The defaults are the 'pronounced' setting -- a very competent
    AI with sharp reliance, so a correct response is genuinely ambiguous between low ability (rode the AI) and
    high (did it alone). The sensitivity study uses the milder ``p_ai=0.92, sr=0.8``.
    """
    std = [item_spec("std", b=b, sr=1.0) for b in (-1.2, -0.4, 0.4, 1.2)]
    ai = [item_spec("ai", b=b, p_ai=p_ai, sr=sr) for b in (-0.4, 0.0, 0.4, 0.8, 1.2, 1.6)]
    return std + ai


def generate(num_students, items, a0, seed=0, ability="normal"):
    rng = np.random.default_rng(seed)
    if ability == "normal":
        eta = np.clip(rng.normal(0.0, 1.0, size=num_students), -H + 1e-3, H - 1e-3)  # IRT-standard ability
        theta01 = (eta + H) / (2 * H)                        # map eta in [-3,3] -> [0,1] for the Bernstein side
    else:
        theta01 = rng.beta(a0, a0, size=num_students)        # bounded ability matching the Beta prior
    P = np.stack([hybrid_p(theta01, it) for it in items], axis=1)  # (B, I)
    y = rng.binomial(1, P)                                    # (B, I)
    return theta01, y


def as_batch(y, cols=None):
    """Padded batch for the estimators. ``cols`` selects a subset of (global) item columns; default = all,
    carrying the *global* item ids so the model's per-item params line up for any subset."""
    B, n_items = y.shape
    cols = list(range(n_items)) if cols is None else list(cols)
    skill = torch.zeros(B, dtype=torch.long)
    item = torch.tensor(cols, dtype=torch.long).expand(B, len(cols)).contiguous()
    answers = torch.as_tensor(np.asarray(y)[:, cols], dtype=torch.long)
    mask = torch.ones(B, len(cols), dtype=torch.bool)
    return skill, item, answers, mask


def held_out_nll(model, batch):
    skill, item, answers, mask = batch
    with torch.no_grad():
        ev = model.log_marginal_evidence(skill, item, answers, mask)
    return float(-ev.mean())


def bern_curve(model, item_idx, theta01):
    """Fitted Bernstein IRF P(correct) on a [0,1] grid for one item."""
    th = torch.tensor(theta01, dtype=torch.float)
    logb = model.irf.log_basis(th)                                   # (Q, n+1)
    item_ids = torch.tensor([[item_idx]])
    skill = torch.zeros(1, 1, dtype=torch.long)
    with torch.no_grad():
        logp = model.irf.log_irf_from_basis(logb, item_ids, skill)   # (1,1,2,Q)
    return logp[0, 0, 1].exp().numpy()


def irt_curve(model, item_idx, theta01):
    """Fitted direct-IRT IRF on the [0,1] grid (evaluate on the R grid eta=6*theta-3, same axis)."""
    eta = torch.tensor(to_eta(theta01), dtype=torch.float).unsqueeze(0)   # (1, Q)
    item_ids = torch.tensor([[item_idx]])
    skill = torch.zeros(1, 1, dtype=torch.long)
    with torch.no_grad():
        logp = model.irf.log_irf(eta, item_ids, skill)               # (1,1,2,Q)
    return logp[0, 0, 1].exp().numpy()


def fig_recovery(items, models, theta01, out, formats):
    """Two representative items: a standard 2PL (all agree) and an AI-mediated valley (only bounded fits)."""
    std_idx = next(i for i, it in enumerate(items) if it["type"] == "std" and abs(it["b"]) < 0.5)
    ai_idx = next(i for i, it in enumerate(items) if it["type"] == "ai" and abs(it["b"]) < 0.1)
    eta = to_eta(theta01)
    fig, axes = plt.subplots(1, 2, figsize=figstyle.figsize("text", 0.46))
    for ax, idx, title in ((axes[0], std_idx, "(a) Standard item (monotone 2PL)"),
                           (axes[1], ai_idx, "(b) AI-mediated item (non-monotone)")):
        ax.plot(eta, hybrid_p(theta01, items[idx]), color=C_TRUE, lw=2.4, ls=":", label="True IRF", zorder=5)
        ax.plot(eta, irt_curve(models["4pl"], idx, theta01), color=C_4PL, lw=1.8, ls="-", label="Monotone 4PL")
        ax.plot(eta, bern_curve(models["mono"], idx, theta01), color=C_MONO, lw=1.8, ls="--",
                label="Monotone Bernstein")
        ax.plot(eta, bern_curve(models["bound"], idx, theta01), color=C_BOUND, lw=2.4, ls="-",
                label="Flexible Bernstein")
        ax.set(xlabel="Ability", ylabel="P(correct)", ylim=(-0.03, 1.03), title=title)
        ax.set_xlim(eta.min() - 0.25, eta.max() + 0.25)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(out / f"ai_mediated_recovery.{fmt}")
    plt.close(fig)
    print(f"  wrote ai_mediated_recovery.{{{','.join(formats)}}}")


def draw_posterior(ax, model, items, *, xlabel="ability  η = 6θ−3",
                   title="Posterior under a recovered AI-mediated IRF", legend_loc="upper center",
                   compact=False):
    """Draw the exact Beta-mixture posterior (multimodal) vs. its Gaussian (mean +/- SD) summary on ``ax``,
    for a respondent who answers all AI-mediated items correctly. ``compact`` uses short legend labels so the
    box fits a narrow (e.g. half-width) panel. Returns (peaks, mean, std)."""
    ai_ids = [i for i, it in enumerate(items) if it["type"] == "ai"]
    skill = torch.zeros(1, dtype=torch.long)
    item = torch.tensor([ai_ids], dtype=torch.long)
    answers = torch.ones(1, len(ai_ids), dtype=torch.long)   # all AI-mediated items correct
    mask = torch.ones(1, len(ai_ids), dtype=torch.bool)
    post = model.posterior_stats(skill, item, answers, mask, level=0.95)
    mix = post.mixture
    a = mix.comp_alphas[0].detach().numpy()
    b = mix.comp_betas[0].detach().numpy()
    pi = mix.log_mix_weights[0].exp().detach().numpy()
    mean, std = float(post.mean), float(post.std)

    th = np.linspace(1e-3, 1 - 1e-3, 600)
    eta = to_eta(th)
    exact = (pi[None, :] * beta_dist.pdf(th[:, None], a[None, :], b[None, :])).sum(-1) / (2 * H)  # density in eta
    gauss = norm_dist.pdf(th, mean, std) / (2 * H)
    peaks = int(((exact[1:-1] > exact[:-2]) & (exact[1:-1] > exact[2:])).sum())

    exact_label = f"Exact ({peaks} modes)" if compact else f"exact Beta-mixture ({peaks} modes)"
    gauss_label = "Gaussian" if compact else "Gaussian (mean ± SD)"
    ax.fill_between(eta, exact, color=C_BOUND, alpha=0.18)
    ax.plot(eta, exact, color=C_BOUND, lw=2.4, label=exact_label)
    ax.plot(eta, gauss, color=C_TRUE, lw=1.8, ls="--", label=gauss_label)
    ax.axvline(to_eta(mean), color=figstyle.OKABE_ITO["grey"], lw=1.0, ls=":")
    ax.set(xlabel=xlabel, ylabel="posterior density", title=title)
    ax.set_xlim(eta.min() - 0.25, eta.max() + 0.25)
    ax.margins(y=0.08)
    ax.legend(fontsize=8.5, loc=legend_loc, frameon=True, framealpha=0.85,
              facecolor="white", edgecolor="none")
    return peaks, mean, std


def fig_posterior(model, items, out, formats):
    """A respondent who answers the AI-mediated items correctly: exact Beta-mixture posterior (multimodal)
    vs. the Gaussian (mean +/- SD) approximation that a point-SE would report."""
    fig, ax = plt.subplots(figsize=figstyle.figsize(3.4, 0.82))
    peaks, mean, std = draw_posterior(ax, model, items)
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(out / f"ai_mediated_posterior.{fmt}")
    plt.close(fig)
    print(f"  wrote ai_mediated_posterior.{{{','.join(formats)}}}  (exact modes={peaks}, "
          f"mean={mean:.3f}, sd={std:.3f})")


LABELS = {"4pl": "4PL IRT (monotone)", "mono": "free Bernstein (monotone)",
          "bound": "free Bernstein (bounded, ours)"}

# The paper's Table 3 reports the MEDIAN over 20 simulation runs. The median is the intended default:
# the free-Bernstein objective is non-convex, so an occasional run settles in a pathological tail fit
# that moves a mean but not a median. Keep `--seeds 20 --agg median` to reproduce the published table.
PAPER_SEEDS = 20
AGG_CHOICES = ("median", "mean")


def add_study_args(p, *, seeds=PAPER_SEEDS):
    """The knobs shared by the multi-run study scripts, so their defaults cannot drift apart."""
    p.add_argument("--students", type=int, default=2000)
    p.add_argument("--degree", type=int, default=12)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seeds", type=int, default=seeds, help="independent simulation runs (paper: 20).")
    p.add_argument("--prior-a", type=float, default=4.0)
    p.add_argument("--agg", choices=AGG_CHOICES, default="median",
                   help="across-run aggregate (paper: median).")
    return p


def priors(a0):
    """The matched prior pair: Beta(a0,a0) on [0,1] and the Normal SD on eta = 2H*theta - H."""
    return (a0, a0), H / np.sqrt(2 * a0 + 1)


EFFECTS = {"pronounced": dict(p_ai=0.95, sr=0.55), "mild": dict(p_ai=0.92, sr=0.80)}


def build_models(students, degree, n_items, prior, prior_std):
    return {
        "4pl": build_quad_irt_estimator(num_users=students, num_skills=1, num_items=n_items,
                                        num_nodes=48, irt_model="4PL", item_weight_rank=2,
                                        prior_mean=0.0, prior_std=prior_std),
        "mono": build_exact_bernstein_estimator_free(degree_n=degree, num_users=students,
                                                     num_skills=1, num_items=n_items, item_weight_rank=2, prior=prior),
        "bound": build_exact_bernstein_estimator_bounded(degree_n=degree, num_users=students,
                                                         num_skills=1, num_items=n_items, item_weight_rank=2, prior=prior),
    }


def fit_and_eval(args, items, n_items, a0, prior, prior_std, seed):
    """Fresh train/test, fit all three models, return (models, {key: {'full':.., 'ai':..}}).

    'full' = held-out marginal NLL over the whole response vector. 'ai' = the conditional NLL of the
    AI-mediated items GIVEN the standard items (chain rule: NLL(all) − NLL(standard-only)) -- it isolates the
    effect on the AI items instead of diluting it across the tied standard items.
    """
    std_cols = [i for i, it in enumerate(items) if it["type"] == "std"]
    _, y_tr = generate(args.students, items, a0, seed=seed, ability=args.ability)
    _, y_te = generate(args.students, items, a0, seed=seed + 10_000, ability=args.ability)
    train = as_batch(y_tr)
    test_full, test_std = as_batch(y_te), as_batch(y_te, std_cols)
    models = build_models(args.students, args.degree, n_items, prior, prior_std)
    nll = {}
    for key, model in models.items():
        fit_mml(model, *train, epochs=args.epochs, lr=0.05)
        full, std = held_out_nll(model, test_full), held_out_nll(model, test_std)
        nll[key] = {"full": full, "ai": full - std}  # ai = conditional NLL of AI items | standard items
    return models, nll


def report_effect_sizes(nlls, metric="full"):
    """Run-level paired effect of the bounded model vs each monotone baseline, over the simulation seeds.

    The unit of replication is the *simulation run* (a fresh dataset + refit), not the student -- so the
    paired test speaks to consistency across data realizations rather than to within-run student count
    (where N is large enough to make any non-zero gap 'significant'). We lead with the magnitude (ΔNLL in
    nats), the win rate, and the standardized effect (Cohen's dz); the p-value is secondary. (The full study
    with both AI-effect strengths and BH-FDR-corrected p-values is :mod:`scripts.ai_mediated_study`.)
    """
    bound = np.array(nlls["bound"])
    print(f"\nrun-level effect size [{metric}]  (ΔNLL = baseline − bounded; > 0 favours the bounded model):")
    for base in ("4pl", "mono"):
        d = np.array(nlls[base]) - bound
        n = len(d)
        m = d.mean()
        sd = d.std(ddof=1)
        se = sd / np.sqrt(n)
        tc = sps.t.ppf(0.975, n - 1)
        dz = m / sd if sd > 0 else float("inf")
        _, p = sps.ttest_rel(nlls[base], bound)
        print(f"  vs {LABELS[base]:<26}: ΔNLL = {m:+.4f}  95% CI [{m - tc * se:+.4f}, {m + tc * se:+.4f}]"
              f"   dz = {dz:.2f}   wins {(d > 0).mean():.0%}   p = {p:.1e}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("figures"))
    p.add_argument("--formats", default="png,pdf")
    p.add_argument("--students", type=int, default=2000,
                   help="matches the study scripts, so the figures and Table 3 share one configuration.")
    p.add_argument("--degree", type=int, default=12)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--prior-a", type=float, default=4.0,
                   help="Beta(a,a) prior for the Bernstein models; the 4PL Normal SD is matched to it.")
    p.add_argument("--ability", choices=["normal", "beta"], default="normal",
                   help="true ability law: 'normal' gives the 4PL its correctly-specified prior (conservative); "
                        "'beta' matches the Bernstein prior instead.")
    p.add_argument("--ai-pa", type=float, default=0.95, help="AI competence on AI-mediated items (higher = deeper valley).")
    p.add_argument("--ai-sr", type=float, default=0.55,
                   help="reliance scale (smaller = sharper ability-graded reliance, deeper valley).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, default=1,
                   help="number of independent simulation runs; >1 reports run-level effect sizes (figures use seed 0).")
    args = p.parse_args()
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    items = make_items(p_ai=args.ai_pa, sr=args.ai_sr)
    n_items = len(items)
    a0 = args.prior_a
    prior, prior_std = priors(a0)  # Beta(a0,a0) on [0,1] <-> matched Normal SD on eta
    spec = "correctly specified for the 4PL" if args.ability == "normal" else "correctly specified for the Bernstein"
    print(f"items: {sum(it['type']=='std' for it in items)} standard (pure 2PL) + "
          f"{sum(it['type']=='ai' for it in items)} AI-mediated; students={args.students} train/test each; "
          f"degree={args.degree}; runs={args.seeds}")
    print(f"true ability ~ {'Normal(0,1)' if args.ability == 'normal' else f'Beta({a0:g},{a0:g})'}; "
          f"student IRF is 2PL logistic, AI mediation makes the realized curve non-logistic")
    print(f"prior: Beta({a0:g},{a0:g}) on [0,1]  <->  Normal(0, {prior_std:.3f}^2) on eta  ({spec})\n")

    if args.seeds <= 1:
        models, nll = fit_and_eval(args, items, n_items, a0, prior, prior_std, args.seed)
        print("held-out NLL  (full vector | AI-conditional):")
        for key in ("4pl", "mono", "bound"):
            print(f"  {LABELS[key]:<34} full = {nll[key]['full']:.4f}   ai = {nll[key]['ai']:.4f}")
    else:
        metrics = ("full", "ai")
        acc = {m: {k: [] for k in ("4pl", "mono", "bound")} for m in metrics}
        models = None
        print(f"running {args.seeds} simulation runs (full-vector NLL per run):")
        for s in range(args.seeds):
            mdl, nll = fit_and_eval(args, items, n_items, a0, prior, prior_std, args.seed + s)
            for m in metrics:
                for k in acc[m]:
                    acc[m][k].append(nll[k][m])
            models = models or mdl  # keep run 0's models for the figures
            print(f"  run {s + 1:2d}/{args.seeds}: "
                  f"4pl={nll['4pl']['full']:.4f}  mono={nll['mono']['full']:.4f}  bound={nll['bound']['full']:.4f}")
        for m in metrics:
            report_effect_sizes(acc[m], metric=m)

    theta01 = np.linspace(1e-3, 1 - 1e-3, 300)
    fig_recovery(items, models, theta01, args.out, formats)
    fig_posterior(models["bound"], items, args.out, formats)
    print(f"\ndone -> {args.out}")


if __name__ == "__main__":
    main()
