"""Illustrative AI-mediated item response function (IRF) under a hybrid student-AI generator.

Generative model for a single item, on the standard ability scale theta in R:

    p_S(theta) = sigmoid(a * (theta - b))          # student-alone success (a 2PL curve)
    p_A        = const in [0, 1]                    # AI-alone success on this item
    r(theta)   = r_max * sigmoid(-(theta - tr)/sr)  # reliance: prob. the student defers to the AI
    P(theta)   = r(theta) * p_A + (1 - r(theta)) * p_S(theta)   # observed (hybrid) IRF

The *shape* of P(theta) is governed entirely by the reliance pattern r(theta):
  - r flat                 -> a flattened, raised-floor / lowered-ceiling curve  (monotone "dilution")
  - r decreasing in theta  -> weaker respondents lean hardest on a competent AI, so the curve can be
                              HIGH at low theta, DIP in the middle, and recover at high theta
                              (a NON-monotone valley -- outside every standard monotone IRF).

This script is illustrative only (no fitting): it draws the curves so we can see the shapes and decide
whether the AI-mediated IRF is worth a figure in the opening.

    python papers/aimecon2026/scripts/ai_mediated_irf.py --out figures/
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for figstyle (sibling module)
import ai_mediated_fit as af
import figstyle

figstyle.apply()

C_STUDENT = figstyle.OKABE_ITO["grey"]        # pure student curve (reference)
C_DILUTE = figstyle.OKABE_ITO["blue"]         # monotone dilution
C_NONMONO = figstyle.OKABE_ITO["vermillion"]  # non-monotone valley
C_AI = figstyle.OKABE_ITO["green"]            # AI competence


def _is_nonmonotone(p, tol=1e-3):
    d = np.diff(p)
    return (d.max() > tol) and (d.min() < -tol)


def _describe(name, theta, p):
    i = int(np.argmin(p))
    flag = "NON-MONOTONE" if _is_nonmonotone(p) else "monotone"
    print(f"  {name:<28} range=[{p.min():.3f}, {p.max():.3f}]  "
          f"min@theta={theta[i]:+.2f}  discrimination(span)={p.max() - p.min():.3f}  [{flag}]")


FIG_ITEM = af.item_spec("ai", b=0.0)   # the paper's pronounced setting, and the item Figure 2 refits


def _fit_bounded(seed=0, students=2000, degree=12, epochs=200, prior_a=4.0):
    """Fit the bounded (non-monotone-capable) Bernstein model on a fresh AI-mediated sample, so the opening
    panel shows a genuine recovered posterior rather than a sketch.

    These are the *study* settings (2000 students, 200 epochs, degree 12) -- the ones behind Table 3 -- not
    ``ai_mediated_fit.py``'s single-run figure defaults (4000 students, 400 epochs), so this panel and the
    standalone posterior figure are fitted on different samples and need not be pixel-identical."""
    import torch

    torch.manual_seed(seed)
    items = af.make_items()                                  # pronounced AI effect (defaults)
    prior, prior_std = af.priors(prior_a)
    _, y_tr = af.generate(students, items, prior_a, seed=seed, ability="normal")
    train = af.as_batch(y_tr)
    model = af.build_models(students, degree, len(items), prior, prior_std)["bound"]
    af.fit_mml(model, *train, epochs=epochs, lr=0.05)
    return model, items


def fig_main(theta, out: Path, formats):
    """Opening figure: (a) AI assistance reshapes the IRF into a non-monotone valley; (b) the genuine
    closed-form posterior it induces is bimodal."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figstyle.figsize("text", 0.42))

    # (a) student-alone vs AI-assisted (non-monotone valley) --------------------------------------
    # Drawn at the SAME settings the paper states for RQ-2 (p_A = 0.95, s_r = 0.55, a = 1.6) and on the
    # same b = 0 AI-mediated item that Figure 2 refits, so panels (a) and (b) and Figure 2 all describe
    # one item under one generative model.
    p_nm, p_s, _ = af.hybrid_parts(theta, FIG_ITEM)
    ax1.plot(theta, p_s, color=C_STUDENT, lw=2.0, ls=":", label="Student alone (2PL)")
    ax1.plot(theta, p_nm, color=C_NONMONO, lw=2.2, label="AI-assisted")
    ax1.set(xlabel="Ability", ylabel="P(correct)", ylim=(0, 1),
            title="(a) AI assistance reshapes the IRF")
    ax1.legend(fontsize=8, loc="lower right")
    for name, p in (("student alone (2PL)", p_s), ("AI-assisted", p_nm)):
        _describe(name, theta, p)

    # (b) the genuine closed-form posterior the valley induces is bimodal --------------------------
    model, items = _fit_bounded()
    peaks, mean, std = af.draw_posterior(ax2, model, items, xlabel="Ability",
                                         title="(b) The induced posterior is bimodal",
                                         legend_loc="upper right", compact=True)
    print(f"  posterior: {peaks} modes, mean={mean:.3f}, sd={std:.3f}")

    fig.tight_layout()
    for fmt in formats:
        fig.savefig(out / f"ai_mediated_irf.{fmt}")
    plt.close(fig)
    print(f"  wrote ai_mediated_irf.{{{','.join(formats)}}}")


def fig_mechanism(theta, out: Path, formats):
    """Decompose the non-monotone case into its ingredients so the dip is legible."""
    p_nm, p_s, r = af.hybrid_parts(theta, FIG_ITEM)
    p_ai = np.full_like(theta, FIG_ITEM["p_ai"])
    fig, ax = plt.subplots(figsize=figstyle.figsize(3.5, 0.82))
    ax.plot(theta, p_s, color=C_STUDENT, lw=1.8, ls=":", label="student alone  $p_S(θ)$")
    ax.plot(theta, p_ai, color=C_AI, lw=1.6, ls="--", label="AI alone  $p_A$")
    ax.plot(theta, p_nm, color=C_NONMONO, lw=2.4, label="hybrid  $P(θ)$")
    ax.fill_between(theta, p_nm, p_s, where=(p_nm >= p_s), color=C_NONMONO, alpha=0.08)
    ax.set(xlabel="ability  θ", ylabel="P(correct)", ylim=(0, 1),
           title="Why the AI-mediated IRF dips")
    axr = ax.twinx()
    axr.plot(theta, r, color="0.55", lw=1.4, ls="-.")
    axr.set_ylabel("reliance  r(θ)", color="0.45")
    axr.set_ylim(0, 1)
    axr.tick_params(axis="y", labelcolor="0.45")
    axr.grid(False)
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(out / f"ai_mediated_mechanism.{fmt}")
    plt.close(fig)
    print(f"  wrote ai_mediated_mechanism.{{{','.join(formats)}}}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("figures"))
    p.add_argument("--formats", default="png,pdf")
    args = p.parse_args()
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    theta = np.linspace(-3.0, 3.0, 400)
    print(f"AI-mediated IRF shapes (a={FIG_ITEM['a']}, b={FIG_ITEM['b']}, "
          f"p_A={FIG_ITEM['p_ai']}, s_r={FIG_ITEM['sr']}) -- the paper's pronounced setting:")
    fig_main(theta, args.out, formats)
    fig_mechanism(theta, args.out, formats)
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
