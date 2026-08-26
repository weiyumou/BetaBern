"""Full effect-size study for the AI-mediated non-monotone simulation.

Runs the bounded vs. monotone comparison across many independent simulation runs, for:
  - two AI-effect strengths: 'pronounced' (p_A=0.95, sr=0.55) and 'mild' (p_A=0.92, sr=0.80), and
  - two metrics: 'full' (held-out NLL over the whole response vector) and 'ai' (the conditional NLL of the
    AI-mediated items given the standard items -- the less-diluted, effect-concentrated metric).

The unit of replication is the simulation run (fresh dataset + refit), so the paired test speaks to
consistency across data realizations, not to within-run student count. All paired t p-values are
Benjamini-Hochberg (FDR) corrected across the whole family of comparisons. Writes per-run NLLs
(``study_runs.csv``) and the corrected effect-size table (``study_effects.csv``) under ``--out``.

    python papers/aimecon2026/scripts/ai_mediated_study.py --out figures/ai_mediated --seeds 20
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ai_mediated_fit as af

EFFECTS = {"pronounced": dict(p_ai=0.95, sr=0.55), "mild": dict(p_ai=0.92, sr=0.80)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("figures/ai_mediated"))
    p.add_argument("--students", type=int, default=2000)
    p.add_argument("--degree", type=int, default=12)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--ability", choices=["normal", "beta"], default="normal")
    p.add_argument("--prior-a", type=float, default=4.0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    a0 = args.prior_a
    prior, prior_std = (a0, a0), af.H / np.sqrt(2 * a0 + 1)

    rows = []
    for effect, params in EFFECTS.items():
        items = af.make_items(**params)
        n_items = len(items)
        ns = SimpleNamespace(students=args.students, degree=args.degree, ability=args.ability, epochs=args.epochs)
        print(f"\n=== effect={effect} {params} | students={args.students} epochs={args.epochs} seeds={args.seeds} ===")
        for s in range(args.seeds):
            _, nll = af.fit_and_eval(ns, items, n_items, a0, prior, prior_std, seed=s)
            for model in ("4pl", "mono", "bound"):
                for metric in ("full", "ai"):
                    rows.append(dict(effect=effect, seed=s, model=model, metric=metric, nll=nll[model][metric]))
            print(f"  {effect:<10} run {s + 1:2d}/{args.seeds}:  "
                  f"full[bound={nll['bound']['full']:.4f} 4pl={nll['4pl']['full']:.4f}]  "
                  f"ai[bound={nll['bound']['ai']:.4f} 4pl={nll['4pl']['ai']:.4f}]")
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "study_runs.csv", index=False)

    # --- paired effect sizes (bounded vs each monotone baseline), BH-FDR across the whole family ---
    comps = []
    for effect in EFFECTS:
        for metric in ("full", "ai"):
            piv = df[(df.effect == effect) & (df.metric == metric)].pivot(index="seed", columns="model", values="nll")
            for base in ("4pl", "mono"):
                d = (piv[base] - piv["bound"]).to_numpy()
                n = len(d)
                m = d.mean()
                sd = d.std(ddof=1)
                se = sd / np.sqrt(n)
                tc = sps.t.ppf(0.975, n - 1)
                _, p = sps.ttest_rel(piv[base], piv["bound"])
                comps.append(dict(effect=effect, metric=metric, baseline=base, n=n, dNLL=m,
                                  ci_lo=m - tc * se, ci_hi=m + tc * se, dz=(m / sd if sd > 0 else np.inf),
                                  win=(d > 0).mean(), p_raw=p))
    cdf = pd.DataFrame(comps)
    cdf["p_fdr"] = multipletests(cdf["p_raw"], method="fdr_bh")[1]
    cdf.to_csv(args.out / "study_effects.csv", index=False)

    print("\n=== effect sizes (ΔNLL = baseline − bounded; > 0 favours bounded; BH-FDR over all "
          f"{len(cdf)} comparisons) ===")
    for _, r in cdf.iterrows():
        print(f"  [{r.effect:<10} {r.metric:>4}]  bounded vs {r.baseline:>4}:  "
              f"ΔNLL={r.dNLL:+.4f}  95%CI[{r.ci_lo:+.4f},{r.ci_hi:+.4f}]  dz={r.dz:5.2f}  "
              f"win={r.win:.0%}  p_raw={r.p_raw:.1e}  p_fdr={r.p_fdr:.1e}")
    print(f"\nwrote {args.out / 'study_runs.csv'} and {args.out / 'study_effects.csv'}")


if __name__ == "__main__":
    main()
