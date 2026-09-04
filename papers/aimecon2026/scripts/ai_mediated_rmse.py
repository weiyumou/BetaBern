"""IRF-recovery RMSE for the AI-mediated study: unweighted RMSE between each fitted IRF and the TRUE
generating IRF, per item type.

For simulated data the true item curve is a cleaner gold standard than the observed (Bernoulli) responses: a
Brier score vs. responses carries an irreducible noise floor and, computed in-sample, is biased toward the
higher-capacity model. RMSE-vs-truth has neither problem -- a flexible model that wiggles spuriously only
*raises* its RMSE -- so it needs no held-out split (the repeated seeds supply the sampling variability). This
is the simulation-specific complement to the held-out NLL and AIC/BIC.

Per item j:  RMSE_j = sqrt( mean_θ[ (P_fit_j(θ) − P_true_j(θ))^2 ] ), averaged uniformly over the ability grid
-- equal weight per θ, hence per η = 6θ − 3, matching the recovery figure's axis (so the number agrees with
what the curves show). We average RMSE within item type (standard / AI-mediated) and report Δ = baseline −
bounded (positive favours bounded) across seeds.

Defaults reproduce the paper's Table 3 (median over 20 runs); pass ``--agg mean`` for the mean instead.

    python papers/aimecon2026/scripts/ai_mediated_rmse.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ai_mediated_fit as af

EFFECTS = af.EFFECTS


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("figures/ai_mediated"))
    af.add_study_args(p)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    a0 = args.prior_a
    prior, prior_std = af.priors(a0)

    grid = np.linspace(1e-3, 1 - 1e-3, 200)              # ability grid on [0,1] (uniform in η = 6θ − 3)
    w = np.ones_like(grid) / len(grid)                   # unweighted: equal weight across the ability range

    def item_rmse(model, idx, spec, kind):
        p_fit = af.irt_curve(model, idx, grid) if kind == "4pl" else af.bern_curve(model, idx, grid)
        p_true = af.hybrid_p(grid, spec)
        return float(np.sqrt(np.sum(w * (p_fit - p_true) ** 2)))

    rows = []
    for effect, params in EFFECTS.items():
        items = af.make_items(**params)
        n_items = len(items)
        print(f"\n=== effect={effect} {params} | {args.seeds} seeds ===")
        for seed in range(args.seeds):
            _, y_tr = af.generate(args.students, items, a0, seed=seed, ability="normal")
            train = af.as_batch(y_tr)
            models = af.build_models(args.students, args.degree, n_items, prior, prior_std)
            for key, m in models.items():
                af.fit_mml(m, *train, epochs=args.epochs, lr=0.05)
                kind = "4pl" if key == "4pl" else "bern"
                for idx, spec in enumerate(items):
                    rows.append(dict(effect=effect, seed=seed, model=key, itype=spec["type"],
                                     rmse=item_rmse(m, idx, spec, kind)))
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "study_rmse.csv", index=False)

    # per-seed mean RMSE within (effect, model, item type), then paired Δ vs bounded across seeds
    per = df.groupby(["effect", "model", "itype", "seed"])["rmse"].mean().reset_index()
    print(f"\n=== IRF-recovery RMSE ({args.agg} over {args.seeds} seeds; unweighted over the ability grid; "
          f"lower = better) ===")
    for effect in EFFECTS:
        print(f"\n[{effect}]")
        print(f"  {'model':<34} {'standard':>10} {'AI-mediated':>12}")
        for key in ("4pl", "mono", "bound"):
            vals = {t: getattr(per[(per.effect == effect) & (per.model == key) & (per.itype == t)]["rmse"],
                               args.agg)() for t in ("std", "ai")}
            print(f"  {af.LABELS[key]:<34} {vals['std']:>10.4f} {vals['ai']:>12.4f}")
        # Δ on AI items (baseline − bounded), paired over seeds
        bvals = per[(per.effect == effect) & (per.model == "bound") & (per.itype == "ai")].set_index("seed")["rmse"]
        for base in ("4pl", "mono"):
            avals = per[(per.effect == effect) & (per.model == base) & (per.itype == "ai")].set_index("seed")["rmse"]
            d = (avals - bvals).to_numpy()
            n = len(d)
            m = d.mean()
            se = d.std(ddof=1) / np.sqrt(n)
            tc = sps.t.ppf(0.975, n - 1)
            print(f"    AI-item ΔRMSE vs {af.LABELS[base]:<26}: {m:+.4f}  95%CI[{m - tc * se:+.4f},{m + tc * se:+.4f}]"
                  f"  win={(d > 0).mean():.0%}")
    print(f"\nwrote {args.out / 'study_rmse.csv'}")


if __name__ == "__main__":
    main()
