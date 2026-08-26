"""Parameter counts + AIC/BIC for the AI-mediated study models — the in-sample, capacity-penalised
complement to the held-out NLL of :mod:`scripts.ai_mediated_study`.

The held-out NLL already controls for overfitting empirically (extra capacity that does not generalise hurts
on fresh data). This script adds the classical psychometric convention: the trainable parameter count `k`,
and AIC `= 2k + 2·NLL_train` / BIC `= k·ln(N) + 2·NLL_train`, with `N` = number of response patterns
(students) — the natural unit for a marginal IRT likelihood (ability integrated out per student). Lower is
better; we report Δ = baseline − bounded (positive favours the bounded model), averaged over seeds.

    python papers/aimecon2026/scripts/ai_mediated_aic.py --seeds 5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ai_mediated_fit as af

EFFECTS = {"pronounced": dict(p_ai=0.95, sr=0.55), "mild": dict(p_ai=0.92, sr=0.80)}


def kcount(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("figures/ai_mediated"))
    p.add_argument("--students", type=int, default=2000)
    p.add_argument("--degree", type=int, default=12)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--prior-a", type=float, default=4.0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    a0 = args.prior_a
    prior, prior_std = (a0, a0), af.H / np.sqrt(2 * a0 + 1)
    N = args.students  # marginal IRT: one likelihood term per student (response pattern)

    rows = []
    for effect, params in EFFECTS.items():
        items = af.make_items(**params)
        n_items = len(items)
        print(f"\n=== effect={effect} {params} | N={N} response patterns, {args.seeds} seeds ===")
        for seed in range(args.seeds):
            _, y_tr = af.generate(args.students, items, a0, seed=seed, ability="normal")
            train = af.as_batch(y_tr)
            models = af.build_models(args.students, args.degree, n_items, prior, prior_std)
            for key, m in models.items():
                af.fit_mml(m, *train, epochs=args.epochs, lr=0.05)
                with torch.no_grad():
                    nll_tot = float(-m.log_marginal_evidence(*train).sum())
                k = kcount(m)
                rows.append(dict(effect=effect, seed=seed, model=key, k=k, nll_train=nll_tot,
                                 aic=2 * k + 2 * nll_tot, bic=k * np.log(N) + 2 * nll_tot))
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "study_aic.csv", index=False)

    print(f"\n=== parameter counts + AIC/BIC (mean over {args.seeds} seeds; N={N}) ===")
    for effect in EFFECTS:
        agg = df[df.effect == effect].groupby("model").mean(numeric_only=True)
        print(f"\n[{effect}]")
        print(f"  {'model':<34} {'k':>4} {'AIC':>10} {'BIC':>10}")
        for key in ("4pl", "mono", "bound"):
            r = agg.loc[key]
            print(f"  {af.LABELS[key]:<34} {int(r.k):>4} {r.aic:>10.1f} {r.bic:>10.1f}")
        kb, ab, bb = agg.loc["bound", ["k", "aic", "bic"]]
        for base in ("4pl", "mono"):
            kk, aa, bb2 = agg.loc[base, ["k", "aic", "bic"]]
            print(f"    Δ vs {af.LABELS[base]:<26}: ΔAIC={aa - ab:+.1f}  ΔBIC={bb2 - bb:+.1f}  (Δk={int(kb - kk):+d})")
    print(f"\nwrote {args.out / 'study_aic.csv'}")


if __name__ == "__main__":
    main()
