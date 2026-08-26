"""
Per-fold paired comparisons for the benchmark.

K-fold rows are *paired* (every model is scored on the same folds), so the right way to ask "is model X
better than the baseline?" is a per-fold difference, not a comparison of two noisy means: the large,
shared fold-to-fold variance (some folds are simply harder for everyone) cancels in the paired delta.
``paired_tests`` turns the tidy ``runs`` table into one row per (dataset, split, metric, model) with the
mean delta vs a baseline, how many folds it wins, the standardized effect size, paired t / Wilcoxon
*difference* p-values, and a TOST *equivalence* p-value.
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.stats.weightstats import ttost_paired

# Prediction metrics and their orientation (recovery metrics are fit once, so they can't be paired).
LOWER_BETTER = {"nll": True, "brier": True, "ece": True, "auc": False, "acc": False}
DEFAULT_BASELINE = "quad_logistic_irt"

# Equivalence (TOST) margins per metric: a |Δ| below this counts as "practically the same". These are
# SUBSTANTIVE, pre-registered choices — set each to the smallest difference you'd actually care about for
# that metric (overridable via the config's ``tost_margins``). Defaults are deliberately modest placeholders.
DEFAULT_TOST_MARGINS = {"nll": 0.01, "brier": 0.005, "ece": 0.01, "auc": 0.01, "acc": 0.01}
TOST_ALPHA = 0.05


def paired_tests(runs: pd.DataFrame, baseline: str = DEFAULT_BASELINE,
                 metrics: tuple = tuple(LOWER_BETTER),
                 fdr_groupby: tuple = ("dataset", "split", "metric"),
                 tost_margins: dict | None = None, tost_alpha: float = TOST_ALPHA) -> pd.DataFrame:
    """Compare every model to ``baseline`` per fold, per (dataset, split, metric).

    One row per (dataset, split, metric, model != baseline) on splits with >= 2 shared folds (recovery,
    fit once, is skipped). ``delta_mean`` is ``model - baseline`` (negative is better for nll/brier/ece,
    positive for auc/acc; ``folds_better`` counts folds in the improving direction). If ``baseline`` is
    absent from a (dataset, split) group, the first model there is used instead.

    Because several models are compared at once, the raw ``p_ttest`` / ``p_wilcoxon`` (difference) *and*
    ``p_tost`` (equivalence) are adjusted for multiple comparisons by Benjamini-Hochberg FDR within each
    ``fdr_groupby`` family (default: the model-vs-baseline tests sharing a (dataset, split, metric)), as
    ``p_ttest_fdr`` / ``p_wilcoxon_fdr`` / ``p_tost_fdr``. Declaring equivalence across many cells is itself a
    multiple-comparison claim, so the equivalence test is corrected on the same footing as the difference tests.

    Two more columns serve the *equivalence* question directly (a non-significant difference is not evidence
    of equivalence): ``cohen_dz`` is the standardized effect size ``mean(Δ) / sd(Δ)``, and ``p_tost`` is a
    two-one-sided-test equivalence p (``statsmodels.ttost_paired`` on the per-fold deltas) against the
    per-metric margin in ``tost_margins`` — ``equivalent`` flags ``p_tost_fdr < tost_alpha`` (the deltas lie
    within ``±margin``, after FDR correction). Read difference p-values as descriptive (few folds, shared
    training data).
    """
    tost_margins = tost_margins or DEFAULT_TOST_MARGINS
    rows = []
    pred = runs[runs["split"] != "recovery"]
    for (dataset, split), group in pred.groupby(["dataset", "split"], sort=False):
        models = list(dict.fromkeys(group["model"]))  # registry order, de-duplicated
        base = baseline if baseline in models else models[0]
        base_by_fold = group[group["model"] == base].set_index("fold")
        if len(base_by_fold) < 2:
            continue
        for model in models:
            if model == base:
                continue
            model_by_fold = group[group["model"] == model].set_index("fold")
            folds = base_by_fold.index.intersection(model_by_fold.index)
            if len(folds) < 2:
                continue
            for metric in metrics:
                if metric not in group.columns:
                    continue
                b = base_by_fold.loc[folds, metric].to_numpy(dtype=float)
                v = model_by_fold.loc[folds, metric].to_numpy(dtype=float)
                if np.isnan(b).any() or np.isnan(v).any():
                    continue
                rows.append(_one(dataset, split, metric, model, base, v, b,
                                 tost_margins.get(metric, 0.0)))
    return _add_fdr(pd.DataFrame(rows), list(fdr_groupby), tost_alpha)


def _bh_fdr(pvals) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values via ``scipy.stats.false_discovery_control``, NaN-safe (NaN
    inputs, e.g. an undefined Wilcoxon, stay NaN and are excluded from the correction)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = ~np.isnan(p)
    if valid.any():
        out[valid] = sps.false_discovery_control(p[valid], method="bh")
    return out


def _add_fdr(df: pd.DataFrame, groupby: list, tost_alpha: float = TOST_ALPHA) -> pd.DataFrame:
    """Add BH-FDR-adjusted p-value columns, correcting within each ``groupby`` family of comparisons.

    Both the *difference* tests (``p_ttest``, ``p_wilcoxon``) and the *equivalence* test (``p_tost``) are
    corrected — declaring equivalence across many cells is itself a multiple-comparison claim, so the TOST
    p is FDR-controlled on the same footing as the difference tests. The ``equivalent`` verdict is then taken
    on the corrected ``p_tost_fdr`` (``< tost_alpha``); NaN p-values stay NaN and resolve to ``False``.
    """
    if df.empty:
        return df
    for col in ("p_ttest", "p_wilcoxon", "p_tost"):
        df[f"{col}_fdr"] = df.groupby(groupby, sort=False)[col].transform(_bh_fdr)
    df["equivalent"] = df["p_tost_fdr"] < tost_alpha  # NaN < alpha -> False (numpy/pandas)
    return df


def _one(dataset, split, metric, model, base, v: np.ndarray, b: np.ndarray, margin: float) -> dict:
    """One paired-comparison row: deltas + effect size + difference (t / Wilcoxon) and the raw equivalence
    (TOST) p. The ``equivalent`` verdict is finalized in :func:`_add_fdr` on the FDR-corrected ``p_tost``."""
    delta = v - b
    better = delta < 0 if LOWER_BETTER.get(metric, True) else delta > 0
    t_stat, p_t = sps.ttest_rel(v, b)
    try:  # Wilcoxon is undefined when every paired difference is zero
        p_w = sps.wilcoxon(v, b).pvalue
    except ValueError:
        p_w = np.nan
    sd = float(delta.std(ddof=1)) if len(delta) > 1 else np.nan
    dz = float(delta.mean() / sd) if sd and np.isfinite(sd) else np.nan  # Cohen's dz (standardized effect)
    return {
        "dataset": dataset, "split": split, "metric": metric, "model": model, "baseline": base,
        "n_folds": int(len(v)), "delta_mean": float(delta.mean()), "delta_std": sd, "cohen_dz": dz,
        "folds_better": int(better.sum()),
        "t_stat": float(t_stat), "p_ttest": float(p_t), "p_wilcoxon": float(p_w),
        "tost_margin": float(margin), "p_tost": _tost(v, b, margin),
    }


def _tost(v: np.ndarray, b: np.ndarray, margin: float) -> float:
    """Two one-sided paired t-tests for equivalence of ``v`` and ``b`` within ``±margin``.

    Returns ``p_tost`` — the larger of the two one-sided p-values (``statsmodels.ttost_paired`` on the per-fold
    deltas): the p for rejecting "the difference is at least ``margin``" on *both* sides. The equivalence
    verdict (``p_tost`` below alpha) is applied *after* FDR correction in :func:`_add_fdr`. NaN when the
    margin is unset or there are too few folds.
    """
    if margin <= 0 or len(v) < 2:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # degenerate (zero-variance) deltas divide-by-zero inside statsmodels
        return float(ttost_paired(np.asarray(v, float), np.asarray(b, float), -margin, margin)[0])
