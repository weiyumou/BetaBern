"""Benchmark report rendering + multiple-comparison correction (report.py, stats.py)."""
import numpy as np
import pandas as pd

from betabern.benchmark.report import render_markdown
from betabern.benchmark.runner import _summarize
from betabern.benchmark.stats import _bh_fdr, paired_tests


def test_bh_fdr_step_up_and_monotone():
    """BH adjusts p_i to min over k>=i of p_(k)*n/k; equal-spaced p's here all map to the largest, n*p_max."""
    p = np.array([0.01, 0.02, 0.03, 0.04])
    np.testing.assert_allclose(_bh_fdr(p), 0.04)              # 0.01*4/1 = 0.02*4/2 = ... = 0.04
    assert np.isnan(_bh_fdr(np.array([np.nan]))[0])           # NaN propagates


def _toy_runs() -> pd.DataFrame:
    """Two models, a 3-fold prediction split + a sim-only recovery split (the case that exposed the bug)."""
    rng = np.random.default_rng(0)
    rows = []
    for model, nll0 in [("quad_logistic_irt", 0.400), ("quad_logistic_bern_irt", 0.402)]:
        for fold in range(3):
            rows.append(dict(dataset="sim", model=model, split="within_random", fold=fold, n_params=100,
                             nll=nll0 + rng.normal(0, 0.004), auc=0.87, brier=0.12, acc=0.80, ece=0.02))
        rows.append(dict(dataset="sim", model=model, split="recovery", fold=0, n_params=100,
                         theta_spearman=0.9, prob_mae=0.05))  # recovery: no prediction metrics
    return pd.DataFrame(rows)


def test_paired_tests_adds_fdr_effect_and_tost_columns():
    paired = paired_tests(_toy_runs(), baseline="quad_logistic_irt")
    assert {"p_ttest_fdr", "p_wilcoxon_fdr", "cohen_dz", "p_tost", "equivalent", "tost_margin"} <= set(paired.columns)
    assert paired[paired["split"] == "recovery"].empty            # recovery is never paired
    fdr = paired["p_ttest_fdr"].dropna()
    assert ((fdr >= 0) & (fdr <= 1)).all()


def _runs_for_tost() -> pd.DataFrame:
    """Baseline + a 'near' model (within margin) + a 'far' model (well beyond), sharing fold difficulty."""
    rng = np.random.default_rng(1)
    rows = []
    for fold in range(8):
        hard = rng.normal(0, 0.02)  # shared per-fold difficulty (cancels in the paired delta)
        for model, gap in [("base", 0.0), ("near", 0.001), ("far", 0.05)]:
            rows.append(dict(dataset="d", model=model, split="within_random", fold=fold, n_params=1,
                             nll=0.40 + hard + gap + rng.normal(0, 5e-4),
                             auc=0.8, brier=0.1, acc=0.8, ece=0.02))
    return pd.DataFrame(rows)


def test_tost_equivalence_vs_difference():
    """TOST flags a within-margin model as equivalent and a beyond-margin one as not; the standardized
    effect size orders them."""
    pt = paired_tests(_runs_for_tost(), baseline="base", tost_margins={"nll": 0.01})
    near = pt[(pt["model"] == "near") & (pt["metric"] == "nll")].iloc[0]
    far = pt[(pt["model"] == "far") & (pt["metric"] == "nll")].iloc[0]
    assert near["equivalent"] and near["p_tost"] < 0.05           # |Δ| ≈ 0.001 < 0.01 margin
    assert not far["equivalent"]                                  # |Δ| ≈ 0.05 >> 0.01 margin
    assert abs(far["cohen_dz"]) > abs(near["cohen_dz"])


def test_report_renders_with_recovery_split(tmp_path):
    """A run with a recovery split (simulated data) must render the prediction comparison, not crash on the
    recovery summary rows (which carry no nll/auc)."""
    runs = _toy_runs()
    runs.to_csv(tmp_path / "runs.csv", index=False)
    _summarize(runs).to_csv(tmp_path / "summary.csv", index=False)
    paired_tests(runs, baseline="quad_logistic_irt").to_csv(tmp_path / "paired.csv", index=False)

    report = render_markdown(tmp_path)
    assert "## Overview" in report and "quad_logistic_bern_irt" in report
    assert "recovery" not in report.split("Scope:")[1].split("\n")[1]  # recovery not used as a family row
