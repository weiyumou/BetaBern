import pandas as pd

from betabern.benchmark.report import render_markdown


def test_report_renders_family_deltas_and_paired_sources(tmp_path):
    run_dir = tmp_path / "20260611_211142"
    run_dir.mkdir()

    summary = pd.DataFrame([
        _summary("art", "quad_logistic_irt", "within_random", 0.40, 0.80, 0.120, 0.82, 0.020),
        _summary("art", "quad_logistic_bern_irt", "within_random", 0.41, 0.79, 0.121, 0.81, 0.018),
        _summary("art", "quad_ogive_irt", "within_random", 0.42, 0.78, 0.122, 0.80, 0.021),
        _summary("art", "quad_ogive_bern_irt", "within_random", 0.41, 0.79, 0.121, 0.81, 0.020),
    ])
    summary.to_csv(run_dir / "summary.csv", index=False)

    runs = pd.DataFrame([
        _run("art", "quad_logistic_irt", "within_random", 0, 0.39, 0.81, 0.120, 0.82, 0.020),
        _run("art", "quad_logistic_irt", "within_random", 1, 0.41, 0.79, 0.120, 0.82, 0.020),
        _run("art", "quad_logistic_bern_irt", "within_random", 0, 0.40, 0.80, 0.121, 0.81, 0.018),
        _run("art", "quad_logistic_bern_irt", "within_random", 1, 0.42, 0.78, 0.121, 0.81, 0.018),
        _run("art", "quad_ogive_irt", "within_random", 0, 0.43, 0.77, 0.123, 0.80, 0.021),
        _run("art", "quad_ogive_irt", "within_random", 1, 0.41, 0.79, 0.121, 0.80, 0.021),
        _run("art", "quad_ogive_bern_irt", "within_random", 0, 0.42, 0.78, 0.122, 0.81, 0.020),
        _run("art", "quad_ogive_bern_irt", "within_random", 1, 0.40, 0.80, 0.120, 0.81, 0.020),
    ])
    runs.to_csv(run_dir / "runs.csv", index=False)

    paired = pd.DataFrame([
        {
            "dataset": "art", "split": "within_random", "metric": "nll",
            "model": "quad_logistic_bern_irt", "baseline": "quad_logistic_irt",
            "n_folds": 2, "delta_mean": 0.01, "delta_std": 0.0, "cohen_dz": 1.5, "folds_better": 0,
            "t_stat": 1.0, "p_ttest": 0.0123, "p_wilcoxon": 0.5,
            "tost_margin": 0.01, "p_tost": 0.3, "equivalent": False,
        },
    ])
    paired.to_csv(run_dir / "paired.csv", index=False)

    report = render_markdown(run_dir)

    assert "# Direct-vs-bridged benchmark report: 20260611_211142" in report
    assert "| logistic | `quad_logistic_irt` | `quad_logistic_bern_irt` | +0.01" in report
    # metric | direct | bridged | Δ | dz | bridge | folds better | p_t | p_fdr | p_w | equiv | source
    assert "| nll | 0.4 +/- 0.001 | 0.41 +/- 0.001 | +0.01 | 1.5 | worse | 0/2 | 0.0123" in report
    assert "no (0.3000)" in report  # TOST equivalence verdict rendered
    assert "| ogive | `quad_ogive_irt` | `quad_ogive_bern_irt` | -0.01" in report
    assert "runs.csv" in report
    assert "Exact vs. quadrature twin" not in report  # no exact models -> faithfulness section omitted
    assert "Exact vs. direct" not in report            # ... and no exact-vs-direct equivalence section


def test_report_exact_vs_quad_twin_section(tmp_path):
    """When a run includes exact models, the report adds an exact-vs-quadrature-twin faithfulness table
    (Δ = exact − quad ≈ 0), recomputing the paired test from runs.csv (the global paired baseline differs)."""
    run_dir = tmp_path / "exact_run"
    run_dir.mkdir()

    summary = pd.DataFrame([
        _summary("art", "quad_logistic_irt", "within_random", 0.40, 0.80, 0.120, 0.82, 0.020),
        _summary("art", "quad_logistic_bern_irt", "within_random", 0.41, 0.79, 0.121, 0.81, 0.018),
        _summary("art", "exact_logistic_bern_irt", "within_random", 0.41, 0.79, 0.121, 0.81, 0.018),
    ])
    summary.to_csv(run_dir / "summary.csv", index=False)

    runs = pd.DataFrame([
        _run("art", "quad_logistic_irt", "within_random", 0, 0.39, 0.81, 0.120, 0.82, 0.020),
        _run("art", "quad_logistic_irt", "within_random", 1, 0.41, 0.79, 0.120, 0.82, 0.020),
        _run("art", "quad_logistic_bern_irt", "within_random", 0, 0.40, 0.80, 0.121, 0.81, 0.018),
        _run("art", "quad_logistic_bern_irt", "within_random", 1, 0.42, 0.78, 0.121, 0.81, 0.018),
        _run("art", "exact_logistic_bern_irt", "within_random", 0, 0.40, 0.80, 0.121, 0.81, 0.018),
        _run("art", "exact_logistic_bern_irt", "within_random", 1, 0.42, 0.78, 0.121, 0.81, 0.018),
    ])
    runs.to_csv(run_dir / "runs.csv", index=False)

    pd.DataFrame([{
        "dataset": "art", "split": "within_random", "metric": "nll",
        "model": "quad_logistic_bern_irt", "baseline": "quad_logistic_irt",
        "n_folds": 2, "delta_mean": 0.01, "delta_std": 0.0, "cohen_dz": 1.5, "folds_better": 0,
        "t_stat": 1.0, "p_ttest": 0.0123, "p_wilcoxon": 0.5, "tost_margin": 0.01, "p_tost": 0.3, "equivalent": False,
    }]).to_csv(run_dir / "paired.csv", index=False)

    report = render_markdown(run_dir)

    assert "Exact vs. quadrature twin" in report
    # exact runs equal the quad twin's -> Δ nll = 0, recomputed from runs.csv
    assert "| art | logistic | 0.41 +/- 0.001 | 0.41 +/- 0.001 | 0 |" in report
    assert "max |Δ nll| across all twins = 0.0000." in report
    # exact-vs-direct equivalence section also renders (direct quad_logistic_irt is present)
    assert "Exact vs. direct (equivalence" in report


def _summary(dataset, model, split, nll, auc, brier, acc, ece):
    row = {"dataset": dataset, "model": model, "split": split, "n_params": 1}
    for metric, value in {"nll": nll, "auc": auc, "brier": brier, "acc": acc, "ece": ece}.items():
        row[f"{metric}_mean"] = value
        row[f"{metric}_std"] = 0.001
    return row


def _run(dataset, model, split, fold, nll, auc, brier, acc, ece):
    return {
        "dataset": dataset, "model": model, "split": split, "fold": fold,
        "n_params": 1, "n": 10, "nll": nll, "auc": auc, "brier": brier, "acc": acc, "ece": ece,
    }
