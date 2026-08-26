"""Paste-friendly reports for benchmark result directories."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from betabern.benchmark import stats

DEFAULT_RUN = Path("results/benchmark/20260611_211142")
DEFAULT_FAMILIES = {
    "logistic": ("quad_logistic_irt", "quad_logistic_bern_irt"),
    "ogive": ("quad_ogive_irt", "quad_ogive_bern_irt"),
    "spline": ("quad_spline_irt", "quad_spline_bern_irt"),
}
METRICS = ("nll", "auc", "brier", "acc", "ece")

# Exact-estimator faithfulness pairs: each exact model vs its *quadrature* twin (same IRF + Bernstein degree).
# Δ ≈ 0 confirms the quadrature reproduces the closed-form posterior. Rendered only when a run has exact models.
EXACT_TWINS = (
    ("logistic", "quad_logistic_bern_irt", "exact_logistic_bern_irt"),
    ("ogive", "quad_ogive_bern_irt", "exact_ogive_bern_irt"),
    ("spline", "quad_spline_bern_irt", "exact_spline_bern_irt"),
    ("free", "quad_bern_free", "exact_bern_free"),  # free-form monotone IRF (no direct-R twin; see below)
)
# Exact-vs-direct equivalence pairs: each exact (closed-form) model against its *direct* IRT counterpart
# (ℝ + Normal). The TOST verdict here is the "closed form at no accuracy cost" headline. Exact models only.
EXACT_VS_DIRECT = (
    ("logistic", "quad_logistic_irt", "exact_logistic_bern_irt"),
    ("ogive", "quad_ogive_irt", "exact_ogive_bern_irt"),
    ("spline", "quad_spline_irt", "exact_spline_bern_irt"),
    # NOTE: no ("free", ...) row — the free-form Bernstein IRF has no direct-ℝ counterpart by construction,
    # so it appears only in EXACT_TWINS (exact == quad faithfulness), not in the exact-vs-direct equivalence.
)


@dataclass(frozen=True)
class FamilyPair:
    family: str
    direct: str
    bridged: str


def default_family_pairs() -> list[FamilyPair]:
    """Return the direct/bridged pairs used in the direct-vs-bridged benchmark sweep."""
    return [FamilyPair(family, direct, bridged) for family, (direct, bridged) in DEFAULT_FAMILIES.items()]


def render_markdown(run_dir: str | Path = DEFAULT_RUN, families: Iterable[FamilyPair] | None = None) -> str:
    """Render a Markdown report for one ``results/benchmark/<run>`` directory.

    The family deltas are always bridged minus direct. Significance comes from ``paired.csv`` when that
    file contains the exact family direct baseline; otherwise the same paired test is recomputed from
    ``runs.csv`` and the row is marked as ``runs.csv`` in the source column.
    """
    run_path = Path(run_dir)
    summary = _read_csv(run_path / "summary.csv")
    paired = _read_csv(run_path / "paired.csv")
    runs = _read_csv(run_path / "runs.csv")
    pairs = list(families or default_family_pairs())
    datasets = [str(d) for d in pd.unique(summary["dataset"])] if "dataset" in summary else [None]

    sections: list[str] = []
    sections.append(f"# Direct-vs-bridged benchmark report: {run_path.name}")
    sections.append(_scope_line(run_path, summary, paired, runs))
    if len(datasets) > 1:  # compact cross-dataset equivalence summary first
        sections.append(_cross_dataset_overview(summary, paired, runs, pairs, datasets))
    # Equivalence/faithfulness tables go right after the overview and before the per-dataset breakdown —
    # they are the headline summary; the dataset-by-dataset detail follows. Both omit when no exact models.
    sections.append(_exact_vs_direct_section(summary, paired, runs, datasets))
    sections.append(_exact_faithfulness_section(summary, paired, runs, datasets))
    for ds in datasets:
        if len(datasets) > 1:
            sections.append(f"# Dataset: {ds}")
        sections.append(_overall_table(summary, paired, runs, pairs, ds))
        for pair in pairs:
            sections.append(_family_section(summary, paired, runs, pair, ds))
    sections.append(_notes(paired, runs))
    return "\n\n".join(s for s in sections if s).rstrip() + "\n"


def _cross_dataset_overview(summary: pd.DataFrame, paired: pd.DataFrame, runs: pd.DataFrame,
                            pairs: list[FamilyPair], datasets: list[str]) -> str:
    """One compact row per (dataset, family): the bridged−direct nll/auc/ece deltas and nll significance —
    the at-a-glance equivalence table for a multi-dataset run."""
    rows = []
    for ds in datasets:
        for pair in pairs:
            direct = _summary_row(summary, pair.direct, ds)
            bridged = _summary_row(summary, pair.bridged, ds)
            if direct is None or bridged is None:
                continue
            p_row, source = _paired_row(paired, runs, direct, pair.bridged, "nll")
            rows.append([ds, pair.family, _delta_cell(direct, bridged, "nll"),
                         _delta_cell(direct, bridged, "auc"), _delta_cell(direct, bridged, "ece"),
                         _p_cell(p_row), _p_cell(p_row, "p_ttest_fdr"), _equiv_cell(p_row), source])
    if not rows:
        return ""
    return "## Overview across datasets (bridged − direct)\n\n" + _markdown_table(
        ["dataset", "family", "Δ nll", "Δ auc", "Δ ece", "nll p_t", "nll p_fdr", "nll equiv", "p source"], rows)


def _exact_vs_direct_section(summary: pd.DataFrame, paired: pd.DataFrame, runs: pd.DataFrame,
                             datasets: list) -> str:
    """One row per (dataset, family) testing whether the EXACT closed-form estimator is TOST-equivalent to the
    *direct* IRT model (ℝ + Normal) — the "closed form at no accuracy cost" headline, on the genuine closed
    form. ``p_diff`` is the two-sided difference test; ``equiv`` is the TOST verdict (with p_tost) at the
    configured margin. Returns "" (omitted) when the run includes no exact models."""
    rows, all_equiv = [], True
    for ds in datasets:
        for fam, direct, exact in EXACT_VS_DIRECT:
            d, e = _summary_row(summary, direct, ds), _summary_row(summary, exact, ds)
            if d is None or e is None:
                continue
            p_row, source = _paired_row(paired, runs, d, exact, "nll")
            if p_row is not None and "equivalent" in p_row and not bool(p_row["equivalent"]):
                all_equiv = False
            rows.append([ds if ds is not None else "-", fam, _mean_sd_cell(d, "nll"), _mean_sd_cell(e, "nll"),
                         _delta_cell(d, e, "nll"), _p_cell(p_row, "p_ttest"), _equiv_cell(p_row), source])
    if not rows:
        return ""
    table = _markdown_table(["dataset", "family", "direct nll", "exact nll", "Δ nll", "p_diff", "equiv (TOST)",
                             "source"], rows)
    verdict = "**all exact ≡ direct (TOST)**" if all_equiv else "**NOT all exact ≡ direct (TOST)**"
    return ("## Exact vs. direct (equivalence: is the closed-form estimator equivalent to standard IRT?)\n\n"
            f"{table}\n\n{verdict} at the configured margin.")


def _exact_faithfulness_section(summary: pd.DataFrame, paired: pd.DataFrame, runs: pd.DataFrame,
                                datasets: list) -> str:
    """One row per (dataset, family) comparing the EXACT estimator to its quadrature twin (same IRF, same
    Bernstein degree): ``Δ = exact − quad`` should be ≈ 0 — the "quadrature is a controllably-exact evaluator
    of the closed form" check. Returns "" (and so is omitted) when the run includes no exact models."""
    rows, deltas = [], []
    for ds in datasets:
        for fam, quad, exact in EXACT_TWINS:
            q, e = _summary_row(summary, quad, ds), _summary_row(summary, exact, ds)
            if q is None or e is None:
                continue
            p_row, source = _paired_row(paired, runs, q, exact, "nll")
            rows.append([ds if ds is not None else "-", fam, _mean_sd_cell(q, "nll"), _mean_sd_cell(e, "nll"),
                         _delta_cell(q, e, "nll"), _delta_cell(q, e, "auc"), _p_cell(p_row),
                         _equiv_cell(p_row), source])
            deltas.append(abs(float(e["nll_mean"]) - float(q["nll_mean"])))
    if not rows:
        return ""
    table = _markdown_table(["dataset", "family", "quad nll", "exact nll", "Δ nll", "Δ auc", "nll p_t",
                             "equiv (TOST)", "source"], rows)
    return ("## Exact vs. quadrature twin (faithfulness: Δ = exact − quad, ≈ 0 ⇒ quadrature reproduces the "
            f"closed form)\n\n{table}\n\nmax |Δ nll| across all twins = {max(deltas):.4f}.")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing benchmark artifact: {path}")
    return pd.read_csv(path)


def _scope_line(run_path: Path, summary: pd.DataFrame, paired: pd.DataFrame, runs: pd.DataFrame) -> str:
    datasets = _join_values(summary.get("dataset", pd.Series(dtype=str)))
    splits = _join_values(summary.get("split", pd.Series(dtype=str)))
    folds = _fold_scope(runs)
    baselines = _join_values(paired.get("baseline", pd.Series(dtype=str)))
    return (
        f"Source: `{run_path}`. Scope: dataset={datasets}, split={splits}, "
        f"folds={folds}. Paired.csv baseline(s): {baselines}."
    )


def _join_values(values: pd.Series) -> str:
    unique = [str(v) for v in pd.unique(values.dropna())]
    return ", ".join(unique) if unique else "n/a"


def _fold_scope(runs: pd.DataFrame) -> str:
    if "fold" not in runs or "split" not in runs:
        return "n/a"
    folds = sorted(int(v) for v in pd.unique(runs.loc[runs["split"] != "recovery", "fold"].dropna()))
    if not folds:
        return "n/a"
    if folds == list(range(folds[0], folds[-1] + 1)):
        return f"{len(folds)} ({folds[0]}-{folds[-1]})"
    return f"{len(folds)} ({', '.join(str(f) for f in folds)})"


def _overall_table(summary: pd.DataFrame, paired: pd.DataFrame, runs: pd.DataFrame,
                   pairs: list[FamilyPair], dataset: str | None = None) -> str:
    rows = []
    for pair in pairs:
        direct = _summary_row(summary, pair.direct, dataset)
        bridged = _summary_row(summary, pair.bridged, dataset)
        if direct is None or bridged is None:
            continue
        nll = _delta_cell(direct, bridged, "nll")
        auc = _delta_cell(direct, bridged, "auc")
        brier = _delta_cell(direct, bridged, "brier")
        acc = _delta_cell(direct, bridged, "acc")
        ece = _delta_cell(direct, bridged, "ece")
        p_row, source = _paired_row(paired, runs, direct, pair.bridged, "nll")
        p, p_fdr = _p_cell(p_row), _p_cell(p_row, "p_ttest_fdr")
        rows.append([pair.family, f"`{pair.direct}`", f"`{pair.bridged}`", nll, auc, brier, acc, ece,
                     p, p_fdr, _equiv_cell(p_row), source])
    if not rows:
        return "## Overview\n\nNo complete direct/bridged family pairs were found."
    return "## Overview\n\n" + _markdown_table(
        ["family", "direct", "bridged", "Δ nll", "Δ auc", "Δ brier", "Δ acc", "Δ ece",
         "nll p_t", "nll p_fdr", "nll equiv", "p source"],
        rows,
    )


def _family_section(summary: pd.DataFrame, paired: pd.DataFrame, runs: pd.DataFrame, pair: FamilyPair,
                    dataset: str | None = None) -> str:
    direct = _summary_row(summary, pair.direct, dataset)
    bridged = _summary_row(summary, pair.bridged, dataset)
    title = f"## {pair.family.title()}: `{pair.direct}` -> `{pair.bridged}`"
    if direct is None or bridged is None:
        missing = pair.direct if direct is None else pair.bridged
        return f"{title}\n\nMissing `{missing}` in `summary.csv`."

    rows = []
    for metric in METRICS:
        p_row, source = _paired_row(paired, runs, direct, pair.bridged, metric)
        rows.append([
            metric,
            _mean_sd_cell(direct, metric),
            _mean_sd_cell(bridged, metric),
            _delta_cell(direct, bridged, metric),
            _num_cell(p_row, "cohen_dz"),
            _direction_label(float(bridged[f"{metric}_mean"] - direct[f"{metric}_mean"]), metric),
            _folds_cell(p_row),
            _p_cell(p_row, "p_ttest"),
            _p_cell(p_row, "p_ttest_fdr"),
            _p_cell(p_row, "p_wilcoxon"),
            _equiv_cell(p_row),
            source,
        ])
    return title + "\n\n" + _markdown_table(
        ["metric", "direct", "bridged", "Δ bridge-direct", "dz", "bridge", "folds better",
         "p_t", "p_fdr", "p_w", "equiv (TOST)", "source"],
        rows,
    )


def _summary_row(summary: pd.DataFrame, model: str, dataset: str | None = None) -> pd.Series | None:
    rows = summary[summary["model"] == model]
    if "split" in rows:  # the report scores prediction splits; recovery (sim-only) is a separate analysis
        rows = rows[rows["split"] != "recovery"]
    if dataset is not None and "dataset" in rows:
        rows = rows[rows["dataset"] == dataset]
    if rows.empty:
        return None
    if len(rows) > 1:
        # Keep the report pasteable for the common one-dataset/one-split case while still being explicit.
        rows = rows.sort_values(["dataset", "split"], kind="stable")
    return rows.iloc[0]


def _paired_row(paired: pd.DataFrame, runs: pd.DataFrame, direct: pd.Series, model: str,
                metric: str) -> tuple[pd.Series | None, str]:
    if not paired.empty and "baseline" in paired.columns:  # a single-model run has an empty (column-less) paired
        paired_match = paired[
            (paired["dataset"] == direct["dataset"])
            & (paired["split"] == direct["split"])
            & (paired["metric"] == metric)
            & (paired["model"] == model)
            & (paired["baseline"] == direct["model"])
            ]
        if not paired_match.empty:
            return paired_match.iloc[0], "paired.csv"

    computed = _family_paired_from_runs(runs, direct, model, metric)
    if computed is not None:
        return computed, "runs.csv"
    return None, "n/a"


def _family_paired_from_runs(runs: pd.DataFrame, direct: pd.Series, model: str, metric: str) -> pd.Series | None:
    subset = runs[(runs["dataset"] == direct["dataset"]) & (runs["split"] == direct["split"])]
    if subset.empty or metric not in subset.columns:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        family_paired = stats.paired_tests(subset, baseline=str(direct["model"]), metrics=(metric,))
    family_paired = family_paired[family_paired["model"] == model]
    if family_paired.empty:
        return None
    return family_paired.iloc[0]


def _mean_sd_cell(row: pd.Series, metric: str) -> str:
    mean = float(row[f"{metric}_mean"])
    std_key = f"{metric}_std"
    if std_key not in row or pd.isna(row[std_key]):
        return _fmt_num(mean)
    return f"{_fmt_num(mean)} +/- {_fmt_num(float(row[std_key]))}"


def _delta_cell(direct: pd.Series, bridged: pd.Series, metric: str) -> str:
    delta = float(bridged[f"{metric}_mean"] - direct[f"{metric}_mean"])
    sign = "+" if delta > 0 else ""
    return f"{sign}{_fmt_num(delta)}"


def _direction_label(delta: float, metric: str) -> str:
    if abs(delta) < 5e-7:
        return "tie"
    lower_better = stats.LOWER_BETTER.get(metric, True)
    improved = delta < 0 if lower_better else delta > 0
    return "better" if improved else "worse"


def _folds_cell(row: pd.Series | None) -> str:
    if row is None or pd.isna(row.get("folds_better", np.nan)):
        return "n/a"
    return f"{int(row['folds_better'])}/{int(row['n_folds'])}"


def _p_cell(row: pd.Series | None, key: str = "p_ttest") -> str:
    if row is None or key not in row or pd.isna(row[key]):
        return "n/a"
    return _fmt_p(float(row[key]))


def _num_cell(row: pd.Series | None, key: str) -> str:
    if row is None or key not in row or pd.isna(row[key]):
        return "n/a"
    return _fmt_num(float(row[key]))


def _equiv_cell(row: pd.Series | None) -> str:
    """TOST equivalence verdict: ``yes``/``no`` (within ``±margin``) with the FDR-corrected equivalence
    p-value ``p_tost_fdr`` (falls back to the raw ``p_tost`` when no corrected column is present)."""
    if row is None or "p_tost" not in row or pd.isna(row.get("p_tost", np.nan)):
        return "n/a"
    p = row["p_tost_fdr"] if "p_tost_fdr" in row and not pd.isna(row.get("p_tost_fdr", np.nan)) else row["p_tost"]
    mark = "yes" if bool(row.get("equivalent", False)) else "no"
    return f"{mark} ({_fmt_p(float(p))})"


def _fmt_num(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.4g}"


def _fmt_p(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _notes(paired: pd.DataFrame, runs: pd.DataFrame) -> str:
    baseline_note = ""
    baselines = sorted(str(v) for v in paired["baseline"].dropna().unique()) if "baseline" in paired else []
    if baselines:
        baseline_note = f"`paired.csv` contains comparisons against: {', '.join(f'`{b}`' for b in baselines)}."
    fallback_note = (
        "Rows marked `runs.csv` recompute the same paired test from per-fold `runs.csv` because "
        "`paired.csv` did not contain that family-direct baseline."
    )
    metrics_note = "Deltas are `bridged - direct`; negative is better for nll/brier/ece, positive is better for auc/acc."
    stats_note = (
        "`dz` is Cohen's standardized effect; `p_t`/`p_fdr`/`p_w` test a *difference* (two-sided t, "
        "BH-FDR-adjusted, and Wilcoxon); `equiv (TOST)` tests *equivalence* — `yes` means the per-fold "
        "deltas lie within the metric's pre-set margin. A non-significant difference is not equivalence; "
        "read `dz` + `equiv` together."
    )
    margins = _join_values(paired.get("tost_margin", pd.Series(dtype=float))) if "tost_margin" in paired else ""
    margin_note = f"TOST margins (per row): {margins}." if margins and margins != "n/a" else ""
    n_rows = len(runs[runs["split"] != "recovery"]) if "split" in runs else len(runs)
    return (f"## Notes\n\n{metrics_note} {stats_note} {margin_note} {baseline_note} {fallback_note} "
            f"Per-fold metric rows summarized: {n_rows}.")
