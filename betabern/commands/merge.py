"""``betabern merge`` — concatenate benchmark result dirs with disjoint datasets.

Recomputes the summary and the paired statistics over the union and concatenates each model's per-row
predictions, so a benchmark can be run dataset-by-dataset (or degree-by-degree, in parallel) and merged
afterwards.

    betabern merge results/benchmark/<run10> results/benchmark/<run20> --out results/benchmark/merged
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from betabern.benchmark import stats
from betabern.benchmark.report import render_markdown
from betabern.benchmark.runner import _summarize


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("dirs", nargs="+", type=Path, help="Result dirs to merge (disjoint datasets).")
    p.add_argument("--out", type=Path, required=True, help="Output dir for the merged run.")
    p.add_argument("--baseline", default=None, help="Paired-test baseline (default: first run's config).")


def main(args) -> None:
    keys = ("dataset", "model", "split", "fold")
    args.out.mkdir(parents=True, exist_ok=True)

    runs = pd.concat([pd.read_csv(d / "runs.csv") for d in args.dirs], ignore_index=True)
    key_cols = [k for k in keys if k in runs.columns]
    if (dups := runs.duplicated(subset=key_cols)).any():
        print(f"WARNING: {int(dups.sum())} duplicate {tuple(key_cols)} rows — do the runs share datasets?")
    models = [set(pd.read_csv(d / "runs.csv")["model"]) for d in args.dirs]
    if any(m != models[0] for m in models[1:]):
        print(f"WARNING: runs have different model lineups: {[sorted(m) for m in models]}")

    cfg = json.loads((args.dirs[0] / "config.json").read_text()) if (args.dirs[0] / "config.json").exists() else {}
    baseline = args.baseline or cfg.get("baseline", stats.DEFAULT_BASELINE)
    summary = _summarize(runs)
    paired = stats.paired_tests(runs, baseline=baseline, tost_margins=cfg.get("tost_margins"))

    runs.to_csv(args.out / "runs.csv", index=False)
    summary.to_csv(args.out / "summary.csv", index=False)
    paired.to_csv(args.out / "paired.csv", index=False)

    # Predictions are one CSV per model; concatenate each model's rows across the runs.
    pred_out = args.out / "predictions"
    pred_out.mkdir(exist_ok=True)
    by_model: dict[str, list[Path]] = {}
    for d in args.dirs:
        for f in sorted((d / "predictions").glob("*.csv")) if (d / "predictions").exists() else []:
            by_model.setdefault(f.name, []).append(f)
    for name, files in by_model.items():
        pd.concat([pd.read_csv(f) for f in files], ignore_index=True).to_csv(pred_out / name, index=False)

    (args.out / "config.json").write_text(json.dumps(
        {"merged_from": [str(d) for d in args.dirs], "baseline": baseline,
         "datasets": sorted(runs["dataset"].unique().tolist())}, indent=2))
    try:
        (args.out / "report.md").write_text(render_markdown(args.out))
    except Exception as exc:  # a report failure must not lose the merged CSVs
        print(f"(report.md skipped: {exc})")

    print(f"merged {len(args.dirs)} runs -> {args.out}  "
          f"({runs['dataset'].nunique()} datasets, {len(runs)} rows, baseline={baseline})")
