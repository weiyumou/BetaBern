"""``betabern benchmark`` — run a model-comparison benchmark config.

Runs a config module (a Python file exposing a ``CONFIG`` dict; see ``examples/benchmark_smoke.py``) and
writes ``config.json`` + ``{runs, summary, paired}.csv`` + per-model prediction CSVs + ``report.md`` to a
result dir. ``betabern report`` re-renders the Markdown report for a result dir; ``betabern merge``
concatenates result dirs with disjoint datasets.

    betabern benchmark --config examples/benchmark_smoke.py
    betabern benchmark --config papers/aimecon2026/configs/benchmark_exact.py --degree 10 --workers 6
"""
import argparse
import importlib.util
import json
import logging
import os
import warnings
from datetime import datetime

from betabern.benchmark import stats


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", required=True, help="Path to a Python module exposing a CONFIG dict.")
    p.add_argument("--out", default=None, help="Output dir (default: results/benchmark/<timestamp>).")
    p.add_argument("--workers", type=int, default=1,
                   help="Datasets to run in parallel processes (1 = serial).")
    p.add_argument("--datasets", default=None, help="Comma-separated subset of the config's datasets.")
    p.add_argument("--models", default=None, help="Comma-separated subset of the config's model keys.")
    p.add_argument("--degree", type=int, default=None, help="Override hyperparams['degree'].")
    p.add_argument("--num-nodes", type=int, default=None, help="Override hyperparams['num_nodes'].")


def quiet_lightning() -> None:
    """Silence Lightning/torch INFO logs and UserWarnings; progress comes from run_benchmark instead."""
    warnings.filterwarnings("ignore")
    for name in ("lightning.pytorch", "lightning.fabric"):
        logging.getLogger(name).setLevel(logging.ERROR)


def load_config(path: str) -> dict:
    spec = importlib.util.spec_from_file_location("benchmark_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def _model_key(entry) -> str:
    """A config model entry is a registry key or ``{"key": ..., "overrides": ...}``."""
    return entry if isinstance(entry, str) else entry.get("key")


def main(args) -> None:
    from betabern.benchmark.runner import run_benchmark

    quiet_lightning()
    config = load_config(args.config)
    # CLI hyperparameter overrides — run several degrees in parallel without editing/copying the config.
    hp_over = {k: v for k, v in (("degree", args.degree), ("num_nodes", args.num_nodes)) if v is not None}
    if hp_over:
        config = {**config, "hyperparams": {**config["hyperparams"], **hp_over}}
        print(f"(hyperparameter override: {hp_over})")
    if args.datasets:  # subset the config's datasets by name (e.g. run only the not-yet-run ones, then merge)
        want = [d.strip() for d in args.datasets.split(",")]
        avail = {d.get("name"): d for d in config["datasets"]}
        if missing := [w for w in want if w not in avail]:
            raise SystemExit(f"--datasets {missing} not in config; available: {list(avail)}")
        config = {**config, "datasets": [avail[w] for w in want]}
    if args.models:  # subset the config's models by key
        want = [m.strip() for m in args.models.split(",")]
        avail = {_model_key(m): m for m in config["models"]}
        if missing := [w for w in want if w not in avail]:
            raise SystemExit(f"--models {missing} not in config; available: {list(avail)}")
        config = {**config, "models": [avail[w] for w in want]}
        baseline = config.get("baseline", stats.DEFAULT_BASELINE)
        if baseline not in want:  # paired tests vs the baseline need the baseline present; ok for merge workflows
            print(f"(note: baseline '{baseline}' is not in --models {want}; paired.csv will omit it — "
                  f"recompute it by merging into a run that has the baseline.)")
    if args.workers > 1:
        from betabern.benchmark.parallel import run_benchmark_parallel
        runs, summary, predictions = run_benchmark_parallel(config, max_workers=args.workers, verbose=True)
    else:
        runs, summary, predictions = run_benchmark(config, verbose=True)
    baseline = config.get("baseline", stats.DEFAULT_BASELINE)
    paired = stats.paired_tests(runs, baseline=baseline, tost_margins=config.get("tost_margins"))

    default_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.degree is not None:  # disambiguate concurrent multi-degree runs that share a start second
        default_name += f"_degree={args.degree}"
    out = args.out or os.path.join("results", "benchmark", default_name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)  # the exact config this run used
    runs.to_csv(os.path.join(out, "runs.csv"), index=False)
    summary.to_csv(os.path.join(out, "summary.csv"), index=False)
    paired.to_csv(os.path.join(out, "paired.csv"), index=False)

    # Row-level predictions (P(correct) + estimated ability per interaction), one CSV per model — the
    # source every statistic above is computed from, kept for calibration plots / re-scoring / audits.
    pred_dir = os.path.join(out, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    for model, g in predictions.groupby("model", sort=False):
        g.to_csv(os.path.join(pred_dir, f"{model}.csv"), index=False)

    # Render the paste-friendly direct-vs-bridged report from the CSVs we just wrote.
    try:
        from betabern.benchmark.report import render_markdown
        with open(os.path.join(out, "report.md"), "w") as f:
            f.write(render_markdown(out))
    except Exception as exc:  # a report failure must not lose the run's CSVs
        print(f"(report.md generation skipped: {exc})")

    print(summary.to_string(index=False))
    if not paired.empty:
        print(f"\nPaired per-fold comparison vs '{baseline}' (Δ = model − baseline):")
        print(paired.to_string(index=False))
    print(f"\nWrote config + {len(runs)} runs + summary + paired + "
          f"{len(predictions)} prediction rows ({predictions['model'].nunique()} models) to {out}")
