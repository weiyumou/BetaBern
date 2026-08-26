# BetaBern

[![CI](https://github.com/weiyumou/BetaBern/actions/workflows/ci.yml/badge.svg)](https://github.com/weiyumou/BetaBern/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reference implementation of **The Beta-Bernstein Bridge: Closed-Form Bayesian Measurement Beyond
Monotone Item Response Functions** (AIME-CON 2026).

Item response theory forces a choice between flexible curves and tractable inference. Writing any
bounded item-response function as a Bernstein polynomial makes the Beta prior conjugate, giving an
**exact closed-form ability posterior as a Beta mixture** — flexible (including non-monotone) curves
with fully tractable Bayesian inference. The package ships the Bridge (exact, quadrature, and
sequential estimators), logistic-IRT baselines, and the benchmark harness used in the paper.

## Installation

Not yet on PyPI — install from source:

```bash
pip install git+https://github.com/weiyumou/BetaBern
```

For development: `uv sync` (creates `.venv` with the package plus the test, lint, and figure
dependencies), `uv run pytest`, and once per clone `git config core.hooksPath .githooks` (a
pre-commit guard that keeps data and run outputs out of the repository).

## Quickstart

The package installs a `betabern` CLI. **Fit a model and get an annotated CSV** — one row per
observed response with `correct_probability`, `ability_mean`, and `ability_var` (the exact
Beta-mixture posterior's mean/variance):

```bash
# on your own response log (user_id, skill_name, item_id, is_correct[, timestamp])
betabern fit --data my_responses.csv --model bern-exact --degree 10

# on a raw IRW-shaped CSV (id, item, resp) — normalized automatically
betabern fit --data my_irw_table.csv --model bern-exact

# straight from the Item Response Warehouse (cached under data/irw/)
betabern fit --irw art --model bern-quad --weights free
```

Models: `bern-exact` (closed-form marginal likelihood), `bern-quad` (Gauss-Jacobi quadrature),
`bern-online` (sequential static-trait), `irt-mml` (Gauss-Hermite logistic IRT baseline), `irt-jml`
(joint MLE baseline). See `betabern fit --help` for the full option list.

Other subcommands:

```bash
betabern simulate --out data/sim.csv                        # simulated log (with ground truth)
betabern fetch-irw art spelling_assessment_study1           # cache IRW tables (needs the `irw` package)
betabern benchmark --config examples/benchmark_smoke.py     # model-comparison benchmark
betabern report <result-dir>                                # re-render a benchmark report
betabern merge <dir1> <dir2> --out <dir>                    # merge disjoint-dataset benchmark runs
```

## Repository layout

| path            | contents | shipped in the wheel |
|-----------------|----------|----------------------|
| `betabern/`  | the installable package (below) | yes |
| `examples/`     | a tiny benchmark config for smoke-testing the harness | no |
| `papers/`       | paper-specific reproduction scripts, configs, and demos (`aimecon2026/`) | no |
| `tests/`        | pytest suite | no |

```
betabern/
├── core/            # model-agnostic machinery
│   ├── model/       #   MonotoneIRF, priors (Beta/Jacobi/Hermite), BetaMixture posterior,
│   │                #   BayesianEstimator (batch MML) and BayesianFilter (sequential) contracts
│   ├── data/        #   ingestion (response logs + IRW tables), datamodule, splitters, simulation
│   ├── harness.py   #   Lightning fit/predict harness -> annotated prediction frame
│   └── task.py      #   Lightning tasks (estimator + filter contracts)
├── bernstein/       # the Bridge: BernsteinIRF family (+ Beta-Bernstein conjugate update),
│                    #   exact / quadrature / online (sequential) estimators
├── irt/             # logistic-IRT baselines: static JML and quadrature MML estimators
├── benchmark/       # benchmark harness: registry, runner, metrics, measurement, reporting
├── commands/        # one module per CLI subcommand
└── cli.py           # the `betabern` entry point
```

## Data policy

No datasets are distributed in this repository. `betabern simulate` generates synthetic response
logs, and the Item Response Warehouse tables used in the paper are fetched on demand with
`betabern fetch-irw` into the git-ignored `data/` directory (see
[`papers/aimecon2026/`](papers/aimecon2026/README.md) for the paper's reproduction steps).

## Relationship to the Filter project

This package is the canonical home of the shared measurement math (`BernsteinIRF` with its conjugate
update, `BetaMixture`, the prior/posterior machinery, and the `BayesianFilter` contract). The dynamic
ability-tracking **Filter** project builds on the conjugacy result implemented here and depends on
this package; it is developed separately.

## Citation

If you use this package in your research, please cite the paper ([CITATION.cff](CITATION.cff)):

```bibtex
@inproceedings{wei2026bernbridge,
  title     = {The Beta-Bernstein Bridge: Closed-Form Bayesian Measurement Beyond Monotone Item Response Functions},
  author    = {Wei, Yumou},
  booktitle = {Proceedings of the AIME Conference (AIME-CON)},
  year      = {2026}
}
```

## License

[MIT](LICENSE)
