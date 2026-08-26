# The Beta-Bernstein Bridge (AIME-CON 2026)

Reproduction material for **The Beta-Bernstein Bridge: Closed-Form Bayesian Measurement Beyond
Monotone Item Response Functions**. The measurement math lives in the installable `betabern`
package; this directory holds the study-specific scripts and configs on top of it. It is tracked
in the repo but never ships in the wheel. Run all commands from the repo root after `uv sync`.

| path | contents |
|---|---|
| `scripts/` | the AI-mediated simulation study (`ai_mediated_*.py`) and the shared figure style (`figstyle.py`) |
| `configs/` | benchmark configs for `betabern benchmark` |

## Paper artifact → command map

| Artifact | Source |
|---|---|
| Table 1 (datasets) | `irw_shortlist.csv` + the summary line printed by `betabern fetch-irw` |
| Table 2 (RQ-1, held-out NLL, TOST equivalence) | `betabern benchmark` with `configs/benchmark_exact.py` (below) |
| Table 3 (RQ-2, AIC/BIC + RMSE) | `scripts/ai_mediated_aic.py` + `scripts/ai_mediated_rmse.py` |
| Figure 1 (`ai_mediated_irf.pdf`) | `scripts/ai_mediated_irf.py` |
| Figure 2 (`ai_mediated_recovery.pdf`) | `scripts/ai_mediated_fit.py` |
| (commented-out robustness table) | `scripts/ai_mediated_study.py` |

## RQ-1: faithfulness benchmark (Table 2)

Fetch the six IRW datasets (needs the `irw` package + a Redivis token; see `irw_shortlist.csv`):

```bash
betabern fetch-irw art spelling_assessment_study1 gilbert_meta_39 gilbert_meta_1 gilbert_meta_2 gilbert_meta_11
```

Run the benchmark at both degrees and merge (the paper's runs were merged this way):

```bash
betabern benchmark --config papers/aimecon2026/configs/benchmark_exact.py --degree 10
betabern benchmark --config papers/aimecon2026/configs/benchmark_exact.py --degree 20
betabern merge results/benchmark/<run10> results/benchmark/<run20> --out results/benchmark/merged
betabern report results/benchmark/merged
```

## RQ-2: AI-mediated simulation (Table 3, Figures 1–2)

Self-contained (no data files needed):

```bash
python papers/aimecon2026/scripts/ai_mediated_irf.py --out figures/    # Figure 1
python papers/aimecon2026/scripts/ai_mediated_fit.py --out figures/    # Figure 2 (+ posterior figure)
python papers/aimecon2026/scripts/ai_mediated_aic.py                   # Table 3: #Params, AIC, BIC
python papers/aimecon2026/scripts/ai_mediated_rmse.py                  # Table 3: RMSE columns
python papers/aimecon2026/scripts/ai_mediated_study.py                 # robustness (not in current draft)
```

`ai_mediated_fit.py` is the shared engine (simulation + fitting); the other `ai_mediated_*` scripts
import it as a sibling module. `figstyle.py` holds the shared publication style (ACL column widths,
Okabe-Ito palette).
