# The Beta-Bernstein Bridge (AIME-CON 2026)

Reproduction material for **The Beta-Bernstein Bridge: Closed-Form Bayesian Measurement Beyond
Monotone Item Response Functions**. The measurement math lives in the installable `betabern`
package; this directory holds the study-specific scripts and configs on top of it. It is tracked
in the repo but never ships in the wheel. Run all commands from the repo root after `uv sync`.

| path | contents |
|---|---|
| `scripts/` | the AI-mediated simulation study (`ai_mediated_*.py`) and the shared figure style (`figstyle.py`) |
| `configs/` | benchmark configs for `betabern benchmark` |
| `ai_mediated_simulation_study.md` | the RQ-2 design record: generative model, protocol, and full result tables |
| `irw_shortlist.csv` | the IRW candidate table behind Table 1 (name, size, construct, licence) |

Outputs land in `figures/` (git-ignored). The AI-mediated scripts are self-contained; only the RQ-1
benchmark needs data.

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

**Table 3 is the median over 20 simulation runs.** That is what `ai_mediated_aic.py` and
`ai_mediated_rmse.py` now do by default (`--seeds 20 --agg median`); pass `--agg mean` for the mean.
Read the `[pronounced]` block — the paper's Table 3 is the pronounced AI effect (`p_A = 0.95`,
`s_r = 0.55`); `[mild]` is the sensitivity check in §5.4 of the design record.

The free-Bernstein objective is non-convex, so the two Bernstein rows are not bit-reproducible across
PyTorch versions: a rerun on a rebuilt environment reproduced the 4PL row exactly and the Bernstein
rows to within ~25-40 AIC units. The 4PL row and every RMSE figure reproduce to the printed precision.

`ai_mediated_fit.py` is the shared engine (simulation + fitting); the other `ai_mediated_*` scripts
import it as a sibling module. `figstyle.py` holds the shared publication style (ACL column widths,
Okabe-Ito palette).

## Expected results

Compare a reproduction against these published values.

**Table 1 (datasets)** — reproduces exactly from the cached IRW tables:

| name | IRW table | students | items | % correct |
|---|---|--:|--:|--:|
| `art` | `art` | 1402 | 50 | 27.08 |
| `spelling` | `spelling_assessment_study1` | 673 | 109 | 56.32 |
| `malawi_math` | `gilbert_meta_39` | 6818 | 10 | 43.97 |
| `online_rct` | `gilbert_meta_1` | 7797 | 30 | 53.72 |
| `content_lit` | `gilbert_meta_2` | 2174 | 20 | 54.87 |
| `vocab` | `gilbert_meta_11` | 2588 | 24 | 44.45 |

**Table 2 (RQ-1, held-out NLL averaged over the six datasets)** — `configs/benchmark_exact.py` at
`--degree 10` and `--degree 20`, merged. "Equiv." counts datasets where the bridged model is
TOST-equivalent to its direct counterpart at a 0.01-nat margin, at *both* degrees.

| IRF family | Direct | Bridged (n=10) | Bridged (n=20) | Equiv. | # Params |
|---|--:|--:|--:|--:|--:|
| Logistic (2PL) | 0.5064 | 0.5073 | 0.5069 | 6/6 | 83 |
| Normal-ogive (2PNO) | 0.5072 | 0.5074 | 0.5072 | 6/6 | 83 |
| I-spline | 0.5051 | 0.5066 | 0.5054 | 6/6 | 116 |
| Free-form | — | 0.5061 | 0.5050 | — | 114 & 144 |

**Table 3 (RQ-2)** — median over 20 runs, pronounced effect; the script defaults produce exactly this:

| Model | # Params | AIC | BIC | RMSE (standard) | RMSE (AI-mediated) |
|---|--:|--:|--:|--:|--:|
| Monotone 4PL | 55 | 22565 | 22873 | 0.056 | 0.164 |
| Monotone Bernstein | 59 | 22613 | 22943 | 0.033 | 0.169 |
| **Flexible Bernstein** | 59 | **22196** | **22526** | **0.028** | **0.033** |

The 4PL row and every RMSE figure reproduce to the printed precision. The two Bernstein rows are a
non-convex fit and are not bit-reproducible across PyTorch versions — expect AIC/BIC within ~25–40
units. Figures 1 and 2 and Table 3 all now share one configuration (2000 students, degree 12, 200
epochs, pronounced `p_A = 0.95`, `s_r = 0.55`, `a = 1.6`).
