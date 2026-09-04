# AI-Mediated Assessment: a Non-Monotone IRF Simulation Study

*Supports the "Conjugate IRT" paper's capability claim that the Beta–Bernstein framework extends beyond
monotone item response functions. This document records the study end-to-end: motivation, data-generating
model, fitted models, evaluation protocol, and results (including two follow-up analyses). All figures are in
[`figures/ai_mediated/`](../figures/ai_mediated/); the runnable code is
[`scripts/ai_mediated_fit.py`](scripts/ai_mediated_fit.py) (single run + figures) and
[`scripts/ai_mediated_study.py`](scripts/ai_mediated_study.py) (multi-seed effect-size study).*

## 1. Motivation

Assessment is increasingly mediated by AI: a respondent answers items with help from a chatbot or copilot, so
the recorded response is no longer a clean signal of the respondent's own ability. We ask a concrete
psychometric question: **can AI mediation reshape an item's characteristic curve into a shape that standard
(monotone) IRT cannot represent — and if so, does the conjugate Beta–Bernstein framework still fit it with a
closed-form posterior?**

The conjugacy of the Beta–Bernstein bridge requires only that each Bernstein weight be a valid probability
(`w_k ∈ [0,1]`); it does **not** require the weights to be monotone. Monotonicity is a modeling prior, not a
conjugacy requirement. This study makes that distinction operational: it constructs a plausible generative
story under which the realized IRF is **non-monotone**, and shows that a *bounded* (non-monotone-capable)
Bernstein IRF recovers it — with an exact, closed-form ability posterior — where every monotone model
(including the 4PL) structurally cannot.

This is a **capability demonstration**, not the paper's core empirical claim. The core results stay on real
data with monotone IRFs; this study is the single simulated panel that shows the framework's reach.

## 2. Data-generating model

For a single item and a respondent with ability `θ ∈ [0,1]` (mapped to the IRT scale by the bridge link
`η = 6θ − 3`), the realized success probability is a **reliance-weighted mixture** of the student acting alone
and the AI acting alone:

```
p_S(θ) = sigmoid(a · (η − b))               # student-alone success — a genuine 2PL logistic curve
r(θ)   = sigmoid(−(η − t_r) / s_r)           # reliance in [0,1]: probability the student defers to the AI
P(θ)   = r(θ) · p_A + (1 − r(θ)) · p_S(θ)    # realized (observed) item response function
```

The **student's own IRF is a genuine 2PL logistic** — only the AI-mediation mixture bends the realized curve
off-logistic. The shape of `P(θ)` is governed by the reliance pattern `r(θ)`:

- **Standard items** set `r ≡ 0`: a plain monotone 2PL.
- **AI-mediated items** use a competent AI (`p_A` high) and an ability-graded reliance that *decreases* in
  `θ` (weaker respondents lean hardest). When the AI is competent and reliance is sharp, this carves a
  **non-monotone "valley"**: low-ability respondents ride the AI to a high score, mid-ability respondents
  (who override the AI but aren't skilled enough) dip, and high-ability respondents recover on their own. A
  correct response is then genuinely ambiguous between low ability (rode the AI) and high ability (did it
  alone).

**Item bank.** 4 standard 2PL items (difficulties `b ∈ {−1.2, −0.4, 0.4, 1.2}`) + 6 AI-mediated items
(`b ∈ {−0.4, 0, 0.4, 0.8, 1.2, 1.6}`), all with discrimination `a = 1.6`. The AI-effect strength is set by
`(p_A, s_r)`: **pronounced** `(0.95, 0.55)` for the headline figures, **mild** `(0.92, 0.80)` for the
sensitivity follow-up.

**Ability distribution (a deliberately conservative choice).** True ability is drawn `η ~ Normal(0, 1)` (the
IRT standard), clipped to the bridge window `[−3, 3]`. This makes the **4PL's Normal prior correctly
specified** and the Bernstein models' `Beta(4, 4)` prior the *mildly misspecified* one — so the design tilts
the prior advantage toward the monotone baseline, not toward the proposed model. (A `--ability beta` option
matches the Bernstein prior instead; it is not used for the headline results.)

The generative shapes (dilution vs. valley, and the reliance-gradient sweep) are illustrated in
[`ai_mediated_irf.png`](../figures/ai_mediated/ai_mediated_irf.png) and decomposed in
[`ai_mediated_mechanism.png`](../figures/ai_mediated/ai_mediated_mechanism.png).

## 3. Models fit

All three are fit by marginal maximum likelihood (ability integrated out), with matched mean/SD priors:

| model | IRF | posterior | prior |
|---|---|---|---|
| **4PL direct IRT** | logistic with lower+upper asymptotes (**monotone**) | Gauss-Hermite quadrature on ℝ | Normal(0, 1) — *correctly specified* |
| **free Bernstein (monotone)** | freely-learned monotone Bernstein weights | exact closed-form Beta mixture | Beta(4, 4) |
| **free Bernstein (bounded, ours)** | freely-learned **bounded** Bernstein weights (no monotone chaining) | exact closed-form Beta mixture | Beta(4, 4) |

The 4PL is the strongest *monotone parametric* baseline (both asymptotes); the monotone Bernstein is the
strongest *monotone nonparametric* baseline — including it isolates **monotonicity**, not parametric form, as
the binding constraint. The bounded model is the new piece: `BoundedBernsteinIRF`
([`bernstein_irf.py`](../../betabern/bernstein/bernstein_irf.py)) sets `w_k = sigmoid(u_k)` independently
per node (no `w_i ≥ w_{i-1}` chaining), so it can represent a non-monotone curve while plugging unchanged into
the exact conjugate fold and the closed-form Beta-mixture posterior
([`build_exact_bernstein_estimator_bounded`](../../betabern/bernstein/estimator/exact.py)).

## 4. Evaluation protocol

- **Fit** on a fresh training sample; **evaluate** held-out marginal NLL on an independent test sample of the
  same size (item parameters are shared; ability is latent/integrated, so held-out students simply use the
  fitted items + prior).
- **Two metrics.** *Full* = held-out marginal NLL over the whole 10-item response vector. *AI-conditional* =
  the conditional NLL of the 6 AI-mediated items **given** the 4 standard items, computed by the chain rule
  `NLL(all) − NLL(standard-only)`. The AI-conditional metric isolates the effect on the AI items instead of
  diluting it across the standard items where all models tie.
- **Replication unit = the simulation run** (a fresh dataset + full refit), not the student. With thousands
  of held-out students, a within-run paired test over per-student NLL is dominated by sample size and would
  call any non-zero gap "significant"; the run level instead measures **consistency across data
  realizations**. We report, across runs: mean ΔNLL with a 95% CI, the win rate, Cohen's `dz`, and a paired
  t-test. **All p-values are Benjamini–Hochberg (FDR) corrected** across the full family of comparisons
  (2 effect strengths × 2 metrics × 2 baselines = 8 tests). We lead with magnitude + win rate; the p-value is
  secondary, and a large run-level `dz` should be read as *reproducibility* (small run-to-run variance
  relative to the mean effect), not as a large per-student effect.
- **Capacity check.** Beyond held-out NLL (which already penalises overfitting empirically), we report the
  trainable parameter count and AIC/BIC as an in-sample, capacity-penalised complement (§5.5), including a
  parameter-matched baseline (the monotone Bernstein has the same `k` as the bounded model).
- **Recovery vs. ground truth.** Because the data are simulated, we also report the unweighted RMSE
  between each fitted IRF and the *true* curve (§5.6). This is unbiased and needs **no held-out split** — a
  spuriously wiggly fit only raises its RMSE-to-truth — so it is the right metric for the simulated regime,
  complementing the prediction-side held-out NLL and AIC/BIC. (A Brier score vs. the binary responses is
  avoided: it has an irreducible noise floor and, in-sample, favours capacity.)

Together the three families of metrics are each unbiased in a different way: **recovery RMSE** (does it
recover the truth? — no CV needed), **held-out NLL** (does it generalise without overfitting?), and
**AIC/BIC + a parameter-matched baseline** (does it survive a capacity penalty?).

## 5. Results

### 5.1 Qualitative — recovery and posterior (headline)

[`ai_mediated_recovery.png`](../figures/ai_mediated/ai_mediated_recovery.png): on a **standard** 2PL item
(panel a) all three models agree, and the bounded model does not hallucinate a dip where there is none. On an
**AI-mediated** item (panel b) the bounded model traces the non-monotone valley almost exactly, while the
4PL and the monotone Bernstein are structurally forced to rise through it — both blind to the dip.

[`ai_mediated_posterior.png`](../figures/ai_mediated/ai_mediated_posterior.png): for a respondent who answers
the AI-mediated items correctly, the exact closed-form ability posterior under the recovered valley IRF is
**bimodal** — a dominant "low ability, rode the AI" mode and a secondary "high ability, did it alone" mode.
A Gaussian (mean ± SD) summary — what a point standard error would report — places its single peak between
the modes, exactly where the true posterior has a trough. This is the closed-form posterior earning its
keep: a non-monotone IRF induces a genuinely non-Gaussian (multimodal) posterior that the exact Beta mixture
captures and a normal approximation qualitatively misses.

### 5.2 Quantitative — effect sizes across 20 runs (BH-FDR corrected)

From the 20-run study ([`study_effects.csv`](../figures/ai_mediated/study_effects.csv)). ΔNLL = baseline −
bounded, in nats; positive favours the bounded model. `n = 20` runs; p-values are BH-FDR corrected across all
8 comparisons.

| AI effect | metric | baseline | ΔNLL (nats) | 95% CI | Cohen's dz | win rate | p (FDR) |
|---|---|---|---:|---|---:|:---:|---:|
| pronounced | full | 4PL | **+0.071** | [0.066, 0.077] | 5.96 | 20/20 | 2e-16 |
| pronounced | full | monotone Bernstein | **+0.072** | [0.066, 0.077] | 6.38 | 20/20 | 1e-16 |
| pronounced | AI-conditional | 4PL | **+0.072** | [0.068, 0.076] | 8.11 | 20/20 | 2e-18 |
| pronounced | AI-conditional | monotone Bernstein | **+0.072** | [0.068, 0.076] | 8.59 | 20/20 | 1e-18 |
| mild | full | 4PL | +0.035 | [0.032, 0.038] | 6.13 | 20/20 | 2e-16 |
| mild | full | monotone Bernstein | +0.031 | [0.029, 0.034] | 5.61 | 20/20 | 6e-16 |
| mild | AI-conditional | 4PL | +0.034 | [0.032, 0.037] | 6.05 | 20/20 | 2e-16 |
| mild | AI-conditional | monotone Bernstein | +0.030 | [0.028, 0.033] | 5.42 | 20/20 | 9e-16 |

Across every comparison the bounded model has lower held-out NLL in **all 20 runs**, with 95% CIs well clear
of zero and FDR-corrected p < 1e-15. The large `dz` values reflect **reproducibility** — the run-to-run
variance of the gap is small relative to its mean — not a large per-student effect; the per-AI-item magnitude
is modest (pronounced: ≈ 0.072 / 6 ≈ 0.012 nats per AI item).

### 5.3 Follow-up 1 — effect concentrated on the AI items

We expected the full-vector ΔNLL to be *diluted* by the 4 standard items where all models tie, and the
**AI-conditional** metric (rows `metric = ai`) to recover a larger effect. The data show a subtler, more
honest picture: **the AI-conditional ΔNLL is essentially identical to the full-vector ΔNLL** (pronounced:
0.072 vs 0.071; mild: 0.034 vs 0.035). The reason is that the standard items are fit *equally well* by all
models, so they contribute ≈ 0 to the between-model **difference** (they cancel: `ΔNLL_ai = ΔNLL_full −
ΔNLL_std`, with `ΔNLL_std ≈ 0`) — the dilution affects the absolute NLL, not the gap. So:

- The full-vector ΔNLL is **not** diluted in magnitude; the entire between-model gap is already attributable
  to the 6 AI-mediated items.
- What the AI-conditional metric *does* buy is a **lower-variance estimate** of the same effect — removing the
  standard-item sampling noise lifts `dz` from ≈ 6.0–6.4 to ≈ 8.1–8.6 in the pronounced case (a tighter, more
  reproducible measurement), without changing the point estimate.

The honest takeaway for the paper: report the effect as "≈ 0.07 nats, concentrated entirely on the
AI-mediated items (≈ 0.012 nats/item), reproducible across all 20 runs" — and note that the standard items
serve as a built-in negative control (everyone ties there, as they should).

### 5.4 Follow-up 2 — sensitivity to the AI-effect strength

Re-running the whole study at a **mild** AI effect (`p_A = 0.92, s_r = 0.80`, rows `effect = mild`) tests how
tuned the result is. The ΔNLL roughly **halves** (0.072 → 0.035 vs the 4PL; 0.072 → 0.031 vs the monotone
Bernstein) — exactly the expected graceful degradation as the valley becomes shallower — **but the bounded
model still wins all 20 runs** against both baselines (`dz ≈ 5.4–6.1`, FDR p < 1e-15). The advantage scales
smoothly with the AI-effect strength rather than being knife-edge, so it is not an artifact of the specific
`(p_A, s_r)` choice.

One asymmetry to flag honestly: the **bimodal posterior requires the pronounced effect**. A shallower (mild)
valley does not split the ability posterior, so the multimodality figure
([`ai_mediated_posterior.png`](../figures/ai_mediated/ai_mediated_posterior.png)) is the strong-effect case,
whereas the *recovery* and *NLL* advantages hold across both strengths.

### 5.5 Parameters and information criteria

Held-out NLL already controls for overfitting empirically (capacity that does not generalise hurts on fresh
data), but for completeness — and to pre-empt "does the bounded model just have more parameters?" — we report
the trainable parameter count `k` and the classical information criteria (AIC `= 2k + 2·NLL_train`,
BIC `= k·ln N + 2·NLL_train`; `N = 2000` response patterns; lower is better), as the **median over 20
seeds** — the same aggregate the paper's Table 3 reports ([`study_aic.csv`](../figures/ai_mediated/study_aic.csv),
via [`scripts/ai_mediated_aic.py`](scripts/ai_mediated_aic.py); the median is the script default).

| model | k | AIC (pron.) | BIC (pron.) | AIC (mild) | BIC (mild) |
|---|--:|--:|--:|--:|--:|
| 4PL IRT (monotone) | 55 | 22565 | 22873 | 23163 | 23471 |
| free Bernstein (monotone) | 59 | 22613 | 22943 | 23170 | 23500 |
| **free Bernstein (bounded, ours)** | 59 | **22196** | **22526** | **22969** | **23299** |

Δ (baseline − bounded; positive favours bounded):

| baseline | Δk | ΔAIC (pron.) | ΔBIC (pron.) | ΔAIC (mild) | ΔBIC (mild) |
|---|--:|--:|--:|--:|--:|
| 4PL IRT | +4 | +369 | +347 | +194 | +171 |
| monotone Bernstein | 0 | +417 | +417 | +201 | +201 |

Three points:

1. **The monotone Bernstein is capacity-matched** to the bounded model (`k = 59 = 59`), so its ΔAIC = ΔBIC
   (the penalties cancel) and the criteria reduce to the pure likelihood gap — the bounded model wins by 417
   (pronounced) / 201 (mild) on both.
2. **Against the 4PL** the bounded model carries only 4 extra parameters, yet still wins AIC by ~369/194 and
   BIC by ~347/171 — the penalty (8 on AIC, ≈ 30 on BIC) is negligible next to the likelihood gain, so the
   advantage survives even BIC's harsher penalty.
3. Tellingly, the **monotone Bernstein is the *worst* on AIC/BIC** despite having the *same* capacity as the
   bounded model: its 4 extra parameters over the 4PL buy nothing, because it is still monotone. **Capacity
   alone does not help — relaxing the monotonicity constraint does.** This is the cleanest rebuttal to "the
   bounded model just has more parameters."

(`N = #response patterns` is the natural unit for a marginal IRT likelihood, where ability is integrated out
per student. Using `N = #responses = 20,000` instead enlarges every penalty by a constant `ln 10` factor and
changes no ranking.)

### 5.6 IRF-recovery RMSE (vs. the true curve)

For simulated data the true item curve is a cleaner gold standard than the observed (Bernoulli) responses, so
we report the **unweighted RMSE between each fitted IRF and the true generating IRF** (equal weight per
point of the ability grid, matching the recovery figure's axis), per item type, as the **median over 20
seeds** ([`study_rmse.csv`](../figures/ai_mediated/study_rmse.csv), via
[`scripts/ai_mediated_rmse.py`](scripts/ai_mediated_rmse.py)). This needs **no held-out split**: a model
that wiggles spuriously only *raises* its RMSE-to-truth, so there is nothing to overfit and the repeated seeds
supply the variability. (A Brier score vs. binary responses would instead carry an irreducible Bernoulli noise
floor and, in-sample, a bias toward capacity — so vs-truth is the better metric here.)

| AI effect | model | standard items | AI-mediated items |
|---|---|--:|--:|
| pronounced | 4PL IRT | 0.056 | 0.164 |
| pronounced | monotone Bernstein | 0.033 | 0.169 |
| pronounced | **bounded Bernstein (ours)** | **0.028** | **0.033** |
| mild | 4PL IRT | 0.035 | 0.152 |
| mild | monotone Bernstein | 0.032 | 0.148 |
| mild | **bounded Bernstein (ours)** | **0.025** | **0.039** |

AI-item ΔRMSE (baseline − bounded; positive favours bounded; mean over 20 seeds):

| AI effect | vs 4PL | vs monotone Bernstein |
|---|--:|--:|
| pronounced | +0.078 [0.076, 0.080], 100% | +0.079 [0.073, 0.086], 100% |
| mild | +0.054 [0.052, 0.057], 100% | +0.048 [0.044, 0.051], 100% |

Reading:

- On **standard** (genuinely monotone) items all models recover the curve well (RMSE ≈ 0.03–0.06), and the
  bounded model is *not worse* — its extra freedom introduces no spurious non-monotonicity where none exists.
  This is an important negative control (matching recovery-figure panel a).
- On **AI-mediated** items the bounded model recovers the curve to within ≈ 0.03 probability units while both
  monotone models are off by ≈ 0.15–0.17 — a **~5× lower error in 95–100% of runs**. This is the recovery
  figure (panel b) quantified, and unbiased by construction.
- The gap shrinks under the mild effect (graceful, consistent with §5.4) but the bounded model still wins
  every run.

### 5.7 Curve prior on the item weights: the soft-monotonicity dial

The bounded model's freedom is double-edged: in **data-sparse regions** its fitted weights are unidentified, so
on real data (e.g. the `art` recognition test) it parks a non-monotone *boundary spike* at θ→0 where ≈3% of
students live — implausible as an item curve, even though it barely affects predictions. The fix is a **prior on
the item weights** (MAP calibration); the closed-form *ability* posterior given the weights is untouched, so
nothing is lost on the headline property. We evaluated two such priors and kept one.

- **Soft-monotonicity prior (kept).** Penalize decreasing steps, `mono_lambda · Σ relu(w_{k-1} − w_k)²` — a dial
  between the free bounded model (`λ = 0`) and the hard monotone model (`λ → ∞`). On `art` it removes the spike
  and converges *exactly to* the monotone fit (val NLL 0.368→0.380 ≈ monotone 0.381; mean P(θ→0) 0.62→0.03 as
  `λ = 0→5→20→100`). On the **AI sim** it trades valley fidelity for monotonicity, as expected — it cannot tell
  the *genuine* valley from a spurious wiggle:

  | soft-mono λ | ΔNLL vs monotone | RMSE (AI items) |
  |---|---:|---:|
  | 0 | +0.062 | 0.038 |
  | 5 | +0.045 | 0.043 |
  | 20 | +0.031 | 0.062 |
  | 100 | +0.002 | 0.090 |
  | (monotone) | 0 | 0.105 |

  So in the **non-monotone regime use `λ = 0`** (the prior is the knob you turn *up* only when you want to impose
  monotonicity); the value of the dial is conceptual — *monotonicity is a tunable prior, not a hard requirement*.
- **Smoothness (curvature) prior (evaluated, then removed to simplify the code).** A second-difference penalty
  `Σ (w_{k-1} − 2w_k + w_{k+1})²` instead *discriminated*: it erased the `art` spike while leaving the AI edge
  almost flat (≈ +0.06–0.07 across λ), because the spurious spike is high-curvature but the genuine valley is
  smooth. It is arguably the better regularizer, but we keep a single prior; the soft-monotonicity dial is the
  one retained because it directly realizes the monotone↔bounded continuum the paper discusses.

*Correction (recorded for honesty):* an earlier version of this sweep was run through `fit_mml`, which **did not
apply the weight penalty** (the hook existed only in the Lightning task) — so its apparent λ-trend was just
random-init noise. `fit_mml` now adds `irf.weight_penalty()` to the loss (matching the task), the λ variants
share an init, and the numbers above are the corrected re-run. The **core study (§5.1–5.6) is unaffected**: it
uses the bounded model at `λ = 0` (no penalty).

## 6. Takeaways

1. **AI mediation can produce non-monotone IRFs.** Under a plausible process — a competent AI plus
   ability-graded reliance (weaker respondents lean harder) — the realized item curve develops a mid-ability
   *valley*, even though the student's own IRF is a clean 2PL logistic.
2. **Monotone models structurally cannot fit it.** Both the 4PL (the strongest monotone parametric form, with
   guessing and slipping asymptotes) and a flexible monotone Bernstein are forced to rise through the dip —
   isolating *monotonicity*, not parametric form, as the binding constraint. It is also not parameter count:
   the monotone Bernstein is parameter-matched to the bounded model (`k = 59`) yet still fails, and is the
   worst model on AIC/BIC (§5.5).
3. **The bounded conjugate Bernstein recovers it with a closed form.** Relaxing only the monotonicity
   constraint (`w_k = sigmoid(u_k)`, bounded but unchained) lets the same exact Beta-mixture machinery trace
   the valley — conjugacy needs only boundedness. It recovers the AI-item curve to within ≈ 0.03 (probability
   units) vs ≈ 0.08–0.11 for the monotone models (§5.6), without harming recovery of the standard items.
4. **The advantage is small but perfectly reproducible.** ≈ 0.07 nats (pronounced) / ≈ 0.03 nats (mild),
   concentrated on the AI items (≈ 0.012 nats/item), with a **100% win rate across all 20 runs in all 8
   comparisons** and FDR-corrected p < 1e-15 — under a conservative prior where the 4PL is *correctly
   specified*, and degrading gracefully with the AI-effect strength.
5. **The closed form matters qualitatively, not just in NLL.** A non-monotone IRF induces a genuinely
   multimodal ability posterior (low-ability-rode-the-AI vs. high-ability-did-it-alone); the exact Beta
   mixture captures both modes while a Gaussian standard error places its single peak in the trough between
   them.
6. **Scope.** This is a *capability demonstration on simulated data* — evidence that the framework's reach
   extends to AI-contaminated, non-monotone item responses — not an empirical claim about any real
   AI-assisted dataset. The paper's core results remain on real data with monotone IRFs.

## 7. Reproduction

```bash
# canonical figures (illustration + recovery + posterior) into figures/ai_mediated/
python papers/aimecon2026/scripts/ai_mediated_irf.py  --out figures/ai_mediated
python papers/aimecon2026/scripts/ai_mediated_fit.py  --out figures/ai_mediated --students 3000 --epochs 300

# full effect-size study (2 effect strengths x 2 metrics x 20 runs, BH-FDR corrected)
python papers/aimecon2026/scripts/ai_mediated_study.py --out figures/ai_mediated --students 2000 --epochs 200 --seeds 20

# parameter counts + AIC/BIC (in-sample, capacity-penalised complement)
python papers/aimecon2026/scripts/ai_mediated_aic.py --out figures/ai_mediated   # 20 seeds, median (defaults)

# IRF-recovery RMSE vs. the true curve (no held-out needed)
python papers/aimecon2026/scripts/ai_mediated_rmse.py --out figures/ai_mediated  # 20 seeds, median (defaults)

# soft-monotonicity-prior sweeps (§5.7): the bounded->monotone dial on the AI sim and on real `art`
# NOTE: ai_mono_check.py / art_mono_sweep.py produced the §5.7 numbers but were never committed to
# either this repo or its predecessor (KT-Pantheon) -- §5.7 is a record, not a reproducible run.
betabern benchmark --config papers/aimecon2026/configs/benchmark_bounded_mono.py --workers 6
```

## 8. Files

| path | role |
|---|---|
| [`scripts/ai_mediated_irf.py`](scripts/ai_mediated_irf.py) | illustrative generative IRF shapes (no fitting) |
| [`scripts/ai_mediated_fit.py`](scripts/ai_mediated_fit.py) | generator + fit + recovery/posterior figures; single & multi-seed |
| [`scripts/ai_mediated_study.py`](scripts/ai_mediated_study.py) | multi-seed, 2-effect, 2-metric effect-size study with BH-FDR |
| [`scripts/ai_mediated_aic.py`](scripts/ai_mediated_aic.py) | parameter counts + AIC/BIC (in-sample, capacity-penalised) |
| [`scripts/ai_mediated_rmse.py`](scripts/ai_mediated_rmse.py) | IRF-recovery RMSE vs. the true curve (no held-out needed) |
| `scripts/ai_mono_check.py` | §5.7 sweep on the AI sim — **not in the repo** (never committed; §5.7 is a record only) |
| `scripts/art_mono_sweep.py` | §5.7 sweep on real `art` — **not in the repo** (never committed) |
| [`configs/benchmark_bounded_mono.py`](configs/benchmark_bounded_mono.py) | 10-fold CV: bounded λ=0 vs λ=20 vs monotone (predict-vs-interpret) |
| [`scripts/figstyle.py`](scripts/figstyle.py) | shared publication figure style (Okabe–Ito, Type-42 PDF) |
| `figures/ai_mediated/*.{png,pdf}` | all figures from this study |
| `figures/ai_mediated/study_runs.csv` | per-run held-out NLLs (full + AI-conditional) |
| `figures/ai_mediated/study_effects.csv` | corrected effect-size table |
| `figures/ai_mediated/study_aic.csv` | per-seed parameter counts + AIC/BIC |
| `figures/ai_mediated/study_rmse.csv` | per-seed IRF-recovery RMSE (per item type) |
