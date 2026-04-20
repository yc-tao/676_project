# RQ1 DLNM Proof-of-Concept — Design Spec

**Date:** 2026-04-20
**Project:** CSCE 676 final project, SatHealth Ohio
**Status:** Draft for review

## 1. Goal

Address the simpler form of RQ1: **is there a relationship between a given environmental exposure (air quality, temperature, greenery) and disease prevalence, and over what lag kernel does that relationship operate?** Deliver a proof-of-concept using a distributed-lag non-linear model (DLNM) with GPU-accelerated Poisson fits. Scope is intentionally narrow; multi-exposure joint adjustment is listed as follow-up.

This is feasibility work, not causal inference. Claims stay correlational.

## 2. Scope

### In scope
- Per-exposure, per-outcome DLNM fits (marginal grid sweep).
- Monthly exposure lags of 0–23 months into the outcome year.
- Poisson likelihood on raw `count` with `log(count_patient)` offset.
- GPU (MPS) acceleration via a small PyTorch model.
- Four ICD L1 outcomes: **J00-J99** (respiratory), **I00-I99** (circulatory), **F01-F99** (mental), **E00-E89** (endo/metabolic).
- Exposure set selected automatically from the monthly env tables (airquality, climate, greenery) after a prune pass.

### Out of scope
- Joint multi-exposure confounder adjustment (noted follow-up).
- Spatial random effects (14 CBSAs is too few).
- SDI adjustment (not available at CBSA per project memory).
- Hyperparameter tuning — fixed `df_var=3`, `df_lag=3`, `ridge=1e-2`.
- Causal claims.

## 3. Data

- **Exposure tables:** `sathealth_dataset/CBSA/{airquality,climate,greenery}.csv`. Key `(CBSAFP, year, month)`. Inner-joined.
- **Landcover** is CBSA-level static; excluded from the lag model (no time dim).
- **Outcomes:** `sathealth_dataset/icdl1_prev_ohio.csv`. Columns used: `CBSAFP, year, code, count, count_patient`. Rate modeled as Poisson with offset.
- **Panel:** 14 disease-CBSAs × 7 years (2016–2022) × 12 months. After dropping the first year to accommodate 23-month lookback, effective outcome panel = 14 × 6 = **84 CBSA-years** per outcome.

### Exposure prune (deterministic)

1. Drop columns with >50% missing in the 14-CBSA × 84-month window.
2. Drop near-zero-variance columns (`std/|mean| < 0.01` or std = 0).
3. Collapse |r| ≥ 0.98 clusters to one representative (highest variance member).
4. Remaining expected: roughly 8–12 exposures (temperature, dewpoint, precipitation, ozone, NO2, NDVI, LAI, surface pressure, wind speed, radiation-sum representative).

### Missingness handling

- Median-impute missing values per `(CBSAFP, month-of-year)` within the 24-month lag window.
- Track imputation fraction per `(CBSAFP, year)` as a covariate. Not part of the cross-basis; enters the GLM linearly.

## 4. Model

### 4.1 Cross-basis construction

For exposure `e` and outcome-year `y`, lag `k ∈ {0..23}` corresponds to month `m = 12 - (k mod 12)` of year `y - (k // 12)` — i.e. lag 0 is December of year `y`, lag 11 is January of year `y`, lag 23 is January of year `y-1`. Exact indexing lives in `data.build_lag_matrix`.

- **Exposure basis:** natural spline (ns), `df_var = 3`, knots at the 25th/50th/75th percentiles of `e` in the training window.
- **Lag basis:** natural spline, `df_lag = 3`, knots at lags 6, 12, 18.
- **Cross-basis matrix:** column-wise tensor product → 3 × 3 = **9 columns per exposure**.

Implementation: roll our own ns on GPU-friendly tensors. No `rpy2`, no `dlnm` R dep. Reference: Wood (2006) §5.3 definition of a natural cubic spline basis.

### 4.2 Poisson GLM

For row `i` with outcome count `y_i` and patient-denominator `n_i`:

```
log μ_i = α + x_i^CB β + γ_cbsa[i] + δ · year_i + θ · miss_frac_i + log n_i
y_i ~ Poisson(μ_i)
```

- `β ∈ ℝ^9`: cross-basis coefficients, ridge-penalized (λ = 1e-2).
- `γ_cbsa`: 14 CBSA fixed effects, one absorbed (reference = smallest CBSAFP). **Unpenalized.**
- `δ`: linear year trend. **Unpenalized.**
- `θ`: imputation-fraction covariate. **Unpenalized.**
- `α`: intercept. **Unpenalized.**

Fit: PyTorch, Adam, lr = 5e-2, 500 steps, `device='mps'`. Convergence check: loss Δ < 1e-6 across 20 consecutive steps or max steps reached. No early stopping heuristics.

### 4.3 Inference

- **Covariance:** analytic observed-information matrix at the MAP (Poisson Fisher info + ridge prior). CIs via Gaussian approx on the log-RR scale.
- **Reference exposure:** median of `e` in training.
- **Cumulative RR at exposure value `v`:** `exp(∑_k s(v) ⊗ w(k) β - ∑_k s(med) ⊗ w(k) β)`, summed across lags.
- **Lag profile at exposure `v`:** RR per lag `k`.
- **Significance:** per-pair two-sided Wald z-test on the cumulative log-RR contrast at the 90th vs 10th percentile of `e` (SE from the delta method applied to `β`'s observed-information covariance). BH-FDR across all (exposure, outcome) pairs, accept at q ≤ 0.1.

## 5. Components

All under `src/rq1_dlnm/` — each module has one clear purpose and is independently testable.

| Module | LOC target | Purpose |
|---|---|---|
| `data.py` | ~80 | Load env & outcomes, build lag matrices, handle missingness. |
| `basis.py` | ~40 | Natural-spline basis and cross-basis tensor product. |
| `model.py` | ~60 | `PoissonDLNM` torch module, `fit()`, `observed_information()`. |
| `predict.py` | ~50 | Exposure-lag surface, cumulative RR, lag profile, CIs. |
| `plot.py` | ~60 | Contour surface, cumulative RR curve, lag profile plot. |
| `run_sweep.py` | ~80 | Orchestrate sweep, write `results/summary.csv` and PNGs. |

Total: ~370 LOC. A thin `rq1_dlnm_demo.ipynb` imports `run_sweep` and renders the summary + top figures.

## 6. Data flow

```
airquality.csv ─┐
climate.csv    ─┼─ data.load_env_monthly() ──▶ monthly_env (14 CBSAs × 84 months × k exposures)
greenery.csv   ─┘                                      │
                                                       ├─ data.build_lag_matrix(e, max_lag=23)
                                                       │       ──▶ L (84 × 24) per exposure
icdl1_prev_ohio.csv ── data.load_outcomes(chapters) ──▶ outcomes (14 × 6 × 4)
                                                       │
                                       basis.cross_basis(L, df_var=3, df_lag=3)
                                                       │       ──▶ X_cb (84 × 9)
                                                       ▼
                                       model.PoissonDLNM.fit(...) on MPS
                                                       │
                                                       ├─ predict.exposure_lag_surface()
                                                       ├─ predict.cumulative_rr()
                                                       └─ plot.render(...)  ──▶ results/
```

## 7. Evaluation

### Sanity checks (must pass before trusting any result)

1. Temperature → respiratory (J00-J99) recovers a roughly U-shaped exposure-response. If linear/flat, basis or lag indexing is wrong.
2. Ridge → 0 (no penalty) reproduces unpenalized MLE to 3 sig figs when the design is well-conditioned.
3. Swapping lag axis direction changes nothing about same-year effects but flips the interpretation of "delayed" lags — verify by testing lag 0 is the outcome-year month, not the first observed month.

### PoC deliverable

- `results/summary.csv`: one row per (exposure, outcome) with cumulative log-RR @ 90th vs 10th pct, SE, BH-adjusted q-value.
- One figure panel per significant pair (q ≤ 0.1): contour + cumulative RR + lag profile.
- Narrative in the demo notebook: which exposures showed non-null lag kernels for which outcomes, and for how long a window.

### Negative results are fine

If no pair survives FDR, report that honestly. The PoC answer ("is there a relationship and at what lag?") is still answered — the answer is "not detectable in this small panel."

## 8. Follow-up (not this PoC)

- Joint multi-exposure DLNM with per-block ridge.
- Bootstrap CIs (CBSA-cluster bootstrap) to replace asymptotic CIs.
- Monthly outcome reconstruction (if we can get sub-annual outcome data later).
- Spatial random effects once the CBSA count is larger.

## 9. Risks & mitigations

- **Tiny panel (84 rows).** Mitigation: ridge penalty, low df, conservative FDR, explicit PoC framing.
- **Lag indexing bugs.** Mitigation: explicit unit tests in `data.py` with synthetic data where the true lag structure is known.
- **Convergence on MPS.** Adam is robust; if any fit hits max steps without Δ < 1e-6, log a warning but report the final estimate (don't silently fail).
- **Multiple testing.** ~40 pairs × 1 cumulative-RR test each → BH-FDR on the 40 p-values. Not claiming individual pair causality.

## 10. Minimal deliverable set

```
src/rq1_dlnm/
  __init__.py
  data.py
  basis.py
  model.py
  predict.py
  plot.py
  run_sweep.py
results/
  summary.csv
  figures/{exposure}_{chapter}.png
rq1_dlnm_demo.ipynb
```

No additional infra, no configs, no CLI flags. The notebook and `run_sweep.py` are the only entry points.
