# 676 Project

CSCE 676 final project. We use the SatHealth dataset (satellite-derived
environmental signals + ICD-chapter disease prevalence across Ohio
CBSAs) to find environment → disease associations that could support
**population-level early warning** for clinical conditions.

## Why this matters

If a satellite-derivable signal — temperature, soil moisture, NO₂, NDVI
— moves the population-level prevalence of a disease chapter weeks or
months ahead of presentation, hospitals can plan capacity (beds,
staffing, supplies) against an exogenous, publicly-available leading
indicator. That is the clinical value we are chasing: not individual
diagnosis, but anticipatory operational planning at the metro-area
level.

## Research question

**RQ1: Which satellite-derived environmental exposures are associated
with elevated CBSA-year prevalence of which ICD-10 chapters, and at
what lag?**

We answer it two ways and require both to agree before we trust a
signal:

- **DLNM** (distributed-lag non-linear model) — Poisson DLNM with a
  cross-basis over exposure value × lag (months 0–23) gives a
  cumulative log relative risk per (exposure, chapter) pair, BH-FDR
  controlled across the sweep.
- **FP-Growth itemset mining** — discretise each CBSA-year into
  high/low quartile tokens for environment and disease, then mine
  association rules whose antecedent is purely environmental and whose
  consequent is a disease chapter.

A small bridge table cross-links the two so that an itemset rule and a
DLNM ranking back the same underlying (exposure, chapter) signal.

## Video walkthrough

📺 https://www.youtube.com/watch?v=ZYlifHTKkA8

## Main notebook

`main_notebook.ipynb` is the entry point. It runs the end-to-end
pipeline (load → DLNM sweep → FP-Growth → bridge), reproduces the
headline figures, and shows the cross-link between the two methods on
the same panel.

The graded checkpoint notebook is `yichen_630007741_ckpt2.ipynb`. A
focused single-pair walkthrough lives in `rq1_dlnm_demo.ipynb`.

## Results summary

The DLNM sweep covers **108 (exposure, chapter) pairs** on a **92
CBSA-year panel** (4 chapters × 27 pruned exposures, minus a few pairs
that lack the 23-month lookback). After BH-FDR control, ~100 pairs
survive q ≤ 0.1 (each gets a three-panel figure in
`results/figures/`).

Top cumulative-RR signals (10th vs 90th percentile of exposure):

| Exposure | Chapter (gloss) | log RR |
|---|---|---|
| `lake_bottom_temperature` | I00–I99 (circulatory) | +2.33 |
| `surface_thermal_radiation_downwards_sum` | F01–F99 (mental) | +1.56 |
| `lake_bottom_temperature` | E00–E89 (endocrine) | +1.31 |
| `lake_bottom_temperature` | F01–F99 (mental) | +1.16 |
| `total_aerosol_optical_depth_at_550nm_surface` | J00–J99 (respiratory) | −0.74 |

FP-Growth on the same panel (92 transactions, min support 0.05, min
confidence 0.5) yields **979 rules** with environmental antecedents
and disease-chapter consequents. The bridge table confirms the top
itemset rules ride exposure-chapter pairs that DLNM independently
ranks high — including `lake_bottom_temperature → I00–I99` and
soil-temperature → respiratory.

Caveats live in the technical report — small-panel asymptotic CIs are
optimistic, and FP-Growth on 92 transactions is a co-occurrence
sketch, not a calibrated effect-size estimator.

## Reports

- `docs/rq1_technical_report.html` — full DLNM + FP-Growth writeup
  with figures and the bridge analysis.
- `docs/rq1_flash.html` — short flash-deck version for class
  presentation.

## Dataset Setup

The `sathealth_dataset/` directory is not included in this repository
due to its size (~320 MB). To set up the project:

1. Obtain the SatHealth dataset.
2. Place the dataset folder in the project root so the structure looks like:

```
676_project/
├── sathealth_dataset/
│   ├── CBSA/
│   ├── County/
│   ├── CT/
│   ├── ZCTA/
│   ├── column_dictionary.csv
│   ├── google_map_points.csv
│   ├── icdl1_prev_ohio.csv
│   ├── icdl2_prev_ohio.csv
│   ├── icdl3_prev_ohio.csv
│   └── README.md
├── main_notebook.ipynb
├── yichen_630007741_ckpt2.ipynb
├── rq1_dlnm_demo.ipynb
├── eda.py
└── ...
```

## Reproducing RQ1

Bootstrap the environment and run the test suite first:

```bash
pixi install
pixi run pytest -q
```

### DLNM sweep

```bash
pixi run python -c "from rq1_dlnm.run_sweep import run; run()"
```

Outputs land in `results/summary.csv` (one row per (exposure,
chapter) pair, with log RR, SE, p, BH-q, convergence flag) and
`results/figures/` (three-panel figure per surviving pair: lag
profile, exposure-response, lag × exposure surface).

The full sweep covers 108 pairs and runs ~4 minutes on Apple Silicon
(MPS). This is a proof-of-concept: confidence intervals come from the
observed-information Hessian at the Adam optimum and are optimistic
on a 92-row panel. Cluster bootstrap CIs are listed as follow-up in
spec §8.

Design spec: `docs/superpowers/specs/2026-04-20-rq1-dlnm-design.md`.
Implementation plan: `docs/superpowers/plans/2026-04-20-rq1-dlnm.md`.

### FP-Growth itemset mining

The itemset pivot lives in `src/rq1_itemset/`:

- `data.py` — build CBSA-year transactions, discretise environmental
  columns into `<col>_Q1..Q4` tokens and disease chapters into
  `<chapter>_high` tokens.
- `mine.py` — FP-Growth + association rules, filtered to
  environmental antecedent → disease consequent (min support 0.05,
  min confidence 0.5, max len 3).
- `bridge.py` — for each rule, look up the DLNM cumulative log-RR for
  the underlying (exposure, chapter) pair so the two pipelines can
  be compared on the same axis.

The mining and bridge are driven from `main_notebook.ipynb`; the unit
tests under `tests/rq1_itemset/` cover the discretiser, miner, and
bridge separately.

## Repository layout

```
src/
├── rq1_dlnm/        # Poisson DLNM sweep (basis, model, predict, plot, run_sweep)
└── rq1_itemset/     # FP-Growth pivot (data, mine, bridge)
tests/
├── rq1_dlnm/
└── rq1_itemset/
docs/
├── rq1_technical_report.html
├── rq1_flash.html
└── superpowers/     # design specs and implementation plans
results/
├── summary.csv      # DLNM sweep output
└── figures/         # three-panel figures per surviving pair
```

## Exploratory Data Analysis

Run the EDA script to generate a Markdown report and figures:

```bash
python eda.py
```

This produces:
- `eda_report.md` — full EDA report with tables and figures
- `eda_output/` — generated PNG visualizations

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
