# 676 Project

Environmental health analysis using the SatHealth dataset, which combines satellite-based environmental data with health outcomes and social determinants across Ohio.

## Dataset Setup

The `sathealth_dataset/` directory is not included in this repository due to its size (~320 MB). To set up the project:

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
├── ckpt1.ipynb
├── eda.py
└── ...
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

## Running RQ1 DLNM (proof-of-concept)

```
pixi install
pixi run pytest -q
pixi run python -c "from rq1_dlnm.run_sweep import run; run()"
```

Outputs land in `results/summary.csv` and `results/figures/`. See the
design spec at `docs/superpowers/specs/2026-04-20-rq1-dlnm-design.md`
and the plan at `docs/superpowers/plans/2026-04-20-rq1-dlnm.md`.

The full sweep runs ~4 minutes on Apple Silicon (MPS) — 108 (exposure,
chapter) pairs. This is a proof-of-concept: confidence intervals come
from the observed-information Hessian at the Adam optimum and are known
to be optimistic on a 78-row panel. Cluster bootstrap CIs are listed as
follow-up in spec §8.
