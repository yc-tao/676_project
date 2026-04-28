"""Orchestrator: fit all (exposure, outcome) pairs and write results/."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from statsmodels.stats.multitest import multipletests

from rq1_dlnm.basis import cross_basis
from rq1_dlnm.data import (
    build_lag_matrix,
    impute_and_flag,
    keep_outcomes_with_lookback,
    load_env_monthly,
    load_outcomes,
    prune_exposures,
)
from rq1_dlnm.model import PoissonDLNM, fit, observed_information
from rq1_dlnm.plot import three_panel
from rq1_dlnm.predict import cumulative_rr_contrast

CHAPTERS = ["J00-J99", "I00-I99", "F01-F99", "E00-E89"]
DF_VAR = 3
DF_LAG = 3
MAX_LAG = 23
RIDGE = 1e-2
STEPS = 2000
LR = 5e-2


@dataclass
class PairResult:
    exposure: str
    chapter: str
    log_rr: float
    se: float
    p: float
    q: float
    converged: bool


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def run(
    dataset_root: Path | str = "sathealth_dataset",
    out_dir: Path | str = "results",
    chapters: list[str] = CHAPTERS,
) -> pd.DataFrame:
    dataset_root = Path(dataset_root)
    out_dir = Path(out_dir)
    figs = out_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    outcomes_all = load_outcomes(dataset_root / "icdl1_prev_ohio.csv", chapters=chapters)
    disease_cbsas = sorted(outcomes_all["CBSAFP"].unique().tolist())
    env = load_env_monthly(dataset_root, cbsa_list=disease_cbsas)

    excluded = {"CBSAFP", "year", "month"}
    candidates = [c for c in env.columns if c not in excluded]
    kept_cols = prune_exposures(env, candidates)

    lag_knots = torch.tensor([6.0, 12.0, 18.0])
    results: list[PairResult] = []
    raw = []  # (PairResult, artifacts for plotting)

    for exp_col in kept_cols:
        for chap in chapters:
            out = outcomes_all[outcomes_all["code"] == chap].reset_index(drop=True)
            out = keep_outcomes_with_lookback(out, env, max_lag=MAX_LAG)
            if len(out) == 0:
                continue
            L = build_lag_matrix(env, out, column=exp_col, max_lag=MAX_LAG)
            L, miss = impute_and_flag(L)
            L_t = torch.tensor(L, dtype=torch.float32)
            # (min, median, max) keeps the standardized polynomial basis in [-1, 1],
            # bounding z^2 and z^3 across observed L.
            v_knots = torch.tensor([
                float(L_t.min()), float(L_t.flatten().median()), float(L_t.max()),
            ])
            X = cross_basis(L_t, var_knots=v_knots, lag_knots=lag_knots)

            cbsa_order = {c: i for i, c in enumerate(sorted(out["CBSAFP"].unique()))}
            cbsa_idx = torch.tensor(out["CBSAFP"].map(cbsa_order).to_numpy(), dtype=torch.long)
            year = torch.tensor((out["year"] - out["year"].mean()).to_numpy(), dtype=torch.float32)
            miss_t = torch.tensor(miss, dtype=torch.float32)
            count = torch.tensor(out["count"].to_numpy(), dtype=torch.float32)
            offset = torch.log(torch.tensor(out["count_patient"].to_numpy(), dtype=torch.float32))

            m = PoissonDLNM(n_cb=X.shape[1], n_cbsa=len(cbsa_order))
            fit_res = fit(
                m, X_cb=X, cbsa_idx=cbsa_idx, year=year, miss=miss_t,
                offset=offset, count=count,
                ridge=RIDGE, steps=STEPS, lr=LR, device=_device(),
            )
            H = observed_information(
                m, X_cb=X, cbsa_idx=cbsa_idx, year=year, miss=miss_t,
                offset=offset, ridge=RIDGE,
            )
            cov = torch.linalg.inv(H)
            beta = m.beta.detach().cpu()

            v_low, v_med, v_high = (float(q) for q in np.quantile(L, [0.10, 0.50, 0.90]))
            contrast = cumulative_rr_contrast(
                beta=beta, cov=cov, v_low=v_low, v_high=v_high,
                var_knots=v_knots, lag_knots=lag_knots, nlag=MAX_LAG + 1,
            )
            pr = PairResult(
                exposure=exp_col, chapter=chap,
                log_rr=contrast.log_rr, se=contrast.se, p=contrast.p, q=np.nan,
                converged=fit_res.converged,
            )
            results.append(pr)
            raw.append((pr, dict(
                beta=beta, cov=cov, v_knots=v_knots, lag_knots=lag_knots,
                L=L, v_low=v_low, v_high=v_high, v_med=v_med,
            )))

    df = pd.DataFrame([r.__dict__ for r in results])
    if len(df) > 0:
        qvals = multipletests(df["p"].to_numpy(), method="fdr_bh")[1]
        df["q"] = qvals
        for r, qv in zip(results, qvals):
            r.q = float(qv)
    df.sort_values("q").to_csv(out_dir / "summary.csv", index=False)

    for pr, art in raw:
        if not np.isnan(pr.q) and pr.q <= 0.1:
            v_grid = np.linspace(art["L"].min(), art["L"].max(), 40)
            three_panel(
                beta=art["beta"], cov=art["cov"],
                var_knots=art["v_knots"], lag_knots=art["lag_knots"], nlag=MAX_LAG + 1,
                v_values=v_grid, v_ref=art["v_med"], v_90=art["v_high"],
                exposure_label=pr.exposure, outcome_label=pr.chapter,
                out_path=figs / f"{pr.exposure}_{pr.chapter}.png",
            )
    return df
