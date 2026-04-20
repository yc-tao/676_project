"""Data loading and lag-matrix construction for RQ1 DLNM."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


KEY = ["CBSAFP", "year", "month"]


def load_env_monthly(dataset_root: Path | str, cbsa_list: Iterable[int]) -> pd.DataFrame:
    """Inner-join airquality, climate, greenery on (CBSAFP, year, month).

    Args:
        dataset_root: path whose child `CBSA/` holds the three CSVs.
        cbsa_list: iterable of CBSAFP ids to keep.
    Returns:
        DataFrame with key columns first, sorted by (CBSAFP, year, month).
    """
    root = Path(dataset_root) / "CBSA"
    air = pd.read_csv(root / "airquality.csv")
    clim = pd.read_csv(root / "climate.csv")
    green = pd.read_csv(root / "greenery.csv")
    df = air.merge(clim, on=KEY, how="inner").merge(green, on=KEY, how="inner")
    df = df[df["CBSAFP"].isin(list(cbsa_list))].copy()
    return df.sort_values(KEY).reset_index(drop=True)


def load_outcomes(path: Path | str, chapters: Iterable[str]) -> pd.DataFrame:
    """Load yearly ICD L1 prevalence and filter to the requested chapters."""
    df = pd.read_csv(path, usecols=["CBSAFP", "year", "code", "count", "count_patient"])
    wanted = set(chapters)
    df = df[df["code"].isin(wanted)].copy()
    return df.sort_values(["CBSAFP", "year", "code"]).reset_index(drop=True)


def build_lag_matrix(
    env: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    column: str,
    max_lag: int = 23,
) -> np.ndarray:
    """Build an (n_outcome_rows, max_lag+1) lag matrix for one env column.

    Lag k -> month (12 - k % 12) of year (y - k // 12).
    """
    lookup = env.set_index(["CBSAFP", "year", "month"])[column]
    n = len(outcomes)
    L = np.empty((n, max_lag + 1), dtype=float)
    for i, row in enumerate(outcomes.itertuples(index=False)):
        c, y = row.CBSAFP, row.year
        for k in range(max_lag + 1):
            year_back = k // 12
            month = 12 - (k % 12)
            L[i, k] = lookup.get((c, y - year_back, month), np.nan)
    return L


def keep_outcomes_with_lookback(
    outcomes: pd.DataFrame, env: pd.DataFrame, *, max_lag: int
) -> pd.DataFrame:
    """Drop outcome rows whose required env lookback is not fully present."""
    need_years = max_lag // 12
    min_env_year = env.groupby("CBSAFP")["year"].min()
    keep = outcomes.apply(
        lambda r: r["year"] - need_years >= min_env_year.get(r["CBSAFP"], r["year"] + 1),
        axis=1,
    )
    return outcomes[keep].reset_index(drop=True)


def impute_and_flag(L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise median imputation. Returns (filled, miss_frac per row).

    All-NaN rows get filled with 0.0 (no row median exists) and marked with
    miss_frac == 1.0, so the downstream GLM can carry the miss-fraction
    covariate but callers should expect these rows to carry no exposure
    signal.
    """
    miss_frac = np.isnan(L).mean(axis=1)
    filled = L.copy()
    for i in range(L.shape[0]):
        row = L[i]
        if not np.isnan(row).any():
            continue
        if miss_frac[i] >= 1.0:
            filled[i] = 0.0
            continue
        med = np.nanmedian(row)
        filled[i] = np.where(np.isnan(row), med, row)
    return filled, miss_frac


def prune_exposures(
    env: pd.DataFrame,
    candidate_cols: list[str],
    *,
    max_missing: float = 0.5,
    min_rel_std: float = 0.01,
    corr_cutoff: float = 0.98,
) -> list[str]:
    """Return the subset of candidate_cols that survives the three-step prune."""
    kept: list[str] = []
    for col in candidate_cols:
        s = env[col]
        if s.isna().mean() > max_missing:
            continue
        mean = s.mean()
        std = s.std(ddof=0)
        denom = abs(mean) if abs(mean) > 1e-12 else 1.0
        if std == 0 or std / denom < min_rel_std:
            continue
        kept.append(col)

    if len(kept) <= 1:
        return kept

    data = env[kept].to_numpy(dtype=float)
    corr = np.corrcoef(data, rowvar=False)
    dropped: set[int] = set()
    variances = env[kept].var(ddof=0).to_numpy()
    # Relative tolerance for treating variances as a tie; tie-break prefers
    # the earlier column in candidate_cols for deterministic output.
    rtol = 1e-3
    for i in range(len(kept)):
        if i in dropped:
            continue
        for j in range(i + 1, len(kept)):
            if j in dropped:
                continue
            if abs(corr[i, j]) >= corr_cutoff:
                vi, vj = variances[i], variances[j]
                scale = max(abs(vi), abs(vj), 1e-12)
                if abs(vi - vj) / scale < rtol or vi >= vj:
                    drop = j
                else:
                    drop = i
                dropped.add(drop)
                if drop == i:
                    break
    return [kept[i] for i in range(len(kept)) if i not in dropped]
