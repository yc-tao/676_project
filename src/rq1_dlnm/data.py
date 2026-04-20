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
