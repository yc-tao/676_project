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
