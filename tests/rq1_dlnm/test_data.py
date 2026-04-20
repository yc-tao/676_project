"""Tests for src/rq1_dlnm/data.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rq1_dlnm import data as dmod


def test_load_env_monthly_joins_three_tables_on_cbsa_year_month(tmp_path: Path):
    # Build three tiny CSVs with the same key grid.
    keys = [(10420, y, m) for y in (2016, 2017) for m in range(1, 13)]
    air = pd.DataFrame(keys, columns=["CBSAFP", "year", "month"])
    air["ozone"] = np.arange(len(air), dtype=float)
    clim = air[["CBSAFP", "year", "month"]].copy()
    clim["temperature_2m"] = np.arange(len(clim), dtype=float) + 100
    green = air[["CBSAFP", "year", "month"]].copy()
    green["NDVI"] = np.arange(len(green), dtype=float) + 1000

    root = tmp_path / "CBSA"
    root.mkdir()
    air.to_csv(root / "airquality.csv", index=False)
    clim.to_csv(root / "climate.csv", index=False)
    green.to_csv(root / "greenery.csv", index=False)

    out = dmod.load_env_monthly(tmp_path, cbsa_list=[10420])

    assert list(out.columns[:3]) == ["CBSAFP", "year", "month"]
    assert {"ozone", "temperature_2m", "NDVI"} <= set(out.columns)
    assert len(out) == 24
    assert out["CBSAFP"].unique().tolist() == [10420]
