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


def test_load_outcomes_filters_chapters_and_returns_counts(tmp_path: Path):
    rows = []
    for c in (10420, 11740):
        for y in (2016, 2017):
            for code in ("J00-J99", "I00-I99", "X99-X99"):
                rows.append({
                    "CBSAFP": c, "year": y, "code": code,
                    "count": 5, "count_patient": 100, "prevalence": 0.05,
                })
    df = pd.DataFrame(rows)
    path = tmp_path / "icdl1_prev_ohio.csv"
    df.to_csv(path, index=False)

    out = dmod.load_outcomes(path, chapters=["J00-J99", "I00-I99"])

    assert set(out["code"].unique()) == {"J00-J99", "I00-I99"}
    assert {"CBSAFP", "year", "code", "count", "count_patient"} <= set(out.columns)
    assert len(out) == 2 * 2 * 2


def test_build_lag_matrix_indexes_lag0_as_december_of_outcome_year():
    rows = []
    for y in (2016, 2017):
        for m in range(1, 13):
            rows.append({
                "CBSAFP": 10420, "year": y, "month": m,
                "temp": float(y * 100 + m),
            })
    env = pd.DataFrame(rows)
    outcomes = pd.DataFrame({
        "CBSAFP": [10420], "year": [2017], "code": ["J00-J99"],
        "count": [5], "count_patient": [100],
    })

    L = dmod.build_lag_matrix(env, outcomes, column="temp", max_lag=23)

    assert L.shape == (1, 24)
    assert L[0, 0] == 2017 * 100 + 12   # lag 0 = Dec of outcome year
    assert L[0, 11] == 2017 * 100 + 1   # lag 11 = Jan of outcome year
    assert L[0, 12] == 2016 * 100 + 12  # lag 12 = Dec of prior year
    assert L[0, 23] == 2016 * 100 + 1   # lag 23 = Jan of prior year


def test_build_lag_matrix_drops_rows_without_full_lookback():
    rows = []
    for y in (2016, 2017):
        for m in range(1, 13):
            rows.append({"CBSAFP": 10420, "year": y, "month": m, "temp": 1.0})
    env = pd.DataFrame(rows)
    outcomes = pd.DataFrame({
        "CBSAFP": [10420, 10420], "year": [2016, 2017],
        "code": ["J00-J99", "J00-J99"], "count": [5, 5], "count_patient": [100, 100],
    })

    kept = dmod.keep_outcomes_with_lookback(outcomes, env, max_lag=23)
    assert kept["year"].tolist() == [2017]  # 2016 can't see 2015, dropped
