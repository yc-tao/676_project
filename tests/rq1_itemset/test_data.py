"""Tests for rq1_itemset.data transaction-panel builder."""
from __future__ import annotations

import numpy as np
import pandas as pd

from rq1_itemset.data import (
    aggregate_env_yearly,
    binarize_outcomes,
    discretize_quartiles,
)


def test_aggregate_env_yearly_collapses_months():
    monthly = pd.DataFrame(
        {
            "CBSAFP": [1, 1, 1, 1, 2, 2, 2, 2],
            "year": [2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020],
            "month": [1, 2, 3, 4, 1, 2, 3, 4],
            "temp": [0, 10, 20, 30, 100, 110, 120, 130],
        }
    )
    out = aggregate_env_yearly(monthly, ["temp"])
    assert len(out) == 2
    assert out.set_index("CBSAFP")["temp"].loc[1] == 15.0
    assert out.set_index("CBSAFP")["temp"].loc[2] == 115.0


def test_discretize_quartiles_emits_quartile_columns():
    df = pd.DataFrame({"x": np.arange(20, dtype=float)})
    out = discretize_quartiles(df, ["x"])
    assert out.shape[0] == 20
    # Expect exactly 4 quartile columns on a uniform range
    assert set(out.columns) == {"x_Q1", "x_Q2", "x_Q3", "x_Q4"}
    # Each row has exactly one True
    assert (out.sum(axis=1) == 1).all()
    # Bottom 5 rows are Q1, top 5 rows are Q4
    assert out.iloc[:5]["x_Q1"].all()
    assert out.iloc[-5:]["x_Q4"].all()


def test_discretize_handles_constant_column():
    df = pd.DataFrame({"x": np.ones(10), "y": np.arange(10, dtype=float)})
    out = discretize_quartiles(df, ["x", "y"])
    # Constant column drops out, variable column contributes quartile cols
    assert not any(c.startswith("x_Q") for c in out.columns)
    assert any(c.startswith("y_Q") for c in out.columns)


def test_binarize_outcomes_tertile_threshold():
    # Two chapters, 9 rows each: high flag should be the top ~1/3.
    rows = []
    for chap, rates in [
        ("J00-J99", [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]),
        ("I00-I99", [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]),
    ]:
        for i, p in enumerate(rates):
            rows.append({"CBSAFP": 100 + i, "year": 2020, "code": chap, "prevalence": p})
    df = pd.DataFrame(rows)
    wide = binarize_outcomes(df, ["J00-J99", "I00-I99"], quantile=2 / 3)
    # With 9 values and q=2/3 threshold is at index 6 (0-indexed); rows with
    # prevalence >= threshold are flagged. Exactly 3 rows should be high per
    # chapter.
    assert wide["J00-J99_high"].sum() == 3
    assert wide["I00-I99_high"].sum() == 3
    # The highest-prevalence CBSA should be flagged in both chapters (same row
    # index across chapters here since we constructed them identically).
    assert wide.loc[wide["CBSAFP"] == 108, "J00-J99_high"].item()
    assert wide.loc[wide["CBSAFP"] == 108, "I00-I99_high"].item()
