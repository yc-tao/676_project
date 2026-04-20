"""Shared synthetic fixtures for RQ1 DLNM tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def synthetic_monthly_env():
    """14 fake CBSAs, 7 years x 12 months each, 2 seasonal exposures (temp, ozone)."""
    rng = np.random.default_rng(0)
    cbsas = [10000 + i for i in range(14)]
    rows = []
    for c in cbsas:
        for year in range(2016, 2023):
            for month in range(1, 13):
                seasonal = np.sin(2 * np.pi * month / 12)
                rows.append({
                    "CBSAFP": c,
                    "year": year,
                    "month": month,
                    "temp": 10 + 15 * seasonal + rng.normal(0, 1),
                    "ozone": 300 + 20 * seasonal + rng.normal(0, 5),
                })
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def synthetic_outcomes():
    """Yearly outcome counts for 14 CBSAs x 7 years, constant rate (no env coupling)."""
    rng = np.random.default_rng(1)
    cbsas = [10000 + i for i in range(14)]
    rows = []
    for c in cbsas:
        for year in range(2016, 2023):
            base = 0.1
            rows.append({
                "CBSAFP": c,
                "year": year,
                "code": "J00-J99",
                "count": int(rng.poisson(1000 * base)),
                "count_patient": 10000,
            })
    return pd.DataFrame(rows)
