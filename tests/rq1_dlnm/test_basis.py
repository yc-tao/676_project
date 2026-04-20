"""Tests for src/rq1_dlnm/basis.py."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from rq1_dlnm import basis as bmod


def test_natural_spline_shape_and_linear_column():
    x = torch.linspace(0.0, 10.0, 50)
    knots = torch.tensor([2.5, 5.0, 7.5])
    B = bmod.natural_spline(x, knots)
    assert B.shape == (50, 3)
    # first column is z = (x - median(knots)) / scale, monotone in x
    corr = torch.corrcoef(torch.stack([B[:, 0], x]))[0, 1]
    assert corr.abs() > 0.99


def test_natural_spline_columns_are_well_scaled_on_knot_range():
    # The basis standardizes against the knot range, so within that range
    # the first column stays in [-1, 1] and higher-power columns stay bounded.
    knots = torch.tensor([2.5, 5.0, 7.5])
    x = torch.linspace(float(knots.min()), float(knots.max()), 40)
    B = bmod.natural_spline(x, knots)
    assert B.shape == (40, 3)
    assert (B[:, 0].abs() <= 1.01).all()       # z in [-1, 1] on knot range
    assert (B[:, 1].abs() <= 1.01).all()       # z^2 in [0, 1] on knot range
    assert (B[:, 2].abs() <= 1.01).all()       # z^3 in [-1, 1] on knot range


def test_cross_basis_shape_and_linear_sanity():
    # lag matrix L: 5 outcome rows x 24 lags
    n, nlag = 5, 24
    L = torch.arange(n * nlag, dtype=torch.float32).reshape(n, nlag)
    var_knots = torch.quantile(L.flatten(), torch.tensor([0.25, 0.5, 0.75]))
    lag_knots = torch.tensor([6.0, 12.0, 18.0])
    X = bmod.cross_basis(L, var_knots=var_knots, lag_knots=lag_knots)
    assert X.shape == (n, 3 * 3)
    # deterministic given same inputs
    X2 = bmod.cross_basis(L, var_knots=var_knots, lag_knots=lag_knots)
    assert torch.equal(X, X2)
