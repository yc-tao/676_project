"""Tests for src/rq1_dlnm/predict.py."""
from __future__ import annotations

import numpy as np
import torch

from rq1_dlnm import basis as bmod
from rq1_dlnm import predict as pmod


def test_cumulative_rr_contrast_is_zero_at_same_exposure():
    var_knots = torch.tensor([1.0, 2.0, 3.0])
    lag_knots = torch.tensor([6.0, 12.0, 18.0])
    nlag = 24
    out = pmod.cumulative_rr_contrast(
        beta=torch.zeros(9),
        cov=torch.eye(9),
        v_low=2.0,
        v_high=2.0,
        var_knots=var_knots,
        lag_knots=lag_knots,
        nlag=nlag,
    )
    assert abs(out.log_rr) < 1e-8
    assert out.se >= 0.0


def test_cumulative_rr_contrast_se_scales_with_cov():
    var_knots = torch.tensor([1.0, 2.0, 3.0])
    lag_knots = torch.tensor([6.0, 12.0, 18.0])
    nlag = 24
    beta = torch.tensor([0.1] * 9)
    cov1 = torch.eye(9)
    cov2 = torch.eye(9) * 4.0  # variance x4 -> SE x2
    o1 = pmod.cumulative_rr_contrast(
        beta=beta, cov=cov1, v_low=1.0, v_high=3.0,
        var_knots=var_knots, lag_knots=lag_knots, nlag=nlag,
    )
    o2 = pmod.cumulative_rr_contrast(
        beta=beta, cov=cov2, v_low=1.0, v_high=3.0,
        var_knots=var_knots, lag_knots=lag_knots, nlag=nlag,
    )
    assert np.isclose(o2.se, 2 * o1.se, atol=1e-6)


def test_exposure_lag_surface_returns_positive_RR_grid():
    var_knots = torch.tensor([1.0, 2.0, 3.0])
    lag_knots = torch.tensor([6.0, 12.0, 18.0])
    nlag = 24
    beta = torch.zeros(9)  # null model -> RR should be ~1 everywhere
    v_grid = np.linspace(0.5, 3.5, 10)
    S = pmod.exposure_lag_surface(
        beta=beta, v_grid=v_grid, v_ref=2.0,
        var_knots=var_knots, lag_knots=lag_knots, nlag=nlag,
    )
    assert S.shape == (10, nlag)
    assert np.allclose(S, 1.0, atol=1e-6)


def test_lag_profile_sums_to_cumulative_rr_log_scale():
    var_knots = torch.tensor([1.0, 2.0, 3.0])
    lag_knots = torch.tensor([6.0, 12.0, 18.0])
    nlag = 24
    beta = torch.tensor([0.05] * 9)
    v_ref, v = 2.0, 3.0

    lp = pmod.lag_profile(
        beta=beta, v=v, v_ref=v_ref,
        var_knots=var_knots, lag_knots=lag_knots, nlag=nlag,
    )
    cum = pmod.cumulative_rr_contrast(
        beta=beta, cov=torch.eye(9) * 0, v_low=v_ref, v_high=v,
        var_knots=var_knots, lag_knots=lag_knots, nlag=nlag,
    )
    assert np.isclose(np.log(lp).sum(), cum.log_rr, atol=1e-5)
