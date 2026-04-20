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
