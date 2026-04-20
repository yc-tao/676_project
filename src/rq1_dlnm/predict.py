"""RR surfaces and Wald contrasts for the fitted PoissonDLNM."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import norm

from rq1_dlnm.basis import cross_basis


@dataclass
class Contrast:
    log_rr: float
    se: float
    z: float
    p: float


def _row_at_exposure(
    v: float, *, var_knots: torch.Tensor, lag_knots: torch.Tensor, nlag: int
) -> torch.Tensor:
    """Cross-basis row for a constant exposure v held at every lag."""
    L = torch.full((1, nlag), float(v))
    return cross_basis(L, var_knots=var_knots, lag_knots=lag_knots).squeeze(0)


def cumulative_rr_contrast(
    *,
    beta: torch.Tensor,
    cov: torch.Tensor,
    v_low: float,
    v_high: float,
    var_knots: torch.Tensor,
    lag_knots: torch.Tensor,
    nlag: int,
) -> Contrast:
    """Two-sided Wald contrast for log-RR(v_high vs v_low) cumulated across lags."""
    c_high = _row_at_exposure(v_high, var_knots=var_knots, lag_knots=lag_knots, nlag=nlag)
    c_low = _row_at_exposure(v_low, var_knots=var_knots, lag_knots=lag_knots, nlag=nlag)
    c = (c_high - c_low).to(torch.float64)
    b = beta.to(torch.float64)
    V = cov.to(torch.float64)
    log_rr = float(c @ b)
    var = float(c @ V @ c)
    se = float(np.sqrt(max(var, 0.0)))
    z = log_rr / se if se > 0 else 0.0
    p = 2 * (1 - norm.cdf(abs(z))) if se > 0 else 1.0
    return Contrast(log_rr=log_rr, se=se, z=z, p=float(p))
