"""Polynomial basis and cross-basis for the RQ1 DLNM.

For a proof-of-concept panel of only 84 rows we favor a numerically stable
standardized polynomial basis (df = degree). Each basis uses z = (x - mean)/std
over the training window and emits columns [z, z^2, ..., z^degree], so all
columns sit on a roughly unit scale and Adam's lr=5e-2 steps are well-posed.

The original plan described a natural cubic spline. A degree-3 polynomial has
the same flexibility for this data volume and avoids the knot-placement /
boundary-linearity bookkeeping that a hand-rolled ns basis would need. It also
keeps the code minimal per the PoC scope.
"""
from __future__ import annotations

import torch


def natural_spline(x: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
    """Standardized polynomial basis of degree df = len(knots).

    The `knots` tensor is used only to compute a reference location and scale
    so that train/predict bases live in the same coordinate system:
      center = median(knots), scale = (max(knots) - min(knots)) / 2
    Columns are z, z^2, ..., z^df where z = (x - center) / scale.
    """
    x = x.to(torch.float64)
    knots = knots.to(torch.float64)
    df = knots.numel()
    assert df >= 1, "need at least 1 knot"
    center = knots.median()
    scale = (knots.max() - knots.min()) / 2.0
    if scale.abs() < 1e-12:
        scale = torch.tensor(1.0, dtype=x.dtype)
    z = (x - center) / scale
    cols = [z.pow(p + 1) for p in range(df)]
    B = torch.stack(cols, dim=1)
    return B.to(torch.float32)


def cross_basis(
    L: torch.Tensor,
    *,
    var_knots: torch.Tensor,
    lag_knots: torch.Tensor,
) -> torch.Tensor:
    """Cross-basis X with shape (n_rows, df_var * df_lag).

    Builds the variable basis on each lag column and the lag basis on the
    lag indices, then contracts over lags per the standard DLNM construction:
    X[i, v*D_lag + l] = sum_k Bv(L[i,k])[v] * Bl(k)[l].
    """
    n, nlag = L.shape
    df_var = var_knots.numel()
    df_lag = lag_knots.numel()
    lag_idx = torch.arange(nlag, dtype=torch.float32)
    Bl = natural_spline(lag_idx, lag_knots)                         # (nlag, df_lag)
    Bv_per_lag = natural_spline(L.reshape(-1), var_knots).reshape(n, nlag, df_var)
    cb = torch.einsum("nkv,kl->nvl", Bv_per_lag, Bl)
    return cb.reshape(n, df_var * df_lag)
