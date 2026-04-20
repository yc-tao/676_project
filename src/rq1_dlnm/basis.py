"""Natural spline and cross-basis for the RQ1 DLNM."""
from __future__ import annotations

import torch


def natural_spline(x: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
    """Natural cubic spline basis, df = len(knots), no intercept column.

    Follows Wood (2006) §5.3: columns are [x - mean(x), h_1(x), ..., h_{k-2}(x)]
    where h_j(x) = d_j(x) - d_{k-1}(x) and
    d_j(x) = ((x - t_j)_+^3 - (x - t_k)_+^3) / (t_k - t_j).
    The last internal knot t_k is the anchor; we emit (k-2) wiggle columns
    plus the linear column, giving df = k - 1 for k internal knots counting
    the anchor. To match df=len(knots) we use knots[:-1] as the "wiggle" set
    and knots[-1] as the anchor.
    """
    x = x.to(torch.float64)
    knots = knots.to(torch.float64)
    k = knots.numel()
    assert k >= 2, "need at least 2 knots"

    def d(x, t_j, t_k):
        num = torch.clamp(x - t_j, min=0.0).pow(3) - torch.clamp(x - t_k, min=0.0).pow(3)
        return num / (t_k - t_j)

    t_k = knots[-1]
    last_d = d(x, knots[-2], t_k)  # d_{k-1}(x)
    linear = x - x.mean()
    cols = [linear]
    for j in range(k - 2):
        cols.append(d(x, knots[j], t_k) - last_d)
    # Ensure we return exactly `k` - 1 wiggle cols + 1 linear = k columns when
    # k >= 2. For df=3 with 3 knots -> 3 columns total.
    B = torch.stack(cols, dim=1)
    # Pad to df = k if underfilled (happens when k == 2 -> only linear col).
    if B.shape[1] < k:
        B = torch.cat([B, torch.zeros(x.shape[0], k - B.shape[1], dtype=B.dtype)], dim=1)
    return B.to(torch.float32)


def cross_basis(
    L: torch.Tensor,
    *,
    var_knots: torch.Tensor,
    lag_knots: torch.Tensor,
) -> torch.Tensor:
    """Cross-basis X with shape (n_rows, df_var * df_lag).

    Builds the variable basis on each lag column and the lag basis on the
    lag indices, then column-wise tensor-products and sums over lags per
    standard DLNM construction: X[i, v*D_lag + l] = sum_k Bv(L[i,k])[v] * Bl(k)[l].
    """
    n, nlag = L.shape
    df_var = var_knots.numel()
    df_lag = lag_knots.numel()
    lag_idx = torch.arange(nlag, dtype=torch.float32)
    Bl = natural_spline(lag_idx, lag_knots)  # (nlag, df_lag)
    # Bv_per_lag[i, k, v] = variable basis for L[i, k]
    Bv_per_lag = natural_spline(L.reshape(-1), var_knots).reshape(n, nlag, df_var)
    # einsum: sum over k of Bv[i,k,v] * Bl[k,l] -> (n, v, l)
    cb = torch.einsum("nkv,kl->nvl", Bv_per_lag, Bl)
    return cb.reshape(n, df_var * df_lag)
