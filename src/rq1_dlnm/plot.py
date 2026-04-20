"""Three-panel figure for a fitted exposure/outcome pair."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from rq1_dlnm.predict import (
    cumulative_rr_contrast,
    exposure_lag_surface,
    lag_profile,
)


def three_panel(
    *,
    beta: torch.Tensor,
    cov: torch.Tensor,
    var_knots: torch.Tensor,
    lag_knots: torch.Tensor,
    nlag: int,
    v_values: np.ndarray,
    v_ref: float,
    v_90: float,
    exposure_label: str,
    outcome_label: str,
    out_path: Path,
) -> None:
    """Save a (contour, cumulative-RR, lag-profile) figure to out_path."""
    S = exposure_lag_surface(
        beta=beta, v_grid=v_values, v_ref=v_ref,
        var_knots=var_knots, lag_knots=lag_knots, nlag=nlag,
    )
    cum_log_rr = np.empty_like(v_values, dtype=float)
    cum_se = np.empty_like(v_values, dtype=float)
    for i, v in enumerate(v_values):
        c = cumulative_rr_contrast(
            beta=beta, cov=cov, v_low=v_ref, v_high=float(v),
            var_knots=var_knots, lag_knots=lag_knots, nlag=nlag,
        )
        cum_log_rr[i] = c.log_rr
        cum_se[i] = c.se
    lp = lag_profile(
        beta=beta, v=v_90, v_ref=v_ref,
        var_knots=var_knots, lag_knots=lag_knots, nlag=nlag,
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    c0 = axes[0].contourf(np.arange(nlag), v_values, S, levels=20, cmap="RdBu_r")
    axes[0].set_xlabel("lag (months)")
    axes[0].set_ylabel(exposure_label)
    axes[0].set_title(f"RR surface | {outcome_label}")
    fig.colorbar(c0, ax=axes[0])

    rr = np.exp(cum_log_rr)
    hi = np.exp(cum_log_rr + 1.96 * cum_se)
    lo = np.exp(cum_log_rr - 1.96 * cum_se)
    axes[1].plot(v_values, rr, label="cumulative RR")
    axes[1].fill_between(v_values, lo, hi, alpha=0.2)
    axes[1].axhline(1.0, color="k", lw=0.5)
    axes[1].set_xlabel(exposure_label)
    axes[1].set_ylabel("cumulative RR (vs median)")
    axes[1].set_title("Cumulative RR")

    axes[2].plot(np.arange(nlag), lp)
    axes[2].axhline(1.0, color="k", lw=0.5)
    axes[2].set_xlabel("lag (months)")
    axes[2].set_ylabel("per-lag RR at 90th pct")
    axes[2].set_title("Lag profile")

    fig.suptitle(f"{exposure_label} -> {outcome_label}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
