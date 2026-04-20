"""Ridge-penalized Poisson DLNM in PyTorch."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class PoissonDLNM(nn.Module):
    """log mu_i = alpha + x_i^CB beta + gamma[cbsa_i] + delta*year_i + theta*miss_i + offset_i.

    gamma[0] is pinned to 0 to make the FE identifiable.
    """

    def __init__(self, n_cb: int, n_cbsa: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(n_cb))
        # n_cbsa - 1 free FE params; gamma[0] pinned to 0
        self.gamma_free = nn.Parameter(torch.zeros(max(n_cbsa - 1, 0)))
        self.delta = nn.Parameter(torch.zeros(1))
        self.theta = nn.Parameter(torch.zeros(1))
        self.n_cbsa = n_cbsa

    def _gamma(self) -> torch.Tensor:
        zero = torch.zeros(1, device=self.gamma_free.device, dtype=self.gamma_free.dtype)
        return torch.cat([zero, self.gamma_free], dim=0)

    def forward(
        self,
        X_cb: torch.Tensor,
        *,
        cbsa_idx: torch.Tensor,
        year: torch.Tensor,
        miss: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor:
        gamma = self._gamma()
        log_mu = (
            self.alpha
            + X_cb @ self.beta
            + gamma[cbsa_idx]
            + self.delta * year
            + self.theta * miss
            + offset
        )
        return torch.exp(log_mu)


@dataclass
class FitResult:
    loss: float
    steps_run: int
    converged: bool


def fit(
    model: PoissonDLNM,
    *,
    X_cb: torch.Tensor,
    cbsa_idx: torch.Tensor,
    year: torch.Tensor,
    miss: torch.Tensor,
    offset: torch.Tensor,
    count: torch.Tensor,
    ridge: float = 1e-2,
    steps: int = 500,
    lr: float = 5e-2,
    device: str = "mps",
    patience: int = 20,
    tol: float = 1e-6,
) -> FitResult:
    """Fit the PoissonDLNM by Adam on negative Poisson log-likelihood + ridge on beta."""
    dev = torch.device(device if _device_available(device) else "cpu")
    model.to(dev)
    X = X_cb.to(dev)
    ci = cbsa_idx.to(dev)
    yr = year.to(dev)
    ms = miss.to(dev)
    off = offset.to(dev)
    y = count.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    prev = float("inf")
    stable = 0
    for step in range(1, steps + 1):
        opt.zero_grad()
        mu = model(X, cbsa_idx=ci, year=yr, miss=ms, offset=off)
        nll = (mu - y * torch.log(mu + 1e-12)).sum()
        penalty = ridge * (model.beta ** 2).sum()
        loss = nll + penalty
        loss.backward()
        opt.step()
        cur = float(loss.detach().cpu())
        if abs(prev - cur) < tol:
            stable += 1
            if stable >= patience:
                return FitResult(loss=cur, steps_run=step, converged=True)
        else:
            stable = 0
        prev = cur
    return FitResult(loss=prev, steps_run=steps, converged=False)


def _device_available(device: str) -> bool:
    if device == "mps":
        return torch.backends.mps.is_available()
    if device == "cuda":
        return torch.cuda.is_available()
    return True


def observed_information(
    model: PoissonDLNM,
    *,
    X_cb: torch.Tensor,
    cbsa_idx: torch.Tensor,
    year: torch.Tensor,
    miss: torch.Tensor,
    offset: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    """Observed information for `beta` at the current params: X^T diag(mu) X + 2*ridge*I."""
    model.eval()
    with torch.no_grad():
        dev = X_cb.device
        mu = model(
            X_cb,
            cbsa_idx=cbsa_idx.to(dev),
            year=year.to(dev),
            miss=miss.to(dev),
            offset=offset.to(dev),
        )
        W = torch.diag(mu)
        H = X_cb.T @ W @ X_cb + 2 * ridge * torch.eye(X_cb.shape[1], device=dev)
        # symmetrize against numerical drift
        H = 0.5 * (H + H.T)
    return H.cpu()
