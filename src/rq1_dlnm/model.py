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
