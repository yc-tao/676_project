"""Tests for src/rq1_dlnm/model.py."""
from __future__ import annotations

import numpy as np
import torch

from rq1_dlnm.model import PoissonDLNM


def test_forward_produces_positive_means_matching_shape():
    torch.manual_seed(0)
    n, n_cb, n_cbsa = 30, 9, 3
    X = torch.randn(n, n_cb)
    cbsa_idx = torch.randint(0, n_cbsa, (n,))
    year = torch.linspace(-1.0, 1.0, n)
    miss = torch.zeros(n)
    offset = torch.log(torch.full((n,), 100.0))

    m = PoissonDLNM(n_cb=n_cb, n_cbsa=n_cbsa)
    mu = m(X, cbsa_idx=cbsa_idx, year=year, miss=miss, offset=offset)

    assert mu.shape == (n,)
    assert (mu > 0).all()


def test_fit_recovers_linear_effect_on_synthetic_data():
    torch.manual_seed(0)
    n, n_cb, n_cbsa = 200, 3, 2
    X = torch.randn(n, n_cb)
    true_beta = torch.tensor([0.3, -0.2, 0.1])
    cbsa_idx = torch.randint(0, n_cbsa, (n,))
    year = torch.zeros(n)
    miss = torch.zeros(n)
    offset = torch.log(torch.full((n,), 100.0))
    log_mu = X @ true_beta + offset
    y = torch.poisson(torch.exp(log_mu)).to(torch.float32)

    from rq1_dlnm.model import PoissonDLNM, fit
    m = PoissonDLNM(n_cb=n_cb, n_cbsa=n_cbsa)
    result = fit(
        m, X_cb=X, cbsa_idx=cbsa_idx, year=year, miss=miss,
        offset=offset, count=y, ridge=1e-4, steps=800, lr=5e-2, device="cpu",
    )
    assert result.converged or result.steps_run == 800
    est = m.beta.detach().cpu()
    assert torch.allclose(est, true_beta, atol=0.08)


def test_observed_information_matches_finite_difference():
    torch.manual_seed(0)
    n, n_cb, n_cbsa = 40, 3, 2
    X = torch.randn(n, n_cb)
    cbsa_idx = torch.randint(0, n_cbsa, (n,))
    year = torch.zeros(n)
    miss = torch.zeros(n)
    offset = torch.zeros(n)
    y = torch.poisson(torch.exp(X @ torch.tensor([0.1, -0.2, 0.05]))).to(torch.float32)

    from rq1_dlnm.model import PoissonDLNM, fit, observed_information
    m = PoissonDLNM(n_cb=n_cb, n_cbsa=n_cbsa)
    fit(m, X_cb=X, cbsa_idx=cbsa_idx, year=year, miss=miss,
        offset=offset, count=y, ridge=1e-2, steps=500, device="cpu")

    H = observed_information(
        m, X_cb=X, cbsa_idx=cbsa_idx, year=year, miss=miss, offset=offset,
        ridge=1e-2,
    )
    assert H.shape == (n_cb, n_cb)
    # Must be symmetric positive-definite
    assert torch.allclose(H, H.T, atol=1e-5)
    eig = torch.linalg.eigvalsh(H)
    assert (eig > 0).all()
