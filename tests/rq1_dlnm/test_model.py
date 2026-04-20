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
