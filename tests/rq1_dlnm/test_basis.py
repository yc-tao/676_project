"""Tests for src/rq1_dlnm/basis.py."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from rq1_dlnm import basis as bmod


def test_natural_spline_shape_and_linear_column():
    x = torch.linspace(0.0, 10.0, 50)
    knots = torch.tensor([2.5, 5.0, 7.5])
    B = bmod.natural_spline(x, knots)
    assert B.shape == (50, 3)
    # first column should be (centered) x itself, so correlation with x ~ 1
    corr = torch.corrcoef(torch.stack([B[:, 0], x]))[0, 1]
    assert corr.abs() > 0.99


def test_natural_spline_is_exactly_linear_outside_boundary_knots():
    # Natural cubic splines are linear beyond the boundary knots.
    # With internal knots at 2.5/5/7.5 and data on [-5, 15], beyond x > 7.5 the
    # spline contribution h_j must be linear in x.
    x = torch.linspace(8.0, 15.0, 20)
    knots = torch.tensor([2.5, 5.0, 7.5])
    B = bmod.natural_spline(x, knots)
    # second differences of each column should be ~0 (linearity check)
    second_diff = B[2:] - 2 * B[1:-1] + B[:-2]
    assert torch.allclose(second_diff, torch.zeros_like(second_diff), atol=1e-4)
