"""
Tests for EWMA covariance estimation (exp_2 methodology).

Verifies the EWMA recursion, realized covariance computation, and loss
surface properties against analytical and known results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp.exp2_fixed_lambda import (
    ewma_lower_triangle_series,
    compute_realized_tril,
    N_ENTRIES,
    TRIL_R,
    TRIL_C,
    N_ASSETS,
    LAMBDAS,
)


RNG = np.random.default_rng(123)
N_OBS = 200
RETURNS = RNG.normal(0, 0.01, size=(N_OBS, N_ASSETS))


# ──────────────────────────────────────────────────────────────────────────────
# ewma_lower_triangle_series
# ──────────────────────────────────────────────────────────────────────────────

def test_ewma_output_shape():
    """EWMA series must have shape (n_obs, 253)."""
    result = ewma_lower_triangle_series(RETURNS, lam=0.94)
    assert result.shape == (N_OBS, N_ENTRIES)


def test_ewma_diagonal_non_negative():
    """Diagonal entries (variances) must be non-negative."""
    result = ewma_lower_triangle_series(RETURNS, lam=0.94)
    # Diagonal indices in the lower-triangle vector
    diag_mask = TRIL_R == TRIL_C
    assert np.all(result[:, diag_mask] >= 0), "Variances must be non-negative"


def test_ewma_lambda_0_equals_outer_product():
    """With lambda=0, the EWMA should equal the current outer product only."""
    returns_single = RETURNS[:10]
    result = ewma_lower_triangle_series(returns_single, lam=0.0)
    # At time t=1, sigma should equal outer product of r[1]
    r1 = returns_single[1]
    expected = np.outer(r1, r1)[TRIL_R, TRIL_C]
    np.testing.assert_allclose(result[1], expected, rtol=1e-10)


def test_ewma_lambda_1_sigma_does_not_change():
    """With lambda=1, the EWMA freezes at the initial estimate."""
    result = ewma_lower_triangle_series(RETURNS[:10], lam=1.0)
    # All rows after the first should equal the first
    np.testing.assert_array_equal(result[1:], result[:-1])


def test_ewma_different_lambdas_differ():
    """Different lambda values should produce different covariance series."""
    result_low = ewma_lower_triangle_series(RETURNS, lam=0.50)
    result_high = ewma_lower_triangle_series(RETURNS, lam=0.99)
    assert not np.allclose(result_low, result_high)


def test_ewma_lambda_in_01_interval():
    """All lambda values in grid must be in (0, 1)."""
    assert np.all(LAMBDAS > 0) and np.all(LAMBDAS < 1)
    assert len(LAMBDAS) == 99


def test_ewma_recursion_manual():
    """Manually verify two-step EWMA recursion against formula."""
    lam = 0.9
    r = np.array([[0.01, -0.02], [0.03, 0.01]])
    S0 = np.outer(r[0], r[0])
    S1_expected = (1 - lam) * np.outer(r[1], r[1]) + lam * S0
    result = ewma_lower_triangle_series(r, lam)
    tril_r, tril_c = np.tril_indices(2)
    S1_actual = result[1]
    np.testing.assert_allclose(S1_actual, S1_expected[tril_r, tril_c], rtol=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# compute_realized_tril
# ──────────────────────────────────────────────────────────────────────────────

def test_realized_tril_output_shape():
    result = compute_realized_tril(RETURNS, horizon=5)
    assert result.shape == (N_OBS, N_ENTRIES)


def test_realized_tril_nan_for_early_dates():
    """First T-1 rows should be NaN (insufficient history)."""
    T = 5
    result = compute_realized_tril(RETURNS, horizon=T)
    assert np.all(np.isnan(result[: T - 1]))


def test_realized_tril_valid_from_T_minus_1():
    """Row at index T-1 should be fully computed."""
    T = 5
    result = compute_realized_tril(RETURNS, horizon=T)
    assert np.all(~np.isnan(result[T - 1]))


def test_realized_tril_diagonal_non_negative():
    """Diagonal entries (sums of squared returns) must be non-negative."""
    result = compute_realized_tril(RETURNS, horizon=5)
    diag_mask = TRIL_R == TRIL_C
    non_nan_rows = ~np.isnan(result[:, 0])
    assert np.all(result[non_nan_rows][:, diag_mask] >= 0)


def test_realized_tril_manual_verification():
    """Verify realized covariance against direct matrix computation."""
    T = 3
    data = np.array([[0.01, 0.02], [-0.01, 0.03], [0.005, -0.015], [0.02, 0.01]])
    result = compute_realized_tril(data, horizon=T)
    # At t=2 (index): window = data[0:3]
    window = data[:T]
    C_expected = window.T @ window
    tril_r, tril_c = np.tril_indices(2)
    np.testing.assert_allclose(result[T - 1], C_expected[tril_r, tril_c], rtol=1e-12)


def test_realized_tril_horizon_1_equals_outer_product():
    """For T=1, realized covariance should equal the outer product at each date."""
    data = RETURNS[:20]
    result = compute_realized_tril(data, horizon=1)
    for t in range(20):
        expected = np.outer(data[t], data[t])[TRIL_R, TRIL_C]
        np.testing.assert_allclose(result[t], expected, rtol=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# Grid and N_ENTRIES validation
# ──────────────────────────────────────────────────────────────────────────────

def test_n_entries_equals_253():
    """22 assets give 22*23/2 = 253 lower-triangle entries."""
    assert N_ENTRIES == 253


def test_lambda_grid_length():
    assert len(LAMBDAS) == 99


def test_lambda_grid_values():
    expected = np.round(np.arange(0.01, 1.00, 0.01), 2)
    np.testing.assert_allclose(LAMBDAS, expected, rtol=1e-10)
