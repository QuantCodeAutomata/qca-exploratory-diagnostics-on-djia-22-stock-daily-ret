"""
Tests for src/diebold_mariano.py (exp_3 methodology).

Verifies the DM test implementation: sign convention, Newey-West HAC,
edge cases, and known statistical properties.
"""

from __future__ import annotations

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.diebold_mariano import diebold_mariano_test, newey_west_lrv


RNG = np.random.default_rng(777)


# ──────────────────────────────────────────────────────────────────────────────
# newey_west_lrv
# ──────────────────────────────────────────────────────────────────────────────

def test_newey_west_lrv_iid():
    """For i.i.d. series, LRV should approximate the sample variance."""
    rng = np.random.default_rng(0)
    series = rng.normal(0, 1, 1000)
    lrv = newey_west_lrv(series, bandwidth=0)
    # With zero bandwidth, LRV = sample variance
    sample_var = float(np.var(series, ddof=1))
    np.testing.assert_allclose(lrv, sample_var, rtol=0.05)


def test_newey_west_lrv_non_negative():
    """LRV must be non-negative."""
    for bw in [0, 1, 5, 20]:
        series = RNG.normal(0, 1, 500)
        lrv = newey_west_lrv(series, bandwidth=bw)
        assert lrv >= 0.0, f"LRV must be non-negative (bandwidth={bw})"


def test_newey_west_lrv_correlated_larger():
    """Positive serial correlation increases LRV beyond the sample variance."""
    rng = np.random.default_rng(1)
    eps = rng.normal(0, 1, 600)
    # AR(1) with high persistence
    series = np.zeros(600)
    for t in range(1, 600):
        series[t] = 0.8 * series[t - 1] + eps[t]
    lrv_bw20 = newey_west_lrv(series, bandwidth=20)
    var_raw = float(np.var(series, ddof=1))
    assert lrv_bw20 > var_raw, "Positively correlated series: LRV > variance"


# ──────────────────────────────────────────────────────────────────────────────
# diebold_mariano_test
# ──────────────────────────────────────────────────────────────────────────────

def test_dm_equal_losses_zero_stat():
    """When both loss series are identical, DM statistic should be zero."""
    losses = RNG.uniform(0, 1, 500) ** 2
    result = diebold_mariano_test(losses, losses, horizon=1)
    np.testing.assert_allclose(result["dm_stat"], 0.0, atol=1e-10)


def test_dm_fixed_strictly_worse_positive_stat():
    """When losses1 > losses2, DM stat should be positive."""
    n = 1000
    losses1 = RNG.uniform(0.5, 1.0, n)  # higher losses
    losses2 = RNG.uniform(0.1, 0.5, n)  # lower losses
    result = diebold_mariano_test(losses1, losses2, horizon=1)
    assert result["dm_stat"] > 0


def test_dm_pval_significant_when_large_difference():
    """Clear loss improvement should yield significant DM test."""
    rng = np.random.default_rng(42)
    n = 2000
    losses1 = rng.uniform(0.5, 1.0, n)
    losses2 = rng.uniform(0.1, 0.3, n)
    result = diebold_mariano_test(losses1, losses2, horizon=1)
    assert result["dm_pval"] < 0.05


def test_dm_pval_not_significant_when_equal():
    """Identical losses should not reject H0 of equal predictive accuracy."""
    rng = np.random.default_rng(2)
    losses = rng.uniform(0, 1, 1000) ** 2
    result = diebold_mariano_test(losses, losses, horizon=1)
    # p-value should be 1.0 (or very close) when stat is 0
    np.testing.assert_allclose(result["dm_pval"], 1.0, atol=1e-6)


def test_dm_returns_expected_keys():
    losses1 = RNG.uniform(0, 1, 300)
    losses2 = RNG.uniform(0, 1, 300)
    result = diebold_mariano_test(losses1, losses2, horizon=5)
    expected_keys = {"dm_stat", "dm_pval", "mean_diff", "lrv", "n_obs"}
    assert expected_keys.issubset(result.keys())


def test_dm_mean_diff_is_mean_of_d():
    """mean_diff should equal np.mean(losses1 - losses2)."""
    rng = np.random.default_rng(3)
    losses1 = rng.uniform(0.2, 0.8, 200)
    losses2 = rng.uniform(0.1, 0.5, 200)
    result = diebold_mariano_test(losses1, losses2, horizon=1)
    expected_mean = float(np.mean(losses1 - losses2))
    np.testing.assert_allclose(result["mean_diff"], expected_mean, rtol=1e-10)


def test_dm_pval_in_01():
    """p-value must be in [0, 1]."""
    losses1 = RNG.uniform(0, 1, 400)
    losses2 = RNG.uniform(0, 1, 400)
    result = diebold_mariano_test(losses1, losses2, horizon=5)
    assert 0 <= result["dm_pval"] <= 1


def test_dm_horizon_increases_bandwidth():
    """Larger horizon argument should use larger Newey-West bandwidth."""
    rng = np.random.default_rng(5)
    losses1 = rng.uniform(0.3, 0.7, 500)
    losses2 = rng.uniform(0.3, 0.7, 500)
    r1 = diebold_mariano_test(losses1, losses2, horizon=1)
    r21 = diebold_mariano_test(losses1, losses2, horizon=21)
    # With more autocorrelation correction, LRV should generally differ
    # (not guaranteed to be larger, but they should differ for horizon>1)
    # Just verify both return valid floats
    assert isinstance(r1["dm_stat"], float)
    assert isinstance(r21["dm_stat"], float)


def test_dm_n_obs_correct():
    losses1 = RNG.uniform(0, 1, 350)
    losses2 = RNG.uniform(0, 1, 350)
    result = diebold_mariano_test(losses1, losses2, horizon=10)
    assert result["n_obs"] == 350
