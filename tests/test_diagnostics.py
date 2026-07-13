"""
Tests for src/diagnostics.py (exp_1 methodology).

Verifies Shapiro-Wilk, ARCH LM, and ADF implementations against known
statistical properties and edge cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.diagnostics import (
    shapiro_wilk,
    arch_lm_test,
    adf_test,
    ibm_supplemental_diagnostics,
    run_diagnostics,
    summarise_diagnostics,
)


# ──────────────────────────────────────────────────────────────────────────────
# shapiro_wilk
# ──────────────────────────────────────────────────────────────────────────────

def test_shapiro_wilk_normal_does_not_reject():
    """Large normal sample should not reject normality at 5%."""
    rng = np.random.default_rng(0)
    series = rng.normal(0, 1, 500)
    result = shapiro_wilk(series)
    assert "sw_stat" in result
    assert "sw_pval" in result
    # SW statistic is in (0, 1]
    assert 0 < result["sw_stat"] <= 1.0
    # p-value is in [0, 1]
    assert 0 <= result["sw_pval"] <= 1.0
    # For a large normal sample, p should be > 0.05 most of the time
    # (not always guaranteed, but with seed=0 and n=500 it holds)
    assert result["sw_pval"] > 0.01, "Normal data should not strongly reject SW"


def test_shapiro_wilk_fat_tailed_rejects():
    """Highly non-normal (Student-t df=2) sample should reject normality."""
    rng = np.random.default_rng(1)
    series = rng.standard_t(df=2, size=300)
    result = shapiro_wilk(series)
    assert result["sw_pval"] < 0.05, "Fat-tailed data should reject normality"


def test_shapiro_wilk_returns_dict_keys():
    rng = np.random.default_rng(2)
    series = rng.normal(0, 0.01, 100)
    result = shapiro_wilk(series)
    assert set(result.keys()) == {"sw_stat", "sw_pval"}


def test_shapiro_wilk_single_observation_edge_case():
    """Single-element series should return a result without error."""
    result = shapiro_wilk(np.array([0.01]))
    assert isinstance(result["sw_stat"], float)


# ──────────────────────────────────────────────────────────────────────────────
# arch_lm_test
# ──────────────────────────────────────────────────────────────────────────────

def test_arch_lm_iid_returns_no_arch():
    """i.i.d. normal returns should NOT show significant ARCH effects."""
    rng = np.random.default_rng(3)
    series = rng.normal(0, 0.01, 500)
    result = arch_lm_test(series, nlags=20)
    assert result["arch_lm_pval"] > 0.01, "i.i.d. returns should not show ARCH"


def test_arch_lm_garch_returns_detects_arch():
    """Simulated GARCH(1,1) series should display significant ARCH effects."""
    rng = np.random.default_rng(4)
    n = 600
    omega, alpha, beta = 1e-6, 0.1, 0.85
    h = np.zeros(n)
    r = np.zeros(n)
    h[0] = omega / (1 - alpha - beta)
    r[0] = np.sqrt(h[0]) * rng.standard_normal()
    for t in range(1, n):
        h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
        r[t] = np.sqrt(h[t]) * rng.standard_normal()

    result = arch_lm_test(r, nlags=20)
    assert result["arch_lm_pval"] < 0.05, "GARCH data should show ARCH effects"


def test_arch_lm_returns_expected_keys():
    rng = np.random.default_rng(5)
    series = rng.normal(0, 0.01, 200)
    result = arch_lm_test(series, nlags=5)
    expected_keys = {"arch_lm_stat", "arch_lm_pval", "arch_f_stat", "arch_f_pval"}
    assert expected_keys.issubset(result.keys())
    # LM stat should be non-negative
    assert result["arch_lm_stat"] >= 0
    # p-values in [0, 1]
    for key in ("arch_lm_pval", "arch_f_pval"):
        assert 0 <= result[key] <= 1


def test_arch_lm_nlags_1():
    """ARCH LM with 1 lag (supplemental IBM diagnostic)."""
    rng = np.random.default_rng(6)
    series = rng.normal(0, 0.01, 300)
    result = arch_lm_test(series, nlags=1)
    assert isinstance(result["arch_lm_stat"], float)


# ──────────────────────────────────────────────────────────────────────────────
# adf_test
# ──────────────────────────────────────────────────────────────────────────────

def test_adf_stationary_series():
    """Return-like i.i.d. normal series should be stationary."""
    rng = np.random.default_rng(7)
    series = rng.normal(0, 0.01, 500)
    result = adf_test(series)
    assert result["adf_pval"] < 0.05, "i.i.d. returns should be stationary"


def test_adf_random_walk_not_stationary():
    """Random walk (unit root) should fail to reject the unit-root H0."""
    rng = np.random.default_rng(8)
    innovations = rng.normal(0, 1, 600)
    random_walk = np.cumsum(innovations)
    result = adf_test(random_walk)
    # For a random walk, ADF p-value should be large (do not reject unit root)
    assert result["adf_pval"] > 0.05, "Random walk should not reject unit root"


def test_adf_returns_expected_keys():
    rng = np.random.default_rng(9)
    series = rng.normal(0, 0.01, 300)
    result = adf_test(series)
    expected_keys = {
        "adf_stat", "adf_pval", "adf_lags", "adf_nobs",
        "adf_crit_1pct", "adf_crit_5pct", "adf_crit_10pct",
    }
    assert expected_keys.issubset(result.keys())
    # ADF statistic for stationary series should be very negative
    assert result["adf_stat"] < 0
    # Lags must be non-negative
    assert result["adf_lags"] >= 0
    # Critical values ordering: 1% < 5% < 10%
    assert result["adf_crit_1pct"] < result["adf_crit_5pct"] < result["adf_crit_10pct"]


# ──────────────────────────────────────────────────────────────────────────────
# ibm_supplemental_diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def test_ibm_supplemental_keys():
    rng = np.random.default_rng(10)
    series = rng.normal(0, 0.01, 400)
    result = ibm_supplemental_diagnostics(series)
    assert "arch_lm1_stat" in result
    assert "arch_lm1_pval" in result
    # Ljung-Box on squared returns at lags 10 and 20
    assert "lb_sq_stat_lag10" in result
    assert "lb_sq_pval_lag10" in result


# ──────────────────────────────────────────────────────────────────────────────
# run_diagnostics (integration)
# ──────────────────────────────────────────────────────────────────────────────

def test_run_diagnostics_output_shape(returns_df):
    """Diagnostic table should have one row per asset."""
    table = run_diagnostics(returns_df, arch_nlags=5, ibm_ticker=None)
    assert len(table) == returns_df.shape[1]


def test_run_diagnostics_correct_columns(returns_df):
    """Diagnostic table should contain all expected column groups."""
    table = run_diagnostics(returns_df, arch_nlags=5, ibm_ticker=None)
    required_cols = [
        "sw_stat", "sw_pval", "sw_reject_normality",
        "arch_lm_stat", "arch_lm_pval", "arch_sig",
        "adf_stat", "adf_pval", "adf_stationary",
    ]
    for col in required_cols:
        assert col in table.columns, f"Missing column: {col}"


def test_run_diagnostics_pvals_in_unit_interval(returns_df):
    """All p-values must be in [0, 1]."""
    table = run_diagnostics(returns_df, arch_nlags=5, ibm_ticker=None)
    for col in ("sw_pval", "arch_lm_pval", "adf_pval"):
        assert (table[col] >= 0).all() and (table[col] <= 1).all(), \
            f"p-values out of [0,1] for column {col}"


def test_run_diagnostics_fat_tailed_rejects_normality(leptokurtic_returns):
    """Fat-tailed returns (Student-t df=3) should mostly reject normality."""
    n_assets_local = leptokurtic_returns.shape[1]
    tickers = [f"A{i:02d}" for i in range(n_assets_local)]
    dates = pd.bdate_range("2000-01-03", periods=len(leptokurtic_returns), freq="B")
    df = pd.DataFrame(leptokurtic_returns, index=dates, columns=tickers)
    table = run_diagnostics(df, arch_nlags=5, ibm_ticker=None)
    # At least 80% should reject normality with fat tails
    assert table["sw_reject_normality"].mean() >= 0.8


def test_summarise_diagnostics_returns_string(returns_df):
    table = run_diagnostics(returns_df, arch_nlags=5, ibm_ticker=None)
    summary = summarise_diagnostics(table)
    assert isinstance(summary, str)
    assert "Diagnostic Summary" in summary


# ──────────────────────────────────────────────────────────────────────────────
# Regression test: known outcome on deterministic data
# ──────────────────────────────────────────────────────────────────────────────

def test_shapiro_wilk_regression_uniform():
    """Uniform(0,1) data should reject normality (known property)."""
    rng = np.random.default_rng(99)
    series = rng.uniform(0, 1, 300)
    result = shapiro_wilk(series)
    assert result["sw_pval"] < 0.05


N_ASSETS = 22  # re-defined locally to avoid fixture naming collision
