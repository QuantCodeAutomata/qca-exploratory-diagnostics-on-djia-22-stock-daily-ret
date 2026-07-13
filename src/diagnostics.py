"""
Statistical diagnostics for daily log return series.

Implements the exploratory analysis from exp_1:
  - Shapiro-Wilk normality test          (scipy.stats.shapiro)
  - Engle ARCH LM test (20 lags)         (statsmodels het_arch)
  - Augmented Dickey-Fuller stationarity (statsmodels adfuller)
  - Supplemental IBM diagnostics         (ARCH LM 1 lag, Ljung-Box on r²)

# Using statsmodels het_arch, adfuller — Context7 confirmed
# Using scipy.stats.shapiro   — Context7 confirmed
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.tsa.stattools import adfuller


# ──────────────────────────────────────────────────────────────────────────────
# Individual test wrappers
# ──────────────────────────────────────────────────────────────────────────────

def shapiro_wilk(series: np.ndarray) -> dict[str, float]:
    """
    Run Shapiro-Wilk normality test.

    Parameters
    ----------
    series : np.ndarray
        1-D array of return observations.

    Returns
    -------
    dict with keys 'sw_stat' and 'sw_pval'.
    """
    stat, pval = stats.shapiro(series)
    return {"sw_stat": float(stat), "sw_pval": float(pval)}


def arch_lm_test(series: np.ndarray, nlags: int = 20) -> dict[str, float]:
    """
    Engle's ARCH LM test for conditional heteroskedasticity.

    # Using statsmodels het_arch — Context7 confirmed

    Parameters
    ----------
    series : np.ndarray
        1-D array of return observations.
    nlags : int
        Number of lags for the LM regression (default 20 per paper).

    Returns
    -------
    dict with keys 'arch_lm_stat', 'arch_lm_pval', 'arch_f_stat', 'arch_f_pval'.
    """
    lm_stat, lm_pval, f_stat, f_pval = het_arch(series, nlags=nlags)
    return {
        "arch_lm_stat": float(lm_stat),
        "arch_lm_pval": float(lm_pval),
        "arch_f_stat": float(f_stat),
        "arch_f_pval": float(f_pval),
    }


def adf_test(series: np.ndarray, autolag: str = "AIC") -> dict[str, object]:
    """
    Augmented Dickey-Fuller unit-root test.

    # Using statsmodels adfuller — Context7 confirmed

    Parameters
    ----------
    series : np.ndarray
        1-D array of return observations.
    autolag : str
        Lag selection criterion passed to adfuller ('AIC', 'BIC', 't-stat',
        or None for fixed lags).  Default is 'AIC', consistent with common
        practice for return series.  The paper does not fully specify this;
        we document 'AIC' for reproducibility.

    Returns
    -------
    dict with keys 'adf_stat', 'adf_pval', 'adf_lags', 'adf_nobs',
    'adf_crit_1pct', 'adf_crit_5pct', 'adf_crit_10pct'.
    """
    result = adfuller(series, autolag=autolag, regression="c")
    adf_stat, pval, used_lags, nobs = result[0], result[1], result[2], result[3]
    crit = result[4]
    return {
        "adf_stat": float(adf_stat),
        "adf_pval": float(pval),
        "adf_lags": int(used_lags),
        "adf_nobs": int(nobs),
        "adf_crit_1pct": float(crit["1%"]),
        "adf_crit_5pct": float(crit["5%"]),
        "adf_crit_10pct": float(crit["10%"]),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Supplemental IBM diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def ibm_supplemental_diagnostics(series: np.ndarray) -> dict[str, object]:
    """
    Supplemental diagnostics for IBM as noted in the paper:
    ARCH LM with 1 lag and Ljung-Box on squared returns.

    These are NOT part of the core replication table.

    Parameters
    ----------
    series : np.ndarray
        1-D daily return series for IBM.

    Returns
    -------
    dict with ARCH LM (1 lag) and Ljung-Box results on squared returns.
    """
    arch1 = arch_lm_test(series, nlags=1)

    sq_returns = series ** 2
    lb_result = acorr_ljungbox(sq_returns, lags=[10, 20], return_df=True)
    lb_dict = {
        f"lb_sq_stat_lag{int(lag)}": float(row["lb_stat"])
        for lag, row in lb_result.iterrows()
    }
    lb_pval_dict = {
        f"lb_sq_pval_lag{int(lag)}": float(row["lb_pvalue"])
        for lag, row in lb_result.iterrows()
    }

    return {
        "arch_lm1_stat": arch1["arch_lm_stat"],
        "arch_lm1_pval": arch1["arch_lm_pval"],
        **lb_dict,
        **lb_pval_dict,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Full diagnostic table
# ──────────────────────────────────────────────────────────────────────────────

def run_diagnostics(
    returns: pd.DataFrame,
    arch_nlags: int = 20,
    adf_autolag: str = "AIC",
    ibm_ticker: Optional[str] = "IBM",
) -> pd.DataFrame:
    """
    Run the three core diagnostics on each column of *returns* and assemble
    a summary table with significance markers.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns; columns are asset tickers.
    arch_nlags : int
        Number of lags for ARCH LM test (default 20).
    adf_autolag : str
        Lag selection method for ADF (default 'AIC').
    ibm_ticker : str or None
        Ticker name for supplemental IBM diagnostics.  Pass None to skip.

    Returns
    -------
    pd.DataFrame
        One row per asset; columns include test statistics, p-values, and
        binary significance flags at the 5% level.
    """
    records: list[dict] = []

    for ticker in returns.columns:
        series = returns[ticker].dropna().values.astype(float)

        sw = shapiro_wilk(series)
        arch = arch_lm_test(series, nlags=arch_nlags)
        adf = adf_test(series, autolag=adf_autolag)

        records.append(
            {
                "ticker": ticker,
                # Shapiro-Wilk
                "sw_stat": sw["sw_stat"],
                "sw_pval": sw["sw_pval"],
                "sw_reject_normality": sw["sw_pval"] < 0.05,
                # ARCH LM
                "arch_lm_stat": arch["arch_lm_stat"],
                "arch_lm_pval": arch["arch_lm_pval"],
                "arch_sig": arch["arch_lm_pval"] < 0.05,
                # ADF
                "adf_stat": adf["adf_stat"],
                "adf_pval": adf["adf_pval"],
                "adf_lags": adf["adf_lags"],
                "adf_stationary": adf["adf_pval"] < 0.05,
                "adf_crit_5pct": adf["adf_crit_5pct"],
            }
        )

    table = pd.DataFrame(records).set_index("ticker")

    # Supplemental IBM diagnostics appended separately
    if ibm_ticker and ibm_ticker in returns.columns:
        ibm_series = returns[ibm_ticker].dropna().values.astype(float)
        ibm_extra = ibm_supplemental_diagnostics(ibm_series)
        for k, v in ibm_extra.items():
            table.loc[ibm_ticker, k] = v  # type: ignore[index]

    return table


def summarise_diagnostics(table: pd.DataFrame) -> str:
    """
    Print a human-readable summary of the diagnostic table.

    Parameters
    ----------
    table : pd.DataFrame
        Output of run_diagnostics().

    Returns
    -------
    str
        Formatted summary string.
    """
    n = len(table)
    sw_pct = table["sw_reject_normality"].mean() * 100
    arch_pct = table["arch_sig"].mean() * 100
    adf_pct = table["adf_stationary"].mean() * 100

    lines = [
        "=" * 60,
        "Diagnostic Summary (exp_1 — 22 DJIA stocks, 1994–1999)",
        "=" * 60,
        f"Assets: {n}",
        f"Non-normal (SW p<0.05):     {sw_pct:.0f}% ({int(table['sw_reject_normality'].sum())}/{n})",
        f"ARCH effects (LM-{20} p<0.05): {arch_pct:.0f}% ({int(table['arch_sig'].sum())}/{n})",
        f"Stationary (ADF p<0.05):    {adf_pct:.0f}% ({int(table['adf_stationary'].sum())}/{n})",
        "=" * 60,
    ]
    return "\n".join(lines)
