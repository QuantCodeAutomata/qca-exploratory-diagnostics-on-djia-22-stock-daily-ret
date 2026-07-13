"""
Diebold-Mariano (DM) test for equal predictive accuracy.

The test is implemented with a Newey-West HAC long-run variance (LRV)
estimator to account for serial correlation in loss differentials caused by
overlapping T-day realized covariance windows.

Bandwidth choice: q = T - 1, motivated by the T-day overlap structure, with
Bartlett kernel weights w_k = 1 - k/(q+1).

Sign convention (consistent with exp_3):
    d_t = loss_fixed_t - loss_adaptive_t
A positive DM statistic indicates the adaptive method yields lower losses.

# Custom — Context7 found no library equivalent for overlap-aware DM test
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def newey_west_lrv(
    series: np.ndarray,
    bandwidth: int,
) -> float:
    """
    Estimate long-run variance of a stationary series using the Newey-West
    Bartlett-kernel HAC estimator.

    LRV = γ_0 + 2 ∑_{k=1}^{q} w_k γ_k,  w_k = 1 - k/(q+1)

    Parameters
    ----------
    series : np.ndarray
        1-D array (the loss differential d_t).
    bandwidth : int
        Truncation lag q.  For T-day overlapping realized covariance,
        use q = T - 1.

    Returns
    -------
    float
        Estimated long-run variance.
    """
    n = len(series)
    demeaned = series - series.mean()

    gamma_0 = float(np.dot(demeaned, demeaned)) / n
    lrv = gamma_0

    for k in range(1, bandwidth + 1):
        gamma_k = float(np.dot(demeaned[k:], demeaned[:-k])) / n
        bartlett_weight = 1.0 - k / (bandwidth + 1)
        lrv += 2.0 * bartlett_weight * gamma_k

    # LRV should be positive; clamp to a small positive if numerical issue
    return max(lrv, 1e-30)


def diebold_mariano_test(
    loss_fixed: np.ndarray,
    loss_adaptive: np.ndarray,
    horizon: int,
) -> dict[str, float]:
    """
    Diebold-Mariano test of equal predictive accuracy.

    H0: E[d_t] = 0, where d_t = loss_fixed_t − loss_adaptive_t.
    Positive DM → adaptive forecasts are better (lower losses).

    Parameters
    ----------
    loss_fixed : np.ndarray
        Daily squared-error losses for the fixed-lambda model.
    loss_adaptive : np.ndarray
        Daily squared-error losses for the adaptive-lambda model.
    horizon : int
        Forecast horizon T, used to set Newey-West bandwidth q = T - 1.

    Returns
    -------
    dict with keys 'dm_stat', 'dm_pval', 'mean_diff', 'n_obs',
    'bandwidth', 'lrv'.
    """
    assert loss_fixed.shape == loss_adaptive.shape, "Loss arrays must have the same length."

    # Drop pairs where either loss is NaN
    valid = np.isfinite(loss_fixed) & np.isfinite(loss_adaptive)
    loss_fixed = loss_fixed[valid]
    loss_adaptive = loss_adaptive[valid]

    d = loss_fixed - loss_adaptive  # positive → fixed is worse
    n = len(d)
    bandwidth = max(horizon - 1, 0)

    lrv = newey_west_lrv(d, bandwidth)
    std_err = np.sqrt(lrv / n)
    dm_stat = float(d.mean() / std_err)
    dm_pval = float(2.0 * (1.0 - stats.norm.cdf(abs(dm_stat))))

    return {
        "dm_stat": dm_stat,
        "dm_pval": dm_pval,
        "mean_diff": float(d.mean()),
        "n_obs": n,
        "bandwidth": bandwidth,
        "lrv": lrv,
    }
