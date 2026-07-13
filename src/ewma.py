"""
EWMA (Exponentially Weighted Moving Average) covariance estimation.

Implements the one-day recursive update and T-day scaling invariance used
throughout exp_2 and exp_3.

Core formula (paper notation):
    \\hat{\\Sigma}_{t+1}(\\lambda) = (1 - \\lambda) r_t r_t' + \\lambda \\hat{\\Sigma}_t

T-day forecast: \\hat{\\Sigma}_{t, T}(\\lambda) = T · \\hat{\\Sigma}_{t+1}(\\lambda)

Realized T-day covariance (no demeaning, no normalization):
    C_{t,T} = \\sum_{k=0}^{T-1} r_{t-k} r_{t-k}'

Loss metric — sum of squared errors over lower triangle (253 unique entries
for N=22 assets):
    SE_{t,T}(\\lambda) = \\sum_{i \\ge j} (\\hat{\\sigma}_{ij,t,T}(\\lambda) - \\sigma_{ij,t,T})^2

# Custom — Context7 found no library equivalent (paper EWMA recursion)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Lower-triangle indices for a 22×22 matrix (253 unique entries)
_N_ASSETS: int = 22
_TRIL_IDX = np.tril_indices(_N_ASSETS)


def _get_tril_idx(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return cached lower-triangle index arrays for an n×n matrix."""
    return np.tril_indices(n)


# ──────────────────────────────────────────────────────────────────────────────
# EWMA recursion (pure NumPy, vectorised over time for a single lambda)
# ──────────────────────────────────────────────────────────────────────────────

def ewma_covariance_series(
    returns: np.ndarray,
    lam: float,
) -> np.ndarray:
    """
    Compute the one-day-ahead EWMA covariance matrix for every date.

    Initialization: the recursion starts from the sample covariance of the
    very first return in the provided series (a 1-observation outer-product).
    With a long pre-2000 burn-in starting from 1994, initialization effects
    are negligible by the year 2000.

    Parameters
    ----------
    returns : np.ndarray, shape (T, N)
        Panel of daily log returns.
    lam : float
        Decay parameter λ ∈ (0, 1).

    Returns
    -------
    np.ndarray, shape (T, N, N)
        sigma[t] is \\hat{\\Sigma}_{t+1}(\\lambda) — the forecast for day t+1
        formed using information up to and including day t.
    """
    n_obs, n_assets = returns.shape
    sigma = np.empty((n_obs, n_assets, n_assets))

    # Initialize Σ_0 with the outer product of the first observation
    r0 = returns[0]
    sigma[0] = np.outer(r0, r0)

    one_minus_lam = 1.0 - lam
    for t in range(1, n_obs):
        rt = returns[t]
        sigma[t] = one_minus_lam * np.outer(rt, rt) + lam * sigma[t - 1]

    return sigma


def realized_tday_covariance(
    returns: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """
    Compute the realized T-day covariance for every date.

    C_{t,T} = \\sum_{k=0}^{T-1} r_{t-k} r_{t-k}'

    No demeaning, no normalization by T or T-1.

    Parameters
    ----------
    returns : np.ndarray, shape (n_obs, N)
        Panel of daily log returns, already time-sorted.
    horizon : int
        Forecast horizon T ∈ {5, 10, 21}.

    Returns
    -------
    np.ndarray, shape (n_obs, N, N)
        realized[t] is C_{t,T}.  For t < horizon-1, realized[t] = NaN matrix.
    """
    n_obs, n_assets = returns.shape
    realized = np.full((n_obs, n_assets, n_assets), np.nan)

    for t in range(horizon - 1, n_obs):
        window = returns[t - horizon + 1 : t + 1]  # shape (T, N)
        realized[t] = window.T @ window             # N×N sum of outer products

    return realized


def compute_daily_loss_matrix(
    returns: np.ndarray,
    eval_mask: np.ndarray,
    lambdas: np.ndarray,
    horizon: int,
    n_assets: int = 22,
) -> np.ndarray:
    """
    Compute the daily squared-error loss SE_{t,T}(lambda) for every
    evaluation date and every candidate lambda on the discrete grid.

    Forecast timing convention
    --------------------------
    At evaluation date index t (where realized covariance uses returns
    t-horizon+1 … t), the forecast must use EWMA information available
    strictly before the realized window begins.  We use sigma[t - horizon],
    which is formed using information up to day t-horizon (i.e., the one-day
    EWMA update as of day t-horizon).  The T-day forecast is then:
        \\hat{\\Sigma}_{t,T}(\\lambda) = T · sigma[t - horizon]

    This ensures no look-ahead bias: all returns used in the forecast
    (indices ≤ t-horizon) precede the realized window (indices t-horizon+1 … t).

    Parameters
    ----------
    returns : np.ndarray, shape (n_total, n_assets)
        Full daily return panel (1994 onward).
    eval_mask : np.ndarray, shape (n_total,), dtype bool
        True for dates that are in the evaluation period.
    lambdas : np.ndarray, shape (n_lambdas,)
        Candidate decay parameters.
    horizon : int
        Forecast horizon T.
    n_assets : int
        Number of assets (22).

    Returns
    -------
    np.ndarray, shape (n_eval, n_lambdas)
        losses[i, k] = SE_{t_i, T}(lambdas[k]).
    """
    n_total = returns.shape[0]
    tril_r, tril_c = _get_tril_idx(n_assets)
    n_entries = len(tril_r)  # 253 for N=22

    eval_indices = np.where(eval_mask)[0]
    n_eval = len(eval_indices)
    n_lambdas = len(lambdas)

    # Pre-compute realized covariance lower-triangle vectors for all dates
    # shape (n_total, n_entries); NaN for t < horizon-1
    realized_tril = np.full((n_total, n_entries), np.nan)
    for t in range(horizon - 1, n_total):
        window = returns[t - horizon + 1 : t + 1]
        C = window.T @ window
        realized_tril[t] = C[tril_r, tril_c]

    losses = np.full((n_eval, n_lambdas), np.nan)

    for k, lam in enumerate(lambdas):
        # Compute full EWMA sigma series (one-day ahead)
        sigma_series = ewma_covariance_series(returns, lam)
        # sigma_series[t] = Σ̂_{t+1} = forecast made using returns[0..t]
        # For evaluation at date t, forecast index = t - horizon
        # T-day forecast = horizon * sigma_series[t - horizon]

        for i, t in enumerate(eval_indices):
            forecast_idx = t - horizon
            if forecast_idx < 0:
                continue
            if np.any(np.isnan(realized_tril[t])):
                continue
            # T-day forecast lower-triangle
            fcast_tril = horizon * sigma_series[forecast_idx][tril_r, tril_c]
            diff = fcast_tril - realized_tril[t]
            losses[i, k] = np.dot(diff, diff)

    return losses


def compute_mse(losses: np.ndarray) -> np.ndarray:
    """
    Average daily squared-error losses over evaluation dates to get MSE_T(lambda).

    Parameters
    ----------
    losses : np.ndarray, shape (n_eval, n_lambdas)
        Daily squared-error losses from compute_daily_loss_matrix.

    Returns
    -------
    np.ndarray, shape (n_lambdas,)
        Mean squared error for each lambda.
    """
    return np.nanmean(losses, axis=0)


def optimal_lambda(mse: np.ndarray, lambdas: np.ndarray) -> tuple[float, float]:
    """
    Return the lambda that minimizes MSE and its corresponding MSE.

    Parameters
    ----------
    mse : np.ndarray, shape (n_lambdas,)
    lambdas : np.ndarray, shape (n_lambdas,)

    Returns
    -------
    (lambda_star, mse_star) tuple.
    """
    idx = int(np.argmin(mse))
    return float(lambdas[idx]), float(mse[idx])
