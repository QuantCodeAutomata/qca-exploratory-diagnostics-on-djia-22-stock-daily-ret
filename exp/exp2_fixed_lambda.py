"""
Exp 2 – Fixed-lambda EWMA covariance forecasting (T=5, 10, 21).

Estimates EWMA covariance forecasts across a 99-point lambda grid (0.01–0.99)
for the full evaluation period 2000-01-03 to 2020-12-31 and identifies the
MSE-minimizing lambda for each horizon.

Usage
-----
    python exp/exp2_fixed_lambda.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.download import load_returns, TICKERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EVAL_START = "2000-01-03"
EVAL_END = "2020-12-31"
HORIZONS = [5, 10, 21]
LAMBDAS = np.round(np.arange(0.01, 1.00, 0.01), 2)  # 99 values: 0.01 … 0.99
N_ASSETS = 22
N_ENTRIES = N_ASSETS * (N_ASSETS + 1) // 2  # 253
TRIL_R, TRIL_C = np.tril_indices(N_ASSETS)

# RiskMetrics benchmark lambda
RISKMETRICS_LAMBDA = 0.97


def ewma_lower_triangle_series(
    returns: np.ndarray,
    lam: float,
) -> np.ndarray:
    """
    Compute lower-triangle EWMA covariance for every date.

    Parameters
    ----------
    returns : np.ndarray, shape (n_obs, N)
    lam : float

    Returns
    -------
    np.ndarray, shape (n_obs, N*(N+1)//2)
        sigma_tril[t] = lower triangle of Σ̂_{t+1}(λ).
    """
    n_obs, n_assets = returns.shape
    tril_r, tril_c = np.tril_indices(n_assets)
    n_entries = len(tril_r)
    one_minus_lam = 1.0 - lam

    sigma_tril = np.empty((n_obs, n_entries))

    # Initialize with outer product of first return
    r0 = returns[0]
    S = np.outer(r0, r0)
    sigma_tril[0] = S[tril_r, tril_c]

    for t in range(1, n_obs):
        rt = returns[t]
        S = one_minus_lam * np.outer(rt, rt) + lam * S
        sigma_tril[t] = S[tril_r, tril_c]

    return sigma_tril


def compute_realized_tril(
    returns: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """
    Compute lower-triangle of realized T-day covariance for every date.

    C_{t,T}[i,j] = \\sum_{k=0}^{T-1} r_{i,t-k} * r_{j,t-k}

    Parameters
    ----------
    returns : np.ndarray, shape (n_obs, N)
    horizon : int

    Returns
    -------
    np.ndarray, shape (n_obs, N*(N+1)//2)
        NaN for t < horizon - 1.
    """
    n_obs, n_assets = returns.shape
    tril_r, tril_c = np.tril_indices(n_assets)
    n_entries = len(tril_r)
    realized = np.full((n_obs, n_entries), np.nan)

    for t in range(horizon - 1, n_obs):
        window = returns[t - horizon + 1 : t + 1]  # (T, N)
        C = window.T @ window
        realized[t] = C[tril_r, tril_c]

    return realized


def compute_loss_surface(
    returns: np.ndarray,
    eval_indices: np.ndarray,
    realized_tril_all: dict[int, np.ndarray],
    horizons: list[int] = HORIZONS,
    lambdas: np.ndarray = LAMBDAS,
) -> dict[int, np.ndarray]:
    """
    Compute daily squared-error loss surface for all horizons and lambdas.

    Parameters
    ----------
    returns : np.ndarray, shape (n_total, N)
        Full daily return panel.
    eval_indices : np.ndarray
        Integer indices of evaluation dates in *returns*.
    realized_tril_all : dict
        Pre-computed lower-triangle realized covariances; key = horizon.
    horizons : list of int
    lambdas : np.ndarray, shape (99,)

    Returns
    -------
    dict mapping horizon → np.ndarray of shape (n_eval, 99)
    """
    n_lambdas = len(lambdas)
    n_eval = len(eval_indices)

    losses: dict[int, np.ndarray] = {T: np.full((n_eval, n_lambdas), np.nan) for T in horizons}

    for k, lam in enumerate(lambdas):
        logger.info("Computing EWMA for lambda=%.2f …", lam)
        sigma_tril = ewma_lower_triangle_series(returns, lam)
        # sigma_tril[t] = forecast made at end of day t (for day t+1)

        for T in horizons:
            realized_tril = realized_tril_all[T]
            for i, t in enumerate(eval_indices):
                # Forecast index: sigma made with info up to day t-T
                fc_idx = t - T
                if fc_idx < 0:
                    continue
                real_vec = realized_tril[t]
                if np.any(np.isnan(real_vec)):
                    continue
                # T-day forecast = T * one-day EWMA
                fcast_vec = T * sigma_tril[fc_idx]
                diff = fcast_vec - real_vec
                losses[T][i, k] = np.dot(diff, diff)

    return losses


def run_experiment(
    returns_full: pd.DataFrame,
) -> dict:
    """
    Run the full fixed-lambda EWMA experiment.

    Parameters
    ----------
    returns_full : pd.DataFrame
        Full daily log returns 1994-2020.

    Returns
    -------
    dict with keys 'mse', 'losses', 'lambda_star', 'mse_star'.
    """
    returns_arr = returns_full.values.astype(np.float64)
    dates = returns_full.index

    # Evaluation date indices
    eval_mask = (dates >= EVAL_START) & (dates <= EVAL_END)
    eval_indices = np.where(eval_mask)[0]
    eval_dates = dates[eval_indices]
    logger.info(
        "Evaluation period: %s – %s (%d dates)",
        eval_dates[0].date(), eval_dates[-1].date(), len(eval_indices),
    )

    # Pre-compute realized covariances for each horizon
    logger.info("Pre-computing realized T-day covariances …")
    realized_tril_all: dict[int, np.ndarray] = {}
    for T in HORIZONS:
        realized_tril_all[T] = compute_realized_tril(returns_arr, T)
        logger.info("Realized covariance T=%d computed.", T)

    # Compute loss surface
    losses = compute_loss_surface(
        returns_arr, eval_indices, realized_tril_all, HORIZONS, LAMBDAS
    )

    # MSE and optimal lambda per horizon
    mse: dict[int, np.ndarray] = {}
    lambda_star: dict[int, float] = {}
    mse_star: dict[int, float] = {}

    for T in HORIZONS:
        mse[T] = np.nanmean(losses[T], axis=0)
        idx = int(np.argmin(mse[T]))
        lambda_star[T] = float(LAMBDAS[idx])
        mse_star[T] = float(mse[T][idx])
        logger.info(
            "Horizon T=%2d | λ*=%.2f | MSE*=%.6f",
            T, lambda_star[T], mse_star[T],
        )

    # RiskMetrics comparison for T=21
    rm_idx = np.where(np.isclose(LAMBDAS, RISKMETRICS_LAMBDA))[0][0]
    mse_rm = float(mse[21][rm_idx])
    mse_opt21 = mse_star[21]
    logger.info(
        "T=21: MSE(λ=0.97)=%.6f, MSE(λ*=%.2f)=%.6f, Δ=%.6f",
        mse_rm, lambda_star[21], mse_opt21, mse_rm - mse_opt21,
    )

    return {
        "mse": mse,
        "losses": losses,
        "lambda_star": lambda_star,
        "mse_star": mse_star,
        "eval_dates": eval_dates,
        "eval_indices": eval_indices,
        "n_eval": len(eval_indices),
        "mse_rm_21": mse_rm,
    }


def save_results(result: dict) -> None:
    """Save MSE tables and plots to the results directory."""
    mse = result["mse"]
    lambda_star = result["lambda_star"]
    mse_star = result["mse_star"]

    # ── MSE table (all lambdas, all horizons) ─────────────────────────────────
    mse_df = pd.DataFrame(
        {f"MSE_T{T}": mse[T] for T in HORIZONS},
        index=LAMBDAS,
    )
    mse_df.index.name = "lambda"
    mse_df.to_csv(RESULTS_DIR / "exp2_mse_table.csv")
    logger.info("MSE table saved.")

    # ── Optimal lambda summary ────────────────────────────────────────────────
    opt_df = pd.DataFrame(
        {
            "horizon": HORIZONS,
            "lambda_star": [lambda_star[T] for T in HORIZONS],
            "mse_star": [mse_star[T] for T in HORIZONS],
        }
    )
    opt_df.to_csv(RESULTS_DIR / "exp2_optimal_lambdas.csv", index=False)
    logger.info("Optimal lambda table saved.")

    # ── MSE vs lambda plots ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    fig.suptitle("MSE vs Lambda – Fixed-Lambda EWMA (2000–2020)", fontsize=13)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for ax, T, color in zip(axes, HORIZONS, colors):
        ax.plot(LAMBDAS, mse[T], color=color, linewidth=1.8)
        ax.axvline(lambda_star[T], color="red", linestyle="--", linewidth=1.2,
                   label=f"λ*={lambda_star[T]:.2f}")
        if T == 21:
            rm_idx = np.where(np.isclose(LAMBDAS, RISKMETRICS_LAMBDA))[0][0]
            ax.axvline(RISKMETRICS_LAMBDA, color="purple", linestyle=":",
                       linewidth=1.2, label=f"RiskMetrics (λ={RISKMETRICS_LAMBDA})")
        ax.set_title(f"T = {T}")
        ax.set_xlabel("Lambda")
        ax.set_ylabel("MSE")
        ax.legend(fontsize=9)
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_mse_vs_lambda.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("MSE plot saved.")

    # ── Print concise result ──────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("Exp 2 Results — Fixed-Lambda EWMA")
    print("=" * 55)
    for T in HORIZONS:
        print(f"T={T:2d}: λ*={lambda_star[T]:.2f}  MSE={mse_star[T]:.6f}")
    print(
        f"T=21: MSE(λ=0.97)={result['mse_rm_21']:.6f}  "
        f"Δ={result['mse_rm_21'] - mse_star[21]:.6f}"
    )
    print(f"Evaluation dates: {result['n_eval']}")
    print("=" * 55)


def main() -> dict:
    """Entry point for exp_2."""
    logger.info("Loading returns for exp_2 …")
    returns_full = load_returns(
        tickers=TICKERS,
        start="1994-01-03",
        end="2020-12-31",
    )
    logger.info("Returns shape: %s", returns_full.shape)

    result = run_experiment(returns_full)
    save_results(result)
    return result


if __name__ == "__main__":
    main()
