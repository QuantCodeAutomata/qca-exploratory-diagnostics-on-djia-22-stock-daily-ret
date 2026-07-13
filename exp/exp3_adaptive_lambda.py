"""
Exp 3 – Lagged time-varying lambda EWMA covariance forecasting (T=5, 10, 21).

Implements the adaptive EWMA strategy where the decay parameter for each date
is the ex-post optimal lambda from the previous evaluation date.  Compares
against the fixed full-sample optimal lambda from exp_2 using the
Diebold-Mariano test with Newey-West HAC long-run variance.

Usage
-----
    python exp/exp3_adaptive_lambda.py
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
from src.diebold_mariano import diebold_mariano_test
from exp.exp2_fixed_lambda import (
    ewma_lower_triangle_series,
    compute_realized_tril,
    HORIZONS,
    LAMBDAS,
    N_ASSETS,
    N_ENTRIES,
    TRIL_R,
    TRIL_C,
    EVAL_START,
    EVAL_END,
    RESULTS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def compute_full_loss_surface(
    returns_arr: np.ndarray,
    eval_indices: np.ndarray,
    realized_tril_all: dict[int, np.ndarray],
    horizons: list[int] = HORIZONS,
    lambdas: np.ndarray = LAMBDAS,
) -> dict[int, np.ndarray]:
    """
    Compute the daily squared-error loss for every evaluation date, horizon,
    and lambda.

    Parameters
    ----------
    returns_arr : np.ndarray, shape (n_total, N)
    eval_indices : np.ndarray
    realized_tril_all : dict  horizon → np.ndarray (n_total, 253)
    horizons : list of int
    lambdas : np.ndarray, shape (n_lambdas,)

    Returns
    -------
    dict  horizon → np.ndarray (n_eval, n_lambdas)
    """
    n_lambdas = len(lambdas)
    n_eval = len(eval_indices)

    losses: dict[int, np.ndarray] = {
        T: np.full((n_eval, n_lambdas), np.nan) for T in horizons
    }

    for k, lam in enumerate(lambdas):
        logger.info("EWMA recursion lambda=%.2f …", lam)
        sigma_tril = ewma_lower_triangle_series(returns_arr, lam)

        for T in horizons:
            realized_tril = realized_tril_all[T]
            for i, t in enumerate(eval_indices):
                fc_idx = t - T
                if fc_idx < 0:
                    continue
                real_vec = realized_tril[t]
                if np.any(np.isnan(real_vec)):
                    continue
                fcast_vec = T * sigma_tril[fc_idx]
                diff = fcast_vec - real_vec
                losses[T][i, k] = np.dot(diff, diff)

    return losses


def run_adaptive_experiment(
    losses_all: dict[int, np.ndarray],
    eval_dates: pd.DatetimeIndex,
    fixed_lambda_star: dict[int, float],
    horizons: list[int] = HORIZONS,
    lambdas: np.ndarray = LAMBDAS,
) -> dict:
    """
    Run the lagged adaptive lambda experiment for all horizons.

    Parameters
    ----------
    losses_all : dict  horizon → np.ndarray (n_eval, n_lambdas)
    eval_dates : pd.DatetimeIndex
    fixed_lambda_star : dict  horizon → float (from exp_2)
    horizons : list of int
    lambdas : np.ndarray

    Returns
    -------
    dict with per-horizon results.
    """
    results = {}

    for T in horizons:
        losses_T = losses_all[T]  # shape (n_eval, n_lambdas)
        n_eval = losses_T.shape[0]

        # Ex-post optimal lambda index for each evaluation date
        expost_lambda_idx = np.nanargmin(losses_T, axis=1)  # shape (n_eval,)
        expost_lambda_vals = lambdas[expost_lambda_idx]

        # ── Adaptive sequence: lambda_t = lambda_{t-1}^* (lagged by 1 date) ──
        # For the first evaluation date, initialize with the fixed optimal lambda
        # (documented fallback per methodology note 4).
        fixed_star = fixed_lambda_star[T]
        fixed_star_idx = int(np.argmin(np.abs(lambdas - fixed_star)))

        adaptive_idx = np.empty(n_eval, dtype=int)
        adaptive_idx[0] = fixed_star_idx  # initialization fallback
        adaptive_idx[1:] = expost_lambda_idx[:-1]  # lag by 1

        # ── Adaptive losses ──────────────────────────────────────────────────
        adaptive_losses = losses_T[np.arange(n_eval), adaptive_idx]
        fixed_losses = losses_T[:, fixed_star_idx]

        # ── MSE ──────────────────────────────────────────────────────────────
        mse_fixed = float(np.nanmean(fixed_losses))
        mse_adaptive = float(np.nanmean(adaptive_losses))

        # ── DM test: d_t = loss_fixed - loss_adaptive (positive → adaptive better)
        dm = diebold_mariano_test(fixed_losses, adaptive_losses, horizon=T)

        logger.info(
            "T=%2d | Fixed MSE=%.6f | Adaptive MSE=%.6f | DM=%.3f (p=%.4f)",
            T, mse_fixed, mse_adaptive, dm["dm_stat"], dm["dm_pval"],
        )

        results[T] = {
            "mse_fixed": mse_fixed,
            "mse_adaptive": mse_adaptive,
            "mse_improvement": mse_fixed - mse_adaptive,
            "fixed_lambda_star": fixed_star,
            "dm_stat": dm["dm_stat"],
            "dm_pval": dm["dm_pval"],
            "mean_loss_diff": dm["mean_diff"],
            "expost_lambda_vals": expost_lambda_vals,
            "adaptive_losses": adaptive_losses,
            "fixed_losses": fixed_losses,
            "eval_dates": eval_dates,
        }

    return results


def save_results(results: dict, horizons: list[int] = HORIZONS) -> None:
    """Save adaptive experiment results and plots."""
    # ── Summary table ─────────────────────────────────────────────────────────
    rows = []
    for T in horizons:
        r = results[T]
        rows.append(
            {
                "horizon": T,
                "fixed_lambda_star": r["fixed_lambda_star"],
                "mse_fixed": r["mse_fixed"],
                "mse_adaptive": r["mse_adaptive"],
                "mse_improvement": r["mse_improvement"],
                "dm_stat": r["dm_stat"],
                "dm_pval": r["dm_pval"],
                "dm_significant": "*" if r["dm_pval"] < 0.05 else "",
            }
        )
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(RESULTS_DIR / "exp3_adaptive_summary.csv", index=False)

    print("\n" + "=" * 65)
    print("Exp 3 Results — Adaptive vs Fixed-Lambda EWMA")
    print("=" * 65)
    print(summary_df.to_string(index=False))
    print("=" * 65)

    # ── Time series of ex-post optimal lambda for T=21 ────────────────────────
    T = 21
    r21 = results[T]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(r21["eval_dates"], r21["expost_lambda_vals"], linewidth=0.8, color="#1f77b4",
            alpha=0.8, label="Ex-post λ*")
    ax.axhline(r21["fixed_lambda_star"], color="red", linestyle="--",
               linewidth=1.4, label=f"Fixed λ*={r21['fixed_lambda_star']:.2f}")
    ax.set_title("Ex-post Optimal λ over Time (T=21, 2000–2020)", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("λ")
    ax.legend()
    ax.set_xlim(r21["eval_dates"][0], r21["eval_dates"][-1])
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "exp3_expost_lambda_T21.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Time series plot for T=21 saved.")

    # ── MSE comparison bar chart ──────────────────────────────────────────────
    x = np.arange(len(horizons))
    width = 0.35
    fixed_mses = [results[T]["mse_fixed"] for T in horizons]
    adaptive_mses = [results[T]["mse_adaptive"] for T in horizons]

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(x - width / 2, fixed_mses, width, label="Fixed λ*", color="#4575b4")
    ax2.bar(x + width / 2, adaptive_mses, width, label="Adaptive λ", color="#d73027")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"T={T}" for T in horizons])
    ax2.set_ylabel("MSE")
    ax2.set_title("Fixed vs Adaptive λ — EWMA Covariance MSE")
    ax2.legend()
    ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    plt.tight_layout()
    fig2.savefig(RESULTS_DIR / "exp3_mse_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logger.info("MSE comparison plot saved.")


def main() -> dict:
    """Entry point for exp_3."""
    logger.info("Loading returns for exp_3 …")
    returns_full = load_returns(
        tickers=TICKERS,
        start="1994-01-03",
        end="2020-12-31",
    )
    returns_arr = returns_full.values.astype(np.float64)
    dates = returns_full.index

    eval_mask = (dates >= EVAL_START) & (dates <= EVAL_END)
    eval_indices = np.where(eval_mask)[0]
    eval_dates = dates[eval_indices]
    logger.info("Evaluation dates: %d", len(eval_indices))

    # Pre-compute realized covariances
    logger.info("Pre-computing realized T-day covariances …")
    realized_tril_all: dict[int, np.ndarray] = {}
    for T in HORIZONS:
        realized_tril_all[T] = compute_realized_tril(returns_arr, T)

    # Compute full loss surface (shared with exp_2 calculation)
    losses_all = compute_full_loss_surface(
        returns_arr, eval_indices, realized_tril_all, HORIZONS, LAMBDAS
    )

    # Fixed optimal lambdas from exp_2 (expected values; computed here afresh)
    fixed_mse = {T: np.nanmean(losses_all[T], axis=0) for T in HORIZONS}
    fixed_lambda_star = {T: float(LAMBDAS[int(np.argmin(fixed_mse[T]))]) for T in HORIZONS}
    logger.info("Fixed optimal lambdas: %s", fixed_lambda_star)

    # Run adaptive experiment
    results = run_adaptive_experiment(
        losses_all, eval_dates, fixed_lambda_star, HORIZONS, LAMBDAS
    )
    save_results(results)
    return results


if __name__ == "__main__":
    main()
