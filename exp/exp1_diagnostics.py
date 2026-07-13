"""
Exp 1 – Exploratory diagnostics on DJIA 22-stock daily returns (1994-1999).

Runs Shapiro-Wilk, Engle ARCH LM (20 lags), and Augmented Dickey-Fuller
tests on each asset and saves a summary table and visualization to results/.

Usage
-----
    python exp/exp1_diagnostics.py
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
import seaborn as sns

# Ensure project root is on the path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.download import load_returns, TICKERS
from src.diagnostics import run_diagnostics, summarise_diagnostics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EVAL_START = "1994-01-03"
EVAL_END = "1999-12-31"


def main() -> pd.DataFrame:
    """Run exp_1 diagnostics and return the summary table."""
    # ── 1. Load returns ────────────────────────────────────────────────────────
    logger.info("Loading returns for exp_1 …")
    returns_full = load_returns(
        tickers=TICKERS,
        start="1994-01-03",
        end="2020-12-31",
    )

    # ── 2. Restrict to 1994-01-03 to 1999-12-31 ──────────────────────────────
    returns = returns_full.loc[EVAL_START:EVAL_END]
    logger.info(
        "Return sample: %s to %s — %d observations, %d assets",
        returns.index[0].date(), returns.index[-1].date(),
        len(returns), returns.shape[1],
    )

    # ── 3. Run diagnostics ────────────────────────────────────────────────────
    logger.info("Running diagnostics …")
    table = run_diagnostics(returns, arch_nlags=20, adf_autolag="AIC", ibm_ticker="IBM")

    # ── 4. Print summary ─────────────────────────────────────────────────────
    summary = summarise_diagnostics(table)
    print(summary)
    print()
    print(table[["sw_pval", "sw_reject_normality",
                 "arch_lm_stat", "arch_lm_pval", "arch_sig",
                 "adf_stat", "adf_pval", "adf_stationary"]].to_string())

    # ── 5. Save table ─────────────────────────────────────────────────────────
    out_csv = RESULTS_DIR / "exp1_diagnostic_table.csv"
    table.to_csv(out_csv)
    logger.info("Diagnostic table saved to %s", out_csv)

    # ── 6. Visualization – heatmap of p-values ────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(
        "DJIA 22-Stock Diagnostic P-values (1994–1999)",
        fontsize=14, fontweight="bold",
    )

    pval_data = {
        "Shapiro-Wilk\n(non-normality)": table["sw_pval"],
        "ARCH LM-20\n(heteroskedasticity)": table["arch_lm_pval"],
        "ADF\n(stationarity)": table["adf_pval"],
    }

    for ax, (title, pvals) in zip(axes, pval_data.items()):
        colors = ["#d73027" if p < 0.05 else "#4575b4" for p in pvals]
        bars = ax.barh(pvals.index[::-1], pvals.values[::-1], color=colors[::-1])
        ax.axvline(x=0.05, color="black", linestyle="--", linewidth=1.2, label="5% threshold")
        ax.set_xlabel("p-value")
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = RESULTS_DIR / "exp1_diagnostic_pvalues.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved to %s", fig_path)

    # ── 7. Heatmap of significance flags ─────────────────────────────────────
    flags = table[["sw_reject_normality", "arch_sig", "adf_stationary"]].astype(int)
    flags.columns = ["Non-normal\n(SW)", "ARCH\n(LM-20)", "Stationary\n(ADF)"]

    fig2, ax2 = plt.subplots(figsize=(6, 8))
    sns.heatmap(
        flags,
        annot=True,
        fmt="d",
        cmap="RdYlGn",
        linewidths=0.5,
        ax=ax2,
        cbar=False,
        vmin=0,
        vmax=1,
    )
    ax2.set_title("Significance Flags (1 = significant at 5%)\n1994–1999", fontsize=12)
    ax2.set_ylabel("")
    fig2.tight_layout()
    fig2_path = RESULTS_DIR / "exp1_significance_heatmap.png"
    fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logger.info("Heatmap saved to %s", fig2_path)

    return table


if __name__ == "__main__":
    result = main()
