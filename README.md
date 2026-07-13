# DJIA 22-Stock EWMA Covariance Forecasting Study

Replication of the EWMA covariance forecasting study on 22 DJIA stocks
covering the period 1994–2020, with full-sample diagnostics, fixed-lambda
optimization, and lagged adaptive lambda selection.

## Experiments

| Experiment | Description |
|------------|-------------|
| **exp_1** | Exploratory diagnostics (Shapiro-Wilk, ARCH LM-20, ADF) on 1994–1999 returns |
| **exp_2** | Fixed-lambda EWMA covariance forecasting; grid search over λ ∈ {0.01,…,0.99} for T ∈ {5,10,21} |
| **exp_3** | Lagged time-varying λ EWMA vs. fixed optimal λ; Diebold-Mariano test |

## Repository Structure

```
.
├── data/             # Data downloading and caching
│   └── download.py
├── src/              # Core algorithms
│   ├── diagnostics.py      # SW, ARCH LM, ADF tests
│   ├── ewma.py             # EWMA covariance recursion
│   └── diebold_mariano.py  # DM test with Newey-West HAC
├── exp/              # Experiment scripts
│   ├── exp1_diagnostics.py
│   ├── exp2_fixed_lambda.py
│   └── exp3_adaptive_lambda.py
├── tests/            # pytest tests
├── results/          # Outputs: tables, plots, RESULTS.md
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install yfinance statsmodels scipy numpy pandas matplotlib seaborn

# Run individual experiments
python exp/exp1_diagnostics.py
python exp/exp2_fixed_lambda.py
python exp/exp3_adaptive_lambda.py

# Run tests
pytest tests/ -v
```

## Data

Daily adjusted closing prices for 22 US large-cap stocks are downloaded from
Yahoo Finance via `yfinance` (with `massive` API as primary if available).
The balanced panel covers 1994-01-03 to 2020-12-31.  A CSV cache is stored
at `data/prices_raw.csv` after the first download.

**Tickers**: MMM, AA, AXP, BA, CAT, DD, XOM, IBM, INTC, JNJ, JPM, MCD, MRK,
MSFT, PFE, PG, KO, HD, DIS, RTX, VZ, WMT

## Key Results

See [`results/RESULTS.md`](results/RESULTS.md) for the full diagnostic table
and MSE optimization results.

### Expected Findings (paper targets)
- **exp_1**: All/most assets reject normality (SW) and show ARCH effects; all are stationary (ADF)
- **exp_2**: λ\*(T=5)≈0.92, λ\*(T=10)≈0.95, λ\*(T=21)≈0.98; RiskMetrics 0.97 nearly ties at T=21
- **exp_3**: Adaptive λ outperforms fixed λ at all horizons; DM ≈ 3.2, 5.9, 5.2 for T=5,10,21

## Methodology Notes

- **Log returns**: r_{i,t} = ln(P_{i,t}/P_{i,t-1}), no demeaning, no annualization
- **EWMA recursion**: Σ̂_{t+1}(λ) = (1−λ) r_t r_t' + λ Σ̂_t
- **T-day scaling**: Σ̂_{t,T}(λ) = T · Σ̂_{t+1}(λ)
- **Realized covariance**: C_{t,T} = Σ_{k=0}^{T-1} r_{t-k} r_{t-k}' (no normalization)
- **Loss**: SE_{t,T}(λ) = Σ_{i≥j} (σ̂_{ij,t,T} − σ_{ij,t,T})²  (253 entries for N=22)
- **DM bandwidth**: Newey-West with q = T−1 lags

## Libraries Used

| Library | Purpose |
|---------|---------|
| `statsmodels` | ARCH LM test, ADF test, Ljung-Box test |
| `scipy.stats` | Shapiro-Wilk test |
| `numpy` / `pandas` | Numerical computation, data handling |
| `yfinance` | Yahoo Finance data download |
| `matplotlib` / `seaborn` | Visualization |
| `pytest` | Testing |
