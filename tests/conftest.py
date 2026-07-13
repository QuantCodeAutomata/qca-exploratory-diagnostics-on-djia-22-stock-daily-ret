"""Shared pytest fixtures."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

N_OBS = 300
N_ASSETS = 22
RNG = np.random.default_rng(42)


@pytest.fixture(scope="session")
def normal_returns() -> np.ndarray:
    """N_OBS × N_ASSETS array of i.i.d. Normal(0,0.01) returns."""
    return RNG.normal(0, 0.01, size=(N_OBS, N_ASSETS))


@pytest.fixture(scope="session")
def leptokurtic_returns() -> np.ndarray:
    """N_OBS × N_ASSETS array of Student-t(3) returns (fat-tailed)."""
    return RNG.standard_t(df=3, size=(N_OBS, N_ASSETS)) * 0.01


@pytest.fixture(scope="session")
def returns_df(leptokurtic_returns) -> pd.DataFrame:
    """Fat-tailed returns as a DataFrame with dummy tickers."""
    tickers = [f"A{i:02d}" for i in range(N_ASSETS)]
    dates = pd.bdate_range("2000-01-03", periods=N_OBS, freq="B")
    return pd.DataFrame(leptokurtic_returns, index=dates, columns=tickers)
