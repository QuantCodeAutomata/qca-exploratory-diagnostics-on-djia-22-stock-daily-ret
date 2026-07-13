"""
Data download and preprocessing for the DJIA 22-stock EWMA covariance study.

Downloads daily adjusted closing prices for 22 DJIA stocks over 1994-01-03
to 2020-12-31 from Yahoo Finance via the massive library or yfinance fallback.
Constructs a balanced panel with common trading-day calendar and computes
daily log returns.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 22 DJIA stocks used in the paper (ticker lineup as of replication)
TICKERS: list[str] = [
    "MMM", "AA", "AXP", "BA", "CAT", "DD", "XOM", "IBM", "INTC",
    "JNJ", "JPM", "MCD", "MRK", "MSFT", "PFE", "PG", "KO", "HD",
    "DIS", "RTX", "VZ", "WMT",
]

START_DATE: str = "1994-01-03"
END_DATE: str = "2020-12-31"
DATA_DIR: Path = Path(__file__).parent

# Ticker lineage note:
# RTX (Raytheon Technologies) was formed from the merger of United Technologies
# (UTX) and Raytheon on 2020-04-03.  yfinance back-fills the full UTX history
# under the RTX ticker back to 1994, so no manual stitching is required here.
# DD (DuPont de Nemours) similarly carries its long back-history through yfinance.
# Both tickers are therefore usable as-is for the full study period.
TICKER_PROXIES: dict[str, tuple[str, str]] = {}  # no stitching required


def _download_massive(
    tickers: list[str],
    start: str,
    end: str,
) -> Optional[pd.DataFrame]:
    """
    Download daily adjusted close prices via massive (Polygon-compatible) API.

    Returns None if the API fails OR if the returned data does not cover the
    requested start date (within a 30-day tolerance), so that the caller can
    fall back to yfinance for long historical studies.

    Parameters
    ----------
    tickers : list of str
        List of stock ticker symbols.
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with tickers as columns and dates as index, or None on failure.
    """
    try:
        from massive import RESTClient  # type: ignore

        api_key = os.getenv("MASSIVE_TOKEN") or os.getenv("MASSIVE_API_KEY", "")
        client = RESTClient(api_key=api_key)

        frames: dict[str, pd.Series] = {}
        for ticker in tickers:
            aggs = []
            for a in client.list_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=start,
                to=end,
                adjusted=True,
                limit=50000,
            ):
                aggs.append(a)
            if not aggs:
                logger.warning("No data returned by massive for %s", ticker)
                return None
            rows = {
                pd.Timestamp(a.timestamp, unit="ms"): a.close
                for a in aggs
                if hasattr(a, "close") and a.close is not None
            }
            frames[ticker] = pd.Series(rows, name=ticker)
            logger.info("Downloaded %d bars for %s via massive", len(rows), ticker)

        prices = pd.DataFrame(frames)
        prices.index = pd.DatetimeIndex(prices.index).normalize()
        prices.sort_index(inplace=True)

        # Validate coverage: check that the earliest date is within 30 trading
        # days of the requested start.  If not, the API lacks the historical
        # depth needed for this study and we fall back to yfinance.
        requested_start = pd.Timestamp(start)
        actual_start = prices.index.min()
        tolerance = pd.Timedelta(days=45)
        if actual_start > requested_start + tolerance:
            logger.warning(
                "massive data starts %s but requested %s — insufficient history; "
                "falling back to yfinance",
                actual_start.date(), requested_start.date(),
            )
            return None

        return prices

    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("massive download failed (%s), falling back to yfinance", exc)
        return None


def _download_yfinance(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Download daily adjusted close prices via yfinance.

    Handles ticker lineage stitching for tickers in TICKER_PROXIES (e.g. RTX/UTX).

    Parameters
    ----------
    tickers : list of str
        Ticker symbols (canonical, post-rebranding names).
    start : str
        Start date 'YYYY-MM-DD'.
    end : str
        End date 'YYYY-MM-DD' (inclusive; yfinance end is exclusive so we add 1 day).

    Returns
    -------
    pd.DataFrame
        Adjusted close prices with dates as index and tickers as columns.
    """
    import yfinance as yf  # type: ignore

    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # Build download list: replace proxied tickers with both the canonical and
    # the proxy so both can be fetched in one call.
    proxy_tickers = [TICKER_PROXIES[t][0] for t in tickers if t in TICKER_PROXIES]
    all_download = list(dict.fromkeys(tickers + proxy_tickers))  # deduplicated, order-preserved

    data = yf.download(
        tickers=all_download,
        start=start,
        end=end_exclusive,
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if isinstance(data.columns, pd.MultiIndex):
        prices_raw = data["Adj Close"]
    else:
        prices_raw = data

    # Stitch proxy series for tickers in TICKER_PROXIES
    prices_out = prices_raw.copy()
    for canonical, (proxy, cutover_str) in TICKER_PROXIES.items():
        if canonical not in tickers:
            continue
        cutover = pd.Timestamp(cutover_str)
        if proxy not in prices_raw.columns:
            logger.warning("Proxy ticker %s not found; using %s as-is", proxy, canonical)
            continue
        # Use proxy (UTX) for dates before the merger, canonical (RTX) after
        proxy_series = prices_raw[proxy]
        canon_series = prices_raw.get(canonical, pd.Series(dtype=float))
        stitched = pd.concat(
            [proxy_series[proxy_series.index < cutover],
             canon_series[canon_series.index >= cutover]]
        ).sort_index()
        prices_out[canonical] = stitched
        logger.info(
            "Stitched %s: %s data until %s, %s data from %s",
            canonical, proxy, cutover.date(), canonical, cutover.date(),
        )

    # Reorder columns to match the canonical ticker list (drop proxy helpers)
    available = [t for t in tickers if t in prices_out.columns]
    prices = prices_out[available]
    return prices


def download_prices(
    tickers: list[str] = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
    cache_path: Optional[Path] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download (or load from cache) daily adjusted close prices for the given tickers.

    Tries massive API first; falls back to yfinance.

    Parameters
    ----------
    tickers : list of str
        Stock ticker symbols.
    start : str
        Start date 'YYYY-MM-DD'.
    end : str
        End date 'YYYY-MM-DD'.
    cache_path : Path, optional
        File path for caching the downloaded prices (CSV).  Defaults to
        data/prices_raw.csv relative to this module.
    force_refresh : bool
        If True, re-download even when the cache file exists.

    Returns
    -------
    pd.DataFrame
        Adjusted close prices; index is DatetimeIndex, columns are ticker symbols.
    """
    if cache_path is None:
        cache_path = DATA_DIR / "prices_raw.csv"

    if not force_refresh and cache_path.exists():
        logger.info("Loading cached prices from %s", cache_path)
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        prices.index = pd.DatetimeIndex(prices.index)
        return prices

    logger.info("Downloading prices for %d tickers (%s–%s)", len(tickers), start, end)

    # Try massive first
    prices = _download_massive(tickers, start, end)

    # Fall back to yfinance
    if prices is None:
        prices = _download_yfinance(tickers, start, end)

    prices.to_csv(cache_path)
    logger.info("Prices saved to %s", cache_path)
    return prices


def build_balanced_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Build a balanced panel by dropping dates with any NaN across all columns.

    Parameters
    ----------
    prices : pd.DataFrame
        Raw adjusted close prices; index is DatetimeIndex, columns are tickers.

    Returns
    -------
    pd.DataFrame
        Balanced adjusted close prices with only complete-data dates.
    """
    before = len(prices)
    prices_clean = prices.dropna(how="any")
    after = len(prices_clean)
    logger.info(
        "Balanced panel: %d → %d dates (dropped %d dates with missing data)",
        before, after, before - after,
    )
    return prices_clean


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns r_{i,t} = ln(P_{i,t} / P_{i,t-1}).

    No demeaning or annualization is applied, consistent with the paper.

    Parameters
    ----------
    prices : pd.DataFrame
        Balanced panel of adjusted close prices.

    Returns
    -------
    pd.DataFrame
        Daily log returns; the first row (NaN) is dropped.
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns


def load_returns(
    tickers: list[str] = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
    cache_path: Optional[Path] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    High-level convenience: download prices, build balanced panel, compute returns.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols.
    start, end : str
        Date range 'YYYY-MM-DD'.
    cache_path : Path, optional
        Cache path for raw prices CSV.
    force_refresh : bool
        Re-download even if cache exists.

    Returns
    -------
    pd.DataFrame
        Daily log returns over the full requested date range.
    """
    prices = download_prices(tickers, start, end, cache_path, force_refresh)
    prices = build_balanced_panel(prices)
    returns = compute_log_returns(prices)
    return returns


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rets = load_returns()
    print(f"Returns shape: {rets.shape}")
    print(rets.tail())
