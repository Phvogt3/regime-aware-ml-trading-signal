"""
Shared test fixtures. Generates SYNTHETIC OHLCV + VIX so the whole test suite
runs offline (no yfinance / network needed). Synthetic data is fine here because
these tests check pipeline CORRECTNESS (leakage, accounting, shapes), not trading
performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _one_ticker(ticker: str, dates: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(dates)
    # geometric random walk for close
    rets = rng.normal(0.0004, 0.012, n)
    close = 100 * np.exp(np.cumsum(rets))
    # build OHLC around close with plausible intraday spread
    spread = np.abs(rng.normal(0, 0.006, n)) * close
    high = close + spread
    low = close - spread
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.002, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({
        "date": dates, "ticker": ticker,
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    })


@pytest.fixture(scope="session")
def synthetic_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2016-01-01", "2024-12-31")  # ~9 years of weekdays
    tickers = ["AAPL", "MSFT", "JPM", "GS", "PG", "KO", "SPY"]
    frames = [_one_ticker(t, dates, seed=i + 1) for i, t in enumerate(tickers)]
    return pd.concat(frames).reset_index(drop=True)


@pytest.fixture(scope="session")
def synthetic_vix() -> pd.DataFrame:
    dates = pd.bdate_range("2016-01-01", "2024-12-31")
    rng = np.random.default_rng(99)
    # mean-reverting-ish VIX around 18 with occasional spikes
    n = len(dates)
    v = np.empty(n)
    v[0] = 18
    for i in range(1, n):
        shock = rng.normal(0, 1.2) + (rng.random() < 0.01) * rng.uniform(10, 30)
        v[i] = max(9.0, 0.94 * v[i - 1] + 0.06 * 18 + shock)
    return pd.DataFrame({"date": dates, "vix_close": v})
