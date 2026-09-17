"""
Data pipeline: pull daily OHLCV for the universe + SPY benchmark, and the VIX
close for the volatility regime filter. Cache to parquet so the (network-bound)
download happens once and every downstream step reads locally.

Run this on a machine with internet access (yfinance hits Yahoo Finance):
    python -m src.data            # downloads and caches
    python -m src.data --force    # re-download even if cache exists

Design notes
------------
- We store prices in LONG format (date, ticker, open/high/low/close/volume).
  Long format keeps per-ticker feature engineering clean and avoids the
  MultiIndex-column headaches yfinance produces for multi-ticker frames.
- We use auto-adjusted prices (splits/dividends) so returns are economically
  meaningful. Historical adjustment factors are not strictly point-in-time at
  corporate-action boundaries; that limitation is disclosed in the README.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import pandas as pd

from . import config


def _cache_path(fname: str) -> str:
    return os.path.join(config.DATA_DIR, fname)


def download_prices(tickers: List[str],
                    start: str = config.START_DATE,
                    end: str = config.END_DATE) -> pd.DataFrame:
    """
    Download OHLCV for a list of tickers and return a tidy long DataFrame with
    columns: date, ticker, open, high, low, close, volume.
    """
    import yfinance as yf  # imported lazily so the rest of the package works offline

    raw = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, progress=False, group_by="ticker",
    )
    frames = []
    # yfinance returns a MultiIndex (ticker, field) when >1 ticker, flat otherwise.
    if isinstance(raw.columns, pd.MultiIndex):
        for tkr in tickers:
            if tkr not in raw.columns.get_level_values(0):
                continue
            sub = raw[tkr].copy()
            sub["ticker"] = tkr
            frames.append(sub)
    else:
        sub = raw.copy()
        sub["ticker"] = tickers[0]
        frames.append(sub)

    if not frames:
        raise RuntimeError("Yahoo Finance returned no usable price series")
    df = pd.concat(frames).reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    df = df.rename(columns={"index": "date"})
    # normalize column set
    keep = ["date", "ticker", "open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]]
    df = df.dropna(subset=["close"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def download_vix(start: str = config.START_DATE,
                 end: str = config.END_DATE) -> pd.DataFrame:
    """Return a DataFrame with columns: date, vix_close."""
    import yfinance as yf

    raw = yf.download(config.VIX_SYMBOL, start=start, end=end,
                      auto_adjust=False, progress=False)
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no VIX data")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    out = raw.reset_index()[["Date", "Close"]]
    out.columns = ["date", "vix_close"]
    out["date"] = pd.to_datetime(out["date"])
    return out.dropna().reset_index(drop=True)


def build_cache(force: bool = False) -> None:
    """Download, validate, and atomically replace parquet caches."""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    prices_path = _cache_path(config.PRICES_FILE)
    vix_path = _cache_path(config.VIX_FILE)

    need_prices = force or not os.path.exists(prices_path)
    need_vix = force or not os.path.exists(vix_path)
    prices = None
    vix = None

    # Finish every requested download before touching a valid existing cache.
    if need_prices:
        tickers = config.TICKERS + [config.MARKET_BENCHMARK]
        print(f"Downloading {len(tickers)} price series ...")
        prices = download_prices(tickers)
    else:
        print(f"Cache exists: {prices_path} (use --force to refresh)")

    if need_vix:
        print("Downloading VIX ...")
        vix = download_vix()
    else:
        print(f"Cache exists: {vix_path} (use --force to refresh)")

    if prices is not None:
        validate(prices)
    if vix is not None:
        if prices is None:
            existing_prices = load_prices()
            validate(existing_prices, vix)
        else:
            validate(prices, vix)

    pending = []
    try:
        if prices is not None:
            tmp = prices_path + ".tmp"
            prices.to_parquet(tmp, index=False)
            pending.append((tmp, prices_path))
        if vix is not None:
            tmp = vix_path + ".tmp"
            vix.to_parquet(tmp, index=False)
            pending.append((tmp, vix_path))
        for tmp, final in pending:
            os.replace(tmp, final)
    finally:
        for tmp, _ in pending:
            if os.path.exists(tmp):
                os.remove(tmp)

    if prices is not None:
        print(f"  wrote {prices_path}  ({len(prices):,} rows, "
              f"{prices['ticker'].nunique()} tickers)")
    if vix is not None:
        print(f"  wrote {vix_path}  ({len(vix):,} rows)")


def load_prices() -> pd.DataFrame:
    """Load cached prices (long format). Raises if cache missing."""
    path = _cache_path(config.PRICES_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.data` first (needs internet).")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_vix() -> pd.DataFrame:
    path = _cache_path(config.VIX_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.data` first (needs internet).")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def validate(prices: pd.DataFrame, vix: Optional[pd.DataFrame] = None) -> None:
    """Cheap sanity checks that catch a broken download early."""
    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"prices frame missing columns: {sorted(missing)}")
    if prices.empty:
        raise ValueError("prices frame is empty")
    if prices[list(required)].isna().any().any():
        raise ValueError("prices frame contains missing values")
    if not prices[["open", "high", "low", "close"]].gt(0).all().all():
        raise ValueError("non-positive prices present")
    if not prices["volume"].ge(0).all():
        raise ValueError("negative volume values present")
    dupes = prices.duplicated(subset=["date", "ticker"]).sum()
    if dupes:
        raise ValueError(f"{dupes} duplicate (date, ticker) rows")
    if vix is not None:
        if not {"date", "vix_close"}.issubset(vix.columns):
            raise ValueError("VIX frame must contain date and vix_close")
        if vix.empty or vix[["date", "vix_close"]].isna().any().any():
            raise ValueError("VIX frame is empty or contains missing values")
        if not vix["vix_close"].gt(0).all():
            raise ValueError("non-positive VIX values")
        vix_dupes = vix.duplicated(subset=["date"]).sum()
        if vix_dupes:
            raise ValueError(f"{vix_dupes} duplicate VIX dates")
    print("validate: OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the local data cache.")
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    args = ap.parse_args()
    build_cache(force=args.force)
    p = load_prices()
    v = load_vix()
    validate(p, v)
    print(f"\nUniverse tickers: {sorted(p['ticker'].unique())}")
    print(f"Date span: {p['date'].min().date()} -> {p['date'].max().date()}")
