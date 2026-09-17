"""
Minimal, dependency-free implementations of the technical indicators this project
uses.

Previously these came from the `ta` package. `ta` ships source-only (no wheel) and
its legacy ``setup.py`` fails to build against modern setuptools, which breaks
clean installs and hosted deployments. The formulas below are standard and are
verified to reproduce `ta`'s output on this project's data (see
``tests/test_indicators.py``), so results are unchanged.

Every function is causal: the value at row t depends only on rows <= t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return close.rolling(window, min_periods=window).mean()


def ema(close: pd.Series, window: int) -> pd.Series:
    """Exponential moving average (recursive form, seeded after `window` obs)."""
    return close.ewm(span=window, min_periods=window, adjust=False).mean()


def macd(close: pd.Series, window_slow: int, window_fast: int, window_sign: int):
    """MACD line, signal line, and their difference (histogram)."""
    line = (close.ewm(span=window_fast, min_periods=window_fast, adjust=False).mean()
            - close.ewm(span=window_slow, min_periods=window_slow, adjust=False).mean())
    signal = line.ewm(span=window_sign, min_periods=window_sign, adjust=False).mean()
    return line, signal, line - signal


def rsi(close: pd.Series, window: int) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing."""
    diff = close.diff(1)
    # NaN and non-positive diffs both map to 0.0, matching the standard definition.
    up_direction = diff.where(diff > 0, 0.0)
    down_direction = -diff.where(diff < 0, 0.0)
    # Wilder smoothing is an EWM with alpha = 1/window.
    emaup = up_direction.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    emadn = down_direction.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    relative_strength = emaup / emadn
    # A window with no down moves is defined as 100 rather than divide-by-zero.
    return pd.Series(
        np.where(emadn == 0, 100.0, 100.0 - (100.0 / (1.0 + relative_strength))),
        index=close.index,
        name="rsi",
    )


def roc(close: pd.Series, window: int) -> pd.Series:
    """Rate of change, in percent."""
    return ((close - close.shift(window)) / close.shift(window)) * 100.0


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               window: int, smooth_window: int):
    """Stochastic oscillator %K and its %D signal line."""
    lowest = low.rolling(window, min_periods=window).min()
    highest = high.rolling(window, min_periods=window).max()
    k = 100.0 * (close - lowest) / (highest - lowest)
    d = k.rolling(smooth_window, min_periods=smooth_window).mean()
    return k, d


def bollinger(close: pd.Series, window: int, window_dev: float):
    """Bollinger middle band, upper band, lower band, and %b."""
    mid = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std(ddof=0)
    hband = mid + window_dev * std
    lband = mid - window_dev * std
    pctb = (close - lband) / (hband - lband)
    return mid, hband, lband, pctb


def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-balance volume, a running signed total of volume.

    Volume is subtracted on a down day and added otherwise, so unchanged closes
    count as up days. That is the conventional definition.
    """
    signed = np.where(close < close.shift(1), -volume, volume)
    return pd.Series(signed, index=close.index, name="obv").cumsum()
