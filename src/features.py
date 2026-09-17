"""
Feature engineering.

Indicators are first calculated causally through each session's close, then shifted
forward one trading session. A feature row dated t therefore contains information
available through the close of t-1. This makes a day-t close-auction trade executable:
the signal is known before the order cutoff rather than being calculated from the
same closing price at which the backtest assumes a fill.

  * All indicators come from src/indicators.py or pandas .rolling()/.ewm(), which
    are backward-looking (the window ends at the current bar). No centered
    windows, no .shift(-k), no full-sample statistics.
  * Per-ticker computation is done inside groupby so a rolling window never bleeds
    across ticker boundaries.
  * The VIX percentile rank is a TRAILING-window rank (default 252 days), NOT a
    full-history rank. A full-history percentile would leak the future
    distribution of volatility into every past row -- a classic, subtle bug.
  * No scaling/standardization happens here. Standardization is fit per
    walk-forward fold on training data only (see walkforward.py), because fitting
    a scaler on the whole series leaks test-period moments into training.

`tests/test_leakage.py` verifies the core guarantee empirically: truncating the
price history at day t leaves every feature value at day t unchanged.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from . import indicators as ind
from .config import FEATURES as F

# Column order is stable so models see the same feature vector every run.
FEATURE_COLUMNS: List[str] = [
    "sma_cross",     # (fast SMA / slow SMA) - 1
    "ema_cross",     # (fast EMA / slow EMA) - 1
    "macd",          # MACD / close (cross-sectionally scale-free)
    "macd_signal",   # MACD signal / close
    "macd_diff",     # MACD histogram / close
    "rsi",
    "roc",
    "stoch_k",
    "stoch_d",
    "bb_width",      # (upper - lower) / middle
    "bb_pctb",       # position of close within the bands, 0..1-ish
    "volatility",    # rolling std of daily returns
    "vol_ratio",     # volume / trailing-mean volume
    "obv_z",         # z-scored on-balance volume (stationary form of OBV)
    "fracdiff",      # fractionally-differenced log price (reduced persistence)
    "vix_level",     # current VIX close
    "vix_pctile",    # trailing-year percentile rank of VIX
]


# Feature families, used by the ablation study (drop one family, retrain, compare).
FEATURE_GROUPS: Dict[str, List[str]] = {
    "trend": ["sma_cross", "ema_cross", "macd", "macd_signal", "macd_diff"],
    "momentum": ["rsi", "roc", "stoch_k", "stoch_d"],
    "volatility": ["bb_width", "bb_pctb", "volatility"],
    "volume": ["vol_ratio", "obv_z"],
    "long_memory": ["fracdiff"],
    "vix": ["vix_level", "vix_pctile"],
}
assert sorted(sum(FEATURE_GROUPS.values(), [])) == sorted(FEATURE_COLUMNS)


def _ffd_weights(d: float, thresh: float, max_width: int) -> np.ndarray:
    """
    Fixed-width-window weights for fractional differencing (Lopez de Prado).
    w_0 = 1 and w_k = -w_{k-1} * (d - k + 1) / k. The series is truncated once a
    weight falls below `thresh` in magnitude (or `max_width` is reached), giving a
    finite causal filter. Returned newest-first: w[0] multiplies the current bar.
    """
    w = [1.0]
    k = 1
    while k <= max_width:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
        k += 1
    return np.array(w)


def frac_diff_ffd(series: pd.Series, d: float, thresh: float,
                  max_width: int) -> pd.Series:
    """
    Fractionally difference a series with the fixed-width-window method.

    The output at time t is a weighted sum of the current and past values only
    (a causal FIR filter), so it never references the future. The first
    (len(weights) - 1) observations lack a full window and are returned as NaN.
    A fractional order in (0, 1) generally reduces persistence while retaining
    more level information than a full first difference. Stationarity is not
    guaranteed by the transform alone.
    """
    w = _ffd_weights(d, thresh, max_width)
    width = len(w)
    x = series.to_numpy(dtype=float)
    # convolve(x, w)[t] = sum_k w[k] * x[t-k]; take the causal, same-length part.
    conv = np.convolve(x, w)[:len(x)]
    conv[:width - 1] = np.nan          # warmup rows without a full window
    return pd.Series(conv, index=series.index)


def trailing_percentile(s: pd.Series, window: int) -> pd.Series:
    """
    Percentile rank of each value within its own TRAILING window (inclusive).
    Returns a value in [0, 1]: the fraction of the last `window` observations
    (including today) that are <= today's value. Causal by construction.
    """
    def _last_rank(x: np.ndarray) -> float:
        today = x[-1]
        return float(np.mean(x <= today))
    return s.rolling(window, min_periods=window // 2).apply(_last_rank, raw=True)


def _features_one_ticker(g: pd.DataFrame) -> pd.DataFrame:
    """Compute all price/volume features for a single ticker's sorted frame."""

    g = g.sort_values("date").copy()
    close, high, low, vol = g["close"], g["high"], g["low"], g["volume"]

    # --- Trend: SMA/EMA crossovers (ratio form, scale-free) ---
    sma_fast = ind.sma(close, F.sma_fast)
    sma_slow = ind.sma(close, F.sma_slow)
    ema_fast = ind.ema(close, F.ema_fast)
    ema_slow = ind.ema(close, F.ema_slow)
    g["sma_cross"] = sma_fast / sma_slow - 1.0
    g["ema_cross"] = ema_fast / ema_slow - 1.0

    # --- Trend: MACD ---
    macd_line, macd_sig, macd_hist = ind.macd(
        close, F.macd_slow, F.macd_fast, F.macd_signal)
    # Normalize the price-unit MACD outputs for a pooled cross-sectional model.
    g["macd"] = macd_line / close
    g["macd_signal"] = macd_sig / close
    g["macd_diff"] = macd_hist / close

    # --- Momentum ---
    g["rsi"] = ind.rsi(close, F.rsi_window)
    g["roc"] = ind.roc(close, F.roc_window)
    stoch_k, stoch_d = ind.stochastic(
        high, low, close, F.stoch_window, F.stoch_smooth)
    g["stoch_k"] = stoch_k
    g["stoch_d"] = stoch_d

    # --- Volatility: Bollinger width + %b, and rolling return std ---
    bb_mid, bb_high, bb_low, bb_pctb = ind.bollinger(close, F.bb_window, F.bb_std)
    g["bb_width"] = (bb_high - bb_low) / bb_mid
    g["bb_pctb"] = bb_pctb
    daily_ret = close.pct_change()
    g["volatility"] = daily_ret.rolling(F.vol_window).std()

    # --- Volume ---
    g["vol_ratio"] = vol / vol.rolling(F.volume_window).mean()
    obv = ind.on_balance_volume(close, vol)
    # OBV is non-stationary (a running total); z-score it over a trailing window
    # so the model sees a stable-scale, causal feature.
    obv_mean = obv.rolling(F.volume_window).mean()
    obv_std = obv.rolling(F.volume_window).std()
    g["obv_z"] = (obv - obv_mean) / obv_std

    # --- Fractional differentiation of log price (reduced persistence) ---
    log_close = np.log(close)
    g["fracdiff"] = frac_diff_ffd(
        log_close, F.fracdiff_d, F.fracdiff_thresh, F.fracdiff_max_width)

    return g


def add_features(prices: pd.DataFrame, vix: pd.DataFrame,
                 n_jobs: int = 1) -> pd.DataFrame:
    """
    Given long-format prices (date, ticker, OHLCV) and VIX (date, vix_close),
    return a long-format frame with every FEATURE_COLUMNS column added, plus the
    raw day-t `close` needed for labeling and the backtest. Indicator columns on a
    row dated t are shifted from t-1 so they are available before a day-t close fill.
    """
    # Per-ticker price/volume features. Iterate groups explicitly (rather than
    # groupby.apply) to keep a rolling window from ever bleeding across tickers
    # and to avoid pandas' grouping-column deprecation warning.
    groups = [g for _, g in prices.groupby("ticker", sort=False)]
    if n_jobs == 1:
        frames = [_features_one_ticker(g) for g in groups]
    else:
        # Each ticker is independent, so the work splits cleanly across CPU cores.
        # This matters for the S&P 500 universe (about 500 tickers).
        from joblib import Parallel, delayed
        frames = Parallel(n_jobs=n_jobs)(delayed(_features_one_ticker)(g) for g in groups)
    feats = pd.concat(frames, ignore_index=True)

    # VIX regime features (shared across tickers, keyed by date).
    vix = vix.sort_values("date").copy()
    vix["vix_level"] = vix["vix_close"]
    vix["vix_pctile"] = trailing_percentile(vix["vix_close"], F.vix_pctile_window)
    feats = feats.merge(vix[["date", "vix_level", "vix_pctile"]], on="date", how="left")

    # Drop indicator warmup rows, then shift every model input by one ticker session.
    # The raw close/date remain on day t so labels measure the executable t -> t+1
    # return while X_t was fully known before the day-t close-auction cutoff.
    feats = feats.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    feats = feats.sort_values(["ticker", "date"]).reset_index(drop=True)
    feats[FEATURE_COLUMNS] = feats.groupby("ticker", sort=False)[FEATURE_COLUMNS].shift(1)
    feats = feats.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    return feats


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract the ordered feature matrix as a numpy array."""
    return df[FEATURE_COLUMNS].to_numpy()
