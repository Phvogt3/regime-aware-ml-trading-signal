"""
Custom day-by-day backtest engine (no black-box backtesting library).

The engine is intentionally signal-agnostic: it takes a matrix of target weights
(date x ticker) and a matrix of realized forward returns, and simulates a
portfolio. The ML strategy, buy-and-hold, and the moving-average crossover are all
just different weight matrices fed to the SAME engine -- which is both DRY and a
clean demonstration that the accounting is identical across strategies.

Timing / no-lookahead contract
------------------------------
`weights.loc[t, i]` is submitted for the close of day t using information
through day t-1. `fwd_returns.loc[t, i]` is the return that position earns from the
close of t to the close of t+1 (i.e. close_{t+1}/close_t - 1). The engine multiplies
the two, so the signal is both causal and available before its assumed fill.

Position sizing
---------------
Fixed fraction of capital per position, equal-weighted across the universe:
each name gets weight max_gross / N_universe when signaled, 0 otherwise. Gross
exposure therefore rises and falls with how many names are active; unused capital
sits in cash earning nothing (a conservative, honest assumption).

Costs
-----
Turnover_t compares target weights with the actual pre-trade weights after the
previous session's market drift. Cost_t = cost_bps/1e4 * Turnover_t, charged against
the step return.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from . import config
from .config import BACKTEST as B


@dataclass
class BacktestResult:
    daily: pd.DataFrame     # index=date; gross_return, cost, net_return, turnover, exposure, equity
    initial_capital: float
    cost_bps: float

    @property
    def equity(self) -> pd.Series:
        return self.daily["equity"]

    @property
    def net_returns(self) -> pd.Series:
        return self.daily["net_return"]


def align(weights: pd.DataFrame, fwd_returns: pd.DataFrame):
    """Align two wide frames on a common (date x ticker) grid, filling zeros."""
    cols = sorted(set(weights.columns) | set(fwd_returns.columns))
    idx = sorted(set(weights.index) | set(fwd_returns.index))
    w = weights.reindex(index=idx, columns=cols).fillna(0.0)
    r = fwd_returns.reindex(index=idx, columns=cols).fillna(0.0)
    return w, r


def run_backtest(weights: pd.DataFrame, fwd_returns: pd.DataFrame,
                 cost_bps: Optional[float] = None,
                 initial_capital: Optional[float] = None) -> BacktestResult:
    """
    Simulate the portfolio day by day. Returns a BacktestResult whose `daily`
    frame is indexed by decision date t (equity[t] = capital after the position
    entered at t's close is marked at t+1's close).
    """
    cost_bps = B.cost_bps if cost_bps is None else float(cost_bps)
    initial_capital = B.initial_capital if initial_capital is None else float(initial_capital)
    if cost_bps < 0:
        raise ValueError("cost_bps must be non-negative")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")

    w, r = align(weights, fwd_returns)
    dates = w.index
    cost_rate = cost_bps / 1e4

    # Actual weights immediately before the next rebalance. These drift after each
    # return; comparing only with the previous target would understate turnover.
    pretrade_w = np.zeros(w.shape[1])
    equity = initial_capital
    records = []

    for t in dates:
        w_t = w.loc[t].to_numpy()
        r_t = r.loc[t].to_numpy()

        turnover = float(np.abs(w_t - pretrade_w).sum())
        cost = cost_rate * turnover
        gross = float(np.dot(w_t, r_t))
        net = gross - cost
        if net <= -1.0:
            raise ValueError(f"portfolio lost 100% or more on {t}")
        equity *= (1.0 + net)

        records.append({
            "date": t,
            "gross_return": gross,
            "cost": cost,
            "net_return": net,
            "turnover": turnover,
            "exposure": float(np.abs(w_t).sum()),
            "equity": equity,
        })

        # Mark target holdings through the realized return to obtain the weights
        # actually present before the next trade. Cost is deducted proportionally
        # from portfolio equity, so it does not change relative post-return weights.
        gross_growth = 1.0 + gross
        if gross_growth <= 0:
            raise ValueError(f"gross portfolio value became non-positive on {t}")
        pretrade_w = w_t * (1.0 + r_t) / gross_growth

    daily = pd.DataFrame.from_records(records).set_index("date")
    return BacktestResult(daily=daily, initial_capital=initial_capital, cost_bps=cost_bps)


# ------------------------------------------------------------------
# Weight-matrix builders (each returns a wide date x ticker frame).
# ------------------------------------------------------------------

def _universe_size(tickers) -> int:
    present = [t for t in tickers if t in config.TICKERS]
    return max(len(present), 1)


def fwd_return_matrix(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot a long frame with ['date','ticker','fwd_return'] to wide."""
    tickers = long_df.loc[long_df["ticker"].isin(config.TICKERS)]
    return tickers.pivot_table(index="date", columns="ticker", values="fwd_return")


def _long_indicator_to_weights(is_long: pd.DataFrame) -> pd.DataFrame:
    """Map a boolean long/flat matrix to fixed-fraction equal weights."""
    n = _universe_size(is_long.columns)
    per_name = B.max_gross_exposure / n
    return is_long.astype(float) * per_name


def ml_weights(predictions: pd.DataFrame, target: str = "binary") -> pd.DataFrame:
    """
    Build target weights from walk-forward predictions. Long when the predicted
    class is 'up' (binary: class 1; 3-class: class 2), flat otherwise. Shorting
    is off by default (long/flat baseline); enable via BacktestConfig.
    """
    preds = predictions[predictions["ticker"].isin(config.TICKERS)].copy()
    if target == "binary":
        preds["is_long"] = preds["pred_class"] == 1
    else:  # 3-class: up == 2
        preds["is_long"] = preds["pred_class"] == 2
    is_long = preds.pivot_table(index="date", columns="ticker",
                                values="is_long", fill_value=False).astype(bool)
    return _long_indicator_to_weights(is_long)


def buy_and_hold_weights(fwd_returns: pd.DataFrame) -> pd.DataFrame:
    """Initial equal-weight purchase whose weights subsequently drift without trades."""
    if fwd_returns.empty:
        return fwd_returns.copy()

    returns = fwd_returns.fillna(0.0)
    first_available = fwd_returns.iloc[0].notna()
    eligible = [c for c in fwd_returns.columns
                if c in config.TICKERS and bool(first_available[c])]
    n = max(len(eligible), 1)
    current = pd.Series(0.0, index=fwd_returns.columns)
    current.loc[eligible] = B.max_gross_exposure / n
    rows = []

    for date in fwd_returns.index:
        rows.append(current.rename(date))
        r_t = returns.loc[date]
        gross = float(np.dot(current.to_numpy(), r_t.to_numpy()))
        if 1.0 + gross <= 0:
            raise ValueError(f"buy-and-hold portfolio became non-positive on {date}")
        current = current * (1.0 + r_t) / (1.0 + gross)

    return pd.DataFrame(rows, index=fwd_returns.index, columns=fwd_returns.columns)


def ma_crossover_weights(features: pd.DataFrame) -> pd.DataFrame:
    """
    Naive rules benchmark: long when the fast SMA is above the slow SMA
    (sma_cross > 0), flat otherwise. Same sizing and costs as the ML strategy,
    so the comparison is apples-to-apples.
    """
    f = features[features["ticker"].isin(config.TICKERS)].copy()
    f["is_long"] = f["sma_cross"] > 0
    is_long = f.pivot_table(index="date", columns="ticker",
                            values="is_long", fill_value=False).astype(bool)
    return _long_indicator_to_weights(is_long)
