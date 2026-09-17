"""
Benchmark strategies the ML model must beat, in increasing order of rigor:

    1. buy_and_hold : equal-weight, always fully invested across the universe.
       Beating the market is table stakes; a strategy that can't beat holding the
       basket is not interesting.
    2. ma_crossover : a naive rules strategy (fast SMA above slow SMA => long).
       Costs nothing to compute. If the ML model can't beat this, the ML is not
       adding value over a one-line trading rule.

Both run through the SAME custom engine as the ML strategy, so any performance
gap is due to the signal, not to accounting differences.
"""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from . import backtest
from .backtest import BacktestResult


def buy_and_hold(features: pd.DataFrame,
                 cost_bps: Optional[float] = None,
                 universe: Optional[Sequence[str]] = None) -> BacktestResult:
    fwd = backtest.fwd_return_matrix(features, universe=universe)
    w = backtest.buy_and_hold_weights(fwd, universe=universe)
    return backtest.run_backtest(w, fwd, cost_bps=cost_bps)


def ma_crossover(features: pd.DataFrame,
                 cost_bps: Optional[float] = None,
                 universe: Optional[Sequence[str]] = None) -> BacktestResult:
    fwd = backtest.fwd_return_matrix(features, universe=universe)
    w = backtest.ma_crossover_weights(features, universe=universe)
    return backtest.run_backtest(w, fwd, cost_bps=cost_bps)
