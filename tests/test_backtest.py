"""
Backtest engine correctness -- hand-checked accounting, not just 'it runs'.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import backtest, evaluation


def test_engine_matches_hand_calculation():
    """Two days, one asset: verify equity, cost, and turnover by hand."""
    dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    weights = pd.DataFrame({"AAPL": [1.0, 1.0]}, index=dates)   # fully long both days
    fwd = pd.DataFrame({"AAPL": [0.02, -0.01]}, index=dates)    # +2% then -1%

    res = backtest.run_backtest(weights, fwd, cost_bps=10.0, initial_capital=1000.0)

    # Day 1: turnover 0->1 = 1.0, cost = 10bps*1 = 0.001, net = 0.02 - 0.001 = 0.019
    assert np.isclose(res.daily["turnover"].iloc[0], 1.0)
    assert np.isclose(res.daily["cost"].iloc[0], 0.001)
    assert np.isclose(res.daily["net_return"].iloc[0], 0.019)
    # Day 2: turnover 1->1 = 0, cost 0, net = -0.01
    assert np.isclose(res.daily["turnover"].iloc[1], 0.0)
    assert np.isclose(res.daily["net_return"].iloc[1], -0.01)
    # Equity = 1000 * 1.019 * 0.99
    assert np.isclose(res.equity.iloc[-1], 1000 * 1.019 * 0.99)


def test_zero_cost_matches_gross():
    dates = pd.bdate_range("2021-01-01", periods=50)
    rng = np.random.default_rng(0)
    weights = pd.DataFrame({"X": np.ones(50)}, index=dates)
    fwd = pd.DataFrame({"X": rng.normal(0, 0.01, 50)}, index=dates)
    res = backtest.run_backtest(weights, fwd, cost_bps=0.0, initial_capital=1.0)
    assert np.allclose(res.daily["net_return"], res.daily["gross_return"])


def test_turnover_uses_drifted_pretrade_weights():
    """Unchanged targets still require rebalancing after assets drift apart."""
    dates = pd.to_datetime(["2021-01-04", "2021-01-05"])
    weights = pd.DataFrame({"AAPL": [0.5, 0.5], "MSFT": [0.5, 0.5]}, index=dates)
    fwd = pd.DataFrame({"AAPL": [0.10, 0.0], "MSFT": [0.0, 0.0]}, index=dates)
    res = backtest.run_backtest(weights, fwd, cost_bps=0.0, initial_capital=1.0)

    drifted_aapl = 0.5 * 1.10 / 1.05
    drifted_msft = 0.5 / 1.05
    expected = abs(0.5 - drifted_aapl) + abs(0.5 - drifted_msft)
    assert np.isclose(res.daily["turnover"].iloc[1], expected)


def test_buy_and_hold_weights_drift_without_rebalancing():
    dates = pd.to_datetime(["2021-01-04", "2021-01-05"])
    fwd = pd.DataFrame({"AAPL": [0.10, 0.0], "MSFT": [0.0, 0.0]}, index=dates)
    weights = backtest.buy_and_hold_weights(fwd)
    res = backtest.run_backtest(weights, fwd, cost_bps=10.0, initial_capital=1.0)

    assert weights["AAPL"].iloc[1] > 0.5
    assert np.isclose(res.daily["turnover"].iloc[0], 1.0)
    assert np.isclose(res.daily["turnover"].iloc[1], 0.0, atol=1e-12)


def test_no_position_no_return():
    dates = pd.bdate_range("2021-01-01", periods=10)
    weights = pd.DataFrame({"X": np.zeros(10)}, index=dates)
    fwd = pd.DataFrame({"X": np.full(10, 0.05)}, index=dates)
    res = backtest.run_backtest(weights, fwd, cost_bps=10.0, initial_capital=1.0)
    assert np.isclose(res.equity.iloc[-1], 1.0)  # never invested => flat


def test_metrics_signs():
    """Sanity: a steadily rising equity has positive Sharpe and small drawdown."""
    eq = pd.Series(np.linspace(1.0, 2.0, 300))
    rets = eq.pct_change().dropna()
    assert evaluation.sharpe(rets) > 0
    assert evaluation.max_drawdown(eq) >= -1e-9  # monotonic up => ~0 drawdown


def test_metrics_include_first_return_and_initial_capital():
    dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    weights = pd.DataFrame({"AAPL": [1.0, 1.0]}, index=dates)
    fwd = pd.DataFrame({"AAPL": [-0.5, 0.0]}, index=dates)
    res = backtest.run_backtest(weights, fwd, cost_bps=0.0, initial_capital=1.0)
    summary = evaluation.performance_summary(res)

    assert np.isclose(summary["total_return"], -0.5)
    assert np.isclose(summary["max_drawdown"], -0.5)
    assert summary["CAGR"] < 0


def test_default_cost_is_read_at_call_time(monkeypatch):
    dates = pd.to_datetime(["2020-01-02"])
    weights = pd.DataFrame({"AAPL": [1.0]}, index=dates)
    fwd = pd.DataFrame({"AAPL": [0.0]}, index=dates)
    monkeypatch.setattr(backtest.B, "cost_bps", 0.0)
    assert backtest.run_backtest(weights, fwd).daily["cost"].iloc[0] == 0.0
