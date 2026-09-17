"""
Central configuration for the trading-strategy classifier project.

Everything tunable lives here so the rest of the codebase reads as pure logic.
Nothing in this file pulls data or does work at import time.

Research question (frame everything against THIS, not "can we predict prices"):
    Does a technical-indicator-based ML signal generate risk-adjusted returns
    above a naive benchmark, after realistic transaction costs, and does that
    edge hold up across different volatility regimes?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# ------------------------------------------------------------------
# Universe: 15 liquid large-caps across three sectors + benchmark + VIX.
# A single-ticker backtest reads as luck; a diversified universe reads as a
# strategy. Sectors are labeled so the app / analysis can group by them.
# ------------------------------------------------------------------
UNIVERSE: Dict[str, str] = {
    # Technology
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "GOOGL": "Technology",
    "AMZN": "Technology",
    "META": "Technology",
    # Financials
    "JPM": "Financials",
    "GS": "Financials",
    "BAC": "Financials",
    "MS": "Financials",
    # Consumer staples / discretionary
    "PG": "Consumer",
    "KO": "Consumer",
    "WMT": "Consumer",
    "HD": "Consumer",
    "COST": "Consumer",
}

TICKERS: List[str] = list(UNIVERSE.keys())

MARKET_BENCHMARK = "SPY"   # market-level buy-and-hold reference
VIX_SYMBOL = "^VIX"        # volatility regime input

# Date range spans multiple regimes on purpose:
#   2015-2019 bull, 2020 COVID crash, 2022 rate-hike bear, 2023-2025 recovery.
START_DATE = "2014-01-01"
END_DATE = "2026-07-01"

# On-disk cache (parquet). Anchor it to the project rather than the process'
# current working directory so CLI and Streamlit launches behave identically.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = str(PROJECT_ROOT / "data")
PRICES_FILE = "prices.parquet"     # long format: date, ticker, OHLCV
VIX_FILE = "vix.parquet"


@dataclass
class FeatureConfig:
    """Windows for the technical indicators. All are backward-looking."""
    sma_fast: int = 10
    sma_slow: int = 50
    ema_fast: int = 20
    ema_slow: int = 100
    rsi_window: int = 14
    roc_window: int = 10
    stoch_window: int = 14
    stoch_smooth: int = 3
    bb_window: int = 20
    bb_std: float = 2.0
    vol_window: int = 20          # rolling std of returns
    volume_window: int = 20       # rolling avg volume
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    vix_pctile_window: int = 252  # trailing-year percentile rank (~1 trading yr)
    # Fractional differentiation: reduce log-price persistence while preserving
    # more level information than integer differencing (plain returns).
    fracdiff_d: float = 0.4       # differencing order in (0, 1)
    fracdiff_thresh: float = 1e-3  # drop filter weights below this magnitude
    fracdiff_max_width: int = 60   # hard cap on the fixed-width window


@dataclass
class LabelConfig:
    """Target design. horizon=1 => predict next-day direction."""
    horizon: int = 1
    # 3-class deadband: |next-day return| < flat_threshold => "flat" (class 0).
    flat_threshold: float = 0.005  # +/- 0.5%


@dataclass
class WalkForwardConfig:
    """Rolling-origin evaluation. Sizes are in trading days (~252/yr)."""
    train_days: int = 504    # ~2 years rolling train window
    test_days: int = 126     # ~6 months out-of-sample test
    step_days: int = 126     # roll forward by the test length (non-overlapping tests)
    min_train_days: int = 252


@dataclass
class BacktestConfig:
    """
    Backtest / execution assumptions.

    Timing convention (the crux of leakage-safety):
      - Features X_t contain information through the close of day t-1.
      - The model predicts the sign of the t -> t+1 return.
      - The signal is therefore known before the day-t close-auction cutoff.
      - We enter at the close of day t and earn the t -> t+1 return.
    """
    initial_capital: float = 100_000.0
    cost_bps: float = 7.5          # round-number cost+slippage per trade, in bps
    max_gross_exposure: float = 1.0  # fully invested when all names signal long


@dataclass
class RegimeConfig:
    """VIX-level regime cutoffs for the differentiating regime analysis."""
    calm_max: float = 20.0     # VIX < 20  => calm
    stressed_min: float = 30.0  # VIX > 30  => stressed
    # 20 <= VIX <= 30 is the "normal" middle regime.


@dataclass
class TuningConfig:
    """Hyperparameter search run inside each walk forward training window."""
    inner_splits: int = 3          # time ordered validation splits per window
    search_trees: int = 100        # smaller forests during the search, full size on refit
    rf_grid: Dict[str, list] = field(default_factory=lambda: {
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [50, 200],
    })
    logreg_grid: Dict[str, list] = field(default_factory=lambda: {
        "C": [0.01, 0.1, 1.0],
    })


# Results written by scripts/run_experiments.py and read by the dashboard.
RESULTS_DIR = str(PROJECT_ROOT / "results")

# Large universe (S&P 500) storage: a Hive partitioned Parquet dataset
# (data/sp500/ticker=AAPL/year=2020/...). Kept out of git because of its size.
LARGE_DATA_DIR = str(PROJECT_ROOT / "data" / "sp500")
LARGE_TICKER_FILE = str(PROJECT_ROOT / "data" / "sp500_tickers.csv")


FEATURES = FeatureConfig()
LABELS = LabelConfig()
WALK = WalkForwardConfig()
BACKTEST = BacktestConfig()
REGIME = RegimeConfig()
TUNING = TuningConfig()

RANDOM_STATE = 42
