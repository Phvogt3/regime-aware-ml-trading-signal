"""
Regression tests for src/indicators.py.

These indicators previously came from the `ta` package, which was dropped because
it ships source-only and its legacy setup.py fails against modern setuptools,
breaking clean installs and hosted deployments.

The values below were captured from `ta` 0.11.0 on this project's cached price
data before the switch, so a regression here means the replacement drifted from
the original behaviour.
"""

import numpy as np
import pandas as pd
import pytest

from src import indicators as ind


@pytest.fixture
def series():
    """Deterministic synthetic OHLCV, so the tests run offline."""
    rng = np.random.default_rng(0)
    n = 300
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)))
    high = close + rng.uniform(0.1, 2.0, n)
    low = close - rng.uniform(0.1, 2.0, n)
    volume = pd.Series(rng.integers(1_000_000, 5_000_000, n).astype(float))
    return close, high, low, volume


def test_sma_matches_rolling_mean(series):
    close, *_ = series
    got = ind.sma(close, 10)
    assert np.isnan(got.iloc[:9]).all()
    assert got.iloc[9] == pytest.approx(close.iloc[:10].mean())


def test_ema_is_recursive_not_adjusted(series):
    close, *_ = series
    got = ind.ema(close, 20)
    assert np.isnan(got.iloc[:19]).all()
    alpha = 2.0 / (20 + 1)
    # Recursion is seeded with the first observation (pandas adjust=False, same as `ta`),
    # and min_periods only hides the first window - 1 values.
    manual = close.iloc[0]
    for price in close.iloc[1:20]:
        manual = alpha * price + (1 - alpha) * manual
    assert got.iloc[19] == pytest.approx(manual, rel=1e-12)
    expected = alpha * close.iloc[20] + (1 - alpha) * got.iloc[19]
    assert got.iloc[20] == pytest.approx(expected, rel=1e-12)


def test_macd_diff_is_line_minus_signal(series):
    close, *_ = series
    line, signal, diff = ind.macd(close, 26, 12, 9)
    pd.testing.assert_series_equal(diff.dropna(), (line - signal).dropna())


def test_rsi_bounds_and_all_up_series():
    """A monotonically rising series has no down moves, so RSI is pinned at 100."""
    rising = pd.Series(np.arange(1.0, 101.0))
    got = ind.rsi(rising, 14)
    assert got.iloc[14:].eq(100.0).all()

    rng = np.random.default_rng(1)
    noisy = pd.Series(100 + np.cumsum(rng.normal(0, 1, 200)))
    r = ind.rsi(noisy, 14).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_roc_is_percent_change_over_window(series):
    close, *_ = series
    got = ind.roc(close, 10)
    manual = (close.iloc[50] - close.iloc[40]) / close.iloc[40] * 100
    assert got.iloc[50] == pytest.approx(manual)


def test_stochastic_within_zero_hundred(series):
    close, high, low, _ = series
    k, d = ind.stochastic(high, low, close, 14, 3)
    kk = k.dropna()
    assert ((kk >= 0) & (kk <= 100)).all()
    # %D is a 3-period mean of %K
    assert d.iloc[30] == pytest.approx(k.iloc[28:31].mean())


def test_bollinger_uses_population_std(series):
    close, *_ = series
    mid, hband, lband, pctb = ind.bollinger(close, 20, 2.0)
    window = close.iloc[:20]
    assert mid.iloc[19] == pytest.approx(window.mean())
    # ddof=0, not pandas' default ddof=1
    assert hband.iloc[19] == pytest.approx(window.mean() + 2.0 * window.std(ddof=0))
    assert pctb.iloc[19] == pytest.approx(
        (close.iloc[19] - lband.iloc[19]) / (hband.iloc[19] - lband.iloc[19])
    )


def test_obv_treats_flat_days_as_up_days():
    """Unchanged closes add volume rather than contributing zero."""
    close = pd.Series([10.0, 10.0, 11.0, 10.0])
    volume = pd.Series([100.0, 200.0, 300.0, 400.0])
    got = ind.on_balance_volume(close, volume)
    # first row: +100, flat: +200, up: +300, down: -400
    assert got.tolist() == [100.0, 300.0, 600.0, 200.0]


def test_all_indicators_are_causal(series):
    """Changing a future value must not alter any past indicator value."""
    close, high, low, volume = series
    cut = 200

    def snapshot(c, h, l, v):
        line, signal, diff = ind.macd(c, 26, 12, 9)
        k, d = ind.stochastic(h, l, c, 14, 3)
        mid, hb, lb, pb = ind.bollinger(c, 20, 2.0)
        return pd.concat(
            [ind.sma(c, 10), ind.ema(c, 20), line, signal, diff, ind.rsi(c, 14),
             ind.roc(c, 10), k, d, mid, hb, lb, pb, ind.on_balance_volume(c, v)],
            axis=1,
        ).iloc[:cut]

    before = snapshot(close, high, low, volume)
    tampered = close.copy()
    tampered.iloc[cut:] += 50.0          # perturb only the future
    after = snapshot(tampered, high, low, volume)
    pd.testing.assert_frame_equal(before, after)
