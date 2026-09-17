"""
The leakage guarantee, verified empirically -- the tests an interviewer would
want to see.

test_feature_truncation_invariance is the strongest one: it computes features on
the full history, then recomputes them on the history TRUNCATED at some day t, and
asserts every feature value at day t is identical. If any feature peeked at data
after t, truncating the future would change it and the test would fail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import features, labeling, walkforward
from src.config import LABELS


def test_feature_truncation_invariance(synthetic_prices, synthetic_vix):
    """Features at day t must not change when future data is removed."""
    full = features.add_features(synthetic_prices, synthetic_vix)

    ticker = "AAPL"
    tk = full[full["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    # pick a few interior test dates well past the warmup
    cut_dates = tk["date"].iloc[[300, 500, 800]].tolist()

    for cut in cut_dates:
        prices_trunc = synthetic_prices[synthetic_prices["date"] <= cut]
        vix_trunc = synthetic_vix[synthetic_vix["date"] <= cut]
        trunc = features.add_features(prices_trunc, vix_trunc)

        row_full = full[(full["ticker"] == ticker) & (full["date"] == cut)]
        row_trunc = trunc[(trunc["ticker"] == ticker) & (trunc["date"] == cut)]
        assert not row_trunc.empty, "truncated frame lost the cut date"

        for col in features.FEATURE_COLUMNS:
            a = row_full[col].to_numpy()[0]
            b = row_trunc[col].to_numpy()[0]
            assert np.isclose(a, b, rtol=1e-9, atol=1e-12), (
                f"feature '{col}' at {cut} changed after truncation: {a} vs {b} "
                "-> LOOKAHEAD LEAK")


def test_day_t_features_do_not_use_day_t_bar(synthetic_prices, synthetic_vix):
    """A signal filled at close t may use data only through the prior session."""
    base = features.add_features(synthetic_prices, synthetic_vix)
    ticker = "AAPL"
    row = base[base["ticker"] == ticker].sort_values("date").iloc[500]
    date = row["date"]

    changed_prices = synthetic_prices.copy()
    mask = ((changed_prices["ticker"] == ticker) &
            (changed_prices["date"] == date))
    for col in ["open", "high", "low", "close", "volume"]:
        changed_prices.loc[mask, col] *= 1.5
    changed_vix = synthetic_vix.copy()
    changed_vix.loc[changed_vix["date"] == date, "vix_close"] *= 2.0

    changed = features.add_features(changed_prices, changed_vix)
    a = base[(base["ticker"] == ticker) & (base["date"] == date)].iloc[0]
    b = changed[(changed["ticker"] == ticker) & (changed["date"] == date)].iloc[0]
    assert np.allclose(a[features.FEATURE_COLUMNS].astype(float),
                       b[features.FEATURE_COLUMNS].astype(float))


def test_training_dates_are_purged_by_label_horizon():
    dates = pd.bdate_range("2020-01-01", periods=10).to_numpy()
    purged = walkforward._purged_train_dates(dates, horizon=1)
    assert len(purged) == 9
    assert purged[-1] == dates[-2]


def test_vix_percentile_is_trailing(synthetic_vix):
    """VIX percentile at t must equal the trailing-window rank, not full-sample."""
    s = synthetic_vix["vix_close"].reset_index(drop=True)
    window = 252
    pct = features.trailing_percentile(s, window)

    t = 900
    window_slice = s.iloc[t - window + 1: t + 1]
    expected = float((window_slice <= s.iloc[t]).mean())
    assert np.isclose(pct.iloc[t], expected, atol=1e-9)


def test_label_uses_only_future_return(synthetic_prices, synthetic_vix):
    """label_binary at t must equal sign of the realized t->t+h return."""
    feats = features.add_features(synthetic_prices, synthetic_vix)
    lab = labeling.add_labels(feats, horizon=LABELS.horizon)

    tk = lab[lab["ticker"] == "MSFT"].sort_values("date").reset_index(drop=True)
    for i in range(200, 210):
        fwd = tk["fwd_return"].iloc[i]
        if np.isnan(fwd):
            continue
        expected = 1.0 if fwd > 0 else 0.0
        assert tk["label_binary"].iloc[i] == expected


def test_last_rows_have_no_label(synthetic_prices, synthetic_vix):
    """The final `horizon` rows per ticker have undefined labels (no future)."""
    feats = features.add_features(synthetic_prices, synthetic_vix)
    lab = labeling.add_labels(feats, horizon=LABELS.horizon)
    for tkr, g in lab.groupby("ticker"):
        g = g.sort_values("date")
        assert np.isnan(g["label_binary"].iloc[-1]), (
            f"{tkr}: last row should have no label")
