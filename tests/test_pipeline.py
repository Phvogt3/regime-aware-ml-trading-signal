"""
End-to-end smoke test: the whole pipeline runs on synthetic data and produces
well-formed outputs. Uses shortened walk-forward windows so it's fast.
"""

from __future__ import annotations

import numpy as np

from src import config, pipeline
from src.features import FEATURE_COLUMNS


def test_features_have_no_nan_and_expected_columns(synthetic_prices, synthetic_vix):
    feats = pipeline.build_features(synthetic_prices, synthetic_vix)
    assert "SPY" not in set(feats["ticker"])
    for col in FEATURE_COLUMNS:
        assert col in feats.columns
        assert feats[col].notna().all(), f"{col} has NaNs after warmup drop"


def test_full_pipeline_runs(synthetic_prices, synthetic_vix, monkeypatch):
    # shrink walk-forward windows so the smoke test is quick
    monkeypatch.setattr(config.WALK, "train_days", 252)
    monkeypatch.setattr(config.WALK, "test_days", 63)
    monkeypatch.setattr(config.WALK, "step_days", 63)
    monkeypatch.setattr(config.WALK, "min_train_days", 200)

    out = pipeline.run(synthetic_prices, synthetic_vix,
                       model_name="logreg", target="binary", cost_bps=3.0)

    # predictions are out-of-sample and non-empty
    assert not out.predictions.empty
    # summary has all three strategies with the primary metrics
    for strat in ["ML strategy", "Buy & hold", "MA crossover"]:
        assert strat in out.summary.index
    for metric in ["sharpe", "sortino", "max_drawdown", "win_rate"]:
        assert metric in out.summary.columns
    # regime table has the three regimes
    assert set(out.regime_table.index) <= {"calm", "normal", "stressed"}
    # accuracy reported in [0,1]
    assert 0.0 <= out.classification["accuracy"] <= 1.0
    # cost drag is a finite fraction
    assert np.isfinite(out.cost_drag)
    assert out.classification["majority_baseline"] >= 0.5
    assert all(result.cost_bps == 3.0 for result in out.results.values())
    assert "buy_hold_sharpe" in out.regime_table.columns
