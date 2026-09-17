"""
Tests for the model checks: ROC-AUC, feature subsets (ablation), PCA inside walk
forward, time ordered tuning splits, and the partitioned S&P 500 storage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, evaluation, experiments, labeling, pipeline, walkforward
from src.features import FEATURE_COLUMNS, FEATURE_GROUPS


@pytest.fixture
def short_windows(monkeypatch):
    monkeypatch.setattr(config.WALK, "train_days", 252)
    monkeypatch.setattr(config.WALK, "test_days", 126)
    monkeypatch.setattr(config.WALK, "step_days", 126)
    monkeypatch.setattr(config.WALK, "min_train_days", 200)


def test_inner_splits_validate_on_later_dates_with_purge():
    dates = np.repeat(pd.bdate_range("2020-01-01", periods=100).to_numpy(), 3)
    for tr_idx, va_idx in walkforward.inner_date_splits(dates, n_splits=3, gap=1):
        train_last = dates[tr_idx].max()
        val_first = dates[va_idx].min()
        assert train_last < val_first
        # one purged date between the end of training and the start of validation
        gap_days = np.unique(dates[(dates > train_last) & (dates < val_first)])
        assert len(gap_days) == 1
        # a date's cross section is never split between train and validation
        assert not set(dates[tr_idx]) & set(dates[va_idx])


def test_roc_auc_perfect_and_random():
    y = pd.Series([0, 1, 0, 1, 1, 0])
    perfect = pd.DataFrame({"y_true": y, "pred_class": y, "prob_up": y * 0.8 + 0.1})
    assert evaluation.classification_metrics(perfect)["roc_auc"] == pytest.approx(1.0)
    flipped = perfect.assign(prob_up=1 - perfect["prob_up"])
    assert evaluation.classification_metrics(flipped)["roc_auc"] == pytest.approx(0.0)


def test_feature_groups_cover_every_feature_once():
    cols = sum(FEATURE_GROUPS.values(), [])
    assert sorted(cols) == sorted(FEATURE_COLUMNS)
    assert len(cols) == len(set(cols))


def test_feature_subset_is_the_only_input(synthetic_prices, synthetic_vix, short_windows):
    feats = pipeline.build_features(synthetic_prices, synthetic_vix)
    labeled = labeling.labeled_frame(labeling.add_labels(feats), target="binary")
    subset = ["rsi", "roc"]
    wf = walkforward.run_walk_forward(labeled, "logreg", feature_columns=subset)
    assert set(wf.importances["feature"]) == set(subset)
    with pytest.raises(ValueError):
        walkforward.run_walk_forward(labeled, "logreg", feature_columns=["not_a_feature"])


def test_pca_and_tuning_run_inside_walk_forward(synthetic_prices, synthetic_vix,
                                                short_windows):
    feats = pipeline.build_features(synthetic_prices, synthetic_vix)
    out = pipeline.run(synthetic_prices, synthetic_vix, model_name="logreg",
                       feats=feats, pca_components=5, tune=True)
    ft = out.fold_table
    assert (ft["pca_components"] == 5).all()
    assert "param_C" in ft and ft["param_C"].isin(config.TUNING.logreg_grid["C"]).all()
    assert ft["test_auc"].between(0, 1).all()
    assert 0 <= out.classification["roc_auc"] <= 1


def test_parallel_features_match_serial(synthetic_prices, synthetic_vix):
    serial = pipeline.build_features(synthetic_prices, synthetic_vix, n_jobs=1)
    parallel = pipeline.build_features(synthetic_prices, synthetic_vix, n_jobs=2)
    key = ["ticker", "date"]
    pd.testing.assert_frame_equal(serial.sort_values(key).reset_index(drop=True),
                                  parallel.sort_values(key).reset_index(drop=True))


def test_partitioned_storage_roundtrip_and_pruning(synthetic_prices, tmp_path, monkeypatch):
    from src import large_data
    monkeypatch.setattr(config, "LARGE_DATA_DIR", str(tmp_path / "sp500"))
    sample = synthetic_prices[synthetic_prices["ticker"].isin(["AAPL", "JPM"])]
    large_data.write_partitions(sample, root=config.LARGE_DATA_DIR)
    assert (tmp_path / "sp500" / "ticker=AAPL" / "year=2020").is_dir()

    back = large_data.load_prices()
    assert len(back) == len(sample)
    only = large_data.load_prices(tickers=["JPM"], start_year=2020, end_year=2021)
    assert set(only["ticker"]) == {"JPM"}
    assert only["date"].dt.year.between(2020, 2021).all()


def test_backtest_trades_the_universe_it_is_given():
    """
    Tickers outside config.TICKERS must still be traded. This was a real bug: the
    weight builders filtered to the 15 default names, so a 500 stock run traded
    only those 15 and its benchmarks did too.
    """
    from src import backtest

    dates = pd.bdate_range("2021-01-01", periods=4)
    preds = pd.DataFrame({
        "date": list(dates) * 2,
        "ticker": ["AAPL"] * 4 + ["ZZZZ"] * 4,     # ZZZZ is not in config.TICKERS
        "pred_class": [1, 1, 0, 1, 1, 0, 1, 1],
        "fwd_return": [0.01, -0.01, 0.02, 0.0] * 2,
    })
    w = backtest.ml_weights(preds)
    assert list(w.columns) == ["AAPL", "ZZZZ"]
    assert w.to_numpy().max() == pytest.approx(0.5)      # 1 / 2 names, not 1 / 15

    fwd = backtest.fwd_return_matrix(preds)
    assert list(fwd.columns) == ["AAPL", "ZZZZ"]
    bh = backtest.buy_and_hold_weights(fwd)
    assert bh.iloc[0].sum() == pytest.approx(1.0)

    # An explicit universe still restricts what can be held.
    only_aapl = backtest.ml_weights(preds, universe=["AAPL"])
    assert list(only_aapl.columns) == ["AAPL"]


def test_ablation_variants_drop_exactly_one_group():
    variants = experiments.ablation_variants()
    assert len(variants["All features"]) == len(FEATURE_COLUMNS)
    for group, cols in FEATURE_GROUPS.items():
        kept = variants[f"Without {group.replace('_', ' ')}"]
        assert not set(kept) & set(cols)
        assert len(kept) == len(FEATURE_COLUMNS) - len(cols)
