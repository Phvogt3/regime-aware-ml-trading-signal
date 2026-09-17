"""
End-to-end pipeline glue: prices+VIX -> features -> labels -> walk-forward
predictions -> backtest -> evaluation + regime analysis.

Kept import-only (no CLI here) so both the orchestration script and the Streamlit
app call the exact same code path -- the app is a live view of the research
pipeline, not a reimplementation of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd

from . import (backtest, benchmarks, config, evaluation, features,
               labeling, regime, walkforward)


@dataclass
class PipelineOutput:
    features: pd.DataFrame
    predictions: pd.DataFrame
    results: Dict[str, object]            # label -> BacktestResult (ml + benchmarks)
    summary: pd.DataFrame                 # primary metrics table
    regime_table: pd.DataFrame            # ml strategy, per VIX regime
    classification: Dict[str, float]      # secondary accuracy diagnostics
    importances: pd.DataFrame
    cost_drag: float
    model_name: str
    target: str
    fold_table: pd.DataFrame = None       # per walk forward window: AUC, tuned params


def build_features(prices: pd.DataFrame, vix: pd.DataFrame,
                   tickers: Optional[List[str]] = None,
                   n_jobs: int = 1) -> pd.DataFrame:
    # SPY is cached as an optional market reference, but it is not part of the
    # stated 15-equity modeling/trading universe and must not enter training or
    # classification metrics.
    model_tickers = list(tickers) if tickers is not None else list(config.TICKERS)
    if not model_tickers:
        raise ValueError("at least one model ticker is required")
    prices = prices[prices["ticker"].isin(model_tickers)]
    if prices.empty:
        raise ValueError("none of the requested model tickers are present in prices")
    return features.add_features(prices, vix, n_jobs=n_jobs)


def run(prices: pd.DataFrame, vix: pd.DataFrame,
        model_name: str = "rf", target: str = "binary",
        tickers: Optional[List[str]] = None,
        start: Optional[str] = None, end: Optional[str] = None,
        cost_bps: Optional[float] = None,
        feature_columns: Optional[Sequence[str]] = None,
        tune: bool = False,
        pca_components: Optional[Union[int, float]] = None,
        n_jobs: int = 1,
        feats: Optional[pd.DataFrame] = None) -> PipelineOutput:
    """
    feature_columns, tune, and pca_components are passed to walk forward validation.
    feats lets experiment scripts compute features once and reuse them across runs.
    """
    if feats is None:
        feats = build_features(prices, vix, tickers=tickers, n_jobs=n_jobs)
    elif tickers is not None:
        feats = feats[feats["ticker"].isin(list(tickers))]
    if start is not None:
        feats = feats[feats["date"] >= pd.Timestamp(start)]
    if end is not None:
        feats = feats[feats["date"] <= pd.Timestamp(end)]

    # Label only after applying the requested date range. This prevents the final
    # in-range observation from earning a return realized beyond the selected end.
    feats = labeling.add_labels(feats)

    # Walk-forward out-of-sample predictions.
    labeled = labeling.labeled_frame(feats, target=target)
    if labeled["date"].nunique() <= config.WALK.train_days:
        raise ValueError(
            f"Selected range has {labeled['date'].nunique()} usable dates; choose more "
            f"than {config.WALK.train_days} for walk-forward evaluation.")
    wf = walkforward.run_walk_forward(labeled, model_name=model_name, target=target,
                                      feature_columns=feature_columns, tune=tune,
                                      pca_components=pca_components)

    # ML backtest on OOS predictions.
    fwd = backtest.fwd_return_matrix(wf.predictions)
    ml_w = backtest.ml_weights(wf.predictions, target=target)
    ml_res = backtest.run_backtest(ml_w, fwd, cost_bps=cost_bps)

    # Benchmarks over the same OOS date span for a fair comparison.
    oos_feats = feats[feats["date"].isin(wf.predictions["date"].unique())]
    bh_res = benchmarks.buy_and_hold(oos_feats, cost_bps=cost_bps)
    ma_res = benchmarks.ma_crossover(oos_feats, cost_bps=cost_bps)

    results = {"ML strategy": ml_res, "Buy & hold": bh_res, "MA crossover": ma_res}
    summary = evaluation.summary_table(results)
    regime_table = regime.regime_comparison(results, vix)
    classification = evaluation.classification_metrics(wf.predictions)
    drag = evaluation.cost_drag(ml_res)

    return PipelineOutput(
        features=feats, predictions=wf.predictions, results=results,
        summary=summary, regime_table=regime_table, classification=classification,
        importances=wf.importances, cost_drag=drag,
        model_name=model_name, target=target, fold_table=wf.fold_table,
    )
