"""
Regime analysis -- the differentiating piece.

The interesting, defensible question is not "does it work?" but "WHEN does it
work?". We split the full backtest into volatility regimes by VIX level and report
performance separately in each:

    calm      : VIX <  calm_max      (default 20)
    normal    : calm_max <= VIX <= stressed_min
    stressed  : VIX >  stressed_min  (default 30)

Regime is assigned using the most recent VIX close strictly before the position's
day-t close fill. This matches the pipeline's one-session feature lag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import evaluation
from .config import REGIME as R


def label_regime(vix_value: float) -> str:
    if np.isnan(vix_value):
        return "unknown"
    if vix_value < R.calm_max:
        return "calm"
    if vix_value > R.stressed_min:
        return "stressed"
    return "normal"


def attach_regime(daily: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """Attach the most recent VIX observation available before each trade date."""
    out = daily.sort_index().copy()
    left = pd.DataFrame({"date": pd.to_datetime(out.index)})
    right = (vix[["date", "vix_close"]].drop_duplicates("date")
             .sort_values("date").copy())
    right["date"] = pd.to_datetime(right["date"])
    matched = pd.merge_asof(
        left.sort_values("date"), right, on="date", direction="backward",
        allow_exact_matches=False)
    v = pd.Series(matched["vix_close"].to_numpy(), index=matched["date"])
    out["vix"] = pd.to_datetime(out.index).map(v)
    out["regime"] = out["vix"].map(label_regime)
    return out


def regime_performance(result, vix: pd.DataFrame) -> pd.DataFrame:
    """
    Per-regime PRIMARY metrics computed on the subset of daily net returns whose
    decision-date VIX falls in each regime. Regimes are non-contiguous, so we
    report annualized return/vol, Sharpe, Sortino, win rate, and cumulative
    within-regime return rather than a peak-to-trough drawdown.
    """
    daily = attach_regime(result.daily, vix)
    rows = []
    for regime in ["calm", "normal", "stressed"]:
        sub = daily[daily["regime"] == regime]
        r = sub["net_return"]
        if len(r) < 2:
            rows.append({"regime": regime, "n_days": len(sub)})
            continue
        ws = evaluation.win_stats(r)
        rows.append({
            "regime": regime,
            "n_days": int(len(sub)),
            "share_of_days": float(len(sub) / len(daily)),
            "ann_return": float(r.mean() * evaluation.TRADING_DAYS),
            "ann_vol": evaluation.ann_vol(r),
            "sharpe": evaluation.sharpe(r),
            "sortino": evaluation.sortino(r),
            "win_rate": ws["win_rate"],
            "cum_return": float((1 + r).prod() - 1.0),
            "avg_vix": float(sub["vix"].mean()),
        })
    return pd.DataFrame(rows).set_index("regime")


def regime_comparison(results: dict, vix: pd.DataFrame) -> pd.DataFrame:
    """ML regime metrics plus like-for-like benchmark and active-return comparisons."""
    ml = results["ML strategy"]
    table = regime_performance(ml, vix)
    ml_daily = attach_regime(ml.daily, vix)

    benchmark_columns = {
        "Buy & hold": "buy_hold",
        "MA crossover": "ma_crossover",
    }
    for label, prefix in benchmark_columns.items():
        if label not in results:
            continue
        bench_daily = attach_regime(results[label].daily, vix)
        aligned = pd.concat([
            ml_daily[["net_return", "regime"]].rename(columns={"net_return": "ml"}),
            bench_daily["net_return"].rename("benchmark"),
        ], axis=1).dropna()
        for regime_name in table.index:
            sub = aligned[aligned["regime"] == regime_name]
            table.loc[regime_name, f"{prefix}_sharpe"] = evaluation.sharpe(
                sub["benchmark"])
            active = sub["ml"] - sub["benchmark"]
            table.loc[regime_name, f"active_ann_return_vs_{prefix}"] = (
                float(active.mean() * evaluation.TRADING_DAYS) if len(active) else np.nan)
            table.loc[regime_name, f"information_ratio_vs_{prefix}"] = (
                evaluation.sharpe(active))

    return table
