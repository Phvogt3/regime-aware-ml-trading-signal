"""
Evaluation framework -- what turns this from a classifier into a finance project.

PRIMARY metrics (risk-adjusted performance, net of costs):
    Sharpe, Sortino, max drawdown, win rate, average win / average loss, CAGR,
    annualized volatility.

SECONDARY metric (reported, explicitly demoted):
    classification accuracy. A model can be 55% accurate and lose money after
    costs, or 45% accurate and profitable with good sizing. Accuracy is a
    diagnostic, not the objective -- the objective is risk-adjusted return.

Annualization uses 252 trading days. Sharpe/Sortino here are excess-of-zero
(risk-free ~ 0 over the sample at daily granularity); this is stated plainly in
the README so the number isn't oversold.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sharpe(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = returns.dropna()
    sd = r.std()
    if sd == 0 or len(r) < 2:
        return np.nan
    return np.sqrt(periods) * r.mean() / sd


def sortino(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = returns.dropna()
    downside = np.minimum(r, 0.0)
    dd = np.sqrt((downside ** 2).mean()) if len(r) else 0.0
    if dd == 0 or len(r) < 2:
        return np.nan
    return np.sqrt(periods) * r.mean() / dd


def _with_initial(equity: pd.Series,
                  initial_value: Optional[float]) -> pd.Series:
    eq = equity.dropna().reset_index(drop=True)
    if initial_value is not None:
        eq = pd.concat([pd.Series([float(initial_value)]), eq], ignore_index=True)
    return eq


def max_drawdown(equity: pd.Series,
                 initial_value: Optional[float] = None) -> float:
    """Most negative peak-to-trough decline of the equity curve (e.g. -0.32)."""
    eq = _with_initial(equity, initial_value)
    if eq.empty:
        return np.nan
    running_max = eq.cummax()
    return float((eq / running_max - 1.0).min())


def cagr(equity: pd.Series, periods: int = TRADING_DAYS,
         initial_value: Optional[float] = None) -> float:
    raw = equity.dropna()
    eq = _with_initial(equity, initial_value)
    if len(eq) < 2 or len(raw) < 1:
        return np.nan
    years = len(raw) / periods
    return float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1.0)


def ann_vol(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(returns.dropna().std() * np.sqrt(periods))


def calmar(equity: pd.Series, periods: int = TRADING_DAYS,
           initial_value: Optional[float] = None) -> float:
    """CAGR divided by the absolute maximum drawdown."""
    mdd = max_drawdown(equity, initial_value=initial_value)
    c = cagr(equity, periods, initial_value=initial_value)
    if mdd is None or np.isnan(mdd) or mdd == 0:
        return np.nan
    return float(c / abs(mdd))


def _align_returns(returns: pd.Series, benchmark: pd.Series):
    df = pd.concat([returns.rename("r"), benchmark.rename("b")], axis=1).dropna()
    return df["r"], df["b"]


def beta(returns: pd.Series, benchmark: pd.Series) -> float:
    """
    Market beta: slope of strategy returns regressed on benchmark returns,
    computed as Cov(r, b) / Var(b). Measures directional co-movement with the
    benchmark (here, the equal-weight buy-and-hold basket).
    """
    r, b = _align_returns(returns, benchmark)
    var_b = b.var()
    if len(r) < 2 or var_b == 0:
        return np.nan
    return float(np.cov(r, b, ddof=1)[0, 1] / var_b)


def alpha(returns: pd.Series, benchmark: pd.Series,
          periods: int = TRADING_DAYS) -> float:
    """Annualized Jensen's alpha: mean(r) minus beta*mean(b), scaled to a year."""
    r, b = _align_returns(returns, benchmark)
    if len(r) < 2:
        return np.nan
    be = beta(returns, benchmark)
    daily_alpha = r.mean() - be * b.mean()
    return float(daily_alpha * periods)


def information_ratio(returns: pd.Series, benchmark: pd.Series,
                      periods: int = TRADING_DAYS) -> float:
    """Annualized mean active return divided by tracking error (std of r minus b)."""
    r, b = _align_returns(returns, benchmark)
    active = r - b
    sd = active.std()
    if len(active) < 2 or sd == 0:
        return np.nan
    return float(np.sqrt(periods) * active.mean() / sd)


def return_t_stat(returns: pd.Series) -> float:
    """
    t-statistic of the mean daily return against zero. A rough gauge of whether
    the average return is statistically distinguishable from zero; it does not
    correct for autocorrelation or multiple testing, so it is read as indicative.
    """
    r = returns.dropna()
    sd = r.std()
    if len(r) < 2 or sd == 0:
        return np.nan
    return float(r.mean() / (sd / np.sqrt(len(r))))


def win_stats(returns: pd.Series) -> Dict[str, float]:
    """Win rate and average win / average loss over active (nonzero) days."""
    r = returns.dropna()
    active = r[r != 0]
    wins, losses = active[active > 0], active[active < 0]
    return {
        "win_rate": float(len(wins) / len(active)) if len(active) else np.nan,
        "avg_win": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss": float(losses.mean()) if len(losses) else np.nan,
        "win_loss_ratio": float(wins.mean() / abs(losses.mean()))
        if len(wins) and len(losses) and losses.mean() != 0 else np.nan,
    }


def performance_summary(result, label: str = "strategy",
                        benchmark_returns: pd.Series = None) -> Dict[str, float]:
    """
    Full PRIMARY-metric summary for a BacktestResult-like object. When a
    benchmark return series is supplied, beta, annualized alpha, and the
    information ratio are computed relative to it.
    """
    rets = result.net_returns
    eq = result.equity
    ws = win_stats(rets)
    row = {
        "strategy": label,
        "CAGR": cagr(eq, initial_value=result.initial_capital),
        "sharpe": sharpe(rets),
        "sortino": sortino(rets),
        "calmar": calmar(eq, initial_value=result.initial_capital),
        "max_drawdown": max_drawdown(eq, initial_value=result.initial_capital),
        "ann_vol": ann_vol(rets),
        "beta": np.nan,
        "alpha": np.nan,
        "information_ratio": np.nan,
        "return_t_stat": return_t_stat(rets),
        "win_rate": ws["win_rate"],
        "avg_win": ws["avg_win"],
        "avg_loss": ws["avg_loss"],
        "win_loss_ratio": ws["win_loss_ratio"],
        "total_return": float(eq.iloc[-1] / result.initial_capital - 1.0)
        if len(eq) else np.nan,
        "avg_daily_cost": float(result.daily["cost"].mean()),
        "avg_exposure": float(result.daily["exposure"].mean()),
    }
    if benchmark_returns is not None:
        row["beta"] = beta(rets, benchmark_returns)
        row["alpha"] = alpha(rets, benchmark_returns)
        row["information_ratio"] = information_ratio(rets, benchmark_returns)
    return row


def summary_table(results: Dict[str, object],
                  benchmark_label: str = "Buy & hold") -> pd.DataFrame:
    """
    Stack performance_summary rows for {label: BacktestResult} into a table.
    Beta, alpha, and information ratio are measured against `benchmark_label`
    (the buy-and-hold basket by default).
    """
    bench = results.get(benchmark_label)
    bench_rets = bench.net_returns if bench is not None else None
    rows = [performance_summary(res, label, benchmark_returns=bench_rets)
            for label, res in results.items()]
    return pd.DataFrame(rows).set_index("strategy")


# ---- Secondary: classification accuracy (explicitly demoted) ----

def classification_metrics(predictions: pd.DataFrame) -> Dict[str, float]:
    """
    SECONDARY diagnostics from walk-forward predictions. Reported, never the
    objective. Directional accuracy = share of days the predicted class matched
    the realized label. ROC-AUC = chance a random "up" day gets a higher predicted
    up probability than a random other day (0.5 = no skill). Unlike accuracy it is
    not inflated by class imbalance, since it ignores the decision threshold.
    """
    p = predictions.dropna(subset=["y_true", "pred_class"])
    if p.empty:
        return {"accuracy": np.nan, "roc_auc": np.nan, "balanced_accuracy": np.nan,
                "majority_baseline": np.nan, "n_classes": 0, "n": 0}
    acc = float((p["y_true"] == p["pred_class"]).mean())
    recalls = []
    for cls, group in p.groupby("y_true"):
        recalls.append(float((group["pred_class"] == cls).mean()))
    majority = float(p["y_true"].value_counts(normalize=True).max())
    auc = np.nan
    if "prob_up" in p.columns:
        from sklearn.metrics import roc_auc_score
        # "up" is class 1 for the binary target and class 2 for the 3-class target.
        up = 2 if p["y_true"].max() >= 2 else 1
        is_up = (p["y_true"] == up).astype(int)
        if is_up.nunique() == 2:
            auc = float(roc_auc_score(is_up, p["prob_up"]))
    return {
        "accuracy": acc,
        "roc_auc": auc,
        "balanced_accuracy": float(np.mean(recalls)),
        "majority_baseline": majority,
        "n_classes": int(p["y_true"].nunique()),
        "n": int(len(p)),
    }


def cost_drag(gross_result, net_result=None) -> float:
    """
    Fraction of gross return eroded by transaction costs -- the honest
    'costs ate X% of gross' number. Pass a single result; computed from its own
    gross vs net cumulative returns.
    """
    d = gross_result.daily
    gross_cum = (1 + d["gross_return"]).prod() - 1
    net_cum = (1 + d["net_return"]).prod() - 1
    if gross_cum == 0:
        return np.nan
    return float((gross_cum - net_cum) / abs(gross_cum))
