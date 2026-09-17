"""
Model checks that go beyond the main backtest. Each one reruns the same pipeline
(same walk forward validation, same backtest, same costs) with one thing changed,
so any difference in results comes from that change.

    ablation : drop one feature family at a time and retrain.
    tuning   : fixed hyperparameters vs a grid search inside each training window.
    pca      : how much the 17 features overlap, and whether a model trained on
               principal components does better or worse.

Results are saved as CSV under results/<universe>/ by scripts/run_experiments.py
and displayed by the dashboard. Features are computed once and reused by every run.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import labeling, pipeline, walkforward
from .features import FEATURE_COLUMNS, FEATURE_GROUPS


def summarize_run(out: pipeline.PipelineOutput, label: str) -> Dict[str, object]:
    """One comparable row of headline numbers for a pipeline run."""
    ml = out.summary.loc["ML strategy"]
    reg = out.regime_table
    row = {
        "variant": label,
        "sharpe": ml["sharpe"],
        "max_drawdown": ml["max_drawdown"],
        "cagr": ml["CAGR"],
        "roc_auc": out.classification["roc_auc"],
        "accuracy": out.classification["accuracy"],
        "buy_hold_sharpe": out.summary.loc["Buy & hold", "sharpe"],
    }
    for name in ["calm", "normal", "stressed"]:
        row[f"sharpe_{name}"] = reg.loc[name, "sharpe"] if name in reg.index else np.nan
    return row


def ablation_variants() -> Dict[str, List[str]]:
    """Full model, then the full model minus each feature family."""
    variants = {"All features": list(FEATURE_COLUMNS)}
    for group, cols in FEATURE_GROUPS.items():
        name = f"Without {group.replace('_', ' ')}"
        variants[name] = [c for c in FEATURE_COLUMNS if c not in cols]
    return variants


def run_ablation(prices, vix, feats, model_name="rf", target="binary",
                 tickers=None, cost_bps=None, log=print) -> pd.DataFrame:
    rows = []
    for label, cols in ablation_variants().items():
        log(f"  ablation: {label} ({len(cols)} features)")
        out = pipeline.run(prices, vix, model_name=model_name, target=target,
                           tickers=tickers, cost_bps=cost_bps,
                           feature_columns=cols, feats=feats)
        row = summarize_run(out, label)
        row["n_features"] = len(cols)
        rows.append(row)
    table = pd.DataFrame(rows).set_index("variant")
    base = table.loc["All features"]
    table["sharpe_change"] = table["sharpe"] - base["sharpe"]
    table["auc_change"] = table["roc_auc"] - base["roc_auc"]
    return table


def run_tuning(prices, vix, feats, model_name="rf", target="binary",
               tickers=None, cost_bps=None, log=print):
    """Returns (comparison table, per window chosen parameters)."""
    rows, params = [], pd.DataFrame()
    for label, tune in [("Fixed settings", False), ("Tuned in each window", True)]:
        log(f"  tuning: {label}")
        out = pipeline.run(prices, vix, model_name=model_name, target=target,
                           tickers=tickers, cost_bps=cost_bps, tune=tune, feats=feats)
        rows.append(summarize_run(out, label))
        if tune:
            params = out.fold_table
    return pd.DataFrame(rows).set_index("variant"), params


def pca_variance(feats: pd.DataFrame, target: str = "binary") -> pd.DataFrame:
    """
    Cumulative share of variance explained by each principal component, averaged
    over the walk forward TRAINING windows (so no test data is used).
    """
    labeled = labeling.labeled_frame(labeling.add_labels(feats), target=target)
    dates = np.sort(labeled["date"].unique())
    curves = []
    for fold in walkforward._date_folds(dates):
        tr = labeled[labeled["date"].isin(fold["train_dates"])]
        X = StandardScaler().fit_transform(tr[FEATURE_COLUMNS].to_numpy())
        curves.append(PCA(random_state=0).fit(X).explained_variance_ratio_)
    ratio = np.mean(curves, axis=0)
    return pd.DataFrame({
        "component": np.arange(1, len(ratio) + 1),
        "variance_share": ratio,
        "cumulative_share": np.cumsum(ratio),
    })


def run_pca(prices, vix, feats, model_name="rf", target="binary", tickers=None,
            cost_bps=None, settings: Sequence[Optional[Union[int, float]]] = (None, 0.90, 5),
            log=print) -> pd.DataFrame:
    rows = []
    for n in settings:
        if n is None:
            label = "No PCA (17 features)"
        elif isinstance(n, float):
            label = f"PCA keeping {n:.0%} of variance"
        else:
            label = f"PCA with {n} components"
        log(f"  pca: {label}")
        out = pipeline.run(prices, vix, model_name=model_name, target=target,
                           tickers=tickers, cost_bps=cost_bps, pca_components=n,
                           feats=feats)
        row = summarize_run(out, label)
        ft = out.fold_table
        row["avg_components"] = (ft["pca_components"].mean()
                                 if "pca_components" in ft else len(FEATURE_COLUMNS))
        rows.append(row)
    return pd.DataFrame(rows).set_index("variant")
