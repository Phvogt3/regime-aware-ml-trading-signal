"""
Walk-forward (rolling-origin) validation for a cross-sectional daily panel.

Why this and not a random 80/20 split: with time series, a random split lets the
model train on the future and test on the past. That inflates every metric and is
the single fastest way to get a finance-ML project dismissed. Here we always train
on a trailing window and test on the immediately following, never-before-seen block
of dates, then roll forward.

Splitting is done by DATE, not by row: at each origin we train on all
ticker-rows whose date falls in the trailing `train_days` window and predict all
ticker-rows in the next `test_days` window. Test blocks are non-overlapping
(step = test length by default), so concatenating them yields one continuous,
gap-free out-of-sample prediction series spanning almost the whole history.

Per fold the StandardScaler (and PCA, when used) is fit on TRAINING ROWS ONLY,
then applied to the test rows. Fitting the scaler on the full sample would leak test-period means/variances
into training -- a quieter cousin of the classic lookahead bug.

Optional extras (all still fit on training rows only):
  * tune=True runs a small grid search inside each training window, using time
    ordered date splits (inner_date_splits), scored by ROC-AUC.
  * pca_components compresses the scaled features with PCA before the model.
  * feature_columns restricts the model to a subset of features (ablation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config, models
from .config import WALK as W
from .config import LABELS as L
from .config import TUNING as T
from .features import FEATURE_COLUMNS
from .labeling import THREE_CLASS_MAP


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame          # OOS rows: date,ticker,y_true,pred_class,signal,prob_up,fwd_return
    importances: pd.DataFrame          # mean feature importance across folds
    n_folds: int
    model_name: str
    target: str
    fold_table: pd.DataFrame = field(default_factory=pd.DataFrame)  # one row per window


def _date_folds(dates: np.ndarray) -> List[dict]:
    """Build rolling (train_idx_range, test_idx_range) folds over unique dates."""
    n = len(dates)
    folds = []
    train_start = 0
    while True:
        train_end = train_start + W.train_days          # exclusive
        test_end = train_end + W.test_days              # exclusive
        if train_end > n:
            break
        if train_end - train_start < W.min_train_days:
            train_start += W.step_days
            continue
        te = min(test_end, n)
        if te <= train_end:
            break
        folds.append({
            "train_dates": dates[train_start:train_end],
            "test_dates": dates[train_end:te],
        })
        train_start += W.step_days
    return folds


def _purged_train_dates(train_dates: np.ndarray,
                        horizon: int = L.horizon) -> np.ndarray:
    """Remove training dates whose forward labels overlap the next test block."""
    if horizon < 0:
        raise ValueError("label horizon must be non-negative")
    if horizon == 0:
        return train_dates
    if len(train_dates) <= horizon:
        return train_dates[:0]
    return train_dates[:-horizon]


def inner_date_splits(row_dates: np.ndarray, n_splits: int,
                      gap: int = L.horizon) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Time ordered splits INSIDE one training window, used for hyperparameter tuning.

    The unique training dates are cut into n_splits + 1 consecutive blocks. Split k
    trains on blocks 0..k-1 and validates on block k, so validation dates are always
    later than training dates. Splitting is by date (never by row) so one day's
    cross section is never divided between train and validation, and the last `gap`
    training dates are purged exactly like the outer walk forward loop.
    Yields (train_row_positions, validation_row_positions).
    """
    unique_dates = np.sort(np.unique(row_dates))
    blocks = np.array_split(unique_dates, n_splits + 1)
    for k in range(1, n_splits + 1):
        train_d = np.concatenate(blocks[:k])
        if gap:
            train_d = train_d[:-gap] if len(train_d) > gap else train_d[:0]
        val_d = blocks[k]
        if len(train_d) == 0 or len(val_d) == 0:
            continue
        yield (np.flatnonzero(np.isin(row_dates, train_d)),
               np.flatnonzero(np.isin(row_dates, val_d)))


def _signal_from_proba(proba: np.ndarray, classes: np.ndarray, target: str) -> np.ndarray:
    """
    Collapse class probabilities into a single directional score in [-1, 1]:
      binary : P(up) - P(down) = 2*P(up) - 1
      3class : P(up) - P(down)   (flat contributes 0)
    Higher => more confident long; used later for optional confidence sizing.
    """
    cls = list(classes)
    if target == "binary":
        up_col = cls.index(1)
        return 2.0 * proba[:, up_col] - 1.0
    up = proba[:, cls.index(THREE_CLASS_MAP["up"])] if THREE_CLASS_MAP["up"] in cls else 0.0
    down = proba[:, cls.index(THREE_CLASS_MAP["down"])] if THREE_CLASS_MAP["down"] in cls else 0.0
    return up - down


def _up_class(target: str) -> int:
    return 1 if target == "binary" else THREE_CLASS_MAP["up"]


def _prob_up(proba: np.ndarray, classes: np.ndarray, target: str) -> np.ndarray:
    """Probability of the "up" class, used for ROC-AUC."""
    cls = list(classes)
    up = _up_class(target)
    return proba[:, cls.index(up)] if up in cls else np.zeros(len(proba))


def up_auc(y_true: np.ndarray, prob_up: np.ndarray, target: str) -> float:
    """ROC-AUC of "up" versus everything else. NaN when only one outcome is present."""
    is_up = (np.asarray(y_true) == _up_class(target)).astype(int)
    if is_up.min() == is_up.max():
        return np.nan
    return float(roc_auc_score(is_up, prob_up))


def build_estimator(model_name: str,
                    pca_components: Optional[Union[int, float]] = None) -> Pipeline:
    """
    Scaler (+ optional PCA) + model as one sklearn Pipeline. Every step is fit on the
    rows passed to .fit only, so inside walk forward (and inside the tuning splits)
    the scaler and PCA never see validation or test rows.
    """
    steps = [("scale", StandardScaler())]
    if pca_components is not None:
        steps.append(("pca", PCA(n_components=pca_components,
                                 random_state=config.RANDOM_STATE)))
    steps.append(("model", models.make_model(model_name)))
    return Pipeline(steps)


def _tune(model_name: str, target: str, X: np.ndarray, y: np.ndarray,
          row_dates: np.ndarray,
          pca_components: Optional[Union[int, float]]) -> Dict[str, object]:
    """Grid search on the training window only, scored by ROC-AUC on later dates."""
    grid = {f"model__{k}": v for k, v in models.param_grid(model_name).items()}
    search_est = build_estimator(model_name, pca_components)
    if model_name == "rf":
        # Smaller forests during the search keep it fast; the refit uses the full size.
        search_est.set_params(model__n_estimators=T.search_trees)
    splits = list(inner_date_splits(row_dates, T.inner_splits))
    if not splits:
        return {}
    search = GridSearchCV(
        search_est, grid, cv=splits, refit=False, n_jobs=1, error_score=np.nan,
        scoring="roc_auc" if target == "binary" else "roc_auc_ovr",
    )
    search.fit(X, y)
    if not np.isfinite(search.best_score_):
        return {}
    return {k.replace("model__", ""): v for k, v in search.best_params_.items()}


def run_walk_forward(df: pd.DataFrame, model_name: str,
                     target: str = "binary",
                     feature_columns: Optional[Sequence[str]] = None,
                     tune: bool = False,
                     pca_components: Optional[Union[int, float]] = None,
                     ) -> WalkForwardResult:
    """
    df must contain the feature columns + ['date','ticker','y','fwd_return','close'].
    Returns concatenated out-of-sample predictions, mean feature importances, and a
    per window table (dates, test ROC-AUC, tuned parameters, PCA components kept).

    feature_columns : subset of FEATURE_COLUMNS to use (ablation studies).
    tune            : grid search hyperparameters inside each training window.
    pca_components  : int = number of components, float in (0, 1) = share of
                      variance to keep, None = no PCA.
    """
    cols = list(feature_columns) if feature_columns is not None else list(FEATURE_COLUMNS)
    unknown = set(cols) - set(FEATURE_COLUMNS)
    if unknown or not cols:
        raise ValueError(f"invalid feature columns: {sorted(unknown) or 'none given'}")

    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    unique_dates = np.sort(df["date"].unique())
    folds = _date_folds(unique_dates)
    if not folds:
        raise ValueError(
            "No walk-forward folds: not enough history for the configured windows.")

    all_preds: List[pd.DataFrame] = []
    imp_rows: List[Dict[str, float]] = []
    fold_rows: List[Dict[str, object]] = []

    for fold in folds:
        # Purge the final label horizon. Without this, the last training label uses
        # the first test date's close and leaks that test outcome into the model.
        train_dates = _purged_train_dates(fold["train_dates"])
        tr = df[df["date"].isin(train_dates)]
        te = df[df["date"].isin(fold["test_dates"])]
        if tr.empty or te.empty or tr["y"].nunique() < 2:
            continue

        X_tr = tr[cols].to_numpy()
        X_te = te[cols].to_numpy()
        y_tr = tr["y"].to_numpy()

        best = _tune(model_name, target, X_tr, y_tr, tr["date"].to_numpy(),
                     pca_components) if tune else {}

        est = build_estimator(model_name, pca_components)
        if best:
            est.set_params(**{f"model__{k}": v for k, v in best.items()})
        est.fit(X_tr, y_tr)                          # scaler/PCA fit on TRAIN ONLY
        model = est.named_steps["model"]

        proba = est.predict_proba(X_te)
        pred_class = model.classes_[np.argmax(proba, axis=1)]
        signal = _signal_from_proba(proba, model.classes_, target)
        prob_up = _prob_up(proba, model.classes_, target)

        block = te[["date", "ticker", "y", "fwd_return", "close"]].copy()
        block = block.rename(columns={"y": "y_true"})
        block["pred_class"] = pred_class
        block["signal"] = signal
        block["prob_up"] = prob_up
        all_preds.append(block)

        row = {
            "test_start": pd.Timestamp(fold["test_dates"][0]),
            "test_end": pd.Timestamp(fold["test_dates"][-1]),
            "test_auc": up_auc(block["y_true"].to_numpy(), prob_up, target),
            "test_accuracy": float((block["y_true"] == block["pred_class"]).mean()),
        }
        row.update({f"param_{k}": v for k, v in best.items()})
        if "pca" in est.named_steps:
            row["pca_components"] = int(est.named_steps["pca"].n_components_)
        fold_rows.append(row)

        if "pca" not in est.named_steps:
            imp = models.feature_importance(model, cols)
            if imp:
                imp_rows.append(imp)

    if not all_preds:
        raise ValueError("All walk-forward folds were empty or had fewer than two classes.")
    predictions = pd.concat(all_preds).sort_values(["date", "ticker"]).reset_index(drop=True)
    importances = (
        pd.DataFrame(imp_rows).mean().sort_values(ascending=False)
        .rename("importance").reset_index().rename(columns={"index": "feature"})
        if imp_rows else pd.DataFrame(columns=["feature", "importance"])
    )
    return WalkForwardResult(
        predictions=predictions, importances=importances,
        n_folds=len(all_preds), model_name=model_name, target=target,
        fold_table=pd.DataFrame(fold_rows),
    )
