"""
Model factory.

Two models, in increasing complexity:
    logreg : Logistic Regression -- interpretable baseline. Its coefficients tell
             you which features actually carry signal.
    rf     : Random Forest -- the main model. Nonlinear, low-tuning, robust.

Standardization is handled OUTSIDE these estimators, per walk-forward fold, so it
can be fit on training data only (see walkforward.py). Tree models don't need
scaling; logistic regression does, which the fold-level scaler provides.
"""

from __future__ import annotations

from typing import Dict, List

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from . import config


def available_models() -> List[str]:
    """Model keys that can be built in this environment."""
    return ["logreg", "rf"]


def make_model(name: str):
    """Construct a fresh, unfitted estimator by key."""
    if name == "logreg":
        # multinomial handles both binary and 3-class; balanced weights guard
        # against the mild class imbalance in up/down/flat labels.
        return LogisticRegression(
            max_iter=2000,
            C=1.0,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=6,         # shallow-ish: daily signals are noisy, guard overfit
            min_samples_leaf=50, # each leaf must generalize over many days
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        )
    raise ValueError(f"unknown model '{name}'")


def param_grid(name: str) -> Dict[str, list]:
    """Search space used when tuning inside each walk forward window."""
    if name == "rf":
        return dict(config.TUNING.rf_grid)
    if name == "logreg":
        return dict(config.TUNING.logreg_grid)
    raise ValueError(f"unknown model '{name}'")


def feature_importance(model, feature_names: List[str]) -> Dict[str, float]:
    """Best-effort importance/coefficient extraction for interpretability."""
    if hasattr(model, "feature_importances_"):
        return dict(zip(feature_names, model.feature_importances_))
    if hasattr(model, "coef_"):
        import numpy as np
        # average |coef| across classes for a single ranking
        imp = np.abs(model.coef_).mean(axis=0)
        return dict(zip(feature_names, imp))
    return {}
