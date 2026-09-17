"""
Target design.

Timing convention (memorize this; an interviewer will push on it):

    Row for day t carries:
        features X_t   -> computed from data through the CLOSE of day t-1
        fwd_return     -> close_{t+h} / close_t - 1   (the future move we predict)
        label          -> a function of fwd_return only

    The label deliberately uses future prices -- that is what a target IS. The
    leakage rule is about FEATURES, not labels: no column in X_t may depend on
    any price on or after day t. The backtest submits the already-known signal for
    the day-t close and earns exactly `fwd_return`, so the fill is executable.

Two labelings:
    binary : 1 if fwd_return > 0 else 0          (next-day up/down)
    three  : +1 / 0 / -1 using a +/- flat_threshold deadband, so genuinely
             flat days are not forced into an up/down guess. Modeled as classes
             {0: down, 1: flat, 2: up} for scikit-learn.

The last `horizon` rows of each ticker have no realized future return and get
label = NaN; they are dropped from the training/eval set but kept for live
signal generation (you can still predict on X_t even when y_t is unknown).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LABELS as L

THREE_CLASS_MAP = {"down": 0, "flat": 1, "up": 2}


def add_labels(df: pd.DataFrame, horizon: int = L.horizon,
               flat_threshold: float = L.flat_threshold) -> pd.DataFrame:
    """
    Add forward return and both labelings to a long-format feature frame.
    Adds columns: fwd_return, label_binary, label_3class.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if flat_threshold < 0:
        raise ValueError("flat_threshold must be non-negative")
    df = df.sort_values(["ticker", "date"]).copy()

    # Future return per ticker. shift(-horizon) looks FORWARD -- this is the
    # target, and it is the ONLY forward-looking operation in the pipeline.
    grp = df.groupby("ticker", group_keys=False)
    df["fwd_return"] = grp["close"].transform(
        lambda c: c.shift(-horizon) / c - 1.0)

    # Binary: next-period up (1) vs down/flat (0).
    df["label_binary"] = (df["fwd_return"] > 0).astype("float")
    df.loc[df["fwd_return"].isna(), "label_binary"] = np.nan

    # 3-class deadband.
    def _three(r: float) -> float:
        if np.isnan(r):
            return np.nan
        if r > flat_threshold:
            return THREE_CLASS_MAP["up"]
        if r < -flat_threshold:
            return THREE_CLASS_MAP["down"]
        return THREE_CLASS_MAP["flat"]

    df["label_3class"] = df["fwd_return"].apply(_three)
    return df


def labeled_frame(df: pd.DataFrame, target: str = "binary") -> pd.DataFrame:
    """
    Return only rows with a defined label for the chosen target, ready for
    training/eval. `target` in {"binary", "3class"}.
    """
    if target not in {"binary", "3class"}:
        raise ValueError("target must be 'binary' or '3class'")
    col = "label_binary" if target == "binary" else "label_3class"
    out = df.dropna(subset=[col]).copy()
    out["y"] = out[col].astype(int)
    return out
