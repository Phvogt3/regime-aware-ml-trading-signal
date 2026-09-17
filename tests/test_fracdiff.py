"""
Tests for the fractional-differentiation feature.

Covers filter causality and the narrower empirical claim that it reduces persistence
while retaining some correlation with the raw series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import features
from src.config import FEATURES as F


def test_ffd_weights_alternate_and_decay():
    w = features._ffd_weights(0.4, 1e-3, 60)
    assert w[0] == 1.0
    assert w[1] < 0                      # first lag weight is negative for d in (0,1)
    assert abs(w[-1]) >= 1e-3            # truncation kept only non-negligible weights
    assert len(w) <= 61


def test_fracdiff_is_causal():
    """frac_diff at t must not change when future values are appended/removed."""
    rng = np.random.default_rng(0)
    x = pd.Series(np.cumsum(rng.normal(0, 1, 800)) + 100.0)
    full = features.frac_diff_ffd(np.log(x), F.fracdiff_d, F.fracdiff_thresh,
                                  F.fracdiff_max_width)
    for t in (300, 500, 700):
        trunc = features.frac_diff_ffd(np.log(x.iloc[:t + 1]), F.fracdiff_d,
                                       F.fracdiff_thresh, F.fracdiff_max_width)
        assert np.isclose(full.iloc[t], trunc.iloc[t], rtol=1e-10, atol=1e-12), (
            f"frac_diff at {t} changed after truncation -> not causal")


def test_fracdiff_reduces_persistence_vs_log_price():
    """
    Fractionally-differenced log price should have smaller lag-1 autocorrelation
    than raw log price while retaining some correlation with it.
    """
    rng = np.random.default_rng(1)
    x = pd.Series(np.cumsum(rng.normal(0.02, 1, 1500)) + 100.0)
    logx = np.log(x)
    fd = features.frac_diff_ffd(logx, F.fracdiff_d, F.fracdiff_thresh,
                                F.fracdiff_max_width).dropna()
    lp = logx.loc[fd.index]
    ac_raw = lp.autocorr(1)
    ac_fd = fd.autocorr(1)
    assert ac_fd < ac_raw                       # much less persistent
    assert abs(np.corrcoef(fd.values, lp.values)[0, 1]) > 0.1  # memory retained


def test_fracdiff_present_in_feature_frame(synthetic_prices, synthetic_vix):
    feats = features.add_features(synthetic_prices, synthetic_vix)
    assert "fracdiff" in feats.columns
    assert feats["fracdiff"].notna().all()      # warmup rows dropped
