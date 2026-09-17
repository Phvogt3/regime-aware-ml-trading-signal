"""
Command-line entry point: run the full research pipeline and print the headline
results. Assumes the data cache exists (run `python -m src.data` first).

    python scripts/run_pipeline.py --model rf --target binary
    python scripts/run_pipeline.py --model logreg --target 3class
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

# allow running as a plain script from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data, pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rf", choices=["logreg", "rf"])
    ap.add_argument("--target", default="binary", choices=["binary", "3class"])
    ap.add_argument("--cost-bps", type=float, default=None,
                    help="transaction cost per unit turnover; defaults to config")
    ap.add_argument("--tune", action="store_true",
                    help="grid search hyperparameters inside each walk forward window")
    ap.add_argument("--pca", type=float, default=None,
                    help="PCA before the model: below 1 = share of variance, else components")
    args = ap.parse_args()

    prices, vix = data.load_prices(), data.load_vix()
    out = pipeline.run(prices, vix, model_name=args.model, target=args.target,
                       cost_bps=args.cost_bps, tune=args.tune,
                       pca_components=(None if args.pca is None else
                                       args.pca if args.pca < 1 else int(args.pca)))

    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", lambda x: f"{x:0.4f}")

    print(f"\n=== Model: {out.model_name} | target: {out.target} | "
          f"OOS dates: {out.predictions['date'].nunique()} ===\n")
    print("PRIMARY metrics (net of costs):")
    print(out.summary.T.to_string())
    print(f"\nSecondary -- classification accuracy: "
          f"{out.classification['accuracy']:.3f} vs majority baseline "
          f"{out.classification['majority_baseline']:.3f} "
          f"(n={out.classification['n']}), ROC-AUC {out.classification['roc_auc']:.3f}")
    print(f"Transaction costs eroded {out.cost_drag*100:.1f}% of gross ML return.\n")
    print("Per-regime ML performance (by VIX level):")
    print(out.regime_table.to_string())
    print("\nTop features (mean walk-forward importance):")
    print(out.importances.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
