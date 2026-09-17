"""
Run the model checks and save the results the dashboard displays.

    python scripts/run_experiments.py                      # 15 stocks, all checks
    python scripts/run_experiments.py --only ablation pca  # pick checks
    python scripts/run_experiments.py --universe sp500     # S&P 500 (run src.large_data first)

Output goes to results/<universe>/ as CSV files plus a meta.json describing the run.
Tuning is the slow one: it trains 18 extra model fits per walk forward window.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, data, experiments, pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="default", choices=["default", "sp500"])
    ap.add_argument("--model", default="rf", choices=["logreg", "rf"])
    ap.add_argument("--target", default="binary", choices=["binary", "3class"])
    ap.add_argument("--only", nargs="*", default=["main", "ablation", "tuning", "pca"],
                    choices=["main", "ablation", "tuning", "pca"])
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="CPU cores for feature engineering")
    args = ap.parse_args()

    out_dir = os.path.join(config.RESULTS_DIR, args.universe)
    os.makedirs(out_dir, exist_ok=True)
    started = time.time()

    vix = data.load_vix()
    if args.universe == "sp500":
        from src import large_data
        prices = large_data.load_prices()
        tickers = sorted(prices["ticker"].unique())
    else:
        prices = data.load_prices()
        tickers = list(config.TICKERS)
    print(f"Universe: {args.universe}, {len(tickers)} tickers, {len(prices):,} price rows")

    t0 = time.time()
    feats = pipeline.build_features(prices, vix, tickers=tickers, n_jobs=args.n_jobs)
    print(f"Features: {len(feats):,} rows in {time.time() - t0:.1f}s")
    kw = dict(model_name=args.model, target=args.target, tickers=tickers)

    if "main" in args.only:
        print("Main run")
        t0 = time.time()
        out = pipeline.run(prices, vix, feats=feats, **kw)
        main_row = experiments.summarize_run(out, "Main model")
        main_row["seconds"] = time.time() - t0
        pd.DataFrame([main_row]).set_index("variant").to_csv(
            os.path.join(out_dir, "main.csv"))
    if "ablation" in args.only:
        print("Ablation study")
        experiments.run_ablation(prices, vix, feats, **kw).to_csv(
            os.path.join(out_dir, "ablation.csv"))
    if "tuning" in args.only:
        print("Tuning comparison")
        table, params = experiments.run_tuning(prices, vix, feats, **kw)
        table.to_csv(os.path.join(out_dir, "tuning.csv"))
        params.to_csv(os.path.join(out_dir, "tuning_params.csv"), index=False)
    if "pca" in args.only:
        print("PCA")
        experiments.pca_variance(feats, target=args.target).to_csv(
            os.path.join(out_dir, "pca_variance.csv"), index=False)
        experiments.run_pca(prices, vix, feats, **kw).to_csv(
            os.path.join(out_dir, "pca.csv"))

    meta_path = os.path.join(out_dir, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    meta.update({
        "universe": args.universe, "model": args.model, "target": args.target,
        "tickers": len(tickers), "price_rows": int(len(prices)),
        "feature_rows": int(len(feats)), "cost_bps": config.BACKTEST.cost_bps,
        "date_start": str(feats["date"].min().date()),
        "date_end": str(feats["date"].max().date()),
        "last_run_minutes": round((time.time() - started) / 60, 1),
    })
    if args.universe == "sp500":
        from src import large_data
        meta["storage"] = large_data.storage_summary()
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
