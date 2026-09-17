"""
S&P 500 universe: download, store, and load about 500 tickers efficiently.

The default 15 stock cache is one small Parquet file. At S&P 500 scale (roughly
1.5 million daily rows) three big data ideas matter:

  1. Partitioned columnar storage. Prices are written as a Hive partitioned Parquet
     dataset, data/sp500/ticker=XYZ/year=2020/part-0.parquet. Parquet stores each
     column separately and compressed, so a query that needs only close and volume
     never reads the other columns.
  2. Partition pruning. load_prices() pushes ticker and year filters down to
     pyarrow, which skips whole folders instead of loading everything and filtering
     in pandas.
  3. Parallel feature engineering. Tickers are independent, so features are
     computed across CPU cores (features.add_features(..., n_jobs=-1)).

Survivorship bias warning: the ticker list is TODAY's index membership, so stocks
that were dropped from the index (often after falling) are missing from history.
Results on this universe are likely flattered and the README says so.

    python -m src.large_data              # download list + prices (needs internet)
    python -m src.large_data --force      # re-download everything
"""

from __future__ import annotations

import argparse
import io
import os
import time
from typing import List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from . import config, data

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def get_sp500_tickers(force: bool = False) -> List[str]:
    """Current S&P 500 tickers from Wikipedia, cached to data/sp500_tickers.csv."""
    path = config.LARGE_TICKER_FILE
    if os.path.exists(path) and not force:
        return pd.read_csv(path)["ticker"].tolist()
    import requests

    html = requests.get(WIKI_URL, timeout=30,
                        headers={"User-Agent": "Mozilla/5.0 (research project)"}).text
    table = pd.read_html(io.StringIO(html))[0]
    # Yahoo uses dashes where the index uses dots (BRK.B -> BRK-B).
    tickers = sorted(table["Symbol"].astype(str).str.replace(".", "-", regex=False))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_csv(path, index=False)
    return tickers


def _existing_tickers() -> set:
    if not os.path.isdir(config.LARGE_DATA_DIR):
        return set()
    return {d.split("=", 1)[1] for d in os.listdir(config.LARGE_DATA_DIR)
            if d.startswith("ticker=")}


def write_partitions(prices: pd.DataFrame, root: str = config.LARGE_DATA_DIR) -> None:
    """Append a long price frame to the ticker/year partitioned dataset."""
    df = prices.copy()
    df["year"] = df["date"].dt.year.astype("int32")
    df["volume"] = df["volume"].astype("float64")
    table = pa.Table.from_pandas(df, preserve_index=False)
    ds.write_dataset(
        table, root, format="parquet",
        partitioning=ds.partitioning(
            pa.schema([("ticker", pa.string()), ("year", pa.int32())]), flavor="hive"),
        existing_data_behavior="overwrite_or_ignore",
        basename_template="part-{i}.parquet",
    )


def download_sp500(force: bool = False, batch_size: int = 50,
                   pause_seconds: float = 2.0) -> None:
    """Download prices in batches and write each batch to the partitioned dataset."""
    tickers = get_sp500_tickers(force=force)
    have = set() if force else _existing_tickers()
    todo = [t for t in tickers if t not in have]
    print(f"{len(tickers)} tickers in the index, {len(todo)} to download")
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        try:
            prices = data.download_prices(batch)
            prices = prices.dropna(subset=["open", "high", "low", "close", "volume"])
            prices = prices[(prices[["open", "high", "low", "close"]] > 0).all(axis=1)]
            write_partitions(prices)
            print(f"  batch {i // batch_size + 1}: {prices['ticker'].nunique()} tickers, "
                  f"{len(prices):,} rows")
        except Exception as exc:  # one bad batch should not stop the whole download
            print(f"  batch {i // batch_size + 1} failed: {exc}")
        time.sleep(pause_seconds)


def load_prices(tickers: Optional[List[str]] = None,
                start_year: Optional[int] = None, end_year: Optional[int] = None,
                columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Read the partitioned dataset with filters pushed down to the file scan, so
    only the matching ticker and year folders (and requested columns) are read.
    """
    if not os.path.isdir(config.LARGE_DATA_DIR):
        raise FileNotFoundError(
            f"{config.LARGE_DATA_DIR} not found. Run `python -m src.large_data` first.")
    dataset = ds.dataset(config.LARGE_DATA_DIR, format="parquet", partitioning="hive")
    flt = None
    if tickers is not None:
        flt = ds.field("ticker").isin(list(tickers))
    if start_year is not None:
        f = ds.field("year") >= start_year
        flt = f if flt is None else flt & f
    if end_year is not None:
        f = ds.field("year") <= end_year
        flt = f if flt is None else flt & f
    cols = columns or ["date", "ticker", "open", "high", "low", "close", "volume"]
    df = dataset.to_table(filter=flt, columns=cols).to_pandas()
    df["ticker"] = df["ticker"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    return (df.drop_duplicates(subset=["date", "ticker"])
              .sort_values(["ticker", "date"]).reset_index(drop=True))


def storage_summary() -> dict:
    """Size on disk and row counts, for the README and dashboard."""
    total_bytes, files = 0, 0
    for root, _, names in os.walk(config.LARGE_DATA_DIR):
        for n in names:
            if n.endswith(".parquet"):
                files += 1
                total_bytes += os.path.getsize(os.path.join(root, n))
    return {"files": files, "megabytes": total_bytes / 1e6,
            "tickers": len(_existing_tickers())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the S&P 500 partitioned dataset.")
    ap.add_argument("--force", action="store_true", help="re-download everything")
    args = ap.parse_args()
    download_sp500(force=args.force)
    print(storage_summary())
