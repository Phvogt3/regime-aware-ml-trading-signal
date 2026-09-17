# Regime-Aware ML Trading Signal

A rigorous, leakage-safe backtest of a technical-indicator machine-learning trading
signal across multiple volatility regimes.

## Research question

> **Does a technical-indicator-based ML signal generate risk-adjusted returns above a
> naive benchmark, after realistic transaction costs, and does that edge hold up across
> different volatility regimes?**

This is deliberately *not* "can I predict stock prices." Framing the project as a
prediction contest is what makes most student versions worthless: a model can be 55%
accurate and lose money to costs, or 45% accurate and profitable with good sizing. The
value here is the **rigor of the evaluation**, not a claim that the strategy beats the
market. The strongest honest finding a project like this usually reaches is a *modest,
regime-dependent* edge, and that is presented as-is, not tuned to sound impressive.

## What's in the box

- **Data pipeline** (`src/data.py`), daily OHLCV for 15 large-caps across tech,
  financials, and consumer, plus SPY as an optional cached reference and the VIX.
- **Feature engineering** (`src/features.py`), SMA/EMA crossovers, MACD, RSI, ROC,
  stochastics, Bollinger width/position, rolling volatility, volume ratio, on-balance
  volume, **fractionally-differenced log price**, VIX level and **trailing-year VIX
  percentile**. Every feature is causal.
- **Labeling** (`src/labeling.py`), binary next-day direction and a 3-class
  up/flat/down deadband (±0.5%).
- **Models** (`src/models.py`), Logistic Regression (interpretable baseline), Random
  Forest (main model).
- **Walk-forward validation** (`src/walkforward.py`), rolling ~2-year train / ~6-month
  test, rolled through history. No random splits.
- **Custom backtest engine** (`src/backtest.py`), a from-scratch day-by-day loop with
  equal-weight sizing, transaction costs, and no lookahead. Benchmarks
  (`src/benchmarks.py`) run through the *same* engine.
- **Evaluation** (`src/evaluation.py`), Sharpe, Sortino, max drawdown, win rate,
  average win/loss; accuracy reported as explicitly secondary.
- **Regime analysis** (`src/regime.py`), performance split by VIX level.
- **Streamlit app** (`app.py`), a live view of the pipeline, styled to match the FOMC
  dashboard.
- **Tests** (`tests/`), including an empirical no-lookahead proof.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt
```

Python 3.10+.

## Running it

```bash
# 1. Build the local data cache (needs internet; hits Yahoo Finance via yfinance)
python -m src.data                 # add --force to re-download

# 2. Run the full research pipeline and print headline results
python scripts/run_pipeline.py --model rf --target binary
python scripts/run_pipeline.py --model logreg --target 3class

# 3. Launch the dashboard
streamlit run app.py

# 4. Run the test suite (works offline on synthetic data)
pytest -q
```

The data download runs on your machine, where yfinance has network access; every
downstream step reads the local cache.

## Methodology and how leakage is prevented

The single most common bug in this kind of project, and the first thing a technical
interviewer probes, is lookahead: letting day *t*'s decision use information that
wasn't knowable until after day *t*. This project defends against it structurally and
then **verifies the defense empirically**.

**Timing convention.** Indicators are calculated causally and shifted forward one
session, so a feature row `X_t` contains data through the *close of t-1*. The signal is
therefore available before the day-*t* close-auction cutoff. The model predicts the sign
of the *t → t+1* close-to-close return, and the backtest enters at the close of *t*.

**Causal features only.** Every indicator uses backward-looking windows
(`.rolling()`, `.ewm()`, or `src/indicators.py`), no centered windows, no `.shift(-k)`
in any feature. The VIX percentile is a **trailing 252-day** rank, not a full-sample
rank; a full-sample percentile would leak the future distribution of volatility into
every past row. The only forward-looking operation in the entire pipeline is the
label itself (which is what a target *is*).

**Per-fold scaling.** Standardization is fit on each walk-forward fold's *training* rows
only, then applied to the test rows. Fitting a scaler on the whole series would leak
test-period means and variances into training, a quieter cousin of the same bug.

**Fractional differentiation.** Most technical features discard the price level's long
memory, while raw prices are highly persistent. The pipeline adds a
fractionally-differenced log-price feature (fixed-width-window method, order *d = 0.4*)
to reduce persistence while retaining some level information. This fixed order is a
modeling assumption, not a guarantee that every transformed series is stationary. The
filter is causal and covered by the same invariance tests as every other feature.

**Walk-forward, not a random split.** Train on a rolling ~2-year window, test on the
next ~6 months, roll forward. Concatenating the non-overlapping test blocks yields one
continuous out-of-sample series. The final label horizon is purged from every training
window so a training target cannot use the first test date's close. A random 80/20 split
on time-series data lets the model train on the future and is a methodological error.

**Verification (`tests/test_leakage.py`).** The key test,
`test_feature_truncation_invariance`, computes features on the full history, then
recomputes them on the history *truncated* at day *t*, and asserts every feature value
at *t* matches to tight numerical tolerance. If any
feature had peeked past *t*, removing the future would change it and the test would fail.
Additional tests confirm the VIX percentile is trailing, labels depend only on the
realized forward return, and the final rows (no known future) carry no label.

## The backtest engine

Built from scratch (no black-box library) so every line is explainable. It takes target
weights and realized forward returns, marks holdings through each return, and computes
the next rebalance against the resulting drifted weights. The buy-and-hold benchmark
makes one initial equal-weight purchase and then lets weights drift without rebalancing.

- **Position sizing:** fixed fraction of capital per position, equal-weighted across the
  universe. Each name gets `1/N` when signaled long, `0` otherwise; unused capital sits
  in cash earning nothing (conservative).
- **Costs:** turnover compares the new target with the actual pre-trade weights after
  market drift, charged at `cost_bps/1e4` per unit (default 7.5 bps, adjustable).

`test_backtest.py` hand-checks the accounting (equity, cost, turnover) on a two-day
example, plus zero-cost and zero-position edge cases.

## Evaluation framing

**Primary:** Sharpe, Sortino, max drawdown, win rate, average win/loss, CAGR,
annualized volatility, all net of costs.

**Secondary:** classification accuracy, reported but explicitly demoted.

**Benchmarks to beat, in order of rigor:** (1) equal-weight buy-and-hold on the same
universe, then (2) a naive moving-average crossover rule. The ML model has to clear
*both*, not just the market.

**Regime split:** performance is reported separately for calm (VIX < 20), normal
(20-30), and stressed (VIX > 30) days using the most recent VIX close available before
each trade. ML performance is shown alongside both benchmarks and active return versus
buy-and-hold; the splits are descriptive rather than formal significance tests.

## Model checks

`scripts/run_experiments.py` reruns the full pipeline with one thing changed at a time
and saves the results to `results/<universe>/`. The dashboard shows them under
**Extra model checks**. Every check uses the same walk forward windows, backtest, and
costs as the main model, and every scaler, PCA, and tuning step is fit on training
rows only.

| Check | What it does | Why it matters |
|---|---|---|
| **ROC-AUC** | Scores how well the predicted up probability ranks up days above other days, overall and per window | Accuracy looks bad when one outcome is more common (always guessing "up" scores about 53%). AUC is not affected by that, so 0.5 cleanly means no skill |
| **Ablation** | Removes one feature family (trend, momentum, volatility, volume, long memory, VIX), retrains, and compares | Shows which inputs actually help. Removing the VIX features is a direct test of H2 |
| **Tuning** | Grid searches tree depth and leaf size inside each two year training window, using three time ordered splits scored by ROC-AUC | Tests whether fixed settings leave performance on the table, without ever using test data to choose settings |
| **PCA** | Measures how much the 17 features overlap, then trains on principal components | Many indicators measure the same trend. Fewer, cleaner inputs can reduce noise |
| **S&P 500** | Runs the same model on about 500 stocks | Checks whether the result holds on a much larger, more varied universe |

```bash
python scripts/run_experiments.py                        # all checks, 15 stocks
python scripts/run_experiments.py --only ablation pca    # choose checks
```

Tuning is the slowest check because it adds 18 model fits per window.

### Scaling to the S&P 500

At about 500 tickers and 1.5 million daily rows, loading one big file into pandas stops
being practical, so `src/large_data.py` uses three big data techniques:

1. **Partitioned Parquet storage.** Prices are written as a Hive partitioned dataset,
   `data/sp500/ticker=AAPL/year=2020/part-0.parquet`. Parquet is columnar and
   compressed, so reading only `close` never touches the other columns.
2. **Partition pruning.** `large_data.load_prices(tickers=..., start_year=...)` pushes
   filters down to pyarrow, which skips whole folders before any data is read.
3. **Parallel feature engineering.** Tickers are independent, so
   `features.add_features(..., n_jobs=-1)` spreads them across CPU cores with joblib.
   A test confirms the parallel output matches the serial output exactly.

```bash
python -m src.large_data                                           # download (needs internet)
python scripts/run_experiments.py --universe sp500 --only main     # run the model
```

The S&P 500 dataset is not committed to git because of its size; the saved results in
`results/sp500/` are.

## Honest limitations

This section exists so the project holds up under questioning rather than falling apart.

- **Daily close-auction execution is idealized.** Signals use prior-session data and can
  be submitted before the close, but real fills still face auction imbalance, market
  impact beyond the flat cost, and operational latency.
- **Survivorship and point-in-time membership.** The universe is today's large-caps; a
  fully point-in-time study would reconstruct index membership historically. Auto-adjusted
  prices also fold splits/dividends back in a way that is uniform but not strictly
  point-in-time at the corporate-action boundary.
- **Costs are a flat estimate.** 7.5 bps per unit turnover is defensible for liquid
  large-caps but is a single number, not a modeled cost curve. Results are shown as a
  function of this assumption in the app.
- **Multiple-comparisons / researcher degrees of freedom.** Choosing indicators,
  windows, thresholds, and the ±0.5% deadband is a form of search. Walk-forward guards
  against in-sample overfitting but not against the meta-overfitting of trying many
  designs; treat a single strong Sharpe with appropriate skepticism.
- **Sharpe/Sortino use a ~0 risk-free rate** at daily granularity, which slightly
  flatters absolute levels; comparisons across strategies (all treated identically) are
  the meaningful part.
- **S&P 500 survivorship bias.** The large universe uses today's index members, so
  companies that were removed (often after falling) are missing from history. Results
  on that universe are likely better than a point in time study would show.
- **No leverage, no intraday, no regime-conditional sizing** in the baseline, these are
  natural extensions, not claims made here.

## Project structure

```
.
├── app.py                     # Streamlit dashboard (live view of the pipeline)
├── requirements.txt
├── scripts/
│   ├── run_pipeline.py        # CLI entry point
│   └── run_experiments.py     # ablation, tuning, PCA, S&P 500 checks
├── results/                   # saved check results shown in the dashboard
├── src/
│   ├── config.py              # universe, windows, walk-forward + backtest params
│   ├── data.py                # yfinance + VIX download and caching
│   ├── indicators.py          # indicator math (replaces the ta package)
│   ├── features.py            # causal technical indicators + feature groups
│   ├── labeling.py            # binary + 3-class targets
│   ├── models.py              # logreg / rf factory
│   ├── walkforward.py         # rolling-origin validation, tuning, PCA, ROC-AUC
│   ├── experiments.py         # ablation, tuning, and PCA comparisons
│   ├── large_data.py          # S&P 500 partitioned Parquet storage
│   ├── backtest.py            # custom day-by-day engine + weight builders
│   ├── benchmarks.py          # buy-and-hold + MA crossover
│   ├── evaluation.py          # Sharpe/Sortino/drawdown/win-rate/accuracy
│   ├── regime.py              # VIX-regime performance split
│   └── pipeline.py            # end-to-end glue (used by CLI and app)
├── tests/
│   ├── conftest.py            # synthetic OHLCV + VIX fixtures (offline)
│   ├── test_leakage.py        # truncation-invariance no-lookahead proof
│   ├── test_backtest.py       # hand-checked accounting
│   ├── test_fracdiff.py        # fractional-difference causality/persistence
│   ├── test_indicators.py     # indicator math matches the original ta values
│   ├── test_extensions.py     # AUC, ablation, PCA, tuning splits, Parquet storage
│   └── test_pipeline.py       # end-to-end smoke test
└── data/                      # parquet cache (created by src/data.py)
```

## A note on the test results you'll see

On synthetic random-walk data (used by the test suite), the pipeline correctly finds
**no edge**, ~50% accuracy and a Sharpe below buy-and-hold. That is the *right* result:
a leaking pipeline would instead report suspiciously strong out-of-sample numbers on
data that contains no signal. Passing that sniff test is part of why the leakage tests
matter.

## Live dashboard

The Streamlit app is deployable as-is on Streamlit Community Cloud. Point a new app at
this repository, branch `main`, main file `app.py`.

A small price and VIX cache is committed under `data/` so the deployed app renders
immediately and does not depend on Yahoo Finance being reachable from the host. To work
with fresher data locally, run `python -m src.data --force` to rebuild the cache.

## Repository layout

```
app.py            Streamlit dashboard, imports the same src/ modules as the CLI
src/              research pipeline (data, features, labeling, models,
                  walkforward, backtest, benchmarks, evaluation, regime, config)
scripts/          command line entry points
tests/            pytest suite, including an empirical no-lookahead check
data/             cached daily prices and VIX (parquet)
```

## Data

Daily OHLCV and VIX history retrieved from Yahoo Finance via `yfinance`. The cached
parquet files in `data/` are a snapshot for reproducibility and convenience.

## License

MIT, see `LICENSE`. This is a research and coursework project. Nothing here is
investment advice.
