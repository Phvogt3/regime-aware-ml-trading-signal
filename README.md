# Regime Aware ML Trading Signal

Can a machine learning model built from technical indicators beat simple strategies after trading costs, and does the answer change when markets get volatile?

**Short answer: no.** The model earns positive returns but well below buy and hold on a risk adjusted basis, and its ROC-AUC of 0.500 means it has no real skill at ranking days. Scaling from 15 stocks to 502 makes the gap worse, not better. The value of this project is that the result is measured honestly, with no lookahead, real trading costs, and a test that proves the model never sees the future.

[Dashboard link](https://phvogt3-regime-aware-ml-trading-signal-app-uophii.streamlit.app/)

Python 3.12

## The question

Traders watch indicators like RSI and moving averages constantly, but each one predicts very little on its own. This project combines 17 of them into one model and asks two things:

1. Does the model earn more return per unit of risk than simply holding the stocks, after paying to trade?
2. Does any advantage depend on whether the market is calm or stressed?

Framing it as "can I predict stock prices" would miss the point. A model can be right 55% of the time and lose money to costs, or right 45% of the time and make money if the wins are bigger. Return per unit of risk is the measure that matters.

## Data

15 large, heavily traded stocks across technology, finance, and consumer companies, plus SPY as a market reference and the VIX, which tracks how much volatility the market expects. Daily prices come from Yahoo Finance, adjusted for splits and dividends, covering January 2014 through mid 2026. That window includes the 2015 to 2019 bull market, the 2020 COVID crash, the 2022 selloff when rates rose, and the recovery after.

A cached copy lives in `data/`, so nothing needs to download for the code to run. There is also a 502 stock S&P 500 version, described further down.

## Features

Every feature for a given day uses only data available by the previous day's close. There are 17, in six families:

| Family | Features |
|---|---|
| Trend | Fast and slow moving average gaps (simple and exponential), MACD line, signal, and histogram |
| Momentum | RSI, rate of change, stochastic oscillator (%K and %D) |
| Volatility | Bollinger Band width, price position inside the bands, rolling return volatility |
| Volume | Volume against its recent average, scaled on balance volume |
| Long memory | Fractionally differenced log price |
| Market conditions | VIX level, VIX rank against the past year |

Two of these deserve a note:

**Fractional differencing.** Raw prices trend and drift, which breaks most models. Daily returns fix that but throw away all memory of the price level. Fractional differencing sits in between: it removes most of the drift while keeping some level information.

**The VIX rank is trailing.** It compares today's VIX to the past 252 days only. Ranking against the whole history would tell the model about future volatility, which is a subtle form of cheating.

## How it is tested

**Target.** Will the stock close higher tomorrow? There is also a three class version that adds a "flat" label for moves under 0.5 percent.

**Models.** Logistic regression as a simple, readable baseline, and a random forest as the main model: 300 trees, max depth 6, at least 50 samples per leaf. Those settings are deliberately restrictive, because daily direction is mostly noise and a deeper model would just memorize it.

**Walk forward validation.** Train on two years, predict the next six months, move the window forward, repeat. Joining the test blocks gives one continuous set of predictions on data the model never trained on. Shuffling time series data randomly lets a model learn from the future and inflates every number, which is the most common mistake in this kind of project.

**Scaling per window.** The feature scaler is fit on training rows only. Fitting it on everything would leak the test period's averages into training.

**Purging.** The last day of each training window is dropped, because its label depends on the first test day's price.

**Backtest.** A custom day by day engine, no outside backtesting library. Each stock the model likes gets an equal share of the portfolio, unused money sits in cash earning nothing, and every trade pays 7.5 basis points by default. Costs are charged on real turnover, comparing the new target against what the portfolio actually holds after prices moved. Both benchmarks run through the same engine, so any difference comes from the signals and not the accounting.

## Proving there is no lookahead

`tests/test_leakage.py` does this empirically. It computes every feature on the full price history, then recomputes it on the history cut off at a chosen day, and requires every value on that day to match. A feature that peeked at the future would change when the future is removed, and the test would fail. Other tests confirm the VIX rank is trailing, labels depend only on realized future returns, and the last rows carry no label.

`tests/test_backtest.py` checks the portfolio math by hand on a two day example, including equity, costs, and turnover.

## Results, 15 stocks

| Strategy | Yearly growth | Sharpe | Max drawdown |
|---|---|---|---|
| ML model | 6.8% | 0.54 | -34.5% |
| Buy and hold | 10.0% | 1.21 | -40.3% |
| Moving average rule | 14.4% | 1.27 | -21.8% |

The model loses on every count. It trails both benchmarks on return per unit of risk, its yearly alpha is -5.2%, and trading costs eat 48 percent of what it earns before costs. Accuracy is 50.2 percent against 53.1 percent for always guessing the more common outcome, and ROC-AUC is 0.500, which means no ranking skill at all.

The one thing it does well is take less risk. Its beta to the basket is 0.39 and it holds cash during weak stretches, so its drawdown is smaller than buy and hold.

**By market volatility:**

| Regime | Trading days | ML Sharpe |
|---|---|---|
| Calm (VIX under 20) | 1,729 | 0.31 |
| Normal (VIX 20 to 30) | 607 | 0.65 |
| Stressed (VIX over 30) | 154 | 1.40 |

Results do get better as volatility rises, which is the pattern the project was built to look for. The catch is that the stressed regime covers only 154 days, and the S&P 500 run below shows that number shrinking once there is more data.

## Results, 502 stocks

| | 15 stocks | S&P 500 |
|---|---|---|
| ML Sharpe | 0.54 | 0.33 |
| Buy and hold Sharpe | 1.21 | 1.01 |
| ML yearly growth | 6.8% | 3.8% |
| Max drawdown | -34.5% | -30.9% |
| ROC-AUC | 0.500 | 0.498 |
| Stressed regime Sharpe | 1.40 | 0.66 |

Nobody can dismiss the 15 stock result as a small or cherry picked sample, because 1.45 million rows say the same thing. The stressed regime Sharpe falling from 1.40 to 0.66 is the most useful detail here: with 33 times more data, the apparent edge in volatile markets mostly disappeared, which is what noise does.

## Model checks

`scripts/run_experiments.py` reruns the whole pipeline with one thing changed and saves results to `results/`, which the dashboard displays. Every check uses the same windows, backtest, and costs.

**Ablation.** Remove one feature family, retrain, compare:

| Removed | Sharpe | Change |
|---|---|---|
| Nothing (all 17) | 0.54 | |
| Trend | 0.49 | -0.05 |
| Momentum | 0.44 | -0.10 |
| Volatility | 0.60 | +0.06 |
| Volume | 0.61 | +0.07 |
| Long memory | 0.62 | +0.08 |
| VIX | 0.67 | +0.13 |

Momentum and trend features help. The rest hurt slightly, and the VIX features hurt most, which is awkward for a project about volatility regimes and is reported as is. ROC-AUC stays at about 0.50 in every version, so none of these changes create real skill.

**Tuning.** Searching tree depth and leaf size inside each training window, scored by ROC-AUC on later dates within that window, raised Sharpe from 0.54 to 0.68 but left AUC at 0.500. The settings it picked jumped around between windows, so the gain looks like luck in which days the model sat in cash rather than a better model.

**PCA.** The first 3 components hold 73 percent of the variation in all 17 features, and 7 reach 90 percent, so the indicators overlap heavily. Training on components did not help: 0.47 keeping 90 percent of variance, 0.58 with 5 components.

## Scaling to the S&P 500

At roughly 1.5 million daily rows, a single file loaded into pandas stops being a good idea, so `src/large_data.py` uses three techniques:

1. **Partitioned columnar storage.** Prices are written as `data/sp500/ticker=AAPL/year=2020/part-0.parquet`. Parquet stores columns separately and compressed, so reading only the closing price never touches the other columns.
2. **Partition pruning.** Ticker and year filters are pushed down to pyarrow, which skips entire folders before reading anything.
3. **Parallel feature engineering.** Stocks are independent, so features are computed across CPU cores. A test confirms the parallel output matches the serial output exactly. Building features for 1.45 million rows takes under 3 seconds.

The dataset itself (about 97 MB across 6,286 files) is not committed. Saved results are.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running it

```bash
streamlit run app.py                                  # the dashboard
python scripts/run_pipeline.py --model rf             # headline results in the terminal
python scripts/run_pipeline.py --model logreg --tune  # baseline model with tuning
python scripts/run_experiments.py                     # all model checks
pytest -q                                             # 38 tests, runs offline
```

The cached data covers everything above. To refresh prices, run `python -m src.data --force`, which needs internet. For the S&P 500 universe, run `python -m src.large_data` first, then `python scripts/run_experiments.py --universe sp500 --only main`.

## Honest limitations

1. **Execution is idealized.** Signals are known before the closing auction, but real fills face order imbalance, market impact beyond the flat cost, and delays.
2. **Survivorship bias.** The stock lists are today's members, so companies that were dropped, often after falling, are missing. This flatters results, and more so for the S&P 500 universe.
3. **Costs are one flat number.** 7.5 basis points is reasonable for large, liquid stocks, but it is an estimate rather than a modeled cost curve. The dashboard slider shows how much the result moves with it.
4. **Many choices were searched.** Picking indicators, windows, and the 0.5 percent flat band is its own form of fitting. Walk forward testing guards against overfitting within a run, not against trying many designs.
5. **Sharpe and Sortino assume a zero risk free rate**, which flatters absolute levels. Comparisons between strategies are unaffected, since all are treated the same way.
6. **No shorting, no leverage, no intraday data, and no sizing by confidence.** Those are natural extensions, not claims made here.
7. **An early version of the backtest traded only the 15 default stocks** no matter which universe was modeled, which made the first S&P 500 run look better than it was. `tests/test_extensions.py` now checks that the portfolio trades whatever universe it is given.

## Project structure

```
app.py                      Streamlit dashboard
src/
  config.py                 universe, windows, costs, tuning grids
  data.py                   price and VIX download and cache
  indicators.py             indicator math
  features.py               the 17 causal features, grouped by family
  labeling.py               up or down, and up, flat, or down targets
  models.py                 logistic regression and random forest
  walkforward.py            time ordered validation, tuning, PCA, ROC-AUC
  backtest.py               day by day portfolio engine
  benchmarks.py             buy and hold, moving average rule
  evaluation.py             Sharpe, Sortino, drawdown, accuracy, AUC
  regime.py                 results split by VIX level
  pipeline.py               end to end glue used by the app and scripts
  experiments.py            ablation, tuning, and PCA comparisons
  large_data.py             S&P 500 partitioned storage
scripts/
  run_pipeline.py           command line run
  run_experiments.py        model checks
tests/                      8 test files, including the no lookahead proof
results/                    saved check results shown in the dashboard
data/                       cached prices and VIX
```

## Data source

Daily prices and VIX from Yahoo Finance via yfinance. This is a research project, not investment advice.
