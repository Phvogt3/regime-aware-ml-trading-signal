"""
ML Trading Signal Research Dashboard

Shows whether a machine learning signal built from technical indicators beats simple
benchmarks after trading costs, and whether the result depends on market volatility.

The dashboard runs the same code in src/ as the command line pipeline, so the
numbers shown here come straight from the research code.
"""

import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src import config, data, evaluation, models, pipeline
from src.config import BACKTEST as BT

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="ML Trading Signal Backtest",
    page_icon=None,
    layout="wide",
)

# Bump this whenever the pipeline output shape or metric set changes; it is part
# of the cache key, so bumping it invalidates any stale @st.cache_data results.
CACHE_VERSION = 3

STRAT_COLORS = {
    "ML strategy": "#38bdf8",
    "Buy & hold": "#f472b6",
    "MA crossover": "#facc15",
}
REGIME_COLORS = {"calm": "#4ade80", "normal": "#facc15", "stressed": "#f87171"}
MODEL_LABELS = {"logreg": "logistic regression", "rf": "random forest"}


def style_fig(fig, height=420):
    """Apply the clean dark theme used in the FOMC dashboard."""
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(size=13),
        hoverlabel=dict(font_size=13),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    return fig


# ============================================================
# DATA + PIPELINE (cached; identical code path to scripts/run_pipeline.py)
# ============================================================

@st.cache_data(show_spinner=False)
def load_cached_data():
    prices_path = os.path.join(config.DATA_DIR, config.PRICES_FILE)
    vix_path = os.path.join(config.DATA_DIR, config.VIX_FILE)
    if not (os.path.exists(prices_path) and os.path.exists(vix_path)):
        return None, None
    return data.load_prices(), data.load_vix()


@st.cache_data(show_spinner=True)
def run_pipeline_cached(model_name, target, tickers, start, end, cost_bps,
                        cache_version=CACHE_VERSION):
    prices, vix = load_cached_data()
    out = pipeline.run(prices, vix, model_name=model_name, target=target,
                       tickers=list(tickers), start=start, end=end,
                       cost_bps=cost_bps)
    return {
        "summary": out.summary,
        "results_daily": {k: v.daily for k, v in out.results.items()},
        "regime_table": out.regime_table,
        "classification": out.classification,
        "importances": out.importances,
        "cost_drag": out.cost_drag,
        "predictions": out.predictions,
        "n_oos_dates": out.predictions["date"].nunique(),
    }


def fmt_pct(x):
    return "n/a" if pd.isna(x) else f"{x*100:.1f}%"


def fmt_num(x, d=2):
    return "n/a" if pd.isna(x) else f"{x:.{d}f}"


def gv(row, key):
    """Safe lookup: return NaN if a metric column is absent (e.g. stale cache)."""
    return row[key] if key in row.index else np.nan


# ============================================================
# UI
# ============================================================

st.title("Can a machine learning trading signal beat simple strategies after "
         "costs, and does it depend on market volatility?")
st.caption(
    "This dashboard tests a trading model built from technical indicators on 15 "
    "large US stocks from 2014 to 2026. The model is trained and tested in time "
    "order, every trade pays a cost, and the main score is return per unit of risk."
)

prices, vix = load_cached_data()

if prices is None:
    st.warning(
        "No saved data was found. Click the button below to download daily prices "
        "for 16 tickers and the VIX from Yahoo Finance. It takes less than a minute."
    )
    if st.button("Download data now", type="primary"):
        with st.spinner("Downloading prices and VIX..."):
            data.build_cache(force=False)
        st.cache_data.clear()
        st.success("Data saved.")
        st.rerun()
    st.stop()

# ---- About this study ----
st.subheader("About this study")
st.markdown(
    """
**Research question:** Traders use technical indicators like RSI and moving averages
all the time, but each one predicts very little on its own. This project combines
many indicators into one machine learning model and asks two questions. Does the
model earn more return per unit of risk than simple strategies after paying trading
costs? And does that result change when the market is calm or stressed?

**Data:** 15 large, heavily traded stocks from technology, finance, and consumer
companies, plus SPY as a market reference and the VIX, which measures how much
volatility the market expects. Daily prices come from Yahoo Finance and are adjusted
for stock splits and dividends. The data runs from January 2014 to mid 2026, which
covers the 2015 to 2019 bull market, the 2020 COVID crash, the 2022 selloff when
interest rates rose, and the recovery after.

**Features:** Every feature for a given day only uses data available by the
previous day's close, so the model never sees the future. The inputs fall into six
groups:

* **Trend:** gaps between short and long moving averages, and MACD.
* **Momentum:** RSI, rate of change, and the stochastic oscillator.
* **Volatility:** Bollinger Band width, where the price sits inside the bands, and
  recent return volatility.
* **Volume:** volume compared to its recent average, and a scaled on balance volume.
* **Long memory:** a fractionally differenced price series, which keeps more price
  history than daily returns while staying stable enough to model.
* **Market conditions:** the VIX level and how it ranks against the past year.

**Hypotheses:**

* **Main:** the model earns a positive return per unit of risk after trading costs.
* **H1:** the model beats both an equal weight buy and hold portfolio and a simple
  moving average crossover rule.
* **H2:** the model's results depend on the volatility regime.
* **H3:** prediction accuracy is a poor measure of whether the model makes money, so
  it is only reported as a side check.

**How the model is tested:** The model predicts whether each stock will go up or
down the next day. A three class version adds a "flat" label for moves smaller than
0.5 percent. Two models are compared: logistic regression as a simple baseline and a
random forest as the main model. Testing uses walk forward validation. The model
trains on two years of data, predicts the next six months, then moves the window
forward and repeats. This keeps everything in time order, because shuffling time
series data lets the model learn from the future and makes results look better than
they are. A custom backtest then turns the predictions into daily trades. Each stock
the model expects to rise gets an equal share of the portfolio, the rest stays in
cash, and every trade pays a cost. Both benchmarks run through the same backtest, so
the comparison is fair.

**Key finding:** Over the full period, the model does not beat buy and hold on
return per unit of risk. It spends more time in cash during weak stretches, so its
swings and its worst loss are smaller. Results differ by volatility regime, but the
regimes with the best results have few trading days, so those results are treated
as a possible pattern that needs more data. The main value of the project is the
testing process, which makes even a weak result believable. Results for your current
settings are at the bottom of the page.
    """
)

with st.expander("Technical details on the models and metrics", expanded=False):
    st.markdown(
        """
**Scaling features.** In each walk forward window, a `StandardScaler` is fit on the
training rows only and then applied to the test rows. Fitting it on all the data
would let averages from the test period leak into training.

**Logistic regression.** The baseline uses an L2 penalty (`C = 1.0`) and balanced
class weights. Its coefficients show which features push predictions up or down.

**Random forest.** The main model uses 300 trees, a max depth of 6, at least 50
samples per leaf, and the square root of the feature count at each split. These
settings keep the trees small on purpose, because daily stock direction is mostly
noise and deep trees would memorize it.

**Walk forward validation.** The model is retrained in every window and predicts
probabilities for the next six months. Joining the test windows gives one continuous
set of predictions on data the model never trained on. The last label period is
dropped from each training window so no training label overlaps the test window.
Each prediction also gets a score from minus 1 to 1, equal to the chance of an up
move minus the chance of a down move, which can be used to size positions.

**Performance metrics.**

* **Sharpe ratio:** average daily return divided by its volatility, scaled to a year.
* **Sortino ratio:** the same idea, counting only downside volatility.
* **Calmar ratio:** yearly growth rate divided by the worst drawdown.
* **Beta:** how much the strategy moves with the buy and hold portfolio.
* **Alpha:** yearly return left over after accounting for beta.
* **Information ratio:** extra return over buy and hold, divided by how much that
  extra return varies.
* **Return t stat:** whether the average daily return is meaningfully different from
  zero. Treat it as a rough guide, since it does not adjust for returns being related
  from one day to the next or for testing many settings.

**Leakage test.** The test suite computes every feature on the full history, then
again on history cut off at a certain day, and the values on that day must match. If
any feature used future data, cutting the history would change it and the test would
fail.
        """
    )

# ---- Sidebar controls ----
st.sidebar.header("Controls")
avail = models.available_models()
model_name = st.sidebar.selectbox(
    "Model", options=avail,
    format_func=lambda m: {"logreg": "Logistic Regression (baseline)",
                           "rf": "Random Forest (main)"}.get(m, m),
    index=avail.index("rf") if "rf" in avail else 0,
)

target = st.sidebar.radio(
    "Target", options=["binary", "3class"],
    format_func=lambda t: {"binary": "Up or down",
                           "3class": "Up, flat, or down"}[t],
)

all_tickers = sorted([t for t in prices["ticker"].unique() if t in config.TICKERS])
picked = st.sidebar.multiselect("Stocks to include",
                                options=all_tickers, default=all_tickers)

dmin = prices["date"].min().to_pydatetime()
dmax = prices["date"].max().to_pydatetime()
date_range = st.sidebar.slider("Date range", min_value=dmin, max_value=dmax,
                               value=(dmin, dmax))

cost_bps = st.sidebar.slider("Trading cost (basis points)",
                             min_value=0.0, max_value=25.0,
                             value=float(BT.cost_bps), step=0.5)
st.sidebar.caption("Charged every time the portfolio trades. One basis point is "
                   "0.01 percent, and 5 to 10 is a realistic cost for a retail "
                   "trader. Set it to 0 to see results before costs.")

if st.sidebar.button("Download latest data"):
    try:
        with st.spinner("Downloading and checking new data..."):
            data.build_cache(force=True)
        st.cache_data.clear()
        st.success("Data updated.")
        st.rerun()
    except Exception as exc:
        st.error(f"Update failed, so the previous data was kept. {exc}")

if len(picked) < 3:
    st.info("Select at least three stocks to run the backtest.")
    st.stop()

try:
    R = run_pipeline_cached(model_name, target, tuple(picked),
                            pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]),
                            cost_bps)
except ValueError as exc:
    st.info(str(exc))
    st.stop()

summary = R["summary"]
ml = summary.loc["ML strategy"]
bh = summary.loc["Buy & hold"]
ma = summary.loc["MA crossover"]

# ---- Overview metrics ----
st.divider()
c1, c2, c3, c4 = st.columns(4)
c1.metric("ML Sharpe (after costs)", fmt_num(gv(ml, "sharpe")),
          help="Return per unit of risk for the ML strategy after trading costs, "
               "scaled to a year. This is the main score.")
c2.metric("Information ratio", fmt_num(gv(ml, "information_ratio")),
          help="Extra yearly return over buy and hold, divided by how much that "
               "extra return varies.")
c3.metric("Beta vs buy and hold", fmt_num(gv(ml, "beta")),
          help="How closely the strategy moves with the equal weight buy and hold "
               "portfolio. A value of 1.0 means it moves exactly with it.")
c4.metric("Costs as share of return",
          "n/a" if pd.isna(R["cost_drag"]) else f"{R['cost_drag']*100:.0f}%",
          help="Share of the strategy's return before costs that trading costs "
               "used up.")

# ---- Equity curve ----
st.divider()
st.subheader("Portfolio value: ML strategy vs benchmarks")
fig = go.Figure()
for label, daily in R["results_daily"].items():
    fig.add_trace(go.Scatter(
        x=daily.index, y=daily["equity"], mode="lines", name=label,
        line=dict(color=STRAT_COLORS.get(label), width=2),
    ))
fig.update_yaxes(title="Portfolio value ($)")
style_fig(fig, height=440)
st.plotly_chart(fig, width="stretch")
st.caption(
    "Each line shows how a $100,000 portfolio grew over the test period. A higher "
    "ending value means more total return, and a smoother line means less risk along "
    "the way. All three strategies use the same backtest and pay the same costs, so "
    "any gap between them comes from the trading signals alone. For the model (blue) "
    "to be worth it, it should finish above buy and hold (pink) and the moving average "
    "rule (yellow), or finish close to them with a smoother path. Check the table "
    "below for risk, since a strategy can earn less in total and still be better per "
    "unit of risk."
)

# ---- Primary metrics table ----
st.divider()
st.subheader("Performance metrics (after costs)")
cols = ["CAGR", "sharpe", "sortino", "calmar", "max_drawdown", "beta", "alpha",
        "information_ratio", "return_t_stat", "win_rate", "ann_vol"]
labels = ["Yearly growth", "Sharpe", "Sortino", "Calmar", "Max drawdown", "Beta",
          "Alpha (yearly)", "Info ratio", "Return t stat", "Win rate",
          "Volatility (yearly)"]
show = summary.reindex(columns=cols).copy()   # reindex tolerates missing columns
show.columns = labels
for c in ["Yearly growth", "Max drawdown", "Alpha (yearly)", "Win rate",
          "Volatility (yearly)"]:
    show[c] = show[c].map(fmt_pct)
for c in ["Sharpe", "Sortino", "Calmar", "Beta", "Return t stat", "Info ratio"]:
    show[c] = show[c].map(lambda x: fmt_num(x))
st.dataframe(show, width="stretch")
st.markdown(
    "**How to read this table.** Each row is a strategy and each column is a measure, "
    "all after costs. Sharpe and Sortino show return per unit of risk, and higher is "
    "better. A value above about 1.0 is strong for a daily strategy. Calmar compares "
    "yearly growth to the worst drawdown. Beta, alpha, and info ratio all compare "
    "against buy and hold. A beta near 1.0 means the strategy moves with the market, "
    "a positive alpha means it earned extra return beyond that, and the info ratio "
    "shows how steady that extra return was. Buy and hold shows n/a for info ratio "
    "because it cannot be compared against itself. A return t stat above about 2.0 "
    "suggests the average daily return is more than luck."
)
class_description = "up or down" if target == "binary" else "up, flat, or down"
st.caption(
    f"The model's {class_description} accuracy was "
    f"{R['classification']['accuracy']*100:.1f} percent, compared to "
    f"{R['classification']['majority_baseline']*100:.1f} percent from always "
    f"guessing the most common outcome, across {R['classification']['n']:,} "
    f"predictions. Its ROC-AUC was {fmt_num(R['classification'].get('roc_auc'), 3)}. "
    "AUC measures how well the model ranks days that went up above days that did not, "
    "where 0.5 is a coin flip and 1.0 is perfect. It is a fairer check than accuracy "
    "when one outcome is more common. Both are only side checks. A model can be right "
    "more than half the time and still lose money after costs, or be right less than "
    "half the time and make money if its correct calls are bigger. Return per unit of "
    "risk, shown above, is what counts."
)

# ---- Regime breakdown ----
st.divider()
st.subheader("Performance by market volatility")
reg = R["regime_table"].reindex(["calm", "normal", "stressed"]).dropna(how="all")
colL, colR = st.columns([1.1, 1])
with colL:
    fig_r = go.Figure()
    for label, column in [("ML strategy", "sharpe"),
                          ("Buy & hold", "buy_hold_sharpe"),
                          ("MA crossover", "ma_crossover_sharpe")]:
        fig_r.add_trace(go.Bar(
            x=reg.index, y=reg[column], name=label,
            marker_color=STRAT_COLORS[label],
            text=[fmt_num(v) for v in reg[column]], textposition="outside"))
    fig_r.add_hline(y=0, line_color="rgba(255,255,255,0.35)", line_width=1)
    fig_r.update_yaxes(title="Sharpe in each regime")
    style_fig(fig_r, height=360)
    fig_r.update_layout(barmode="group")
    st.plotly_chart(fig_r, width="stretch")
with colR:
    reg_show = reg[["n_days", "ann_return", "sharpe", "buy_hold_sharpe",
                    "ma_crossover_sharpe", "information_ratio_vs_buy_hold",
                    "avg_vix"]].copy()
    reg_show.columns = ["Days", "ML yearly return", "ML Sharpe", "Buy and hold Sharpe",
                        "MA rule Sharpe", "ML info ratio vs buy and hold", "Avg VIX"]
    reg_show["ML yearly return"] = reg_show["ML yearly return"].map(fmt_pct)
    for c in ["ML Sharpe", "Buy and hold Sharpe", "MA rule Sharpe",
              "ML info ratio vs buy and hold"]:
        reg_show[c] = reg_show[c].map(lambda x: fmt_num(x))
    reg_show["Avg VIX"] = reg_show["Avg VIX"].map(lambda x: fmt_num(x, 1))
    st.dataframe(reg_show, width="stretch")

# Dynamic interpretation of the regime pattern actually observed.
_reg_note = ""
if not reg.empty and reg["sharpe"].notna().any():
    best = reg["sharpe"].idxmax()
    best_s = reg.loc[best, "sharpe"]
    best_days = int(reg.loc[best, "n_days"])
    total_days = int(reg["n_days"].sum())
    _reg_note = (
        f" In this run, the best Sharpe ratio was in the {best} regime at "
        f"{fmt_num(best_s)}. That regime covers {best_days} of {total_days} trading "
        f"days, and regimes with fewer days give less reliable numbers.")
st.caption(
    "This splits results by how volatile the market was before each trade, using "
    "the previous day's VIX close. Calm is below 20, normal is 20 to 30, and stressed "
    "is above 30. A bar above zero means a positive Sharpe ratio in that regime. The "
    "table adds the benchmark Sharpes and the model's info ratio against buy and "
    "hold. This shows whether the model works in all conditions or only in one, which "
    "is the main question of the project."
    + _reg_note
)

# ---- Signal exposure over time (smoothed for readability) ----
st.divider()
st.subheader("How much of the portfolio is invested")
ml_daily = R["results_daily"]["ML strategy"].copy()
ml_daily["exposure_ma"] = ml_daily["exposure"].rolling(21, min_periods=5).mean()
fig_e = go.Figure()
# Raw daily exposure kept faint in the background for honesty about variability.
fig_e.add_trace(go.Scatter(
    x=ml_daily.index, y=ml_daily["exposure"], mode="lines", name="Daily",
    line=dict(color="rgba(56,189,248,0.20)", width=1),
))
# 21-day moving average as the primary, readable series.
fig_e.add_trace(go.Scatter(
    x=ml_daily.index, y=ml_daily["exposure_ma"], mode="lines",
    name="21 day average", line=dict(color="#38bdf8", width=2.5),
))
fig_e.update_yaxes(title="Share invested", range=[0, 1.05])
style_fig(fig_e, height=340)
st.plotly_chart(fig_e, width="stretch")
st.caption(
    "The faint line is the share of the portfolio invested each day, and the solid "
    "line is its 21 day average. Each stock gets an equal slice when the model "
    "expects it to rise, and the rest stays in cash. When the average drops, the "
    "model sees fewer good trades and holds more cash, which lowers risk on its own."
)

# ---- Interactive per-stock explorer ----
st.divider()
st.subheader("Single stock explorer")
preds = R["predictions"]
sel = st.selectbox("Pick a stock", options=sorted(picked))
sub = preds[preds["ticker"] == sel].sort_values("date").copy()

if len(sub) > 5:
    up_class = 1 if target == "binary" else 2
    sub["is_long"] = (sub["pred_class"] == up_class).astype(float)
    turnover = sub["is_long"].diff().abs().fillna(sub["is_long"])
    sub["net_ret"] = sub["is_long"] * sub["fwd_return"] - (cost_bps / 1e4) * turnover
    sub["equity_signal"] = (1.0 + sub["net_ret"]).cumprod()
    sub["equity_hold"] = (1.0 + sub["fwd_return"]).cumprod()
    acc = float((sub["pred_class"] == sub["y_true"]).mean())
    pct_long = float(sub["is_long"].mean())
    tot_sig = float(sub["equity_signal"].iloc[-1] - 1.0)
    tot_hold = float(sub["equity_hold"].iloc[-1] - 1.0)
    # Risk context: the timing overlay trades total return for a smoother path.
    sig_dd = evaluation.max_drawdown(sub["equity_signal"], initial_value=1.0)
    hold_dd = evaluation.max_drawdown(sub["equity_hold"], initial_value=1.0)
    sig_vol = evaluation.ann_vol(sub["net_ret"])
    hold_vol = evaluation.ann_vol(sub["fwd_return"])
    sig_shp = evaluation.sharpe(sub["net_ret"])
    hold_shp = evaluation.sharpe(sub["fwd_return"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{sel} accuracy", f"{acc*100:.1f}%",
              help="Share of test days where the model's prediction for this stock "
                   "was correct.")
    m2.metric("Days invested", f"{pct_long*100:.0f}%",
              help="Share of days the model held this stock instead of cash.")
    m3.metric("Signal Sharpe", fmt_num(sig_shp),
              delta=f"{sig_shp - hold_shp:+.2f} vs holding",
              help="Return per unit of risk from following the model on this stock, "
                   "compared to just holding it. This is the fair comparison, since "
                   "the model is often in cash.")
    m4.metric("Signal max drawdown", fmt_pct(sig_dd),
              delta=f"{(abs(hold_dd) - abs(sig_dd))*100:+.1f} points vs holding",
              delta_color="normal",
              help="Largest drop from a peak when following the model on this stock. "
                   "A smaller drop than holding is the main benefit the model can offer.")

    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(
        x=sub["date"], y=sub["equity_signal"], mode="lines",
        name=f"{sel} following the model", line=dict(color="#38bdf8", width=2)))
    fig_s.add_trace(go.Scatter(
        x=sub["date"], y=sub["equity_hold"], mode="lines",
        name=f"{sel} buy and hold", line=dict(color="#f472b6", width=2)))
    fig_s.update_yaxes(title="Growth of $1")
    style_fig(fig_s, height=380)
    st.plotly_chart(fig_s, width="stretch")
    vol_word = "lowered" if abs(sig_vol) <= abs(hold_vol) else "raised"
    dd_word = "lowered" if abs(sig_dd) <= abs(hold_dd) else "raised"
    st.caption(
        f"Over this period, holding {sel} returned {fmt_pct(tot_hold)}, while "
        f"following the model returned {fmt_pct(tot_sig)}. The model {vol_word} "
        f"yearly volatility from {fmt_pct(hold_vol)} to {fmt_pct(sig_vol)} and "
        f"{dd_word} the max drawdown from {fmt_pct(hold_dd)} to {fmt_pct(sig_dd)}. "
        "On a stock that rises steadily, the model usually earns less in total because "
        "it sits in cash on many up days, and that alone does not mean it failed. The "
        "real test is whether it improves return per unit of risk, which the "
        "portfolio metrics above answer."
    )
else:
    st.info("Not enough test data for this stock with the current settings.")

# ---- Feature importance ----
if not R["importances"].empty:
    st.divider()
    st.subheader("Which features the model relies on")
    imp = R["importances"].head(12).iloc[::-1]
    fig_i = go.Figure(go.Bar(
        x=imp["importance"], y=imp["feature"], orientation="h",
        marker_color="#38bdf8"))
    style_fig(fig_i, height=380)
    fig_i.update_layout(showlegend=False)
    st.plotly_chart(fig_i, width="stretch")
    st.caption(
        "Longer bars mean the feature had more influence on the model's predictions, "
        "averaged across all walk forward windows. For the random forest this is tree "
        "importance, and for logistic regression it is the size of each coefficient. "
        "Influence is different from profit. A feature can help predict direction and "
        "still add little return once costs are paid."
    )

# ============================================================
# EXTRA MODEL CHECKS (precomputed by scripts/run_experiments.py)
# ============================================================

def load_results(universe):
    folder = os.path.join(config.RESULTS_DIR, universe)
    if not os.path.isdir(folder):
        return {}
    out = {}
    for name in ["main", "ablation", "tuning", "tuning_params", "pca", "pca_variance"]:
        path = os.path.join(folder, f"{name}.csv")
        if os.path.exists(path):
            has_index = name not in ("tuning_params", "pca_variance")
            out[name] = pd.read_csv(path, index_col=0 if has_index else None)
    meta_path = os.path.join(folder, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            out["meta"] = json.load(fh)
    return out


def check_table(df, extra=None):
    cols = {"sharpe": "Sharpe", "roc_auc": "ROC-AUC", "max_drawdown": "Max drawdown",
            "sharpe_calm": "Calm Sharpe", "sharpe_normal": "Normal Sharpe",
            "sharpe_stressed": "Stressed Sharpe"}
    cols.update(extra or {})
    t = df.reindex(columns=list(cols)).copy()
    t.columns = list(cols.values())
    for c in t.columns:
        if c == "Max drawdown":
            t[c] = t[c].map(fmt_pct)
        elif c in ("ROC-AUC", "AUC change"):
            t[c] = t[c].map(lambda x: fmt_num(x, 3))
        elif c == "Features":
            t[c] = t[c].map(lambda x: "n/a" if pd.isna(x) else f"{x:.0f}")
        else:
            t[c] = t[c].map(fmt_num)
    t.index.name = None
    return t


checks = load_results("default")
st.divider()
st.subheader("Extra model checks")
if not checks:
    st.info("Run `python scripts/run_experiments.py` to generate these checks.")
else:
    meta = checks.get("meta", {})
    st.caption(
        "These checks were run ahead of time because some take a long time to train. "
        f"They use the random forest, the up or down target, all 15 stocks, and a "
        f"trading cost of {meta.get('cost_bps', BT.cost_bps):g} basis points, so they "
        "do not change with the sidebar. Each check reruns the same walk forward "
        "validation and backtest with one thing changed.")

    tab_abl, tab_tune, tab_pca, tab_big = st.tabs(
        ["Removing features", "Tuning settings", "PCA", "S&P 500"])

    with tab_abl:
        abl = checks.get("ablation")
        if abl is None:
            st.info("Ablation results not found.")
        else:
            order = abl.drop(index="All features").sort_values("sharpe_change")
            fig_a = go.Figure(go.Bar(
                x=order["sharpe_change"], y=order.index, orientation="h",
                marker_color=["#f87171" if v < 0 else "#4ade80"
                              for v in order["sharpe_change"]],
                text=[f"{v:+.2f}" for v in order["sharpe_change"]],
                textposition="outside"))
            fig_a.add_vline(x=0, line_color="rgba(255,255,255,0.35)", line_width=1)
            fig_a.update_xaxes(title="Change in Sharpe vs using all features")
            style_fig(fig_a, height=330)
            fig_a.update_layout(showlegend=False)
            st.plotly_chart(fig_a, width="stretch")
            st.dataframe(check_table(abl, {"n_features": "Features",
                                           "auc_change": "AUC change"}),
                         width="stretch")
            vix_row = "Without vix"
            vix_note = ""
            if vix_row in abl.index:
                d = abl.loc[vix_row, "sharpe_change"]
                vix_note = (
                    f" Removing the two VIX features changed the Sharpe ratio by "
                    f"{d:+.2f}. This speaks directly to H2: if the model's results "
                    "depend on volatility, taking away its view of volatility should "
                    "hurt.")
            st.caption(
                "An ablation study removes one group of features, retrains the model "
                "from scratch, and measures what changed. A red bar means results got "
                "worse without that group, so the group was helping. A green bar means "
                "results improved without it, so the group was adding noise. Small "
                "changes in either direction can be random variation, since each "
                "retrain sees slightly different data splits." + vix_note)

    with tab_tune:
        tun = checks.get("tuning")
        params = checks.get("tuning_params")
        if tun is None:
            st.info("Tuning results not found.")
        else:
            st.dataframe(check_table(tun), width="stretch")
            if params is not None and not params.empty:
                pcols = [c for c in params.columns if c.startswith("param_")]
                if pcols:
                    combo = params[pcols].astype(str).agg(", ".join, axis=1)
                    counts = combo.value_counts().rename("Windows").to_frame()
                    counts.index = [
                        ", ".join(f"{c.replace('param_', '').replace('_', ' ')} {v}"
                                  for c, v in zip(pcols, idx.split(", ")))
                        for idx in counts.index]
                    st.markdown("**Settings chosen across the walk forward windows**")
                    st.dataframe(counts, width="stretch")
            st.caption(
                "The fixed version uses the same random forest settings in every "
                "window. The tuned version tries 6 combinations of tree depth and "
                "minimum leaf size inside each two year training window. It splits "
                "that window into three time ordered parts, trains on earlier dates, "
                "scores ROC-AUC on later dates, and keeps the best combination. Test "
                "data is never used to pick settings. If the chosen settings jump "
                "around from window to window, there is no stable best setting, which "
                "is common with noisy daily stock data.")

    with tab_pca:
        var = checks.get("pca_variance")
        pca_tbl = checks.get("pca")
        if var is None or pca_tbl is None:
            st.info("PCA results not found.")
        else:
            fig_p = go.Figure()
            fig_p.add_trace(go.Bar(x=var["component"], y=var["variance_share"],
                                   name="Each component", marker_color="#38bdf8"))
            fig_p.add_trace(go.Scatter(x=var["component"], y=var["cumulative_share"],
                                       name="Running total", mode="lines+markers",
                                       line=dict(color="#facc15", width=2)))
            fig_p.add_hline(y=0.9, line_dash="dot",
                            line_color="rgba(255,255,255,0.4)")
            fig_p.update_xaxes(title="Principal component", dtick=1)
            fig_p.update_yaxes(title="Share of variance", range=[0, 1.05],
                               tickformat=".0%")
            style_fig(fig_p, height=360)
            st.plotly_chart(fig_p, width="stretch")
            n90 = int((var["cumulative_share"] < 0.9).sum() + 1)
            top3 = float(var["cumulative_share"].iloc[min(2, len(var) - 1)])
            st.dataframe(check_table(pca_tbl, {"avg_components": "Features"}),
                         width="stretch")
            st.caption(
                "PCA combines features that move together into new summary features "
                "called components, ordered from most to least informative. The chart "
                f"shows that the first 3 components hold {top3:.0%} of the variation in "
                f"all 17 features, and {n90} components reach 90 percent (dotted line). "
                "That means many indicators measure nearly the same thing, which makes "
                "sense because moving averages, MACD, and rate of change all track "
                "trend. PCA is fit only on each training window. The table shows "
                "whether training on the smaller set of components helps or hurts.")

    with tab_big:
        big = load_results("sp500")
        if not big or "main" not in big:
            st.info(
                "Not run yet. On a machine with internet, run "
                "`python -m src.large_data` to download the S&P 500, then "
                "`python scripts/run_experiments.py --universe sp500 --only main`.")
        else:
            bmeta = big.get("meta", {})
            both = pd.concat([checks.get("main", pd.DataFrame()).assign(Universe="15 stocks"),
                              big["main"].assign(Universe="S&P 500")])
            both.index = both.pop("Universe")
            st.dataframe(check_table(both, {"buy_hold_sharpe": "Buy and hold Sharpe"}),
                         width="stretch")
            storage = bmeta.get("storage", {})
            st.caption(
                f"The same model on {bmeta.get('tickers', 'n/a')} stocks and "
                f"{bmeta.get('feature_rows', 0):,} daily rows. Prices are stored as "
                f"{storage.get('files', 'n/a')} Parquet files "
                f"({storage.get('megabytes', 0):.0f} MB) split into folders by ticker "
                "and year, so loading a few stocks or years skips every other folder. "
                "Features are computed on all CPU cores at once. The stock list is "
                "today's S&P 500, so companies that left the index are missing, which "
                "likely makes these results look better than they really were.")

# ============================================================
# RESULTS
# ============================================================

st.divider()
st.subheader("Results")

ml_sh, bh_sh, ma_sh = gv(ml, "sharpe"), gv(bh, "sharpe"), gv(ma, "sharpe")
ml_alpha = gv(ml, "alpha")
beats_bh = ml_sh > bh_sh
beats_ma = ml_sh > ma_sh
acc = R["classification"]["accuracy"]

model_label = MODEL_LABELS.get(model_name, model_name)

# ---- Main result for this run
main_v = "Positive in this run" if ml_sh > 0 else "Not positive in this run"
main_h = ("that a machine learning model built from technical indicators earns a "
          "positive return per unit of risk after trading costs")
if ml_sh > 0:
    main = (
        f"**Main ({main_v}):** {main_h}. The {model_label} model had a Sharpe ratio of "
        f"{fmt_num(ml_sh)} after costs and a yearly alpha of {fmt_pct(ml_alpha)}, so it "
        f"earned a positive risk adjusted return in this test. This is an estimate from "
        f"one sample and does not prove the edge will last. H1 checks whether it beats "
        f"simpler options.")
else:
    main = (
        f"**Main ({main_v}):** {main_h}. The {model_label} model had a Sharpe ratio of "
        f"{fmt_num(ml_sh)} after costs, so it did not earn a positive risk adjusted "
        f"return over the full period.")

# ---- H1: does it beat both benchmarks?
if beats_bh and beats_ma:
    h1_v = "Observed"
    h1_tail = ("The model beat both, so it added value over just holding and over a "
               "simple rule.")
elif beats_bh or beats_ma:
    h1_v = "Partly observed"
    which = "buy and hold" if beats_bh else "the moving average rule"
    other = "the moving average rule" if beats_bh else "buy and hold"
    h1_tail = f"The model beat {which} and trailed {other}."
else:
    h1_v = "Not observed"
    h1_tail = ("The model trailed both, so the extra modeling did not improve on just "
               "holding or on a simple moving average rule with these settings.")
h1 = (
    f"**H1 ({h1_v}):** that the model beats both an equal weight buy and hold "
    f"portfolio and a simple moving average crossover rule. The model's Sharpe ratio "
    f"was {fmt_num(ml_sh)}, compared to {fmt_num(bh_sh)} for buy and hold and "
    f"{fmt_num(ma_sh)} for the moving average rule. {h1_tail}")

# ---- H2: how results vary by regime
h2 = ""
if not reg.empty and reg["sharpe"].notna().sum() >= 2:
    best = reg["sharpe"].idxmax()
    worst = reg["sharpe"].idxmin()
    best_active_ir = reg["information_ratio_vs_buy_hold"].idxmax()
    h2 = (
        f"**H2 (Descriptive):** results changed across volatility regimes. The Sharpe "
        f"ratio ranged from {fmt_num(reg.loc[worst, 'sharpe'])} in the {worst} regime "
        f"({int(reg.loc[worst, 'n_days'])} days) to {fmt_num(reg.loc[best, 'sharpe'])} "
        f"in the {best} regime ({int(reg.loc[best, 'n_days'])} days). Compared to buy "
        f"and hold, the best info ratio was in the {best_active_ir} regime at "
        f"{fmt_num(reg.loc[best_active_ir, 'information_ratio_vs_buy_hold'])}. These "
        f"numbers come from smaller slices of the data, and no statistical test here "
        f"shows the pattern will continue.")

# ---- H3: accuracy as a side check
h3 = (
    f"**H3 (Side check):** accuracy was {acc*100:.1f} percent, compared to "
    f"{R['classification']['majority_baseline']*100:.1f} percent from always "
    f"guessing the most common outcome, and ROC-AUC was "
    f"{fmt_num(R['classification'].get('roc_auc'), 3)}, where 0.5 means no ranking "
    f"skill. Whether the model makes money also depends on "
    f"how big the moves are, how much is invested, and costs. Accuracy alone cannot "
    f"answer the investment question, so return per unit of risk is the main score.")

# ---- Risk and cost context
ml_beta = gv(ml, "beta")
risk_word = "less" if pd.notna(ml_beta) and ml_beta < 1 else "about as much or more"
context = (
    f"**Risk and costs:** the model had a beta of {fmt_num(ml_beta)} against buy and "
    f"hold and a max drawdown of {fmt_pct(gv(ml, 'max_drawdown'))}, compared to "
    f"{fmt_pct(gv(bh, 'max_drawdown'))} for buy and hold, so it took on {risk_word} "
    f"market risk. At {cost_bps:.1f} basis points per trade, costs used up about "
    f"{R['cost_drag']*100:.0f} percent of the return before costs. Move the cost "
    f"slider to see how sensitive the results are.")

st.markdown("\n\n".join(p for p in [main, h1, h2, h3, context] if p))

st.caption(
    "These results update when you change any setting in the sidebar. See the README "
    "for the full methods and limitations."
)

st.divider()
st.caption(
    "Data: daily prices and VIX from Yahoo Finance. Methods: features that only use "
    "past data, walk forward validation, and a custom backtest with trading costs."
)
