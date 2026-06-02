"""
NEPSE Signals — live Streamlit app.

Loads the SWAPPABLE LightGBM boosters from models/ and scores the latest data on
every visit. Replace models/lgbm_h*.txt (+ push) and users immediately get the
new model's outputs — no code change.

Run locally:  streamlit run app.py
Deploy free:  Hugging Face Spaces (SDK: streamlit) or Streamlit Community Cloud.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import streamlit as st

HERE = Path(__file__).parent
st.set_page_config(page_title="NEPSE Signals", page_icon="📈", layout="wide")


@st.cache_data
def load_meta():
    return json.loads((HERE / "data/metrics.json").read_text())


@st.cache_data
def load_data():
    feats = pd.read_parquet(HERE / "data/latest_features.parquet")
    risk = pd.read_parquet(HERE / "data/risk.parquet")
    return feats, risk


@st.cache_resource
def load_models(horizons):
    return {h: lgb.Booster(model_file=str(HERE / f"models/lgbm_h{h}.txt")) for h in horizons}


meta = load_meta()
feats, risk = load_data()
models = load_models(meta["horizons"])
FEATURES, H = meta["features"], meta["horizons"]
m = meta["metrics"]; m10 = m["10"] if "10" in m else m[10]

# ---- live scoring (this is what makes model-swap change outputs) --------- #
scores = feats[["symbol"]].copy()
for h in H:
    scores[f"p{h}"] = models[h].predict(feats[FEATURES].values)
scores = scores.merge(risk, on="symbol", how="left")

# =========================== UI =========================================== #
st.title("📈 NEPSE Signals")
st.caption(f"Personalized, honest machine-learning recommendations for the Nepal Stock Exchange · "
           f"{meta['n_stocks']} stocks · {meta['yrs']} years · signals as of **{meta['as_of']}**")
nerd = st.toggle("🤓 Nerd mode (show technical detail)", value=False)

st.info(f"**The honest verdict:** the model is right about **{m10['acc']:.1f}%** of the time on "
        f"10-day direction — small but real (beats the naive baseline by +{m10['edge']:.1f} pts), and "
        f"rises to **{m10['acc20']:.0f}%** on its most confident calls.", icon="🎯")
st.warning("Not financial advice. Educational research tool. A ~55% edge means it is wrong nearly "
           "half the time. Never invest money you can't afford to lose.", icon="⚠️")

# headline metric cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("10-day accuracy (OOS)", f"{m10['acc']:.1f}%", f"+{m10['edge']:.1f} pts vs baseline")
c2.metric("On high-conviction calls", f"{m10['acc20']:.0f}%", "top-20% confidence")
c3.metric("Overfitting prob. (PBO)", f"{m10['pbo']:.2f}", "≈0 = trustworthy", delta_color="off")
c4.metric("Backtest 5d (net)", f"{m['5']['strat_x'] if '5' in m else m[5]['strat_x']:.1f}×",
          f"vs {m['5']['bh_x'] if '5' in m else m[5]['bh_x']:.1f}× buy&hold")

if nerd:
    st.markdown(
        f"> 🤓 **Method:** global LightGBM panel classifier across {meta['n_stocks']} stocks, "
        f"walk-forward OOS ({m10['n']:,} predictions) with h-day embargo to block leakage. "
        f"Edge +{m10['edge']:.1f}pts (95% CI ±{m10['ci']:.1f}). PBO via CSCV (Bailey & López de Prado). "
        f"Prices back-adjusted for bonus/rights. Live scores = boosters in `models/` applied to the "
        f"latest bar — swap the file to change these outputs.")

st.divider()

# ---- personalized recommendation (the main event) ----------------------- #
st.subheader("🎯 Get your personalized recommendation")
st.caption("The model turns its forecasts into a suggested basket tailored to you — your risk "
           "appetite, horizon and capital. An idea generator, **not** advice (see the warning above).")
with st.form("reco"):
    a, b, c = st.columns(3)
    risk_app = a.radio("Risk appetite", ["Conservative", "Balanced", "Aggressive"], index=1,
                       help="Caps how volatile the suggested stocks can be.")
    hlabel = b.selectbox("Holding horizon", ["~1 week (5d)", "~2 weeks (10d)", "~1 month (20d)"], index=1)
    capital = c.number_input("Amount to invest (NPR)", min_value=1000, value=100000, step=10000)
    d, e = st.columns(2)
    nstocks = d.slider("How many stocks to hold", 3, 15, 6)
    weighting = e.radio("How to split the money", ["Conviction-weighted", "Equal", "Lower-risk first"],
                        index=0, horizontal=True)
    go = st.form_submit_button("🎯 Build my recommendation", use_container_width=True)

if go:
    h = {"~1 week (5d)": 5, "~2 weeks (10d)": 10, "~1 month (20d)": 20}[hlabel]
    pcol = f"p{h}"
    pool = scores.dropna(subset=["vol10"]).copy()
    qcap = {"Conservative": 0.40, "Balanced": 0.70, "Aggressive": 1.0}[risk_app]
    vcap = pool["vol10"].quantile(qcap)
    cand = pool[(pool[pcol] > 0.5) & (pool["vol10"] <= vcap)].copy()
    cand["conviction"] = (cand[pcol] - .5) * 100
    cand["score"] = cand["conviction"] / cand["vol10"] if weighting == "Lower-risk first" else cand["conviction"]
    cand = cand.sort_values("score", ascending=False).head(nstocks)
    if len(cand) == 0:
        st.error(f"At a **{risk_app.lower()}** risk level over **{hlabel}**, the model currently leans "
                 f"bullish on **no stocks within your risk limit**. That's a real signal — the market looks "
                 f"weak right now, so **staying in cash may be the prudent move**. Try a longer horizon or a "
                 f"higher risk appetite to see more speculative ideas.")
    else:
        if weighting == "Equal":
            w = np.ones(len(cand))
        elif weighting == "Lower-risk first":
            w = 1.0 / cand["vol10"].values
        else:
            w = cand["conviction"].values
        w = w / w.sum()
        out = pd.DataFrame({
            "Stock": cand["symbol"].values,
            "P(up)": (cand[pcol].values * 100).round(1),
            "Conviction": cand["conviction"].values.round(1),
            "Risk 10d vol%": cand["vol10"].values.round(1),
            "Allocation %": (w * 100).round(1),
            "Allocation NPR": (w * capital).round(0).astype(int)})
        st.success(f"Suggested **{len(cand)}-stock basket** for a **{risk_app.lower()}** investor · "
                   f"**{hlabel}** · **NPR {capital:,.0f}**")
        st.dataframe(out, hide_index=True, use_container_width=True,
                     column_config={"P(up)": st.column_config.NumberColumn(format="%.1f%%"),
                                    "Allocation NPR": st.column_config.NumberColumn(format="NPR %d")})
        k1, k2, k3 = st.columns(3)
        k1.metric("Avg model conviction", f"{float((w*cand['conviction']).sum()):.1f} pts")
        k2.metric("Basket risk (~10d vol)", f"{float((w*cand['vol10']).sum()):.1f}%")
        k3.metric("Diversification", f"{len(cand)} stocks")
        st.caption("Allocations come from the model's conviction & GARCH risk forecasts. Not advice — "
                   "the model is wrong ~45% of the time; only invest what you can afford to lose.")
        if nerd:
            st.markdown(f"> 🤓 Pool = stocks with P(up)>0.5 at h={h} and forecast vol ≤ the "
                        f"{int(qcap*100)}th percentile ({vcap:.1f}%). Ranked by "
                        f"{'conviction/vol' if weighting=='Lower-risk first' else 'conviction'}, top {nstocks}. "
                        f"Weights: {weighting.lower()}, normalised to your capital.")

st.divider()

# today's signals
st.subheader(f"📍 Today's signals — as of {meta['as_of']}")
st.caption("Ranked by 10-day confidence. Green = model leans up, red = leans down. "
           "Conviction = P(up) − 50%.")
s10 = scores.sort_values("p10", ascending=False)
def fmt(d):
    d = d.copy()
    d["P(up)"] = (d["p10"] * 100).round(1).astype(str) + "%"
    d["Conviction"] = ((d["p10"] - .5) * 100).round(1)
    d["Risk (10d vol)"] = d["vol10"].round(1).astype(str) + "%"
    return d[["symbol", "P(up)", "Conviction", "Risk (10d vol)"]].rename(columns={"symbol": "Stock"})
left, right = st.columns(2)
with left:
    st.markdown("### ▲ Top 10 — leans **UP**")
    st.dataframe(fmt(s10.head(10)).reset_index(drop=True), use_container_width=True, hide_index=True)
with right:
    st.markdown("### ▼ Bottom 10 — leans **DOWN**")
    st.dataframe(fmt(s10.tail(10).iloc[::-1]).reset_index(drop=True), use_container_width=True, hide_index=True)

# watchlist — user builds their own
st.subheader("⭐ Build your own watchlist")
all_syms = sorted(scores["symbol"].unique())
picks = st.multiselect(
    f"Search and pick any of the {len(all_syms)} stocks to track:",
    all_syms, default=[],
    help="The model's conviction (P(up) − 50%) at each horizon, plus its risk forecast, "
         "is shown for the stocks you choose.")
if picks:
    wl = []
    for s in picks:
        r = scores[scores.symbol == s]
        row = {"Stock": s}
        for h in H:
            row[f"{h}-day"] = round((float(r[f"p{h}"].values[0]) - .5) * 100, 1) if len(r) else None
        v = r["vol10"].values[0] if len(r) and pd.notna(r["vol10"].values[0]) else None
        row["Risk (10d vol)"] = f"{v:.1f}%" if v is not None else "—"
        wl.append(row)
    st.dataframe(pd.DataFrame(wl), use_container_width=True, hide_index=True)
    st.caption("Conviction (P(up) − 50%). Positive = model leans up, negative = leans down.")
else:
    st.caption("⬆️ Type a ticker above (e.g. NABIL, NMB, ADBL) to add it to your watchlist.")

st.divider()

# evidence
st.subheader("🔬 The evidence")
tab1, tab2, tab3 = st.tabs(["Does it work? (credibility)", "What to do (decision)", "Data cleaning"])
with tab1:
    st.image(str(HERE / "assets/dashboard_performance.png"))
    st.caption("Accuracy vs baseline · confidence gating · stability over time · cost-aware equity "
               "curve · overfitting (PBO) · top signals.")
    if nerd:
        st.markdown("> 🤓 Top features are market momentum/volatility/breadth — the edge is largely "
                    "**regime timing**, not stock-specific alpha. The accuracy-vs-coverage curve is the "
                    "honest version of '70%': reachable only on a small high-conviction slice.")
with tab2:
    st.image(str(HERE / "assets/dashboard_decision.png"))
    st.caption("Buy/avoid ranking · watchlist · volatility forecast · conviction-vs-risk.")
with tab3:
    st.image(str(HERE / "assets/adjustment_validation.png"))
    st.caption("Bonus/rights back-adjustment: the blue line is continuous through corporate actions; "
               "training on raw red prices would mislabel every bonus issue as a crash.")

st.divider()
with st.expander("ℹ️ How it works & why you can trust the numbers"):
    st.markdown(f"""
- **Out-of-sample**: judged only on data unseen in training, with a time-gap (embargo) so it can't peek ahead.
- **Baseline-compared**: we always show the naive "guess the usual outcome" number ({m10['maj']:.1f}%) so the
  real added skill (+{m10['edge']:.1f} pts) is visible.
- **Overfitting-checked**: PBO {m10['pbo']:.2f} (≈0) means the edge isn't luck from settings-tuning.
- **After costs**: backtest subtracts ~{meta['cost']:.1f}% per round-trip trade.

**Honest bottom line:** a reliable after-cost *70%-every-day* NEPSE predictor does not exist. A small,
statistically-sound *~{m10['acc']:.1f}% directional edge* that grows with conviction — that's real. Use it as
**one input** to your judgement, never a guarantee.
""")
st.caption(f"Educational use only · not financial advice · data as of {meta['as_of']} · "
           "model: global LightGBM · swap models/lgbm_h*.txt to update.")
