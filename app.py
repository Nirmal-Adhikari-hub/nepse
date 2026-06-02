"""
NEPSE Signals — live Streamlit app (SaaS-grade UI).

Loads the SWAPPABLE LightGBM boosters from models/ and scores the latest data on
every visit. Replace models/lgbm_h*.txt (+ push) and users immediately get the
new model's outputs — no code change.

Run locally:  streamlit run app.py
Deploy free:  Hugging Face Spaces (sdk: streamlit) or Streamlit Community Cloud.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import streamlit as st

HERE = Path(__file__).parent
st.set_page_config(page_title="NEPSE Signals — AI stock forecasts",
                   page_icon="📈", layout="wide", initial_sidebar_state="collapsed")


# --------------------------------------------------------------------------- #
# data / models
# --------------------------------------------------------------------------- #
@st.cache_data
def load_meta():
    return json.loads((HERE / "data/metrics.json").read_text())


@st.cache_data
def load_data():
    return (pd.read_parquet(HERE / "data/latest_features.parquet"),
            pd.read_parquet(HERE / "data/risk.parquet"))


@st.cache_resource
def load_models(horizons):
    return {h: lgb.Booster(model_file=str(HERE / f"models/lgbm_h{h}.txt")) for h in horizons}


meta = load_meta()
feats, risk = load_data()
models = load_models(meta["horizons"])
FEATURES, H = meta["features"], meta["horizons"]
m = meta["metrics"]; m10 = m["10"] if "10" in m else m[10]; m5 = m["5"] if "5" in m else m[5]

scores = feats[["symbol"]].copy()
for h in H:
    scores[f"p{h}"] = models[h].predict(feats[FEATURES].values)
scores = scores.merge(risk, on="symbol", how="left")


# --------------------------------------------------------------------------- #
# styling
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
html, body, [class*="css"], .stApp, button, input, textarea { font-family:'Inter',-apple-system,sans-serif; }

/* hide Streamlit chrome */
#MainMenu, header[data-testid="stHeader"], footer, [data-testid="stToolbar"] { display:none !important; }
[data-testid="stDecoration"] { display:none; }

/* background + centered, width-constrained column (fixes full-screen stretch) */
.stApp { background:
   radial-gradient(1100px 500px at 50% -8%, #18233f 0%, rgba(11,16,32,0) 60%),
   linear-gradient(180deg, #0a0e1a 0%, #0a0e1a 100%); }
.block-container { max-width: 1120px !important; padding: 1.2rem 1.2rem 5rem !important; margin: 0 auto; }

/* brand bar */
.nav { display:flex; align-items:center; justify-content:space-between; padding:6px 2px 14px; }
.brand { font-weight:800; font-size:1.15rem; letter-spacing:-.02em; }
.brand .dot { color:#22c55e; }
.nav .tag { font-size:.72rem; color:#8b95a7; border:1px solid #2a3450; border-radius:999px; padding:4px 10px; }

/* hero */
.hero { position:relative; border:1px solid #243049; border-radius:20px; padding:34px 30px;
  background: linear-gradient(135deg, rgba(34,197,94,.10), rgba(56,139,255,.08) 60%, rgba(20,27,46,.4));
  box-shadow: 0 20px 60px -30px rgba(0,0,0,.8); overflow:hidden; }
.hero:before { content:""; position:absolute; inset:0; background:
  radial-gradient(600px 200px at 90% 0%, rgba(34,197,94,.12), transparent 70%); pointer-events:none; }
.hero h1 { font-size: clamp(1.7rem, 4vw, 2.7rem); font-weight:900; line-height:1.1; margin:0 0 10px;
  letter-spacing:-.03em; }
.hero h1 .grad { background:linear-gradient(90deg,#22c55e,#5eead4); -webkit-background-clip:text;
  background-clip:text; -webkit-text-fill-color:transparent; }
.hero p { font-size: clamp(.98rem,2vw,1.15rem); color:#c3cbd9; max-width:680px; margin:0 0 16px; }
.pills { display:flex; flex-wrap:wrap; gap:8px; }
.pill { font-size:.8rem; color:#cdd5e3; background:rgba(255,255,255,.04); border:1px solid #2a3450;
  border-radius:999px; padding:5px 12px; }
.pill b { color:#fff; }

/* metric cards */
.cardgrid { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:14px; margin:18px 0 6px; }
.mcard { background:linear-gradient(180deg, rgba(255,255,255,.035), rgba(255,255,255,.01));
  border:1px solid #243049; border-radius:16px; padding:18px 18px 16px; position:relative;
  transition:transform .15s ease, border-color .15s ease; }
.mcard:hover { transform:translateY(-3px); border-color:#34507a; }
.mcard .top { font-size:.78rem; color:#8b95a7; font-weight:600; text-transform:uppercase; letter-spacing:.04em; }
.mcard .val { font-size:2rem; font-weight:800; margin:6px 0 2px; letter-spacing:-.02em; }
.mcard .sub { font-size:.82rem; color:#9aa4b6; }
.green { color:#22c55e; } .blue { color:#5eead4; } .amber { color:#fbbf24; } .red { color:#f87171; }

/* section headings */
.sec { display:flex; align-items:center; gap:10px; margin:34px 0 6px; font-size:1.35rem; font-weight:800;
  letter-spacing:-.02em; }
.sec:before { content:""; width:5px; height:24px; border-radius:3px; background:linear-gradient(#22c55e,#0e7a3a); }
.lead { color:#9aa4b6; margin:0 0 14px; font-size:.95rem; }

/* buttons */
.stButton>button, .stFormSubmitButton>button { background:linear-gradient(90deg,#22c55e,#16a34a);
  color:#04210f; font-weight:800; border:0; border-radius:12px; padding:.6rem 1rem;
  box-shadow:0 10px 25px -12px rgba(34,197,94,.8); transition:filter .15s; }
.stButton>button:hover, .stFormSubmitButton>button:hover { filter:brightness(1.08); }

/* form container */
[data-testid="stForm"] { background:rgba(255,255,255,.02); border:1px solid #243049 !important;
  border-radius:18px; padding:18px 18px 6px; }

/* dataframes */
[data-testid="stDataFrame"] { border:1px solid #243049; border-radius:14px; overflow:hidden; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap:6px; }
.stTabs [data-baseweb="tab"] { background:rgba(255,255,255,.03); border:1px solid #243049;
  border-radius:10px 10px 0 0; padding:8px 14px; }
.stTabs [aria-selected="true"] { background:rgba(34,197,94,.12); border-color:#2f7a4d; }

/* banners */
.disc { background:linear-gradient(90deg, rgba(251,191,36,.10), rgba(251,191,36,.02));
  border:1px solid #5a4a1e; border-radius:14px; padding:13px 16px; font-size:.88rem; color:#e8d9ad; margin:14px 0; }

/* footer */
.foot { margin-top:40px; padding-top:18px; border-top:1px solid #1e2740; font-size:.82rem; color:#7a849a; }
.foot a { color:#5eead4; text-decoration:none; }

/* steps */
.steps { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin-top:8px; }
.step { background:rgba(255,255,255,.025); border:1px solid #243049; border-radius:14px; padding:14px 16px; }
.step .n { font-weight:800; color:#22c55e; font-size:.85rem; } .step .t { font-weight:700; margin:4px 0; }
.step .d { color:#9aa4b6; font-size:.85rem; }

@media (max-width:640px){
  .block-container { padding:.8rem .8rem 4rem !important; }
  .hero { padding:24px 18px; border-radius:16px; }
  .mcard .val { font-size:1.7rem; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# header + hero
# --------------------------------------------------------------------------- #
st.markdown(
    f'<div class="nav"><div class="brand">📈 NEPSE<span class="dot">·</span>Signals</div>'
    f'<div class="tag">updated {meta["as_of"]}</div></div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="hero">
  <h1>Can a machine predict the <span class="grad">Nepal Stock Exchange?</span></h1>
  <p>A little — honestly about <b>{m10['acc']:.1f}%</b> of the time on 10-day direction. Small but real,
  and it gets sharper on high-conviction calls. Get a <b>personalized, risk-tuned</b> basket below —
  built from {meta['n_stocks']} stocks and {meta['yrs']} years of data, tested the honest way.</p>
  <div class="pills">
    <span class="pill">🏦 <b>{meta['n_stocks']}</b> stocks</span>
    <span class="pill">🗓️ <b>{meta['yrs']}</b> years</span>
    <span class="pill">🔬 out-of-sample tested</span>
    <span class="pill">⚙️ overfitting-checked (PBO {m10['pbo']:.2f})</span>
  </div>
</div>
""", unsafe_allow_html=True)

cc = st.container()
with cc:
    left, rightt = st.columns([3, 1])
    left.markdown('<div class="disc">⚠️ <b>Not financial advice.</b> Educational research tool. '
                  'A ~55% edge means it is wrong nearly half the time. Never invest money you can\'t afford to lose.</div>',
                  unsafe_allow_html=True)
    nerd = rightt.toggle("🤓 Nerd mode", value=False, help="Show the technical detail behind each number.")

# metric cards
st.markdown(f"""
<div class="cardgrid">
  <div class="mcard"><div class="top">10-day accuracy</div>
    <div class="val green">{m10['acc']:.1f}%</div>
    <div class="sub">+{m10['edge']:.1f} pts vs naive baseline</div></div>
  <div class="mcard"><div class="top">High-conviction calls</div>
    <div class="val green">{m10['acc20']:.0f}%</div>
    <div class="sub">accuracy on its top-20% confident picks</div></div>
  <div class="mcard"><div class="top">Overfitting (PBO)</div>
    <div class="val blue">{m10['pbo']:.2f}</div>
    <div class="sub">≈0 → the edge is real, not curve-fit</div></div>
  <div class="mcard"><div class="top">Backtest · 5-day, net</div>
    <div class="val green">{m5['strat_x']:.1f}×</div>
    <div class="sub">vs {m5['bh_x']:.1f}× buy &amp; hold</div></div>
</div>
""", unsafe_allow_html=True)

if nerd:
    st.caption(f"🤓 Global LightGBM panel classifier over {meta['n_stocks']} stocks, walk-forward OOS "
               f"({m10['n']:,} predictions) with an h-day embargo to block leakage. Edge +{m10['edge']:.1f}pts "
               f"(95% CI ±{m10['ci']:.1f}). PBO via CSCV (Bailey & López de Prado). Prices back-adjusted for "
               f"bonus/rights. Live scores = boosters in models/ applied to the latest bar — swap to change outputs.")


# --------------------------------------------------------------------------- #
# personalized recommendation
# --------------------------------------------------------------------------- #
st.markdown('<div class="sec">🎯 Get your personalized recommendation</div>', unsafe_allow_html=True)
st.markdown('<p class="lead">Tell us your risk appetite, horizon and budget — the model builds a suggested '
            'basket with allocations. An idea generator, not advice.</p>', unsafe_allow_html=True)

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
    go = st.form_submit_button("🎯  Build my recommendation", use_container_width=True)

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
                                    "Allocation NPR": st.column_config.NumberColumn(format="NPR %d"),
                                    "Conviction": st.column_config.ProgressColumn(
                                        format="%.1f", min_value=0, max_value=float(max(cand["conviction"].max(), 1)))})
        st.markdown(f"""
<div class="cardgrid">
  <div class="mcard"><div class="top">Avg conviction</div><div class="val green">{float((w*cand['conviction']).sum()):.1f}</div><div class="sub">points above coin-flip</div></div>
  <div class="mcard"><div class="top">Basket risk</div><div class="val amber">{float((w*cand['vol10']).sum()):.1f}%</div><div class="sub">≈10-day volatility</div></div>
  <div class="mcard"><div class="top">Diversification</div><div class="val blue">{len(cand)}</div><div class="sub">stocks</div></div>
</div>""", unsafe_allow_html=True)
        st.caption("Allocations come from the model's conviction & GARCH risk forecasts. Not advice — "
                   "the model is wrong ~45% of the time; only invest what you can afford to lose.")
        if nerd:
            st.caption(f"🤓 Pool = P(up)>0.5 at h={h} and forecast vol ≤ the {int(qcap*100)}th percentile "
                       f"({vcap:.1f}%); ranked by {'conviction/vol' if weighting=='Lower-risk first' else 'conviction'}, "
                       f"top {nstocks}; weights = {weighting.lower()}.")


# --------------------------------------------------------------------------- #
# today's signals
# --------------------------------------------------------------------------- #
st.markdown(f'<div class="sec">📍 Today\'s signals</div>', unsafe_allow_html=True)
st.markdown('<p class="lead">Every stock ranked by the model\'s 10-day confidence. '
            'Green leans up, red leans down. Conviction = P(up) − 50%.</p>', unsafe_allow_html=True)
s10 = scores.sort_values("p10", ascending=False)
def fmt(d):
    return pd.DataFrame({
        "Stock": d["symbol"].values,
        "P(up)": (d["p10"].values * 100).round(1),
        "Conviction": ((d["p10"].values - .5) * 100).round(1),
        "Risk 10d vol%": d["vol10"].values.round(1)})
cfg = {"P(up)": st.column_config.NumberColumn(format="%.1f%%")}
left, right = st.columns(2)
with left:
    st.markdown("##### ▲ Top 10 — leans **UP**")
    st.dataframe(fmt(s10.head(10)), hide_index=True, use_container_width=True, column_config=cfg)
with right:
    st.markdown("##### ▼ Bottom 10 — leans **DOWN**")
    st.dataframe(fmt(s10.tail(10).iloc[::-1]), hide_index=True, use_container_width=True, column_config=cfg)


# --------------------------------------------------------------------------- #
# watchlist
# --------------------------------------------------------------------------- #
st.markdown('<div class="sec">⭐ Build your own watchlist</div>', unsafe_allow_html=True)
all_syms = sorted(scores["symbol"].unique())
picks = st.multiselect(f"Search and pick any of the {len(all_syms)} stocks to track:", all_syms, default=[],
                       help="The model's conviction at each horizon, plus its risk forecast, for your picks.")
if picks:
    wl = []
    for s in picks:
        r = scores[scores.symbol == s]; row = {"Stock": s}
        for h in H:
            row[f"{h}-day"] = round((float(r[f"p{h}"].values[0]) - .5) * 100, 1) if len(r) else None
        v = r["vol10"].values[0] if len(r) and pd.notna(r["vol10"].values[0]) else None
        row["Risk 10d vol%"] = round(float(v), 1) if v is not None else None
        wl.append(row)
    st.dataframe(pd.DataFrame(wl), hide_index=True, use_container_width=True)
    st.caption("Conviction (P(up) − 50%). Positive = model leans up.")
else:
    st.caption("⬆️ Type a ticker (e.g. NABIL, NMB, ADBL) to add it to your watchlist.")


# --------------------------------------------------------------------------- #
# evidence
# --------------------------------------------------------------------------- #
st.markdown('<div class="sec">🔬 The evidence</div>', unsafe_allow_html=True)
t1, t2, t3 = st.tabs(["Does it work?", "What to do", "Data cleaning"])
with t1:
    st.image(str(HERE / "assets/dashboard_performance.png"), use_container_width=True)
    st.caption("Accuracy vs baseline · confidence gating · stability over time · cost-aware equity curve · "
               "overfitting test (PBO) · which signals matter.")
    if nerd:
        st.caption("🤓 Top features are market momentum/volatility/breadth — the edge is largely regime "
                   "timing, not stock-specific alpha. Accuracy-vs-coverage is the honest version of '70%'.")
with t2:
    st.image(str(HERE / "assets/dashboard_decision.png"), use_container_width=True)
    st.caption("Buy/avoid ranking · watchlist · volatility forecast · conviction-vs-risk.")
with t3:
    st.image(str(HERE / "assets/adjustment_validation.png"), use_container_width=True)
    st.caption("Bonus/rights back-adjustment: the blue line is continuous through corporate actions; "
               "training on raw red prices would mislabel every bonus issue as a crash.")


# --------------------------------------------------------------------------- #
# how it works
# --------------------------------------------------------------------------- #
st.markdown('<div class="sec">🧠 How it works</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="steps">
  <div class="step"><div class="n">01</div><div class="t">Learn</div>
    <div class="d">A LightGBM model studies {meta['yrs']} yrs of {meta['n_stocks']} stocks — momentum, volatility, market regime.</div></div>
  <div class="step"><div class="n">02</div><div class="t">Test honestly</div>
    <div class="d">Judged only on unseen data with a time-gap (embargo), vs a naive baseline, with an overfitting check.</div></div>
  <div class="step"><div class="n">03</div><div class="t">Recommend</div>
    <div class="d">Turns per-stock direction + risk into a basket tuned to your appetite, horizon and budget.</div></div>
</div>
""", unsafe_allow_html=True)

with st.expander("ℹ️ Why you can trust the numbers (and their limits)"):
    st.markdown(f"""
- **Out-of-sample** — judged only on data unseen in training, with an embargo so it can't peek ahead.
- **Baseline-compared** — we always show the naive number ({m10['maj']:.1f}%) so the real skill (+{m10['edge']:.1f} pts) is visible.
- **Overfitting-checked** — PBO {m10['pbo']:.2f} (≈0): the edge isn't luck from settings-tuning.
- **After costs** — backtest subtracts ~{meta['cost']:.1f}% per round-trip trade.

**Honest bottom line:** a reliable after-cost *70%-every-day* NEPSE predictor does not exist. A small,
statistically-sound *~{m10['acc']:.1f}% directional edge* that grows with conviction is real. Use it as **one
input** to your judgement, never a guarantee.
""")

st.markdown(f'<div class="foot">📈 <b>NEPSE Signals</b> · educational research, not financial advice · '
            f'data as of {meta["as_of"]} · global LightGBM · '
            f'<a href="https://huggingface.co/spaces/Nirmal590/nepse-signals">about</a></div>',
            unsafe_allow_html=True)
