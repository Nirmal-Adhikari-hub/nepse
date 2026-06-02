"""
NEPSE Signals — live multi-page Streamlit app (interactive, no static images).

Pages:
  🏠 Home              — hero, personalized recommendation, today's signals, watchlist
  🔍 Explore a stock   — price chart + forecast cone + per-metric DEEP-DIVE + relative rank
  📊 Model & Evidence  — fully INTERACTIVE charts (accuracy, coverage, equity, PBO, …) + explanations
  💬 Assistant         — personalized chatbot grounded in the live model data

Model runs live: LightGBM boosters in models/ score the latest bar on every visit.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import assistant

HERE = Path(__file__).parent
st.set_page_config(page_title="NEPSE Signals — AI stock forecasts", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")
GREEN, RED, BLUE, AMBER, GREY = "#22c55e", "#f87171", "#5eead4", "#fbbf24", "#7f8aa0"


@st.cache_data
def load_meta():   return json.loads((HERE / "data/metrics.json").read_text())
@st.cache_data
def load_evidence(): return json.loads((HERE / "data/evidence.json").read_text())
@st.cache_data
def load_data():
    return (pd.read_parquet(HERE / "data/latest_features.parquet"),
            pd.read_parquet(HERE / "data/risk.parquet"),
            pd.read_parquet(HERE / "data/prices.parquet"))
@st.cache_resource
def load_models(hz): return {h: lgb.Booster(model_file=str(HERE / f"models/lgbm_h{h}.txt")) for h in hz}


meta = load_meta(); EV = load_evidence()
feats, risk, prices = load_data()
models = load_models(meta["horizons"])
FEATURES, H = meta["features"], meta["horizons"]
m = meta["metrics"]; m10 = m.get("10", m.get(10)); m5 = m.get("5", m.get(5))

scores = feats[["symbol"]].copy()
for h in H:
    scores[f"p{h}"] = models[h].predict(feats[FEATURES].values)
scores = scores.merge(risk, on="symbol", how="left")
ASOF = meta.get("asof", {})
scores["asof"] = scores["symbol"].map(ASOF).fillna(meta["as_of"])
scores["fresh"] = scores["asof"] >= "2026-01-01"
fresh_scores = scores[scores["fresh"]].copy()
PRICE_SYMS = sorted(set(prices["symbol"]).intersection(scores["symbol"]))

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
html, body, [class*="css"], .stApp, button, input, textarea { font-family:'Inter',-apple-system,sans-serif; }
#MainMenu, header[data-testid="stHeader"], footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }
.stApp { background: radial-gradient(1100px 500px at 50% -8%, #18233f 0%, rgba(11,16,32,0) 60%), #0a0e1a; }
.block-container { max-width:1140px !important; padding:1.1rem 1.2rem 5rem !important; }
[data-testid="stSidebar"] { background:#0c1220; border-right:1px solid #1c2740; }
[data-testid="stSidebar"] .sb-brand { font-weight:900; font-size:1.15rem; padding:6px 4px 2px; }
[data-testid="stSidebar"] .sb-sub { color:#7a849a; font-size:.78rem; padding:0 4px 10px; }
.nav { display:flex; align-items:center; justify-content:space-between; padding:2px 2px 14px; }
.brand { font-weight:800; font-size:1.1rem; } .brand .dot{color:#22c55e;}
.tag { font-size:.72rem; color:#8b95a7; border:1px solid #2a3450; border-radius:999px; padding:4px 10px; }
.hero { position:relative; border:1px solid #243049; border-radius:20px; padding:34px 30px;
  background:linear-gradient(135deg, rgba(34,197,94,.10), rgba(56,139,255,.08) 60%, rgba(20,27,46,.4)); overflow:hidden; }
.hero h1 { font-size:clamp(1.7rem,4vw,2.7rem); font-weight:900; line-height:1.1; margin:0 0 10px; letter-spacing:-.03em; }
.hero h1 .grad { background:linear-gradient(90deg,#22c55e,#5eead4); -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }
.hero p { font-size:clamp(.98rem,2vw,1.15rem); color:#c3cbd9; max-width:700px; margin:0 0 16px; }
.pills { display:flex; flex-wrap:wrap; gap:8px; } .pill{ font-size:.8rem; color:#cdd5e3; background:rgba(255,255,255,.04); border:1px solid #2a3450; border-radius:999px; padding:5px 12px; } .pill b{color:#fff;}
.cardgrid { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:14px; margin:18px 0 6px; }
.mcard { background:linear-gradient(180deg, rgba(255,255,255,.035), rgba(255,255,255,.01)); border:1px solid #243049; border-radius:16px; padding:18px; transition:transform .15s, border-color .15s; }
.mcard:hover { transform:translateY(-3px); border-color:#34507a; }
.mcard .top { font-size:.76rem; color:#8b95a7; font-weight:600; text-transform:uppercase; letter-spacing:.04em; }
.mcard .val { font-size:1.9rem; font-weight:800; margin:6px 0 2px; } .mcard .sub{ font-size:.82rem; color:#9aa4b6; }
.green{color:#22c55e;} .blue{color:#5eead4;} .amber{color:#fbbf24;} .red{color:#f87171;}
.sec { display:flex; align-items:center; gap:10px; margin:30px 0 6px; font-size:1.35rem; font-weight:800; }
.sec:before { content:""; width:5px; height:24px; border-radius:3px; background:linear-gradient(#22c55e,#0e7a3a); }
.lead { color:#9aa4b6; margin:0 0 14px; font-size:.95rem; }
.stButton>button, .stFormSubmitButton>button { background:linear-gradient(90deg,#22c55e,#16a34a); color:#04210f; font-weight:800; border:0; border-radius:12px; padding:.6rem 1rem; }
[data-testid="stForm"] { background:rgba(255,255,255,.02); border:1px solid #243049 !important; border-radius:18px; padding:18px 18px 6px; }
[data-testid="stDataFrame"] { border:1px solid #243049; border-radius:14px; overflow:hidden; }
.disc { background:linear-gradient(90deg, rgba(251,191,36,.10), rgba(251,191,36,.02)); border:1px solid #5a4a1e; border-radius:14px; padding:13px 16px; font-size:.88rem; color:#e8d9ad; margin:14px 0; }
.readout { background:rgba(255,255,255,.03); border:1px solid #243049; border-left:4px solid #22c55e; border-radius:12px; padding:14px 16px; font-size:1.02rem; }
.expl { background:rgba(86,139,255,.06); border:1px solid #29406a; border-radius:12px; padding:12px 16px; font-size:.92rem; color:#cdd9ef; margin:6px 0 18px; }
.metric-row { display:flex; align-items:center; justify-content:space-between; padding:9px 2px; border-bottom:1px solid #1b2440; }
.metric-row .lab { font-weight:600; } .metric-row .txt { color:#9aa4b6; font-size:.86rem; }
.badge { font-size:.72rem; font-weight:800; padding:3px 10px; border-radius:999px; }
.foot { margin-top:36px; padding-top:18px; border-top:1px solid #1e2740; font-size:.82rem; color:#7a849a; }
@media (max-width:640px){ .block-container{padding:.8rem .8rem 4rem !important;} .hero{padding:24px 18px;} .mcard .val{font-size:1.6rem;} }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------- shared helpers ------------------------------------------------- #
def conviction_word(p):
    c = (p - .5) * 100
    if c >= 6:  return "strongly bullish", GREEN
    if c >= 2:  return "leans bullish", GREEN
    if c > -2:  return "neutral", AMBER
    if c > -6:  return "leans bearish", RED
    return "strongly bearish", RED


def theme(fig, h=360, legend=True):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=h, margin=dict(l=10, r=10, t=30, b=10), font=dict(color="#c3cbd9"),
        hovermode="x unified", showlegend=legend, legend=dict(orientation="h", y=1.12, x=0))
    fig.update_xaxes(gridcolor="rgba(255,255,255,.05)"); fig.update_yaxes(gridcolor="rgba(255,255,255,.05)")
    return fig


def forecast_figure(sym, h):
    hist = prices[prices.symbol == sym].sort_values("date").tail(120).copy()
    row = scores[scores.symbol == sym].iloc[0]
    p_up = float(row[f"p{h}"]); vol10 = float(row["vol10"]) if pd.notna(row["vol10"]) else 6.0
    P0 = float(hist["close"].iloc[-1]); last_date = pd.Timestamp(hist["date"].iloc[-1])
    sig_d = (vol10 / 100) / np.sqrt(10); fut = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=h)
    t = np.arange(1, h + 1); sig_h = sig_d * np.sqrt(h)
    central = P0 * (1 + (2 * p_up - 1) * 0.5 * sig_h * (t / h)); st_ = sig_d * np.sqrt(t)
    col = GREEN if p_up >= .5 else RED; fx = [last_date] + list(fut); b = lambda a: [P0] + list(a)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.76, .24], vertical_spacing=.04)
    fig.add_trace(go.Candlestick(x=hist["date"], open=hist["open"], high=hist["high"], low=hist["low"],
        close=hist["close"], increasing_line_color=GREEN, decreasing_line_color=RED, showlegend=False), 1, 1)
    fc = "rgba(34,197,94,.09)" if p_up >= .5 else "rgba(248,113,113,.09)"
    fc1 = "rgba(34,197,94,.18)" if p_up >= .5 else "rgba(248,113,113,.18)"
    fig.add_trace(go.Scatter(x=fx, y=b(central*(1+2*st_)), line=dict(width=0), hoverinfo="skip", showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=fx, y=b(central*(1-2*st_)), line=dict(width=0), fill="tonexty", fillcolor=fc, hoverinfo="skip", name="95% range"), 1, 1)
    fig.add_trace(go.Scatter(x=fx, y=b(central*(1+st_)), line=dict(width=0), hoverinfo="skip", showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=fx, y=b(central*(1-st_)), line=dict(width=0), fill="tonexty", fillcolor=fc1, hoverinfo="skip", name="68% range"), 1, 1)
    fig.add_trace(go.Scatter(x=fx, y=b(central), line=dict(color=col, width=2, dash="dash"), name="model lean"), 1, 1)
    vcol = np.where(hist["close"] >= hist["open"], "rgba(34,197,94,.5)", "rgba(248,113,113,.5)")
    fig.add_trace(go.Bar(x=hist["date"], y=hist["volume"], marker_color=vcol, showlegend=False), 2, 1)
    theme(fig, 520); fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="Price (NPR, adj)", row=1, col=1)
    return fig, dict(P0=P0, p_up=p_up, vol10=vol10, lo=float(central[-1]*(1-2*st_[-1])), hi=float(central[-1]*(1+2*st_[-1])))


def analyze_stock(sym):
    """Per-metric interpretation of one stock's latest features."""
    row = feats[feats.symbol == sym].iloc[0]
    pct = {c: float((feats[c] < row[c]).mean()) for c in FEATURES if c in feats}
    items = []
    def add(group, label, bull, text):
        v, c = {1: ("Bullish", GREEN), 0: ("Neutral", AMBER), -1: ("Bearish", RED)}[bull]
        items.append(dict(group=group, label=label, verdict=v, color=c, text=text))
    r20, r60 = row["ret20"], row["ret60"]
    add("Trend & Momentum", "1-month return", 1 if r20 > .03 else -1 if r20 < -.03 else 0, f"{r20:+.1%} over ~1 month")
    add("Trend & Momentum", "3-month return", 1 if r60 > .05 else -1 if r60 < -.05 else 0, f"{r60:+.1%} over ~3 months")
    mar20, mar60 = row["mar20"], row["mar60"]
    add("Trend & Momentum", "Price vs 20-day avg", 1 if mar20 > .01 else -1 if mar20 < -.01 else 0,
        f"{abs(mar20):.1%} {'above' if mar20 >= 0 else 'below'} its 20-day average")
    add("Trend & Momentum", "Price vs 60-day avg", 1 if mar60 > .01 else -1 if mar60 < -.01 else 0,
        f"{abs(mar60):.1%} {'above' if mar60 >= 0 else 'below'} its 60-day average")
    fh, fl = row["from_hi"], row["from_lo"]
    add("Strength vs range", "Distance from 52-wk high", 1 if fh > -.1 else -1 if fh < -.35 else 0, f"{fh:.0%} from its 52-week high")
    add("Strength vs range", "Above 52-wk low", 1 if fl > .4 else -1 if fl < .1 else 0, f"{fl:+.0%} above its 52-week low")
    rsi = row["rsi14"] * 100
    add("Strength vs range", "RSI (14)", -1 if rsi > 70 else 1 if rsi < 30 else 0,
        f"RSI {rsi:.0f} — {'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'}")
    add("Risk & Volatility", "20-day volatility", -1 if pct.get('vol20', .5) > .7 else 1 if pct.get('vol20', .5) < .3 else 0,
        f"more volatile than {pct.get('vol20', .5)*100:.0f}% of stocks")
    vz = row["volz"]
    add("Volume & Interest", "Volume vs normal", 1 if vz > 1 else -1 if vz < -1 else 0,
        f"{'well above' if vz > 1 else 'below' if vz < -1 else 'around'} its normal volume")
    mm, br = row["mkt_mom20"], row["breadth"]
    add("Market regime", "Market momentum (1-mo)", 1 if mm > .01 else -1 if mm < -.01 else 0, f"overall market {mm:+.1%} over ~1 month")
    add("Market regime", "Market breadth", 1 if br > .55 else -1 if br < .45 else 0, f"{br*100:.0f}% of stocks rising lately")
    return items


# =========================================================================== #
def page_home():
    st.markdown(f'<div class="nav"><div class="brand">📈 NEPSE<span class="dot">·</span>Signals</div>'
                f'<div class="tag">updated {meta["as_of"]}</div></div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="hero"><h1>Can a machine predict the <span class="grad">Nepal Stock Exchange?</span></h1>
  <p>A little — honestly about <b>{m10['acc']:.1f}%</b> of the time on 10-day direction. Small but real.
  Get a <b>personalized basket</b> below, <b>Explore</b> any stock's chart + forecast, or ask the <b>Assistant</b>.</p>
  <div class="pills"><span class="pill">🏦 <b>{meta['n_stocks']}</b> stocks</span><span class="pill">🗓️ <b>{meta['yrs']}</b> yrs</span>
  <span class="pill">🔬 out-of-sample</span><span class="pill">⚙️ PBO {m10['pbo']:.2f}</span></div></div>""", unsafe_allow_html=True)
    st.markdown('<div class="disc">⚠️ <b>Not financial advice.</b> Educational tool. ~55% accuracy means it is wrong nearly half the time.</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="cardgrid">
  <div class="mcard"><div class="top">10-day accuracy</div><div class="val green">{m10['acc']:.1f}%</div><div class="sub">+{m10['edge']:.1f} pts vs baseline</div></div>
  <div class="mcard"><div class="top">High-conviction</div><div class="val green">{m10['acc20']:.0f}%</div><div class="sub">top-20% confident calls</div></div>
  <div class="mcard"><div class="top">Overfitting (PBO)</div><div class="val blue">{m10['pbo']:.2f}</div><div class="sub">≈0 → real, not curve-fit</div></div>
  <div class="mcard"><div class="top">Backtest 5d, net</div><div class="val green">{m5['strat_x']:.1f}×</div><div class="sub">vs {m5['bh_x']:.1f}× buy &amp; hold</div></div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec">🎯 Get your personalized recommendation</div>', unsafe_allow_html=True)
    with st.form("reco"):
        a, b, c = st.columns(3)
        risk_app = a.radio("Risk appetite", ["Conservative", "Balanced", "Aggressive"], index=1)
        hlabel = b.selectbox("Holding horizon", ["~1 week (5d)", "~2 weeks (10d)", "~1 month (20d)"], index=1)
        capital = c.number_input("Amount to invest (NPR)", min_value=1000, value=100000, step=10000)
        d, e = st.columns(2)
        nstocks = d.slider("How many stocks to hold", 3, 15, 6)
        weighting = e.radio("Split the money", ["Conviction-weighted", "Equal", "Lower-risk first"], index=0, horizontal=True)
        go_btn = st.form_submit_button("🎯  Build my recommendation", use_container_width=True)
    if go_btn:
        h = {"~1 week (5d)": 5, "~2 weeks (10d)": 10, "~1 month (20d)": 20}[hlabel]; pcol = f"p{h}"
        pool = fresh_scores.dropna(subset=["vol10"]).copy()
        qcap = {"Conservative": .4, "Balanced": .7, "Aggressive": 1.0}[risk_app]
        cand = pool[(pool[pcol] > .5) & (pool["vol10"] <= pool["vol10"].quantile(qcap))].copy()
        cand["conviction"] = (cand[pcol] - .5) * 100
        cand["score"] = cand["conviction"] / cand["vol10"] if weighting == "Lower-risk first" else cand["conviction"]
        cand = cand.sort_values("score", ascending=False).head(nstocks)
        if not len(cand):
            st.error(f"At a **{risk_app.lower()}** level over **{hlabel}**, the model leans bullish on **no stocks within "
                     f"your risk limit** — the market looks weak, so **cash may be prudent**. Try a longer horizon or higher risk.")
        else:
            w = (np.ones(len(cand)) if weighting == "Equal" else 1/cand["vol10"].values if weighting == "Lower-risk first" else cand["conviction"].values)
            w = w / w.sum()
            out = pd.DataFrame({"Stock": cand["symbol"].values, "P(up)": (cand[pcol].values*100).round(1),
                "Conviction": cand["conviction"].values.round(1), "Risk 10d vol%": cand["vol10"].values.round(1),
                "Allocation %": (w*100).round(1), "Allocation NPR": (w*capital).round(0).astype(int)})
            st.success(f"Suggested **{len(cand)}-stock basket** · **{risk_app.lower()}** · **{hlabel}** · **NPR {capital:,.0f}**")
            st.dataframe(out, hide_index=True, use_container_width=True,
                column_config={"P(up)": st.column_config.NumberColumn(format="%.1f%%"), "Allocation NPR": st.column_config.NumberColumn(format="NPR %d")})
            st.caption("From the model's conviction & GARCH risk. Not advice — wrong ~45% of the time.")

    st.markdown('<div class="sec">📍 Today\'s signals</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="lead">Ranked by 10-day confidence across the <b>{len(fresh_scores)}</b> current stocks. '
                f'(Full {meta["n_stocks"]}-stock universe in <b>Explore</b>, each dated.)</p>', unsafe_allow_html=True)
    s10 = fresh_scores.sort_values("p10", ascending=False)
    f = lambda d: pd.DataFrame({"Stock": d["symbol"].values, "P(up)": (d["p10"].values*100).round(1),
        "Conviction": ((d["p10"].values-.5)*100).round(1), "Risk%": d["vol10"].values.round(1)})
    cfg = {"P(up)": st.column_config.NumberColumn(format="%.1f%%")}
    L, R = st.columns(2)
    L.markdown("##### ▲ Top 10 — leans UP"); L.dataframe(f(s10.head(10)), hide_index=True, use_container_width=True, column_config=cfg)
    R.markdown("##### ▼ Bottom 10 — leans DOWN"); R.dataframe(f(s10.tail(10).iloc[::-1]), hide_index=True, use_container_width=True, column_config=cfg)
    st.info("👉 Open **Explore a stock** for a full per-stock breakdown, or **Assistant** to just ask.", icon="💬")
    st.markdown(f'<div class="foot">📈 <b>NEPSE Signals</b> · educational research, not financial advice · data as of {meta["as_of"]}</div>', unsafe_allow_html=True)


# =========================================================================== #
def page_explore():
    st.markdown('<div class="nav"><div class="brand">🔍 Explore a stock</div></div>', unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])
    default = PRICE_SYMS.index("NABIL") if "NABIL" in PRICE_SYMS else 0
    sym = c1.selectbox("Stock", PRICE_SYMS, index=default)
    hlabel = c2.selectbox("Forecast horizon", ["1 week (5d)", "2 weeks (10d)", "1 month (20d)"], index=1)
    h = {"1 week (5d)": 5, "2 weeks (10d)": 10, "1 month (20d)": 20}[hlabel]
    row = scores[scores.symbol == sym].iloc[0]; p_up = float(row[f"p{h}"]); word, col = conviction_word(p_up)
    sym_asof = ASOF.get(sym, meta["as_of"])
    if sym_asof < "2026-01-01":
        st.markdown(f'<div class="disc">🕗 <b>{sym}</b> has data through <b>{sym_asof}</b> in our sources — '
                    f'this forecast is computed as of that date (historical, not live).</div>', unsafe_allow_html=True)
    # relative rank within the same freshness cohort
    cohort = fresh_scores if row["fresh"] else scores
    rank = int((cohort[f"p{h}"] > p_up).sum()) + 1; ntot = len(cohort)
    fig, info = forecast_figure(sym, h)
    st.markdown(f"""<div class="cardgrid">
  <div class="mcard"><div class="top">Last price</div><div class="val">{info['P0']:.0f}</div><div class="sub">NPR (adjusted)</div></div>
  <div class="mcard"><div class="top">P(up) over {h}d</div><div class="val" style="color:{col}">{p_up*100:.0f}%</div><div class="sub">{word}</div></div>
  <div class="mcard"><div class="top">Relative rank</div><div class="val blue">#{rank}</div><div class="sub">of {ntot} (top {rank/ntot*100:.0f}%) by conviction</div></div>
  <div class="mcard"><div class="top">Risk (10d vol)</div><div class="val amber">{info['vol10']:.1f}%</div><div class="sub">GARCH forecast</div></div></div>""", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    arrow = "↑" if p_up >= .5 else "↓"
    st.markdown(f'<div class="readout">Over the next <b>{hlabel}</b>, the model <b style="color:{col}">{word}</b> on '
                f'<b>{sym}</b> {arrow} — P(up) <b>{p_up*100:.0f}%</b>, ranked <b>#{rank} of {ntot}</b>. A typical '
                f'outcome lands roughly <b>{info["lo"]:.0f}–{info["hi"]:.0f}</b> NPR. The dashed line is the directional '
                f'lean (not a price target); the cone is the realistic range.</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec">🔬 What the model sees — metric by metric</div>', unsafe_allow_html=True)
    st.markdown('<p class="lead">Each signal the model reads for this stock, in plain English. '
                'Green = supports up, red = supports down.</p>', unsafe_allow_html=True)
    items = analyze_stock(sym)
    bull = sum(1 for it in items if it["verdict"] == "Bullish" and it["group"] != "Market regime")
    bear = sum(1 for it in items if it["verdict"] == "Bearish" and it["group"] != "Market regime")
    tone = ("a net-bullish picture" if bull > bear + 1 else "a net-bearish picture" if bear > bull + 1 else "a mixed picture")
    st.markdown(f'<div class="readout">Across its signals, {sym} shows <b>{tone}</b> ({bull} bullish vs {bear} bearish on '
                f'stock-specific metrics). The model weighs these together into its {p_up*100:.0f}% up-probability.</div>', unsafe_allow_html=True)
    groups = ["Trend & Momentum", "Strength vs range", "Risk & Volatility", "Volume & Interest", "Market regime"]
    for g in groups:
        gi = [it for it in items if it["group"] == g]
        if not gi: continue
        with st.expander(f"**{g}**", expanded=(g in ["Trend & Momentum", "Strength vs range"])):
            for it in gi:
                st.markdown(f'<div class="metric-row"><div><span class="lab">{it["label"]}</span><br>'
                            f'<span class="txt">{it["text"]}</span></div>'
                            f'<span class="badge" style="background:{it["color"]}22;color:{it["color"]};'
                            f'border:1px solid {it["color"]}55">{it["verdict"]}</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec">Across horizons</div>', unsafe_allow_html=True)
    cols = st.columns(len(H))
    for cc, hh in zip(cols, H):
        pu = float(row[f"p{hh}"]); cc.metric(f"{hh}-day P(up)", f"{pu*100:.0f}%", f"{(pu-.5)*100:+.1f} pts")
    st.caption("Forecast = directional lean ± realistic volatility (GARCH). Not a guarantee; model wrong ~45% of the time.")


# =========================================================================== #
def page_evidence():
    st.markdown('<div class="nav"><div class="brand">📊 Model &amp; Evidence</div></div>', unsafe_allow_html=True)
    st.markdown('<p class="lead">Everything here is interactive and out-of-sample — the proof the model is real, not curve-fit. Hover any chart.</p>', unsafe_allow_html=True)
    hz = [str(h) for h in H]

    st.markdown('<div class="sec">Accuracy vs a naive baseline</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_bar(name="Model", x=[f"{h}d" for h in hz], y=[EV["per_h"][h]["acc"] for h in hz], marker_color=GREEN,
                error_y=dict(type="data", array=[EV["per_h"][h]["ci"] for h in hz]))
    fig.add_bar(name="Naive baseline", x=[f"{h}d" for h in hz], y=[EV["per_h"][h]["maj"] for h in hz], marker_color=GREY)
    fig.add_hline(y=50, line_dash="dash", line_color="white", opacity=.4)
    fig.update_yaxes(range=[45, max(EV["per_h"][h]["acc"] for h in hz)+4], title="accuracy %")
    st.plotly_chart(theme(fig), use_container_width=True, config={"displayModeBar": False})
    st.markdown(f'<div class="expl">📖 The model beats the "just guess the usual outcome" baseline at every horizon — '
                f'by <b>+{EV["per_h"]["20"]["edge"]:.1f} pts</b> at 20 days. Error bars are 95% confidence; they sit above '
                f'50%, so the edge is statistically real, not luck.</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec">Accuracy vs coverage — the honest path to high accuracy</div>', unsafe_allow_html=True)
    fig = go.Figure()
    for h, cl in zip(hz, [BLUE, GREEN, AMBER]):
        fig.add_scatter(x=EV["per_h"][h]["cov"], y=EV["per_h"][h]["cov_acc"], name=f"{h}d", line=dict(color=cl, width=2))
    fig.add_hline(y=50, line_dash="dash", line_color="white", opacity=.4)
    fig.add_hline(y=70, line_dash="dot", line_color=GREEN, opacity=.6)
    fig.update_xaxes(title="coverage % (act on only the most confident)", autorange="reversed"); fig.update_yaxes(title="accuracy %")
    st.plotly_chart(theme(fig), use_container_width=True, config={"displayModeBar": False})
    st.markdown('<div class="expl">📖 If you act on <b>every</b> signal, accuracy ≈55%. If you act on only the model\'s '
                '<b>most confident</b> calls (move left), accuracy climbs. This is how a "modest" model becomes useful — '
                'and the honest version of a "70%" target: reachable on a small, high-conviction slice, not every day.</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec">Is the edge stable over time?</div>', unsafe_allow_html=True)
    e = EV["per_h"]["10"]
    fig = go.Figure()
    fig.add_scatter(x=e["roll_dates"], y=e["roll_acc"], line=dict(color=GREEN, width=1.6), name="60-day rolling acc")
    fig.add_hline(y=50, line_dash="dash", line_color="white", opacity=.4)
    fig.add_hline(y=e["acc"], line_dash="dot", line_color=BLUE, opacity=.6)
    fig.update_yaxes(title="accuracy %")
    st.plotly_chart(theme(fig), use_container_width=True, config={"displayModeBar": False})
    st.markdown('<div class="expl">📖 Rolling 10-day-horizon accuracy over the out-of-sample years. It wobbles with market '
                'regimes but mostly stays above the 50% coin-flip line — the edge persists, it isn\'t a one-off fluke.</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec">Does it make money after costs?</div>', unsafe_allow_html=True)
    e = EV["per_h"]["10"]
    fig = go.Figure()
    fig.add_scatter(x=e["eq_dates"], y=e["eq_strat"], name=f"Top-5 strategy (net)", line=dict(color=GREEN, width=2))
    fig.add_scatter(x=e["eq_dates"], y=e["eq_bench"], name="Buy & hold", line=dict(color=GREY, width=1.8))
    fig.update_yaxes(title="growth of 1 unit")
    st.plotly_chart(theme(fig), use_container_width=True, config={"displayModeBar": False})
    st.markdown(f'<div class="expl">📖 Holding the 5 highest-conviction stocks each rebalance — net of ~{meta["cost"]:.1f}% '
                f'round-trip cost — grew to <b>{e["strat_x"]:.1f}×</b> vs <b>{e["bh_x"]:.1f}×</b> for buy-and-hold '
                f'(Sharpe {e["sharpe"]:.2f}). The ranking signal is more useful than the raw 55% suggests.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sec">Overfitting check (PBO)</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(x=[f"{h}d" for h in hz], y=[EV["per_h"][h]["pbo"] for h in hz],
            marker_color=[GREEN if EV["per_h"][h]["pbo"] < .5 else RED for h in hz]))
        fig.add_hline(y=.5, line_dash="dash", line_color="white", opacity=.4)
        fig.update_yaxes(range=[0, 1], title="PBO")
        st.plotly_chart(theme(fig, 300, False), use_container_width=True, config={"displayModeBar": False})
        st.markdown('<div class="expl">📖 Probability of Backtest Overfitting (López de Prado). ≈0 means the edge would '
                    '<b>survive</b> if we\'d just gotten lucky picking settings — i.e. it\'s real.</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="sec">What signals matter most</div>', unsafe_allow_html=True)
        imp = EV["importance"]
        fig = go.Figure(go.Bar(x=imp["gain"][::-1], y=imp["feature"][::-1], orientation="h", marker_color=BLUE))
        st.plotly_chart(theme(fig, 300, False), use_container_width=True, config={"displayModeBar": False})
        st.markdown('<div class="expl">📖 Feature importance. Market momentum/volatility/breadth dominate — the edge is '
                    'largely <b>timing the overall NEPSE regime</b>, not stock-specific magic.</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec">Why we adjust prices (bonus/rights)</div>', unsafe_allow_html=True)
    adj = EV["adjust"]
    fig = go.Figure()
    fig.add_scatter(x=adj["dates"], y=adj["raw"], name="raw close", line=dict(color=RED, width=1.4))
    fig.add_scatter(x=adj["dates"], y=adj["adj"], name="adjusted close", line=dict(color=BLUE, width=1.8))
    fig.update_yaxes(title="price (NPR)")
    st.plotly_chart(theme(fig), use_container_width=True, config={"displayModeBar": False})
    st.markdown(f'<div class="expl">📖 {adj["symbol"]} — NEPSE firms issue bonus/rights shares, which make the <b>raw</b> '
                f'price (red) drop sharply on the ex-date (an accounting event, not a real loss). We back-adjust (blue) so '
                f'the series is continuous; training on raw prices would mislabel every bonus issue as a crash.</div>', unsafe_allow_html=True)


# =========================================================================== #
def page_assistant():
    st.markdown('<div class="nav"><div class="brand">💬 Assistant</div></div>', unsafe_allow_html=True)
    mode = "llm" if assistant.get_provider() else "grounded"
    if mode == "grounded":
        st.markdown('<div class="expl">💡 Running in <b>grounded</b> mode (free, no API key) — answers come straight from the '
                    'live model data. For full ChatGPT-style chat, add a free <code>GROQ_API_KEY</code> as a Space secret.</div>', unsafe_allow_html=True)
    st.markdown('<p class="lead">Ask about any stock, how accurate the model is, how it works, or how to invest. '
                'Grounded in live predictions — not financial advice.</p>', unsafe_allow_html=True)
    if "chat" not in st.session_state:
        st.session_state.chat = [("assistant", "Hi! I'm the NEPSE Signals assistant 🤖 Ask me things like "
                                  "*“How does NABIL look?”*, *“How accurate is the model?”*, or *“What should a "
                                  "cautious investor do?”*")]
    for role, content in st.session_state.chat:
        with st.chat_message(role, avatar="📈" if role == "assistant" else None):
            st.markdown(content)
    if q := st.chat_input("Ask about a stock or the model…"):
        st.session_state.chat.append(("user", q))
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant", avatar="📈"):
            with st.spinner("thinking…"):
                resp, used = assistant.answer(q, st.session_state.chat[:-1], scores, meta)
            st.markdown(resp)
        st.session_state.chat.append(("assistant", resp))


# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown('<div class="sb-brand">📈 NEPSE·Signals</div><div class="sb-sub">AI forecasts for the Nepal Stock Exchange</div>', unsafe_allow_html=True)
nav = st.navigation([
    st.Page(page_home, title="Home", icon="🏠", default=True),
    st.Page(page_explore, title="Explore a stock", icon="🔍"),
    st.Page(page_evidence, title="Model & Evidence", icon="📊"),
    st.Page(page_assistant, title="Assistant", icon="💬"),
])
nav.run()
