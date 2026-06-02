"""
NEPSE Signals — live multi-page Streamlit app (SaaS-grade UI).

Pages:
  • Home          — hero, personalized recommendation, today's signals, watchlist, evidence
  • Explore       — pick a stock → interactive price chart + honest forecast cone + signals

The model RUNS live on every visit: LightGBM boosters in models/ score the latest
data. Swap models/lgbm_h*.txt (+ push) → new outputs, no code change.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).parent
st.set_page_config(page_title="NEPSE Signals — AI stock forecasts", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")


# --------------------------------------------------------------------------- #
@st.cache_data
def load_meta():
    return json.loads((HERE / "data/metrics.json").read_text())

@st.cache_data
def load_data():
    return (pd.read_parquet(HERE / "data/latest_features.parquet"),
            pd.read_parquet(HERE / "data/risk.parquet"),
            pd.read_parquet(HERE / "data/prices.parquet"))

@st.cache_resource
def load_models(horizons):
    return {h: lgb.Booster(model_file=str(HERE / f"models/lgbm_h{h}.txt")) for h in horizons}


meta = load_meta()
feats, risk, prices = load_data()
models = load_models(meta["horizons"])
FEATURES, H = meta["features"], meta["horizons"]
m = meta["metrics"]; m10 = m["10"] if "10" in m else m[10]; m5 = m["5"] if "5" in m else m[5]

scores = feats[["symbol"]].copy()
for h in H:
    scores[f"p{h}"] = models[h].predict(feats[FEATURES].values)
scores = scores.merge(risk, on="symbol", how="left")
PRICE_SYMS = sorted(set(prices["symbol"]).intersection(scores["symbol"]))

GREEN, RED, BLUE, AMBER, BG, CARD = "#22c55e", "#f87171", "#5eead4", "#fbbf24", "#0a0e1a", "#141b2e"


# --------------------------------------------------------------------------- #
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
.brand { font-weight:800; font-size:1.1rem; letter-spacing:-.02em; } .brand .dot{color:#22c55e;}
.tag { font-size:.72rem; color:#8b95a7; border:1px solid #2a3450; border-radius:999px; padding:4px 10px; }
.hero { position:relative; border:1px solid #243049; border-radius:20px; padding:34px 30px;
  background:linear-gradient(135deg, rgba(34,197,94,.10), rgba(56,139,255,.08) 60%, rgba(20,27,46,.4));
  box-shadow:0 20px 60px -30px rgba(0,0,0,.8); overflow:hidden; }
.hero h1 { font-size:clamp(1.7rem,4vw,2.7rem); font-weight:900; line-height:1.1; margin:0 0 10px; letter-spacing:-.03em; }
.hero h1 .grad { background:linear-gradient(90deg,#22c55e,#5eead4); -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }
.hero p { font-size:clamp(.98rem,2vw,1.15rem); color:#c3cbd9; max-width:680px; margin:0 0 16px; }
.pills { display:flex; flex-wrap:wrap; gap:8px; } .pill{ font-size:.8rem; color:#cdd5e3; background:rgba(255,255,255,.04); border:1px solid #2a3450; border-radius:999px; padding:5px 12px; } .pill b{color:#fff;}
.cardgrid { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:14px; margin:18px 0 6px; }
.mcard { background:linear-gradient(180deg, rgba(255,255,255,.035), rgba(255,255,255,.01)); border:1px solid #243049; border-radius:16px; padding:18px; transition:transform .15s, border-color .15s; }
.mcard:hover { transform:translateY(-3px); border-color:#34507a; }
.mcard .top { font-size:.76rem; color:#8b95a7; font-weight:600; text-transform:uppercase; letter-spacing:.04em; }
.mcard .val { font-size:1.9rem; font-weight:800; margin:6px 0 2px; letter-spacing:-.02em; } .mcard .sub{ font-size:.82rem; color:#9aa4b6; }
.green{color:#22c55e;} .blue{color:#5eead4;} .amber{color:#fbbf24;} .red{color:#f87171;}
.sec { display:flex; align-items:center; gap:10px; margin:32px 0 6px; font-size:1.35rem; font-weight:800; letter-spacing:-.02em; }
.sec:before { content:""; width:5px; height:24px; border-radius:3px; background:linear-gradient(#22c55e,#0e7a3a); }
.lead { color:#9aa4b6; margin:0 0 14px; font-size:.95rem; }
.stButton>button, .stFormSubmitButton>button { background:linear-gradient(90deg,#22c55e,#16a34a); color:#04210f; font-weight:800; border:0; border-radius:12px; padding:.6rem 1rem; box-shadow:0 10px 25px -12px rgba(34,197,94,.8); }
.stButton>button:hover, .stFormSubmitButton>button:hover { filter:brightness(1.08); }
[data-testid="stForm"] { background:rgba(255,255,255,.02); border:1px solid #243049 !important; border-radius:18px; padding:18px 18px 6px; }
[data-testid="stDataFrame"] { border:1px solid #243049; border-radius:14px; overflow:hidden; }
.stTabs [data-baseweb="tab"] { background:rgba(255,255,255,.03); border:1px solid #243049; border-radius:10px 10px 0 0; padding:8px 14px; }
.stTabs [aria-selected="true"] { background:rgba(34,197,94,.12); border-color:#2f7a4d; }
.disc { background:linear-gradient(90deg, rgba(251,191,36,.10), rgba(251,191,36,.02)); border:1px solid #5a4a1e; border-radius:14px; padding:13px 16px; font-size:.88rem; color:#e8d9ad; margin:14px 0; }
.readout { background:rgba(255,255,255,.03); border:1px solid #243049; border-left:4px solid #22c55e; border-radius:12px; padding:14px 16px; font-size:1.02rem; }
.foot { margin-top:40px; padding-top:18px; border-top:1px solid #1e2740; font-size:.82rem; color:#7a849a; } .foot a{color:#5eead4; text-decoration:none;}
.steps { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; }
.step { background:rgba(255,255,255,.025); border:1px solid #243049; border-radius:14px; padding:14px 16px; }
.step .n{font-weight:800;color:#22c55e;font-size:.85rem;} .step .t{font-weight:700;margin:4px 0;} .step .d{color:#9aa4b6;font-size:.85rem;}
@media (max-width:640px){ .block-container{padding:.8rem .8rem 4rem !important;} .hero{padding:24px 18px;} .mcard .val{font-size:1.6rem;} }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
def conviction_word(p):
    c = (p - .5) * 100
    if c >= 6:  return "strongly bullish", GREEN
    if c >= 2:  return "leans bullish", GREEN
    if c > -2:  return "neutral", AMBER
    if c > -6:  return "leans bearish", RED
    return "strongly bearish", RED


def forecast_figure(sym, h):
    hist = prices[prices.symbol == sym].sort_values("date").tail(120).copy()
    row = scores[scores.symbol == sym].iloc[0]
    p_up = float(row[f"p{h}"]); vol10 = float(row["vol10"]) if pd.notna(row["vol10"]) else 6.0
    P0 = float(hist["close"].iloc[-1]); last_date = pd.Timestamp(hist["date"].iloc[-1])
    sig_d = (vol10 / 100) / np.sqrt(10)                      # per-day vol
    sig_h = sig_d * np.sqrt(h)
    drift_total = (2 * p_up - 1) * 0.5 * sig_h               # lean up to ±½σ over horizon
    fut = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=h)
    t = np.arange(1, h + 1)
    central = P0 * (1 + drift_total * (t / h))
    st_ = sig_d * np.sqrt(t)
    up1, lo1 = central * (1 + st_), central * (1 - st_)
    up2, lo2 = central * (1 + 2 * st_), central * (1 - 2 * st_)
    x0 = [last_date]; col = GREEN if p_up >= .5 else RED

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.76, 0.24],
                        vertical_spacing=0.04)
    fig.add_trace(go.Candlestick(x=hist["date"], open=hist["open"], high=hist["high"],
        low=hist["low"], close=hist["close"], name="price",
        increasing_line_color=GREEN, decreasing_line_color=RED, showlegend=False), row=1, col=1)
    # forecast cone (start each band at the last actual close for continuity)
    fx = list(x0) + list(fut)
    band = lambda arr: [P0] + list(arr)
    fig.add_trace(go.Scatter(x=fx, y=band(up2), line=dict(width=0), hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=fx, y=band(lo2), line=dict(width=0), fill="tonexty",
        fillcolor="rgba(34,197,94,.08)" if p_up>=.5 else "rgba(248,113,113,.08)", hoverinfo="skip",
        name="95% range", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=fx, y=band(up1), line=dict(width=0), hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=fx, y=band(lo1), line=dict(width=0), fill="tonexty",
        fillcolor="rgba(34,197,94,.16)" if p_up>=.5 else "rgba(248,113,113,.16)", hoverinfo="skip",
        name="68% range", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=fx, y=band(central), line=dict(color=col, width=2, dash="dash"),
        name="model lean", mode="lines"), row=1, col=1)
    # volume
    vcol = np.where(hist["close"] >= hist["open"], "rgba(34,197,94,.5)", "rgba(248,113,113,.5)")
    fig.add_trace(go.Bar(x=hist["date"], y=hist["volume"], marker_color=vcol, showlegend=False), row=2, col=1)
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=520, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False,
        hovermode="x unified", legend=dict(orientation="h", y=1.06, x=0),
        font=dict(color="#c3cbd9"))
    fig.update_xaxes(gridcolor="rgba(255,255,255,.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.05)", row=1, col=1, title_text="Price (NPR, adj)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.05)", row=2, col=1, title_text="Vol")
    rng = (float(lo2[-1]), float(up2[-1]))
    return fig, dict(p_up=p_up, vol10=vol10, P0=P0, lo=rng[0], hi=rng[1], h=h)


# =========================================================================== #
# PAGE: HOME
# =========================================================================== #
def page_home():
    st.markdown(f'<div class="nav"><div class="brand">📈 NEPSE<span class="dot">·</span>Signals</div>'
                f'<div class="tag">updated {meta["as_of"]}</div></div>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="hero">
  <h1>Can a machine predict the <span class="grad">Nepal Stock Exchange?</span></h1>
  <p>A little — honestly about <b>{m10['acc']:.1f}%</b> of the time on 10-day direction. Small but real.
  Get a <b>personalized, risk-tuned</b> basket below, or open <b>Explore</b> to chart any stock with its forecast.</p>
  <div class="pills"><span class="pill">🏦 <b>{meta['n_stocks']}</b> stocks</span>
    <span class="pill">🗓️ <b>{meta['yrs']}</b> years</span><span class="pill">🔬 out-of-sample tested</span>
    <span class="pill">⚙️ overfitting-checked (PBO {m10['pbo']:.2f})</span></div>
</div>""", unsafe_allow_html=True)

    l, r = st.columns([3, 1])
    l.markdown('<div class="disc">⚠️ <b>Not financial advice.</b> Educational tool. A ~55% edge means it is '
               'wrong nearly half the time. Never invest money you can\'t afford to lose.</div>', unsafe_allow_html=True)
    nerd = r.toggle("🤓 Nerd mode", value=False)

    st.markdown(f"""<div class="cardgrid">
  <div class="mcard"><div class="top">10-day accuracy</div><div class="val green">{m10['acc']:.1f}%</div><div class="sub">+{m10['edge']:.1f} pts vs baseline</div></div>
  <div class="mcard"><div class="top">High-conviction</div><div class="val green">{m10['acc20']:.0f}%</div><div class="sub">top-20% confident picks</div></div>
  <div class="mcard"><div class="top">Overfitting (PBO)</div><div class="val blue">{m10['pbo']:.2f}</div><div class="sub">≈0 → real, not curve-fit</div></div>
  <div class="mcard"><div class="top">Backtest 5d, net</div><div class="val green">{m5['strat_x']:.1f}×</div><div class="sub">vs {m5['bh_x']:.1f}× buy &amp; hold</div></div>
</div>""", unsafe_allow_html=True)
    if nerd:
        st.caption(f"🤓 Global LightGBM panel over {meta['n_stocks']} stocks, walk-forward OOS "
                   f"({m10['n']:,} preds) with h-day embargo. Edge +{m10['edge']:.1f}pts (±{m10['ci']:.1f}). "
                   f"PBO via CSCV. Live scoring of models/ on the latest bar.")

    st.markdown('<div class="sec">🎯 Get your personalized recommendation</div>', unsafe_allow_html=True)
    st.markdown('<p class="lead">Your risk appetite, horizon and budget → a suggested basket with allocations.</p>', unsafe_allow_html=True)
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
        pool = scores.dropna(subset=["vol10"]).copy()
        qcap = {"Conservative": .4, "Balanced": .7, "Aggressive": 1.0}[risk_app]
        vcap = pool["vol10"].quantile(qcap)
        cand = pool[(pool[pcol] > .5) & (pool["vol10"] <= vcap)].copy()
        cand["conviction"] = (cand[pcol] - .5) * 100
        cand["score"] = cand["conviction"] / cand["vol10"] if weighting == "Lower-risk first" else cand["conviction"]
        cand = cand.sort_values("score", ascending=False).head(nstocks)
        if not len(cand):
            st.error(f"At a **{risk_app.lower()}** level over **{hlabel}**, the model leans bullish on **no stocks "
                     f"within your risk limit**. That's a real signal — the market looks weak; **cash may be prudent**. "
                     f"Try a longer horizon or higher risk appetite.")
        else:
            w = (np.ones(len(cand)) if weighting == "Equal"
                 else 1/cand["vol10"].values if weighting == "Lower-risk first" else cand["conviction"].values)
            w = w / w.sum()
            out = pd.DataFrame({"Stock": cand["symbol"].values, "P(up)": (cand[pcol].values*100).round(1),
                "Conviction": cand["conviction"].values.round(1), "Risk 10d vol%": cand["vol10"].values.round(1),
                "Allocation %": (w*100).round(1), "Allocation NPR": (w*capital).round(0).astype(int)})
            st.success(f"Suggested **{len(cand)}-stock basket** · **{risk_app.lower()}** · **{hlabel}** · **NPR {capital:,.0f}**")
            st.dataframe(out, hide_index=True, use_container_width=True,
                column_config={"P(up)": st.column_config.NumberColumn(format="%.1f%%"),
                               "Allocation NPR": st.column_config.NumberColumn(format="NPR %d")})
            st.caption("Allocations from the model's conviction & GARCH risk. Not advice — wrong ~45% of the time.")

    st.markdown('<div class="sec">📍 Today\'s signals</div>', unsafe_allow_html=True)
    s10 = scores.sort_values("p10", ascending=False)
    f = lambda d: pd.DataFrame({"Stock": d["symbol"].values, "P(up)": (d["p10"].values*100).round(1),
        "Conviction": ((d["p10"].values-.5)*100).round(1), "Risk%": d["vol10"].values.round(1)})
    cfg = {"P(up)": st.column_config.NumberColumn(format="%.1f%%")}
    L, R = st.columns(2)
    L.markdown("##### ▲ Top 10 — leans UP"); L.dataframe(f(s10.head(10)), hide_index=True, use_container_width=True, column_config=cfg)
    R.markdown("##### ▼ Bottom 10 — leans DOWN"); R.dataframe(f(s10.tail(10).iloc[::-1]), hide_index=True, use_container_width=True, column_config=cfg)
    st.info("👉 Want a chart + forecast for one stock? Open **Explore a stock** in the left sidebar.", icon="🔍")

    st.markdown('<div class="sec">🔬 The evidence</div>', unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["Does it work?", "What to do", "Data cleaning"])
    t1.image(str(HERE / "assets/dashboard_performance.png"), use_container_width=True)
    t2.image(str(HERE / "assets/dashboard_decision.png"), use_container_width=True)
    t3.image(str(HERE / "assets/adjustment_validation.png"), use_container_width=True)

    st.markdown(f'<div class="foot">📈 <b>NEPSE Signals</b> · educational research, not financial advice · '
                f'data as of {meta["as_of"]} · global LightGBM</div>', unsafe_allow_html=True)


# =========================================================================== #
# PAGE: EXPLORE
# =========================================================================== #
def page_explore():
    st.markdown(f'<div class="nav"><div class="brand">🔍 Explore a stock</div>'
                f'<div class="tag">updated {meta["as_of"]}</div></div>', unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])
    default = PRICE_SYMS.index("NABIL") if "NABIL" in PRICE_SYMS else 0
    sym = c1.selectbox("Stock", PRICE_SYMS, index=default)
    hlabel = c2.selectbox("Forecast horizon", ["1 week (5d)", "2 weeks (10d)", "1 month (20d)"], index=1)
    h = {"1 week (5d)": 5, "2 weeks (10d)": 10, "1 month (20d)": 20}[hlabel]

    row = scores[scores.symbol == sym].iloc[0]
    p_up = float(row[f"p{h}"]); word, col = conviction_word(p_up)
    fig, info = forecast_figure(sym, h)

    # signal cards
    st.markdown(f"""<div class="cardgrid">
  <div class="mcard"><div class="top">Last price</div><div class="val">{info['P0']:.0f}</div><div class="sub">NPR (adjusted)</div></div>
  <div class="mcard"><div class="top">P(up) over {h}d</div><div class="val" style="color:{col}">{p_up*100:.0f}%</div><div class="sub">{word}</div></div>
  <div class="mcard"><div class="top">Risk (10d vol)</div><div class="val amber">{info['vol10']:.1f}%</div><div class="sub">GARCH forecast</div></div>
  <div class="mcard"><div class="top">Likely {h}d range</div><div class="val" style="font-size:1.3rem">{info['lo']:.0f}–{info['hi']:.0f}</div><div class="sub">95% cone (NPR)</div></div>
</div>""", unsafe_allow_html=True)

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    arrow = "↑" if p_up >= .5 else "↓"
    st.markdown(f'<div class="readout">Over the next <b>{hlabel}</b>, the model <b style="color:{col}">{word}</b> '
                f'on <b>{sym}</b> {arrow} — probability of an up-move <b>{p_up*100:.0f}%</b>. '
                f'With its forecast volatility, a typical outcome lands roughly between '
                f'<b>{info["lo"]:.0f}</b> and <b>{info["hi"]:.0f}</b> NPR. The dashed line is the model\'s '
                f'directional lean, not a precise price target — the shaded cone is the realistic range.</div>',
                unsafe_allow_html=True)

    # all-horizon mini view
    st.markdown('<div class="sec">Across horizons</div>', unsafe_allow_html=True)
    cols = st.columns(len(H))
    for cc, hh in zip(cols, H):
        pu = float(row[f"p{hh}"])
        cc.metric(f"{hh}-day P(up)", f"{pu*100:.0f}%", f"{(pu-.5)*100:+.1f} pts")

    st.caption("Forecast cone = model's directional lean ± realistic volatility range (GARCH). Not a guarantee; "
               "the model is wrong ~45% of the time. Educational tool, not financial advice.")
    st.markdown('<div class="foot">📈 <b>NEPSE Signals</b> · prices are bonus/rights-adjusted · '
                f'data as of {meta["as_of"]}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown('<div class="sb-brand">📈 NEPSE·Signals</div>'
                '<div class="sb-sub">AI forecasts for the Nepal Stock Exchange</div>', unsafe_allow_html=True)
nav = st.navigation([
    st.Page(page_home, title="Home", icon="🏠", default=True),
    st.Page(page_explore, title="Explore a stock", icon="🔍"),
])
nav.run()
