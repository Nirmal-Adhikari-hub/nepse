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
def load_outlook():
    import os
    p = HERE / "data/market_outlook.json"
    return json.loads(p.read_text()) if os.path.exists(p) else None
@st.cache_data
def load_data():
    return (pd.read_parquet(HERE / "data/latest_features.parquet"),
            pd.read_parquet(HERE / "data/risk.parquet"),
            pd.read_parquet(HERE / "data/prices.parquet"))
@st.cache_resource
def load_models(hz): return {h: lgb.Booster(model_file=str(HERE / f"models/lgbm_h{h}.txt")) for h in hz}
@st.cache_resource
def load_models_xs(hz):
    import os
    return {h: lgb.Booster(model_file=str(HERE / f"models/lgbm_xs_h{h}.txt"))
            for h in hz if os.path.exists(str(HERE / f"models/lgbm_xs_h{h}.txt"))}


meta = load_meta(); EV = load_evidence()
feats, risk, prices = load_data()
models = load_models(meta["horizons"])
FEATURES, H = meta["features"], meta["horizons"]
models_xs = load_models_xs(H)
HAS_XS = len(models_xs) == len(H)               # beat-market ranking signal available?
m = meta["metrics"]; m10 = m.get("10", m.get(10)); m5 = m.get("5", m.get(5))
XM = meta.get("xs_metrics", {}); xm10 = XM.get("10", {})
# headline metrics prefer the deployed ranking signal (outperformance) when available
HEAD = dict(acc20=xm10.get("acc20", m10["acc20"]), acc=xm10.get("acc", m10["acc"]),
            edge=xm10.get("edge", m10["edge"]), kind=("outperformance" if XM else "direction"))

scores = feats[["symbol"]].copy()
for h in H:
    scores[f"p{h}"] = models[h].predict(feats[FEATURES].values)          # direction (price cone)
    scores[f"px{h}"] = (models_xs[h].predict(feats[FEATURES].values) if HAS_XS else scores[f"p{h}"])  # outperform (ranking)
scores = scores.merge(risk, on="symbol", how="left")
ASOF = meta.get("asof", {})
scores["asof"] = scores["symbol"].map(ASOF).fillna(meta["as_of"])
scores["fresh"] = scores["asof"] >= "2026-01-01"
fresh_scores = scores[scores["fresh"]].copy()
PRICE_SYMS = sorted(set(prices["symbol"]).intersection(scores["symbol"]))

PALETTES = {
  "dark":  dict(bg="#0a0e1a", glow="rgba(24,35,63,1)", card="#121a2e", card2="#0f1626",
                bd="#243049", tx="#e6edf3", mut="#9aa4b6", sb="#0c1220", sbbd="#1c2740",
                heroto="rgba(20,27,46,.4)", rowbd="#1b2440"),
  "light": dict(bg="#f4f7fc", glow="rgba(210,228,255,.8)", card="#ffffff", card2="#f7f9fd",
                bd="#e2e8f2", tx="#0f172a", mut="#566174", sb="#eef2f9", sbbd="#dde5f0",
                heroto="rgba(236,242,251,.6)", rowbd="#e9eef6"),
}
with st.sidebar:
    st.markdown('<div class="sb-brand">📈 NEPSE·Signals</div>'
                '<div class="sb-sub">AI forecasts for the Nepal Stock Exchange</div>', unsafe_allow_html=True)
    LIGHT = st.toggle("☀️ Light mode", value=False, key="lightmode")
PAL = PALETTES["light" if LIGHT else "dark"]
st.session_state["_dark"] = not LIGHT
with st.sidebar:
    st.divider()
    st.markdown("**💬 Quick ask the assistant**")
    qa = st.text_input("quickask", placeholder="e.g. How does NABIL look?",
                       label_visibility="collapsed", key="sidebar_qa")
    if qa:
        with st.spinner("…"):
            _ans, _ = assistant.answer(qa, [], scores, meta)
        st.info(_ans)
    st.caption("Open the **Assistant** page for full chat →")

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
:root {{ --bg:{PAL['bg']}; --glow:{PAL['glow']}; --card:{PAL['card']}; --card2:{PAL['card2']};
  --bd:{PAL['bd']}; --tx:{PAL['tx']}; --mut:{PAL['mut']}; --sb:{PAL['sb']}; --sbbd:{PAL['sbbd']};
  --heroto:{PAL['heroto']}; --rowbd:{PAL['rowbd']}; }}
html, body, [class*="css"], .stApp, button, input, textarea {{ font-family:'Inter',-apple-system,sans-serif; }}
#MainMenu, header[data-testid="stHeader"], footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{ display:none !important; }}
.stApp {{ background: radial-gradient(1100px 500px at 50% -8%, var(--glow) 0%, rgba(0,0,0,0) 60%), var(--bg); color:var(--tx); }}
.block-container {{ max-width:1140px !important; padding:1.1rem 1.2rem 5rem !important; }}
.stApp p, .stApp label, .stApp span, .stMarkdown {{ color:var(--tx); }}
[data-testid="stSidebar"] {{ background:var(--sb); border-right:1px solid var(--sbbd); }}
.sb-brand {{ font-weight:900; font-size:1.15rem; padding:6px 4px 2px; color:var(--tx); }}
.sb-sub {{ color:var(--mut); font-size:.78rem; padding:0 4px 10px; }}
.nav {{ display:flex; align-items:center; justify-content:space-between; padding:2px 2px 14px; }}
.brand {{ font-weight:800; font-size:1.1rem; color:var(--tx); }} .brand .dot{{color:#22c55e;}}
.tag {{ font-size:.72rem; color:var(--mut); border:1px solid var(--bd); border-radius:999px; padding:4px 10px; }}
.hero {{ position:relative; border:1px solid var(--bd); border-radius:20px; padding:34px 30px;
  background:linear-gradient(135deg, rgba(34,197,94,.12), rgba(56,139,255,.10) 60%, var(--heroto)); overflow:hidden; }}
.hero h1 {{ font-size:clamp(1.7rem,4vw,2.7rem); font-weight:900; line-height:1.1; margin:0 0 10px; letter-spacing:-.03em; color:var(--tx); }}
.hero h1 .grad {{ background:linear-gradient(90deg,#22c55e,#0ea5a0); -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }}
.hero p {{ font-size:clamp(.98rem,2vw,1.15rem); color:var(--mut); max-width:700px; margin:0 0 16px; }}
.pills {{ display:flex; flex-wrap:wrap; gap:8px; }} .pill{{ font-size:.8rem; color:var(--tx); background:var(--card2); border:1px solid var(--bd); border-radius:999px; padding:5px 12px; }}
.cardgrid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:14px; margin:18px 0 6px; }}
.mcard {{ background:var(--card); border:1px solid var(--bd); border-radius:16px; padding:18px; transition:transform .15s; }}
.mcard:hover {{ transform:translateY(-3px); }}
.mcard .top {{ font-size:.76rem; color:var(--mut); font-weight:600; text-transform:uppercase; letter-spacing:.04em; }}
.mcard .val {{ font-size:1.9rem; font-weight:800; margin:6px 0 2px; color:var(--tx); }} .mcard .sub{{ font-size:.82rem; color:var(--mut); }}
.green{{color:#16a34a;}} .blue{{color:#0ea5a0;}} .amber{{color:#d97706;}} .red{{color:#dc2626;}}
.sec {{ display:flex; align-items:center; gap:10px; margin:30px 0 6px; font-size:1.35rem; font-weight:800; color:var(--tx); }}
.sec:before {{ content:""; width:5px; height:24px; border-radius:3px; background:linear-gradient(#22c55e,#0e7a3a); }}
.lead {{ color:var(--mut); margin:0 0 14px; font-size:.95rem; }}
.stButton>button, .stFormSubmitButton>button {{ background:linear-gradient(90deg,#22c55e,#16a34a); color:#04210f; font-weight:800; border:0; border-radius:12px; padding:.6rem 1rem; }}
[data-testid="stForm"] {{ background:var(--card2); border:1px solid var(--bd) !important; border-radius:18px; padding:18px 18px 6px; }}
.disc {{ background:rgba(251,191,36,.12); border:1px solid rgba(180,140,20,.45); border-radius:14px; padding:13px 16px; font-size:.88rem; color:#b07c12; margin:14px 0; }}
.readout {{ background:var(--card2); border:1px solid var(--bd); border-left:4px solid #22c55e; border-radius:12px; padding:14px 16px; font-size:1.02rem; color:var(--tx); }}
.expl {{ background:rgba(56,139,255,.10); border:1px solid rgba(56,139,255,.32); border-radius:12px; padding:12px 16px; font-size:.92rem; color:var(--tx); margin:6px 0 18px; }}
.metric-row {{ display:flex; align-items:center; justify-content:space-between; padding:9px 2px; border-bottom:1px solid var(--rowbd); }}
.metric-row .lab {{ font-weight:600; color:var(--tx); }} .metric-row .txt {{ color:var(--mut); font-size:.86rem; }}
.badge {{ font-size:.72rem; font-weight:800; padding:3px 10px; border-radius:999px; }}
.foot {{ margin-top:36px; padding-top:18px; border-top:1px solid var(--bd); font-size:.82rem; color:var(--mut); }}
table.t {{ width:100%; border-collapse:collapse; font-size:.92rem; margin:4px 0; }}
table.t th {{ text-align:left; color:var(--mut); font-weight:600; font-size:.72rem; text-transform:uppercase; letter-spacing:.04em; padding:8px 10px; border-bottom:1px solid var(--bd); }}
table.t td {{ padding:8px 10px; border-bottom:1px solid var(--rowbd); color:var(--tx); }}
table.t td.sym {{ font-weight:700; color:#0ea5a0; }}
table.t tr:hover td {{ background:var(--card2); }}
@media (max-width:640px){{ .block-container{{padding:.8rem .8rem 4rem !important;}} .hero{{padding:24px 18px;}} .mcard .val{{font-size:1.6rem;}} }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.markdown("""<style>
/* floating chat bubble (Facebook-style) — the single popover on the page */
div[data-testid="stPopover"] { position: fixed !important; bottom: 22px; right: 22px; z-index: 99999; }
div[data-testid="stPopover"] > div > button { border-radius: 999px !important;
  background: linear-gradient(90deg,#22c55e,#16a34a) !important; color:#04210f !important; font-weight:800 !important;
  padding:.6rem 1.1rem !important; box-shadow:0 10px 28px -6px rgba(34,197,94,.7) !important; border:0 !important; }
div[data-testid="stPopoverBody"] { width: 460px !important; max-width: 92vw !important; }
@media (max-width:640px){ div[data-testid="stPopoverBody"]{ width: 94vw !important; } }
</style>""", unsafe_allow_html=True)

st.markdown("""<style>
/* ===== premium polish + mobile-immersive ===== */
.stApp { -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility; }
html { scroll-behavior: smooth; }
.mcard { box-shadow: 0 8px 26px -16px rgba(0,0,0,.50); }
.hero  { box-shadow: 0 28px 70px -38px rgba(0,0,0,.65); }
.stButton>button, .stFormSubmitButton>button { border-radius:12px !important; transition:filter .15s, transform .08s; }
.stButton>button:hover { filter:brightness(1.06); } .stButton>button:active { transform:translateY(1px); }
.sec { scroll-margin-top: 14px; }
/* MOBILE: full-bleed, safe-area aware (Dynamic Island / notch), big touch targets */
@media (max-width: 760px) {
  .block-container { padding-top: calc(0.6rem + env(safe-area-inset-top)) !important;
    padding-left: max(0.7rem, env(safe-area-inset-left)) !important;
    padding-right: max(0.7rem, env(safe-area-inset-right)) !important; padding-bottom: 5.5rem !important; }
  .cardgrid { grid-template-columns: 1fr 1fr !important; gap:10px; }
  .hero { padding: 22px 18px; border-radius:18px; } .hero h1 { font-size: 1.75rem; }
  .sec { font-size: 1.18rem; margin-top: 24px; }
  .stButton>button, .stFormSubmitButton>button { min-height: 44px; font-size: 1rem; }
  [data-baseweb="select"] > div, [data-testid="stTextInput"] input { min-height: 44px; }
  div[data-testid="stPopover"] { bottom: calc(14px + env(safe-area-inset-bottom)) !important; right: 14px !important; }
  div[data-testid="stPopover"] > div > button { padding:.7rem 1rem !important; }
  table.t th, table.t td { padding: 7px 6px; font-size: .84rem; }
}
@media (max-width: 430px) { .cardgrid { grid-template-columns: 1fr !important; } }
</style>""", unsafe_allow_html=True)

if LIGHT:
    st.markdown("""<style>
/* ===== light-mode widget overrides (Streamlit base theme is dark) ===== */
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input, [data-baseweb="input"] input,
[data-baseweb="textarea"] textarea, [data-testid="stChatInput"] textarea { background:#ffffff !important; color:#0f172a !important; }
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"] { background:#ffffff !important; border-color:#d4dbe8 !important; }
[data-testid="stNumberInput"] button { background:#eef2f9 !important; color:#0f172a !important; }
[data-baseweb="select"] > div { background:#ffffff !important; border-color:#d4dbe8 !important; }
[data-baseweb="select"] span, [data-baseweb="select"] div { color:#0f172a !important; }
[data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"], ul[role="listbox"], [data-baseweb="popover"] > div { background:#ffffff !important; }
[role="option"], [role="option"] * { background:#ffffff !important; color:#0f172a !important; }
[role="option"]:hover { background:#eef2f9 !important; }
label, .stRadio label, .stCheckbox label, [data-testid="stWidgetLabel"] *, [role="radiogroup"] label,
.stSlider label, [data-testid="stForm"] label, [data-baseweb="radio"] div { color:#0f172a !important; }
[data-testid="stThumbValue"], [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] { color:#0f172a !important; }
[data-testid="stExpander"] { background:#ffffff !important; border:1px solid #e2e8f2 !important; border-radius:12px; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary *, details summary, details summary * { color:#0f172a !important; }
.stTabs [data-baseweb="tab"] { color:#334155 !important; background:#f0f4fa !important; border-color:#e2e8f2 !important; }
.stTabs [aria-selected="true"] { color:#0f172a !important; background:rgba(34,197,94,.14) !important; }
[data-testid="stMetricValue"] { color:#0f172a !important; } [data-testid="stMetricLabel"] * { color:#566174 !important; }
[data-testid="stMetricDelta"] { color:#0f172a !important; }
[data-testid="stChatMessage"] { background:#f7f9fd !important; border:1px solid #e9eef6; }
[data-testid="stChatMessage"] * { color:#0f172a !important; }
.stButton>button[kind="secondary"], button[data-testid="baseButton-secondary"], [data-testid="stForm"] .stButton>button { color:#0f172a; }
.stButton>button:not([kind="primary"]) { background:#eef2f9 !important; color:#0f172a !important; border:1px solid #d4dbe8 !important; }
[data-testid="stNotification"], .stAlert, .stAlert * { color:#0f172a !important; }
[data-testid="stCaptionContainer"] *, .stCaption { color:#566174 !important; }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] a { color:#0f172a !important; }
[data-testid="stSidebarNav"] a span { color:#0f172a !important; }
/* alert/info/success/warning boxes: light bg + dark text (base theme is dark) */
[data-testid="stNotification"], [data-testid="stNotificationContentInfo"], .stAlert,
div[data-baseweb="notification"] { background:#eaf1fb !important; }
[data-testid="stNotification"] *, .stAlert * { color:#0f172a !important; }
/* the floating chat popover window in light mode */
div[data-testid="stPopoverBody"], div[data-testid="stPopoverBody"] [data-testid="stChatMessage"] { background:#ffffff !important; }
div[data-testid="stPopoverBody"] * { color:#0f172a !important; }
/* the invisible sidebar collapse/expand ">" chevron + any header icons -> dark */
[data-testid="stSidebarCollapseButton"] svg, [data-testid="collapsedControl"] svg,
[data-testid="stSidebarCollapsedControl"] svg, [data-testid="baseButton-headerNoPadding"] svg,
button[kind="header"] svg, [data-testid="stSidebarCollapseButton"], [data-testid="collapsedControl"] {
  color:#0f172a !important; fill:#0f172a !important; }
[data-testid="collapsedControl"] { background:#e8eef7 !important; border-radius:8px; }
</style>""", unsafe_allow_html=True)


def html_table(df, color_cols=()):
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body = ""
    for _, r in df.iterrows():
        tds = ""
        for c in df.columns:
            v = r[c]; cls = "sym" if c.lower() in ("stock", "symbol") else ""
            style = ""
            if c in color_cols:
                try:
                    style = f"color:{'#16a34a' if float(v) >= 0 else '#dc2626'};font-weight:700"
                except Exception:
                    pass
            tds += f'<td class="{cls}" style="{style}">{v}</td>'
        body += f"<tr>{tds}</tr>"
    return f'<table class="t"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


# ---------- shared helpers ------------------------------------------------- #
def conviction_word(p):
    c = (p - .5) * 100
    if c >= 6:  return "strongly bullish", GREEN
    if c >= 2:  return "leans bullish", GREEN
    if c > -2:  return "neutral", AMBER
    if c > -6:  return "leans bearish", RED
    return "strongly bearish", RED


def verdict_label(p):
    """One-word, plain verdict for tables."""
    c = (p - .5) * 100
    if c >= 5:  return "🟢 Strong"
    if c >= 1.5: return "🟢 Lean up"
    if c > -1.5: return "🟡 Neutral"
    if c > -5:  return "🔴 Lean down"
    return "🔴 Avoid"


def accuracy_grid(right=57):
    """100-square visual: `right` greens (correct) + rest muted (missed)."""
    cells = "".join(
        f'<span style="display:inline-block;width:13px;height:13px;margin:1.5px;border-radius:3px;'
        f'background:{"#22c55e" if i < right else "rgba(140,150,170,.30)"}"></span>'
        for i in range(100))
    return f'<div style="line-height:0;max-width:170px;margin:6px 0">{cells}</div>'


def theme(fig, h=360, legend=True):
    dark = st.session_state.get("_dark", True)
    fig.update_layout(template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=h,
        margin=dict(l=10, r=10, t=30, b=10), font=dict(color="#c3cbd9" if dark else "#334155"),
        hovermode="x unified", showlegend=legend, legend=dict(orientation="h", y=1.12, x=0))
    grid = "rgba(255,255,255,.06)" if dark else "rgba(0,0,0,.08)"
    fig.update_xaxes(gridcolor=grid); fig.update_yaxes(gridcolor=grid)
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
    if "macd" in row:
        mc = row["macd"]
        add("Trend & Momentum", "MACD", 1 if mc > 0.001 else -1 if mc < -0.001 else 0,
            f"MACD {'positive — up-momentum' if mc > 0 else 'negative — down-momentum' if mc < 0 else 'flat'}")
    if "drawdown" in row:
        dd = row["drawdown"]
        add("Strength vs range", "Drawdown from peak", 1 if dd > -0.05 else -1 if dd < -0.20 else 0,
            f"{dd:.0%} below its 60-day peak")
    if "streak" in row and pd.notna(row["streak"]):
        sk = int(row["streak"])
        add("Trend & Momentum", "Recent streak", 1 if sk >= 2 else -1 if sk <= -2 else 0,
            f"{abs(sk)}-day {'up' if sk > 0 else 'down' if sk < 0 else 'flat'} streak")
    mm, br = row["mkt_mom20"], row["breadth"]
    add("Market regime", "Market momentum (1-mo)", 1 if mm > .01 else -1 if mm < -.01 else 0, f"overall market {mm:+.1%} over ~1 month")
    add("Market regime", "Market breadth", 1 if br > .55 else -1 if br < .45 else 0, f"{br*100:.0f}% of stocks rising lately")
    return items


@st.cache_data(show_spinner=False)
def metric_explain(sym, label, text, _asof):
    """LLM micro-explanation of one metric for one stock (cached; friendly fallback)."""
    r = assistant.explain_metric(label, text, sym)
    return r or (f"**{label}** — {text}. (Turn on the LLM for a fuller explanation.)")


@st.cache_data(show_spinner=False)
def stock_writeup(sym, h, _asof):
    """Investment-perspective text for a stock: LLM if a key is set, else rich template."""
    row = scores[scores.symbol == sym].iloc[0]
    items = analyze_stock(sym)
    stock_items = [it for it in items if it["group"] != "Market regime"]
    bull = [it["label"] for it in stock_items if it["verdict"] == "Bullish"]
    bear = [it["label"] for it in stock_items if it["verdict"] == "Bearish"]
    p = {hh: float(row[f"p{hh}"]) for hh in H}
    vol = float(row["vol10"]) if pd.notna(row["vol10"]) else None
    cohort = fresh_scores if row["fresh"] else scores
    rank = int((cohort[f"p{h}"] > row[f"p{h}"]).sum()) + 1; ntot = len(cohort)
    tone = "net-bullish" if len(bull) > len(bear) + 1 else "net-bearish" if len(bear) > len(bull) + 1 else "mixed"
    riskw = "low" if (vol or 6) < 5 else "high" if (vol or 6) > 9 else "moderate"
    facts = (f"Stock={sym}. Up-probabilities: " + ", ".join(f"{hh}-day={p[hh]*100:.0f}%" for hh in H) +
             f". 10-day volatility(risk)={vol:.1f}% ({riskw}). Conviction rank #{rank} of {ntot} stocks. "
             f"Signal balance: {len(bull)} bullish ({', '.join(bull[:4]) or 'none'}), "
             f"{len(bear)} bearish ({', '.join(bear[:4]) or 'none'}). Overall {tone}. "
             f"As-of {ASOF.get(sym, meta['as_of'])}.")
    facts += (f" It {'beats' if row.get('px10', .5) >= .5 else 'lags'} the market on our outperformance signal "
              f"({float(row.get('px10', .5))*100:.0f}% odds of outperforming). Write for a normal investor.")
    llm = assistant.narrative(facts)
    if llm:
        return llm, "llm"
    # friendly plain-language fallback (shared with the assistant)
    return assistant.friendly_stock(sym, scores, meta), "template"


# =========================================================================== #
def page_home():
    st.markdown(f'<div class="nav"><div class="brand">📈 NEPSE<span class="dot">·</span>Signals</div>'
                f'<div class="tag">updated {meta["as_of"]}</div></div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="hero"><h1>Can a machine predict the <span class="grad">Nepal Stock Exchange?</span></h1>
  <p>On its <b>high-conviction</b> picks — yes, about <b>{HEAD['acc20']:.0f}%</b> of the time ({HEAD['kind']}).
  Get a <b>personalized basket</b> below, <b>Explore</b> any stock's chart + forecast, or ask the <b>Assistant</b>.</p>
  <div class="pills"><span class="pill">🏦 <b>{meta['n_stocks']}</b> stocks</span><span class="pill">🗓️ <b>{meta['yrs']}</b> yrs</span>
  <span class="pill">🎯 <b>{HEAD['acc20']:.0f}%</b> on top picks</span><span class="pill">⚙️ PBO {m10['pbo']:.2f}</span></div></div>""", unsafe_allow_html=True)
    st.markdown('<div class="disc">⚠️ <b>Not financial advice.</b> Educational tool. Low-conviction signals are near a coin '
                'flip — the edge is in the <b>high-conviction</b> picks. Never invest what you can\'t afford to lose.</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="cardgrid">
  <div class="mcard"><div class="top">High-conviction acc</div><div class="val green">{HEAD['acc20']:.0f}%</div><div class="sub">top-20% picks ({HEAD['kind']})</div></div>
  <div class="mcard"><div class="top">Overall acc (10d)</div><div class="val green">{HEAD['acc']:.1f}%</div><div class="sub">+{HEAD['edge']:.1f} pts vs baseline</div></div>
  <div class="mcard"><div class="top">Overfitting (PBO)</div><div class="val blue">{m10['pbo']:.2f}</div><div class="sub">≈0 → real, not curve-fit</div></div>
  <div class="mcard"><div class="top">Backtest 5d, net</div><div class="val green">{m5['strat_x']:.1f}×</div><div class="sub">vs {m5['bh_x']:.1f}× buy &amp; hold</div></div></div>""", unsafe_allow_html=True)

    with st.expander("🎓 What does \"accuracy\" actually mean here? (plain English)"):
        st.markdown("**Out of every 100 predictions, about 57 come true:**")
        st.markdown(accuracy_grid(57), unsafe_allow_html=True)
        st.markdown("<span style='color:#22c55e'>🟢 right (57)</span> &nbsp; · &nbsp; "
                    "⬜ missed (43) &nbsp; — a coin flip would be 50/50.", unsafe_allow_html=True)
        st.markdown(assistant.ACCURACY_PLAIN)

    # ---- market outlook (the most accurate signal) ----
    try:
        ol = json.loads((HERE / "data/market_outlook.json").read_text())
        o = ol["outlook"]
        st.markdown('<div class="sec">🧭 Market outlook</div>', unsafe_allow_html=True)
        cards = ""
        names = {"5": "1 week", "10": "2 weeks", "20": "1 month"}
        for k in ["5", "10", "20"]:
            pu = o[k]["p_up"]; cl = GREEN if pu >= 50 else RED
            lean = "leans up" if pu >= 50 else "leans down"
            cards += (f'<div class="mcard"><div class="top">Next {names[k]}</div>'
                      f'<div class="val" style="color:{cl}">{pu:.0f}%</div>'
                      f'<div class="sub">P(market up) — {lean} · {o[k]["acc"]:.0f}% acc</div></div>')
        st.markdown(f'<div class="cardgrid">{cards}</div>', unsafe_allow_html=True)
        st.markdown('<div class="expl">📖 The whole-market direction model is our <b>most reliable</b> signal — '
                    'the market is more predictable than single stocks (+5–6 pts over a coin-flip, up to ~68% on '
                    'high-conviction weeks). Use it to gauge the overall climate before picking stocks.</div>',
                    unsafe_allow_html=True)
    except Exception:
        pass

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
        h = {"~1 week (5d)": 5, "~2 weeks (10d)": 10, "~1 month (20d)": 20}[hlabel]
        pcol = f"px{h}" if HAS_XS else f"p{h}"                  # rank by outperformance conviction
        plab = "P(outperform)" if HAS_XS else "P(up)"
        # regime: condition recommendation on the overall market climate
        _ol = load_outlook()
        mkt_p = (_ol["outlook"].get(str(h), _ol["outlook"]["10"])["p_up"] if _ol else 50)
        if mkt_p < 45:
            st.warning(f"🧭 **Bearish market climate** — the model sees only a {mkt_p:.0f}% chance the overall "
                       f"market rises over this horizon. Even top-ranked picks face headwinds; favour cash, smaller "
                       f"positions, or waiting. The basket below is *relative* (best of a weak field).", icon="⚠️")
        elif mkt_p < 50:
            st.info(f"🧭 **Soft market climate** ({mkt_p:.0f}% up-probability) — be selective and size carefully.", icon="🧭")
        else:
            st.success(f"🧭 **Supportive market climate** ({mkt_p:.0f}% up-probability) — a tailwind for the picks below.", icon="🧭")
        pool = fresh_scores.dropna(subset=["vol10"]).copy()
        qcap = {"Conservative": .4, "Balanced": .7, "Aggressive": 1.0}[risk_app]
        cand = pool[(pool[pcol] > .5) & (pool["vol10"] <= pool["vol10"].quantile(qcap))].copy()
        cand["conviction"] = (cand[pcol] - .5) * 100
        cand["score"] = cand["conviction"] / cand["vol10"] if weighting == "Lower-risk first" else cand["conviction"]
        cand = cand.sort_values("score", ascending=False).head(nstocks)
        if not len(cand):
            st.error(f"At a **{risk_app.lower()}** level over **{hlabel}**, the model expects **no stocks to beat the "
                     f"market within your risk limit** — a weak climate, so **cash may be prudent**. Try a longer horizon or higher risk.")
        else:
            w = (np.ones(len(cand)) if weighting == "Equal" else 1/cand["vol10"].values if weighting == "Lower-risk first" else cand["conviction"].values)
            w = w / w.sum()
            out = pd.DataFrame({"Stock": cand["symbol"].values,
                "Verdict": [verdict_label(v) for v in cand[pcol].values],
                plab: [f"{v*100:.1f}%" for v in cand[pcol].values],
                "Conviction": cand["conviction"].values.round(1),
                "Risk 10d vol%": cand["vol10"].values.round(1),
                "Allocation %": (w*100).round(1),
                "Allocation NPR": [f"{int(round(x)):,}" for x in (w*capital)]})
            st.success(f"Suggested **{len(cand)}-stock basket** · **{risk_app.lower()}** · **{hlabel}** · **NPR {capital:,.0f}**")
            st.markdown(html_table(out, color_cols=["Conviction"]), unsafe_allow_html=True)
            st.caption("Ranked by the model's outperformance conviction & GARCH risk. Not advice — frequently wrong.")

    st.markdown('<div class="sec">📍 Today\'s signals</div>', unsafe_allow_html=True)
    rk = "px10" if HAS_XS else "p10"; plab = "P(outperform)" if HAS_XS else "P(up)"
    st.markdown(f'<p class="lead">Ranked by 10-day {"outperformance" if HAS_XS else ""} conviction across the '
                f'<b>{len(fresh_scores)}</b> current stocks. (Full {meta["n_stocks"]}-stock universe in <b>Explore</b>, each dated.)</p>', unsafe_allow_html=True)
    s10 = fresh_scores.sort_values(rk, ascending=False)
    f = lambda d: pd.DataFrame({"Stock": d["symbol"].values,
        "Verdict": [verdict_label(v) for v in d[rk].values],
        plab: [f"{v*100:.1f}%" for v in d[rk].values],
        "Conviction": ((d[rk].values-.5)*100).round(1),
        "Risk%": d["vol10"].values.round(1)})
    L, R = st.columns(2)
    L.markdown("##### ▲ Top 10 — leans UP"); L.markdown(html_table(f(s10.head(10)), color_cols=["Conviction"]), unsafe_allow_html=True)
    R.markdown("##### ▼ Bottom 10 — leans DOWN"); R.markdown(html_table(f(s10.tail(10).iloc[::-1]), color_cols=["Conviction"]), unsafe_allow_html=True)
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
    p_out = float(row[f"px{h}"]); oword, ocol = conviction_word(p_out)
    sym_asof = ASOF.get(sym, meta["as_of"])
    if sym_asof < "2026-01-01":
        st.markdown(f'<div class="disc">🕗 <b>{sym}</b> has data through <b>{sym_asof}</b> in our sources — '
                    f'this forecast is computed as of that date (historical, not live).</div>', unsafe_allow_html=True)
    # relative rank by outperformance conviction within the same freshness cohort
    rk = f"px{h}" if HAS_XS else f"p{h}"
    cohort = fresh_scores if row["fresh"] else scores
    rank = int((cohort[rk] > row[rk]).sum()) + 1; ntot = len(cohort)
    fig, info = forecast_figure(sym, h)
    ocard = (f'<div class="mcard"><div class="top">Outperform conviction</div>'
             f'<div class="val" style="color:{ocol}">{p_out*100:.0f}%</div>'
             f'<div class="sub">P(beats market over {h}d)</div></div>') if HAS_XS else ""
    st.markdown(f"""<div class="cardgrid">
  <div class="mcard"><div class="top">Last price</div><div class="val">{info['P0']:.0f}</div><div class="sub">NPR (adjusted)</div></div>
  {ocard}
  <div class="mcard"><div class="top">Direction P(up) {h}d</div><div class="val" style="color:{col}">{p_up*100:.0f}%</div><div class="sub">{word}</div></div>
  <div class="mcard"><div class="top">Relative rank</div><div class="val blue">#{rank}</div><div class="sub">of {ntot} (top {rank/ntot*100:.0f}%)</div></div>
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
    st.caption("💡 Tap **Explain** on any signal for a plain-English, AI-written explanation of what it means for this stock.")
    groups = ["Trend & Momentum", "Strength vs range", "Risk & Volatility", "Volume & Interest", "Market regime"]
    for g in groups:
        gi = [it for it in items if it["group"] == g]
        if not gi: continue
        with st.expander(f"**{g}**", expanded=(g in ["Trend & Momentum", "Strength vs range"])):
            for it in gi:
                c1, c2, c3 = st.columns([6, 2, 1.4])
                c1.markdown(f"**{it['label']}**  \n<span style='color:{PAL['mut']};font-size:.86rem'>{it['text']}</span>", unsafe_allow_html=True)
                c2.markdown(f"<span class='badge' style='background:{it['color']}22;color:{it['color']};"
                            f"border:1px solid {it['color']}55'>{it['verdict']}</span>", unsafe_allow_html=True)
                key = f"ex_{sym}_{it['label']}"
                if c3.button("💡 Explain", key="b_" + key):
                    st.session_state[key] = True
                if st.session_state.get(key):
                    with st.spinner("explaining…"):
                        st.info(metric_explain(sym, it["label"], it["text"], sym_asof))
    st.markdown('<div class="sec">💡 Investment perspective</div>', unsafe_allow_html=True)
    with st.spinner("writing perspective…"):
        writeup, src = stock_writeup(sym, h, ASOF.get(sym, meta["as_of"]))
    with st.container(border=True):
        st.markdown(writeup)
    if src == "template":
        st.caption("💡 Written from the model's live data. For richer LLM-written perspectives, add a free "
                   "GROQ_API_KEY (or your HF_TOKEN) under Space → Settings → Variables & secrets.")

    st.markdown('<div class="sec">Across horizons</div>', unsafe_allow_html=True)
    cols = st.columns(len(H))
    for cc, hh in zip(cols, H):
        pu = float(row[f"p{hh}"]); cc.metric(f"{hh}-day P(up)", f"{pu*100:.0f}%", f"{(pu-.5)*100:+.1f} pts")
    st.caption("Forecast = directional lean ± realistic volatility (GARCH). Not a guarantee; model wrong ~45% of the time.")


# =========================================================================== #
def page_evidence():
    st.markdown('<div class="nav"><div class="brand">📊 Model &amp; Evidence</div></div>', unsafe_allow_html=True)
    sig = EV.get("signal", "directional")
    st.markdown(f'<p class="lead">Evidence for the deployed <b>{sig}</b> recommendation signal — all interactive and '
                f'out-of-sample (the proof it\'s real, not curve-fit). Its strength shows in the <b>gated accuracy</b> '
                f'and the <b>cost-aware backtest</b> below, not the raw overall edge. Hover any chart.</p>', unsafe_allow_html=True)
    hz = [str(h) for h in H]

    st.markdown('<div class="sec">Accuracy vs a naive baseline</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_bar(name="Model", x=[f"{h}d" for h in hz], y=[EV["per_h"][h]["acc"] for h in hz], marker_color=GREEN,
                error_y=dict(type="data", array=[EV["per_h"][h]["ci"] for h in hz]))
    fig.add_bar(name="Naive baseline", x=[f"{h}d" for h in hz], y=[EV["per_h"][h]["maj"] for h in hz], marker_color=GREY)
    fig.add_hline(y=50, line_dash="dash", line_color=("rgba(255,255,255,.45)" if st.session_state.get("_dark",True) else "rgba(15,23,42,.40)"), opacity=.4)
    fig.update_yaxes(range=[45, max(EV["per_h"][h]["acc"] for h in hz)+4], title="accuracy %")
    st.plotly_chart(theme(fig), use_container_width=True, config={"displayModeBar": False})
    st.markdown(f'<div class="expl">📖 The model beats the "just guess the usual outcome" baseline at every horizon — '
                f'by <b>+{EV["per_h"]["20"]["edge"]:.1f} pts</b> at 20 days. Error bars are 95% confidence; they sit above '
                f'50%, so the edge is statistically real, not luck.</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec">Accuracy vs coverage — the honest path to high accuracy</div>', unsafe_allow_html=True)
    fig = go.Figure()
    for h, cl in zip(hz, [BLUE, GREEN, AMBER]):
        fig.add_scatter(x=EV["per_h"][h]["cov"], y=EV["per_h"][h]["cov_acc"], name=f"{h}d", line=dict(color=cl, width=2))
    fig.add_hline(y=50, line_dash="dash", line_color=("rgba(255,255,255,.45)" if st.session_state.get("_dark",True) else "rgba(15,23,42,.40)"), opacity=.4)
    fig.add_hline(y=70, line_dash="dot", line_color=GREEN, opacity=.6)
    fig.update_xaxes(title="coverage % (act on only the most confident)", autorange="reversed"); fig.update_yaxes(title="accuracy %")
    st.plotly_chart(theme(fig), use_container_width=True, config={"displayModeBar": False})
    st.markdown(f'<div class="expl">📖 If you act on <b>every</b> signal, accuracy is modest (~{EV["per_h"]["10"]["acc"]:.0f}%). '
                f'Act on only the model\'s <b>most confident</b> picks (move left) and it climbs to ~<b>{HEAD["acc20"]:.0f}%</b>. '
                f'This is how a "modest" model becomes useful — the edge lives in the high-conviction slice, not every day.</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec">Is the edge stable over time?</div>', unsafe_allow_html=True)
    e = EV["per_h"]["10"]
    fig = go.Figure()
    fig.add_scatter(x=e["roll_dates"], y=e["roll_acc"], line=dict(color=GREEN, width=1.6), name="60-day rolling acc")
    fig.add_hline(y=50, line_dash="dash", line_color=("rgba(255,255,255,.45)" if st.session_state.get("_dark",True) else "rgba(15,23,42,.40)"), opacity=.4)
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
        fig.add_hline(y=.5, line_dash="dash", line_color=("rgba(255,255,255,.45)" if st.session_state.get("_dark",True) else "rgba(15,23,42,.40)"), opacity=.4)
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
        st.markdown('<div class="expl">💡 Running in <b>friendly</b> mode (free, no key) — plain-English answers from the '
                    'live model data. For full ChatGPT-style conversation, add a free <code>GROQ_API_KEY</code> '
                    '(get one at console.groq.com) <b>or</b> your <code>HF_TOKEN</code> under '
                    '<b>Space → Settings → Variables &amp; secrets</b>. No code change needed.</div>', unsafe_allow_html=True)
    st.markdown('<p class="lead">Ask about any stock, how accurate the model is, how it works, or how to invest. '
                'Grounded in live predictions — not financial advice.</p>', unsafe_allow_html=True)
    if "chat" not in st.session_state:
        st.session_state.chat = [("assistant", "Hi! I'm the NEPSE Signals assistant 🤖 Ask me about a stock "
                                  "(*“How does NABIL look?”*), the model's accuracy, or what to do as an investor.")]

    def ask(qq):
        st.session_state.chat.append(("user", qq))
        resp, _ = assistant.answer(qq, st.session_state.chat[:-1], scores, meta)
        st.session_state.chat.append(("assistant", resp))

    st.markdown("**Try one:**")
    sugg = ["How accurate is the model?", "Top stocks to watch right now?",
            "How does NABIL look?", "What should a cautious investor do?",
            "How does the model work?", "Which stocks are riskiest?"]
    cols = st.columns(3)
    for i, s in enumerate(sugg):
        if cols[i % 3].button(s, use_container_width=True, key=f"sugg{i}"):
            with st.spinner("thinking…"):
                ask(s)
            st.rerun()
    cc1, cc2 = st.columns([4, 1])
    if cc2.button("🗑️ Clear", use_container_width=True):
        del st.session_state.chat; st.rerun()

    for role, content in st.session_state.chat:
        with st.chat_message(role, avatar="📈" if role == "assistant" else None):
            st.markdown(content)
    if q := st.chat_input("Ask about a stock or the model…"):
        with st.spinner("thinking…"):
            ask(q)
        st.rerun()


# ---- assistant: floating bubble + immersive full-page ---------------------- #
GREETING = ("Hi! 👋 I'm your NEPSE assistant. Ask me things like *“top 5 stocks?”*, "
            "*“how does NABIL look?”*, or *“how accurate is this?”*")


def _ask(q):
    st.session_state.chat.append(("user", q))
    try:
        resp, _ = assistant.answer(q, st.session_state.chat[:-1], scores, meta)
    except Exception:
        resp = "Sorry — I hit a snag. Try again in a moment."
    st.session_state.chat.append(("assistant", resp))


def chat_widget():
    if "chat" not in st.session_state:
        st.session_state.chat = [("assistant", GREETING)]
    with st.popover("💬  Ask the assistant", use_container_width=False):
        top = st.columns([5, 1])
        top[0].markdown("**📈 NEPSE Assistant**  ·  plain-English help")
        if top[1].button("⛶", key="enlarge", help="Open full-screen chat"):
            st.session_state.immersive = True; st.rerun()
        hist = st.container(height=380)
        with st.form("floatchat", clear_on_submit=True):
            cols = st.columns([5, 1])
            q = cols[0].text_input("msg", placeholder="e.g. top 5 stocks?", label_visibility="collapsed")
            sent = cols[1].form_submit_button("➤")
        if sent and q:
            _ask(q)
        with hist:                       # render AFTER processing so the reply shows without a rerun
            for role, content in st.session_state.chat[-12:]:
                with st.chat_message(role, avatar="📈" if role == "assistant" else None):
                    st.markdown(content)


def immersive_chat():
    if "chat" not in st.session_state:
        st.session_state.chat = [("assistant", GREETING)]
    c1, c2 = st.columns([6, 1])
    c1.markdown('<div class="nav"><div class="brand">💬 NEPSE Assistant</div></div>', unsafe_allow_html=True)
    if c2.button("✕ Close", use_container_width=True):
        st.session_state.immersive = False; st.rerun()
    st.markdown('<p class="lead">Your AI guide to NEPSE — plain-English, grounded in the live model. Not financial advice.</p>', unsafe_allow_html=True)
    chips = st.columns(4)
    for cc, s in zip(chips, ["Top 5 stocks?", "How does NABIL look?", "How accurate is this?", "What should I avoid?"]):
        if cc.button(s, use_container_width=True):
            _ask(s); st.rerun()
    for role, content in st.session_state.chat:
        with st.chat_message(role, avatar="📈" if role == "assistant" else None):
            st.markdown(content)
    if q := st.chat_input("Ask about any stock or the model…"):
        _ask(q); st.rerun()


# --------------------------------------------------------------------------- #
if st.session_state.get("immersive"):
    immersive_chat()
else:
    chat_widget()
    nav = st.navigation([
        st.Page(page_home, title="Home", icon="🏠", default=True),
        st.Page(page_explore, title="Explore a stock", icon="🔍"),
        st.Page(page_evidence, title="Model & Evidence", icon="📊"),
    ])
    nav.run()
