---
title: NEPSE Signals
emoji: 📈
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
license: mit
---

# 📈 NEPSE Signals

Honest machine-learning **direction** forecasts for the Nepal Stock Exchange — a global
LightGBM model trained across ~106 stocks and 23 years, validated fully out-of-sample.

**Not financial advice.** Educational research tool. ~55% directional accuracy means it is
wrong nearly half the time.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy for free (pick one)

### A) Hugging Face Spaces (recommended)
1. Create a free account at huggingface.co (no card).
2. New Space → SDK: **Streamlit** → name it.
3. Push this folder to the Space repo:
   ```bash
   git init && git add . && git commit -m "nepse signals"
   git remote add origin https://huggingface.co/spaces/<you>/<space>
   git push -u origin main
   ```
4. Live at `https://<you>-<space>.hf.space` in ~2 min.

### B) Streamlit Community Cloud
1. Push this folder to a public GitHub repo.
2. share.streamlit.io → "New app" → pick the repo → `app.py` → Deploy.
3. Live at `https://<app>.streamlit.app`.

Free tiers sleep when idle (~30s cold start on first visit). That's the only catch.

## Updating the model (the whole point)
The app **loads and scores** the boosters in `models/` on every visit — it does not hard-code
outputs. To ship a better model later:
1. Retrain offline, save `models/lgbm_h{5,10,20}.txt` (same feature order as `data/metrics.json`).
2. Replace the files, commit, push → users instantly get the new model's outputs. No code change.

## Keeping data fresh (optional, free)
`.github/workflows/refresh.yml` re-pulls the daily NEPSE mirror, regenerates
`data/latest_features.parquet` + `data/risk.parquet`, and commits — which triggers a redeploy.
Runs on GitHub's free Actions minutes.

## What's inside
```
app.py                       Streamlit UI (loads models, scores latest bar live)
models/lgbm_h{5,10,20}.txt   swappable LightGBM boosters
data/latest_features.parquet latest bar per stock (live-scored)
data/risk.parquet            GARCH 10-day volatility per stock
data/metrics.json            OOS credibility metrics + backtest + feature list
assets/*.png                 dashboards
```
