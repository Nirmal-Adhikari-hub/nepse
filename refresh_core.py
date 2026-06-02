"""
Daily data refresh (runs in CI, or locally).

Pulls the latest NEPSE end-of-day data from the public mirror, back-adjusts for
bonus/rights, rebuilds features, and regenerates the small app bundles:
    data/latest_features.parquet  + data/risk.parquet  + bumps metrics.json as_of

It does NOT retrain the model — it re-scores the latest bar with whatever
boosters are in models/. So signals stay current automatically; model quality
changes only when you swap models/lgbm_h*.txt.

Usage:  python refresh_core.py
"""
import warnings; warnings.filterwarnings("ignore")
import os, json, glob, shutil, urllib.request
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MIRROR_API = "https://api.github.com/repos/Aabishkar2/nepse-data/git/trees/main?recursive=1"
MIRROR_RAW = "https://raw.githubusercontent.com/Aabishkar2/nepse-data/main/data/company-wise"
DATA_DIR = os.path.join(HERE, "nepse_data", "company-wise")


def fetch_mirror():
    os.makedirs(DATA_DIR, exist_ok=True)
    tree = json.loads(urllib.request.urlopen(MIRROR_API, timeout=60).read())
    syms = sorted({p["path"].split("/")[-1][:-4] for p in tree["tree"]
                   if p["path"].startswith("data/company-wise/") and p["path"].endswith(".csv")})
    print(f"mirror has {len(syms)} symbols; downloading ...")
    for i, s in enumerate(syms, 1):
        try:
            urllib.request.urlretrieve(f"{MIRROR_RAW}/{s}.csv", f"{DATA_DIR}/{s}.csv")
        except Exception as e:
            print(f"  skip {s}: {e}")
        if i % 30 == 0:
            print(f"  {i}/{len(syms)}")
    # optional: drop any extra CSVs in seed/ into the universe (none by default)
    for f in glob.glob(os.path.join(HERE, "seed", "*.csv")):
        shutil.copy(f, DATA_DIR)
    print(f"data dir now has {len(os.listdir(DATA_DIR))} CSVs")


def main():
    fetch_mirror()
    # reuse the exact pipeline modules shipped in this repo
    import nepse_data, features, lightgbm as lgb
    from arch import arch_model
    nepse_data.DATA_DIR = DATA_DIR
    panel, dropped = nepse_data.load_panel()
    df = features.build(panel)
    meta = json.loads(open(os.path.join(HERE, "data/metrics.json")).read())
    FEAT = meta["features"]
    last = df["date"].max()

    latest = df[df["date"] == last].dropna(subset=FEAT)[["symbol"] + FEAT].copy()
    latest.to_parquet(os.path.join(HERE, "data/latest_features.parquet"))

    risk = []
    for sym, g in df.groupby("symbol"):
        r = g["adj_return"].dropna().values * 100
        if len(r) < 300:
            continue
        try:
            fc = arch_model(r[-1000:], vol="GARCH", p=1, q=1, dist="t").fit(disp="off")\
                .forecast(horizon=10, reindex=False)
            risk.append(dict(symbol=sym, vol10=round(float(np.sqrt(fc.variance.values[-1].sum())), 2)))
        except Exception:
            pass
    pd.DataFrame(risk).to_parquet(os.path.join(HERE, "data/risk.parquet"))

    meta["as_of"] = str(pd.Timestamp(last).date())
    meta["n_stocks"] = int(df.symbol.nunique())
    json.dump(meta, open(os.path.join(HERE, "data/metrics.json"), "w"), indent=2)
    print(f"refreshed: as_of={meta['as_of']} stocks={len(latest)} risk={len(risk)}")


if __name__ == "__main__":
    main()
