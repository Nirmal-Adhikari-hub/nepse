"""
Feature engineering + multi-horizon labels for the NEPSE panel.

Design rules (anti-leakage):
  * every FEATURE at bar t uses only information available at close t.
  * every LABEL is a FORWARD quantity (t -> t+h) derived from adj_close, so
    bonus/rights are already handled (adj_close is continuous).
  * cross-sectional ranks at date t use only that date's cross-section.

Label families (h in {5,10,20} trading days):
  y_dir_h   : 1 if forward return > 0 else 0              (direction)
  y_dz_h    : 1 / 0 for moves beyond +/-band, else NaN    (dead-zone: skip noise)
  y_xs_h    : 1 if stock beats the market over h else 0   (relative / ranking)
  fwd_ret_h : raw forward return (for backtest PnL)
"""
import numpy as np
import pandas as pd

from nepse_data import load_panel

HORIZONS = [5, 10, 20]
DZ_BAND = {5: 0.03, 10: 0.05, 20: 0.07}     # dead-zone half-width per horizon


def _rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / (dn + 1e-9))


def _per_stock_features(g):
    """Backward-looking technical features for a single stock (sorted by date)."""
    c = g["adj_close"]
    r = g["adj_return"]
    f = pd.DataFrame(index=g.index)
    # momentum / returns
    for k in [1, 2, 3, 5, 10, 20, 60]:
        f[f"ret{k}"] = c.pct_change(k)
    # moving-average ratios (trend)
    for w in [5, 10, 20, 60]:
        f[f"mar{w}"] = c / c.rolling(w).mean() - 1
    # realised volatility
    for w in [10, 20, 60]:
        f[f"vol{w}"] = r.rolling(w).std()
    # RSI + return distribution shape
    f["rsi14"] = _rsi(c) / 100
    f["skew20"] = r.rolling(20).skew()
    f["kurt20"] = r.rolling(20).kurt()
    # distance from 52-week (252d) high / low
    f["from_hi"] = c / c.rolling(252, min_periods=60).max() - 1
    f["from_lo"] = c / c.rolling(252, min_periods=60).min() - 1
    # liquidity / volume
    lv = np.log1p(g["volume"])
    f["volz"] = (lv - lv.rolling(20).mean()) / (lv.rolling(20).std() + 1e-9)
    lt = np.log1p(g["turnover"])
    f["logturn"] = lt
    f["turnz"] = (lt - lt.rolling(20).mean()) / (lt.rolling(20).std() + 1e-9)
    # intraday range
    f["range"] = (g["high"] - g["low"]) / c
    f["range10"] = f["range"].rolling(10).mean()
    return f


def build(panel=None, min_rows=400):
    if panel is None:
        panel, _ = load_panel(min_rows=min_rows)
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)

    # ---- per-stock features ----
    feat = panel.groupby("symbol", group_keys=False).apply(_per_stock_features)
    df = pd.concat([panel[["symbol", "date", "adj_close", "adj_return"]], feat], axis=1)

    # ---- market aggregate (equal-weight across the cross-section each date) ----
    mkt = df.groupby("date")["adj_return"].agg(
        mkt_ret="mean", breadth=lambda x: (x > 0).mean(), mkt_vol="std").reset_index()
    mkt["mkt_mom5"]  = mkt["mkt_ret"].rolling(5).sum().values
    mkt["mkt_mom20"] = mkt["mkt_ret"].rolling(20).sum().values
    df = df.merge(mkt, on="date", how="left")

    # ---- cross-sectional ranks within each date (relative strength) ----
    for col in ["ret5", "ret20", "vol20", "volz", "mar20"]:
        df[f"xs_{col}"] = df.groupby("date")[col].rank(pct=True)

    # ---- forward labels ----
    g = df.groupby("symbol", group_keys=False)
    for h in HORIZONS:
        fwd = g["adj_close"].apply(lambda s: s.shift(-h) / s - 1)
        df[f"fwd_ret_{h}"] = fwd.values
        df[f"y_dir_{h}"] = (df[f"fwd_ret_{h}"] > 0).astype(float)
        band = DZ_BAND[h]
        dz = np.where(df[f"fwd_ret_{h}"] > band, 1.0,
                      np.where(df[f"fwd_ret_{h}"] < -band, 0.0, np.nan))
        df[f"y_dz_{h}"] = dz
        # market forward return for the same window -> relative (beat market) label
        mfwd = df.groupby("date")[f"fwd_ret_{h}"].transform("mean")
        df[f"y_xs_{h}"] = (df[f"fwd_ret_{h}"] > mfwd).astype(float)
        # where forward return is NaN (end of series), labels must be NaN
        df.loc[df[f"fwd_ret_{h}"].isna(), [f"y_dir_{h}", f"y_xs_{h}"]] = np.nan

    return df


FEATURES = [
    "ret1", "ret2", "ret3", "ret5", "ret10", "ret20", "ret60",
    "mar5", "mar10", "mar20", "mar60",
    "vol10", "vol20", "vol60", "rsi14", "skew20", "kurt20",
    "from_hi", "from_lo", "volz", "logturn", "turnz", "range", "range10",
    "mkt_ret", "breadth", "mkt_vol", "mkt_mom5", "mkt_mom20",
    "xs_ret5", "xs_ret20", "xs_vol20", "xs_volz", "xs_mar20",
]

if __name__ == "__main__":
    df = build()
    print(f"Feature matrix: {df.shape[0]:,} rows x {len(FEATURES)} features, "
          f"{df.symbol.nunique()} stocks")
    print(f"Date range: {df.date.min().date()} -> {df.date.max().date()}")
    # label balance
    for h in HORIZONS:
        d = df.dropna(subset=[f"y_dir_{h}"])
        dz = df.dropna(subset=[f"y_dz_{h}"])
        print(f"h={h:2d}: dir up-rate={d[f'y_dir_{h}'].mean():.3f} | "
              f"dead-zone rows={len(dz):,} (up-rate {dz[f'y_dz_{h}'].mean():.3f}) | "
              f"usable rows={len(d):,}")
    df.to_parquet("nepse_features.parquet")
    print("saved nepse_features.parquet")
