"""
NEPSE data layer + bonus/rights ADJUSTMENT.

Source: Aabishkar2/nepse-data mirror in ./nepse_data/company-wise/<SYM>.csv
columns: published_date, open, high, low, close, per_change, traded_quantity,
         traded_amount, status

WHY adjustment matters: NEPSE firms issue bonus/rights shares constantly. On the
ex-date the raw price drops sharply (e.g. a 1:1 bonus ~halves it) - that is an
ACCOUNTING event, not a -50% return. Training on raw prices makes every such
event look like a crash and poisons direction labels. We back-adjust.

Detection is reliable here: NEPSE has a daily circuit limit (<=10%), so any
overnight move beyond ~-15% cannot be ordinary trading -> it is a corporate
action (or a data error, which we clean first).
"""
import glob
import os
import numpy as np
import pandas as pd

DATA_DIR = "nepse_data/company-wise"
CORP_ACTION_THRESH = -0.15      # overnight return below this = corp action (beyond circuit)
MIN_PRICE = 1.0                 # below this = data error / not yet listed
MIN_ROWS = 400                  # need enough history to be useful


def _raw(sym):
    d = pd.read_csv(f"{DATA_DIR}/{sym}.csv")
    d = d.rename(columns={"published_date": "date", "traded_quantity": "volume",
                          "traded_amount": "turnover"})
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume", "turnover"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    # data errors: zero / sub-rupee closes -> treat as missing, forward fill
    d.loc[d["close"] < MIN_PRICE, ["open", "high", "low", "close"]] = np.nan
    d[["open", "high", "low", "close"]] = d[["open", "high", "low", "close"]].ffill()
    d = d.dropna(subset=["close"]).reset_index(drop=True)
    return d


def adjust(sym, thresh=CORP_ACTION_THRESH, return_events=False):
    """Return one stock's OHLCV with a back-adjusted 'adj_close' column.

    Back-adjustment: walking newest->oldest, when an ex-date gap is detected,
    every PRIOR price is scaled by the gap ratio so the adjusted series is
    continuous (no artificial cliff). Volume is left raw.
    """
    d = _raw(sym)
    c = d["close"].values
    overnight = np.empty(len(c)); overnight[0] = 0.0
    overnight[1:] = c[1:] / c[:-1] - 1.0
    event_idx = np.where(overnight < thresh)[0]          # index of the ex-date bar

    factor = np.ones(len(c))                              # multiplicative back-adjust factor
    events = []
    for i in event_idx:
        ratio = c[i] / c[i - 1]                            # <1 ; the artificial drop
        factor[:i] *= ratio                                # scale all prior bars down
        events.append(dict(sym=sym, date=d["date"].iloc[i], drop_pct=round(overnight[i] * 100, 1),
                           ratio=round(ratio, 3)))
    d["adj_close"] = c * factor
    d["adj_return"] = d["adj_close"].pct_change()
    # neutralise the residual gap on ex-dates themselves (set to NaN -> not a label)
    for i in event_idx:
        d.loc[i, "adj_return"] = np.nan
    if return_events:
        return d, pd.DataFrame(events)
    return d


def list_symbols(min_rows=MIN_ROWS):
    syms = []
    for f in sorted(glob.glob(f"{DATA_DIR}/*.csv")):
        sym = os.path.basename(f)[:-4]
        try:
            n = sum(1 for _ in open(f)) - 1
            if n >= min_rows:
                syms.append(sym)
        except Exception:
            pass
    return syms


def load_panel(symbols=None, min_rows=MIN_ROWS, last_date_min="2026-01-01"):
    """Load all (adjusted) stocks into one long/tidy DataFrame.

    Filters out delisted names (last trade before last_date_min) so the panel is
    a live, tradeable universe.
    """
    if symbols is None:
        symbols = list_symbols(min_rows)
    frames = []
    dropped = []
    for sym in symbols:
        try:
            d = adjust(sym)
            if len(d) < min_rows:
                dropped.append((sym, "short")); continue
            if d["date"].iloc[-1] < pd.Timestamp(last_date_min):
                dropped.append((sym, "delisted")); continue
            d["symbol"] = sym
            frames.append(d)
        except Exception as e:
            dropped.append((sym, str(e)[:30]))
    panel = pd.concat(frames, ignore_index=True)
    return panel, dropped


if __name__ == "__main__":
    panel, dropped = load_panel()
    print(f"Loaded {panel.symbol.nunique()} stocks, {len(panel):,} rows")
    print(f"Date range: {panel.date.min().date()} -> {panel.date.max().date()}")
    print(f"Dropped: {len(dropped)} "
          f"(delisted={sum(1 for _,r in dropped if r=='delisted')}, "
          f"short={sum(1 for _,r in dropped if r=='short')})")
    # total corp-action events adjusted
    tot = 0
    for s in panel.symbol.unique():
        _, ev = adjust(s, return_events=True)
        tot += len(ev)
    print(f"Corporate-action events back-adjusted: {tot}")
