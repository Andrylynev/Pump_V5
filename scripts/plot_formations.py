"""Render candlestick charts annotated with the V5 detector's view of each
formation: channel bounds, spark/twix marks, accumulation window, VAH target,
and a verdict (KEPT / why-DROPPED). Saves PNGs to docs/charts/.
"""
import sys, time, requests, os
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
sys.path.insert(0, "/root/Pump_V5")
import yaml
from app.detectors.methodic_detector import MethodicDetector

CFG = yaml.safe_load(open("/root/Pump_V5/app/config/default.yaml"))
DET = MethodicDetector(detector_cfg=CFG["detector"])
S = requests.Session()
OUT = "/root/Pump_V5/docs/charts"
os.makedirs(OUT, exist_ok=True)
FRESH_DAYS = 75; MAX_BELOW = 0.15; AP = 0.08; BC = 3; NEAR = 0.03


def daily(sym, days=400):
    now = int(time.time()*1000); start = now - days*86400000
    j = S.get("https://api.bybit.com/v5/market/kline", params={"category":"linear","symbol":sym,
              "interval":"D","start":start,"end":now,"limit":1000}, timeout=15).json()
    rows = j["result"]["list"]
    df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume","turnover"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True).iloc[:-1].reset_index(drop=True)


def vah_from_vp(w):
    """Upper value-area boundary (VAH) of a fixed-range volume profile over the
    window (prior-high typical-price binning). Simple, self-contained."""
    price = (w["high"] + w["low"] + w["close"]) / 3.0
    vol = w["volume"].to_numpy()
    lo, hi = float(w["low"].min()), float(w["high"].max())
    if hi <= lo:
        return hi
    bins = 100
    edges = np.linspace(lo, hi, bins+1)
    idx = np.clip(np.digitize(price, edges) - 1, 0, bins-1)
    hist = np.zeros(bins)
    for k, vv in zip(idx, vol):
        hist[k] += vv
    total = hist.sum()
    if total <= 0:
        return hi
    poc = int(np.argmax(hist))
    inc = {poc}; covered = hist[poc]
    lo_i = hi_i = poc
    while covered < 0.70 * total and (lo_i > 0 or hi_i < bins-1):
        left = hist[lo_i-1] if lo_i > 0 else -1
        right = hist[hi_i+1] if hi_i < bins-1 else -1
        if right >= left:
            hi_i += 1; covered += hist[hi_i]
        else:
            lo_i -= 1; covered += hist[lo_i]
    vah = edges[hi_i+1]
    return float(vah)


def verdict(sym, df, f):
    last = float(df["close"].iloc[-1]); lts = df["timestamp"].iloc[-1]
    acc_end = pd.Timestamp(f.accumulation_end)
    if (lts - acc_end).days > FRESH_DAYS:
        return "DROP: устарела (>75д)"
    ub, lb = f.upper_bound, f.lower_bound
    post = df[df["timestamp"] > acc_end]; pc = post["close"].to_numpy(); ph = post["high"].to_numpy()
    if lb > 0 and len(pc) >= BC:
        below = pc < lb; tr = 0
        for k in range(len(below)-1, -1, -1):
            if below[k]: tr += 1
            else: break
        if tr >= BC: return "DROP: дно потеряно (3+ закр. ниже)"
    if lb > 0 and last < lb: return "DROP: цена ниже дна сейчас"
    if lb > 0 and last < lb*(1-MAX_BELOW): return "DROP: обвал ниже дна"
    if len(ph) and ub > 0 and float(ph.max()) > ub*(1+AP): return "DROP: памп уже прошёл"
    st = "broke_out" if last > ub else ("near_upper" if last >= ub*(1-NEAR) else "accumulating")
    return f"KEEP: {st}"


def plot(sym, note=""):
    df = daily(sym)
    forms = DET.detect_accumulations(sym, df)
    if not forms:
        print(f"{sym}: нет окна детектора — рисую только свечи"); 
        f = None
    else:
        f = forms[0]
    d = df.set_index("timestamp")[["open","high","low","close","volume"]].copy()
    d.index.name = "Date"
    # Zoom: show accumulation window + ~30% margin before and a tail after, so
    # the formation is legible instead of buried in 400d of history.
    if f is not None:
        a_s = pd.Timestamp(f.accumulation_start); a_e = pd.Timestamp(f.accumulation_end)
        span = a_e - a_s
        lo_t = a_s - 0.30 * span
        d = d[d.index >= lo_t]
    apds = []; vlines = {}; vahline = None; title = sym
    if f is not None:
        ub, lb = f.upper_bound, f.lower_bound
        a_s = pd.Timestamp(f.accumulation_start); a_e = pd.Timestamp(f.accumulation_end)
        w = df[(df["timestamp"] >= a_s) & (df["timestamp"] <= a_e)]
        vah = vah_from_vp(w)
        # channel lines across full chart
        upper = pd.Series(ub, index=d.index)
        lower = pd.Series(lb, index=d.index)
        vahs = pd.Series(vah, index=d.index)
        apds.append(mpf.make_addplot(upper, color="#d62728", width=1.2, linestyle="--"))
        apds.append(mpf.make_addplot(lower, color="#2ca02c", width=1.2, linestyle="--"))
        apds.append(mpf.make_addplot(vahs, color="#9467bd", width=1.0, linestyle=":"))
        # spark / twix markers within window — aligned to the ZOOMED index d.
        sp = np.full(len(d), np.nan); tw = np.full(len(d), np.nan)
        marks = DET._precompute_marks(df)
        is_sp = marks["is_spark"]; is_tw = marks["is_twix"]
        ts_to_pos = {ts: p for p, ts in enumerate(df["timestamp"].values)}
        d_ts = d.index.values
        for p in range(len(d)):
            i = ts_to_pos.get(d_ts[p])
            if i is None:
                continue
            ts = df["timestamp"].iloc[i]
            if ts < a_s or ts > a_e:
                continue
            if is_sp[i]:
                sp[p] = df["low"].iloc[i] * 0.97
            elif is_tw[i]:
                tw[p] = df["low"].iloc[i] * 0.97
        if np.any(~np.isnan(sp)):
            apds.append(mpf.make_addplot(sp, type="scatter", marker="^", markersize=70, color="#1f77b4"))
        if np.any(~np.isnan(tw)):
            apds.append(mpf.make_addplot(tw, type="scatter", marker="o", markersize=45, color="#ff7f0e"))
        v = verdict(sym, df, f)
        title = (f"{sym}  [{v}]  {note}\n"
                 f"канал {a_s.date()}→{a_e.date()} ({f.diagnostics['trend']}, ширина {f.channel_width:.2f}, "
                 f"баллы {f.score})  низ={lb:.4g} верх={ub:.4g} VAH={vah:.4g}")
        vlines = dict(vlines=[a_s.to_pydatetime(), a_e.to_pydatetime()], colors="#888", linestyle="-", linewidths=0.8)
    fname = f"{OUT}/{sym}.png"
    kwargs = dict(type="candle", style="yahoo", volume=True, addplot=apds, figratio=(16,9),
                  figscale=1.3, title=title, savefig=dict(fname=fname, dpi=110, bbox_inches="tight"),
                  tight_layout=True, datetime_format="%m-%d")
    if f is not None:
        # shade the accumulation window between the channel bounds
        yb = np.where((d.index >= a_s) & (d.index <= a_e), lb, np.nan)
        yt = np.where((d.index >= a_s) & (d.index <= a_e), ub, np.nan)
        kwargs["fill_between"] = dict(y1=list(yb), y2=list(yt), alpha=0.08, color="#1f77b4")
    if vlines:
        kwargs["vlines"] = vlines
    mpf.plot(d, **kwargs)
    print(f"saved {fname}")


if __name__ == "__main__":
    targets = [
        ("RLUSDUSDT", "✓ живой тугой канал — приоритет"),
        ("CCUSDT", "✓ нисходящий→боковик"),
        ("ETHBTCUSDT", "✓ нисходящее накопление"),
        ("POLUSDT", "✓ нисходящее накопление"),
        ("ASTERUSDT", "✓ нисходящий→боковик"),
        ("LTCUSDT", "✗ слом вниз (Никита)"),
        ("GLMUSDT", "✗ слом вниз (Никита)"),
        ("ATOMUSDT", "✗ памп/откат (Никита)"),
        ("ASPUSDT", "? в ручном, не в коде"),
        ("KASUSDT", "? в ручном, не в коде"),
        ("QNTUSDT", "? в ручном, не в коде"),
    ]
    for sym, note in targets:
        try:
            plot(sym, note)
        except Exception as e:
            print(f"{sym}: ERR {e}")
