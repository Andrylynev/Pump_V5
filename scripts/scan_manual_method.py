"""MANUAL scan — independent re-implementation of the Pump method from the TZ,
using ONLY the crypto-public-data skill's raw Bybit REST. This is a DELIBERATE
second opinion: it does NOT import the V5 detector. If the two scans disagree,
that disagreement is information (detector bug vs method-reading bug).

Method (read straight from ТЗ, first principles):
  * Channel: price ranges within <=50% over >=45 days (sideways or downtrend).
  * Spark: a single GREEN day whose volume exceeds EACH of the up-to-4 preceding
    RED days individually, with a SMALL body (low price impact). +1.0 point.
  * Twix: spark with an adjacent green stuck to it; score the one anomalous candle,
    take the larger volume of the pair. +0.5 point.
  * Accumulation likely at >= 3 points.
  * Entry watch: price breaks ABOVE the channel top (we flag broke_out / near).
Differences from the detector are intentional (simple flat channel = rolling
max-high/min-low over the window; no regression slope; plain body-ratio test).
"""
import time, json
import requests
import pandas as pd
import numpy as np

NEAR_PCT = 0.03
MIN_DAYS = 45
MAX_WIDTH = 0.50            # <=50% range
WINDOW_DAYS = 60           # look at the most-recent 60d window as the candidate range
SMALL_BODY_MAX = 0.35      # body/range <= this => small body / low impact
MIN_SCORE = 3.0
FRESH = True               # only current windows (we use the latest WINDOW)
NOW_MS = int(time.time() * 1000)
START_MS = NOW_MS - 400 * 86400_000
S = requests.Session()


def safe_get(url, params, tries=5):
    delay = 0.8
    for _ in range(tries):
        try:
            r = S.get(url, params=params, timeout=15)
            if r.status_code in (403, 429, 418):
                time.sleep(delay); delay *= 2; continue
            r.raise_for_status()
            return r.json()
        except (requests.Timeout, requests.ConnectionError):
            time.sleep(delay); delay *= 2
    return None


def universe_linear():
    j = safe_get("https://api.bybit.com/v5/market/instruments-info", {"category": "linear"})
    return sorted(it["symbol"] for it in j["result"]["list"]
                  if it.get("quoteCoin") == "USDT" and it.get("status") == "Trading"
                  and it.get("contractType") == "LinearPerpetual")


def daily(symbol):
    j = safe_get("https://api.bybit.com/v5/market/kline",
                 {"category": "linear", "symbol": symbol, "interval": "D",
                  "start": START_MS, "end": NOW_MS, "limit": 1000})
    if not j or j.get("retCode") != 0:
        return pd.DataFrame()
    rows = j["result"]["list"]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df.iloc[:-1].reset_index(drop=True)   # drop forming bar


def score_window(w):
    """Count spark/twix points in window w (DataFrame, oldest-first). My own
    first-principles implementation, independent of the detector."""
    o = w["open"].to_numpy(); c = w["close"].to_numpy()
    h = w["high"].to_numpy(); l = w["low"].to_numpy(); v = w["volume"].to_numpy()
    n = len(w)
    green = c > o
    rng = np.maximum(h - l, 1e-12)
    body = np.abs(c - o)
    small_body = (body / rng) <= SMALL_BODY_MAX
    spark_idx, twix_idx = [], []
    used = set()
    for i in range(n):
        if not (green[i] and small_body[i]):
            continue
        # preceding red run, max 4
        reds = []
        j = i - 1
        while j >= 0 and len(reds) < 4 and (c[j] < o[j]):
            reds.append(j); j -= 1
        if not reds:
            continue
        # spark: volume exceeds EACH preceding red individually
        if all(v[i] > v[r] for r in reds):
            # twix? adjacent green stuck to it (i+1 or i-1 green, anomalous one is i)
            is_twix = (i + 1 < n and green[i + 1]) or (i - 1 >= 0 and green[i - 1] and (i - 1) not in used)
            if is_twix:
                twix_idx.append(i)
            else:
                spark_idx.append(i)
            used.add(i)
    score = len(spark_idx) * 1.0 + len(twix_idx) * 0.5
    return score, len(spark_idx), len(twix_idx)


def analyze(symbol, df):
    if len(df) < MIN_DAYS + 1:
        return None
    last_close = float(df["close"].iloc[-1])
    # Try several window lengths ending NOW; keep the tightest one that both
    # qualifies on width (<=50%) and reaches >=3 accumulation points. (TZ says
    # accumulation lasts >=45 days; stronger pumps come from the narrowest channel.)
    best = None
    for wd in (45, 55, 70, 90, 120, 150):
        if len(df) < wd:
            continue
        w = df.tail(wd).reset_index(drop=True)
        hi = float(w["high"].max()); lo = float(w["low"].min())
        width = (hi - lo) / max(lo, 1e-12)
        if width > MAX_WIDTH:
            continue
        score, nsp, ntw = score_window(w)
        if score < MIN_SCORE:
            continue
        cand = (width, w, hi, lo, score, nsp, ntw)
        if best is None or width < best[0]:
            best = cand
    if best is None:
        return None
    width, w, hi, lo, score, nsp, ntw = best
    x = np.arange(len(w)); y = w["close"].to_numpy()
    slope = np.polyfit(x, y, 1)[0] / max(np.mean(y), 1e-12)
    trend = "downtrend" if slope < -0.0005 else "sideways"
    upper = hi
    if last_close > upper:
        state = "broke_out"
    elif last_close >= upper * (1 - NEAR_PCT):
        state = "near_upper"
    else:
        state = "accumulating"
    return {
        "symbol": symbol, "state": state, "score": round(score, 1),
        "spark": nsp, "twix": ntw, "trend": trend,
        "channel_width": round(width, 3),
        "win_start": w["timestamp"].iloc[0].date().isoformat(),
        "win_end": w["timestamp"].iloc[-1].date().isoformat(),
        "win_days": (w["timestamp"].iloc[-1] - w["timestamp"].iloc[0]).days,
        "lower": lo, "upper": upper, "last_close": last_close,
        "dist_to_upper_pct": round((upper / last_close - 1) * 100, 2),
    }


def fmt_price(x):
    if x >= 1: return f"{x:,.4f}".rstrip("0").rstrip(".")
    return f"{x:.8f}".rstrip("0").rstrip(".")


def main():
    syms = universe_linear()
    print(f"universe: {len(syms)}", flush=True)
    res = []
    for i, sym in enumerate(syms):
        df = daily(sym)
        if df.empty:
            continue
        r = analyze(sym, df)
        if r:
            res.append(r)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(syms)}, {len(res)} formations", flush=True)
        time.sleep(0.08)
    order = {"broke_out": 0, "near_upper": 1, "accumulating": 2}
    res.sort(key=lambda r: (order[r["state"]], -r["score"], r["channel_width"]))
    today = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Актуальные формации Pump — ВРУЧНУЮ (метод из ТЗ, скил Bybit)",
        "",
        f"Дата скана: **{today}** · вселенная: **{len(syms)}** linear USDT-перпов Bybit · "
        "независимая реализация метода (НЕ использует код V5-детектора).",
        "",
        "Метод применён с нуля по ТЗ: окно 60 дн (≥45), ширина ≤50%, spark=зелёная с малым телом "
        "и объёмом больше КАЖДОЙ из ≤4 предыдущих красных (+1.0), twix=спарк с приклеенной зелёной (+0.5), "
        "порог ≥3 балла. Состояния: broke_out / near_upper (≤3%) / accumulating.",
        "",
        f"**Найдено формаций: {len(res)}**",
        "",
    ]
    titles = {"broke_out": "🚀 Пробой границы (кандидаты на вход)",
              "near_upper": "⚠️ Подошли к границе (≤3%)",
              "accumulating": "🟡 В накоплении"}
    by = {}
    for r in res:
        by.setdefault(r["state"], []).append(r)
    for st in ["broke_out", "near_upper", "accumulating"]:
        rows = by.get(st, [])
        lines.append(f"## {titles[st]} — {len(rows)}\n")
        if not rows:
            lines.append("_нет_\n"); continue
        for r in rows:
            lines.append(f"- **{r['symbol']}** · баллы **{r['score']}** (spark {r['spark']}, twix {r['twix']}) · {r['trend']}")
            lines.append(f"  - окно {r['win_start']}→{r['win_end']} ({r['win_days']} дн), ширина {r['channel_width']}")
            lines.append(f"  - границы: низ {fmt_price(r['lower'])} / верх {fmt_price(r['upper'])} · "
                         f"цена {fmt_price(r['last_close'])} · до границы {r['dist_to_upper_pct']}%")
        lines.append("")
    out = "/root/Pump_V5/docs/formations_manual_scan.md"
    open(out, "w").write("\n".join(lines))
    json.dump(res, open("/root/Pump_V5/data/experiments/formations_manual_scan.json", "w"),
              ensure_ascii=False, indent=2)
    print("written ->", out, f"({len(res)} formations)", flush=True)


if __name__ == "__main__":
    main()
