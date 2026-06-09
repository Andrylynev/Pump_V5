"""CODE scan — current active Pump formations on Bybit (fresh data to today).

Fetches FRESH daily klines for the full Bybit linear (USDT-perp) universe,
runs the V5 MethodicDetector, and keeps formations whose accumulation window
is STILL ACTIVE (ends within `fresh_days` of the latest bar). Each is classified
by state vs the current price:
  - accumulating       : price inside channel, below upper bound
  - near_upper         : price within `near_pct` of the upper bound
  - broke_out          : last close already above the upper bound (watch для входа)

Writes a markdown report. Public no-auth Bybit v5 REST.
"""
import sys, time, json
import requests
import pandas as pd
import numpy as np

sys.path.insert(0, "/root/Pump_V5")
from app.detectors.methodic_detector import MethodicDetector
import yaml

CFG = yaml.safe_load(open("/root/Pump_V5/app/config/default.yaml"))
DET = MethodicDetector(detector_cfg=CFG["detector"])
NEAR_PCT = 0.03          # within 3% of upper bound = "near"
FRESH_DAYS = 75          # keep formations still active now. The detector finds the
                         # tightest historical sub-window, so a CURRENTLY-valid
                         # channel can have accumulation_end up to ~2.5mo back while
                         # price is still inside it (cross-check vs manual scan caught
                         # RLUSDUSDT/POL/LTC being dropped at the old 10d cutoff).
                         # We additionally require price is still inside/at the channel
                         # (not collapsed far below the lower bound) — see filter below.
MAX_BELOW_LOWER = 0.15   # drop if last close fell >15% under the lower bound (channel dead)
MIN_BARS = CFG["detector"]["min_range_days"]
NOW_MS = int(time.time() * 1000)
LOOKBACK_DAYS = 400
START_MS = NOW_MS - LOOKBACK_DAYS * 86400_000

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
    j = safe_get("https://api.bybit.com/v5/market/instruments-info",
                 {"category": "linear"})
    out = []
    for it in j["result"]["list"]:
        if (it.get("quoteCoin") == "USDT" and it.get("status") == "Trading"
                and it.get("contractType") == "LinearPerpetual"):
            out.append(it["symbol"])
    return sorted(out)


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
    for c in ["open", "high", "low", "close", "volume", "turnover"]:
        df[c] = df[c].astype(float)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    # drop the still-forming last bar
    return df.iloc[:-1].reset_index(drop=True)


def scan():
    syms = universe_linear()
    print(f"universe: {len(syms)} linear USDT perps", flush=True)
    results = []
    for i, sym in enumerate(syms):
        df = daily(sym)
        if df.empty or len(df) < MIN_BARS:
            continue
        last_ts = df["timestamp"].iloc[-1]
        last_close = float(df["close"].iloc[-1])
        try:
            forms = DET.detect_accumulations(sym, df)
        except Exception:
            continue
        for f in forms:
            acc_end = pd.Timestamp(f.accumulation_end)
            age_days = (last_ts - acc_end).days
            if age_days > FRESH_DAYS:
                continue  # stale formation, not current
            ub = float(f.upper_bound); lb = float(f.lower_bound)
            # Channel must still be alive: price not collapsed far below the floor.
            if lb > 0 and last_close < lb * (1 - MAX_BELOW_LOWER):
                continue
            if last_close > ub:
                state = "broke_out"
            elif last_close >= ub * (1 - NEAR_PCT):
                state = "near_upper"
            else:
                state = "accumulating"
            results.append({
                "symbol": sym,
                "state": state,
                "score": round(float(f.score), 1),
                "spark": int(f.spark_count),
                "twix": int(f.twix_count),
                "trend": f.diagnostics.get("trend"),
                "downtrend_to_range": bool(f.diagnostics.get("downtrend_to_range")),
                "channel_width": round(float(f.channel_width), 3),
                "acc_start": pd.Timestamp(f.accumulation_start).date().isoformat(),
                "acc_end": acc_end.date().isoformat(),
                "acc_days": (acc_end - pd.Timestamp(f.accumulation_start)).days,
                "lower": lb, "upper": ub, "last_close": last_close,
                "dist_to_upper_pct": round((ub / last_close - 1) * 100, 2),
            })
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(syms)} scanned, {len(results)} formations so far", flush=True)
        time.sleep(0.08)
    return results, len(syms)


def fmt_price(x):
    if x >= 1: return f"{x:,.4f}".rstrip("0").rstrip(".")
    return f"{x:.8f}".rstrip("0").rstrip(".")


def write_md(results, n_univ):
    order = {"broke_out": 0, "near_upper": 1, "accumulating": 2}
    results.sort(key=lambda r: (order[r["state"]], -r["score"], r["channel_width"]))
    today = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Актуальные формации Pump — КОД (V5 detector)",
        "",
        f"Дата скана: **{today}** · вселенная: **{n_univ}** linear USDT-перпов Bybit · "
        f"свежие данные (дневки до вчера) · формации, активные на сейчас (накопление завершилось ≤{FRESH_DAYS} дн назад).",
        "",
        "Метод V5: канал ≤45 дн (боковик плоский / нисходящий наклонный), spark=1 балл, twix=0.5, "
        "вход от ≥3 баллов. Состояния: **broke_out** (закрылся выше границы — кандидат на вход), "
        "**near_upper** (в пределах 3% от границы), **accumulating** (ещё в канале).",
        "",
        f"**Найдено формаций: {len(results)}**",
        "",
    ]
    by_state = {}
    for r in results:
        by_state.setdefault(r["state"], []).append(r)
    titles = {"broke_out": "🚀 Пробой границы (закрепление — кандидаты на вход)",
              "near_upper": "⚠️ Подошли к верхней границе (≤3%)",
              "accumulating": "🟡 В накоплении (в канале)"}
    for st in ["broke_out", "near_upper", "accumulating"]:
        rows = by_state.get(st, [])
        lines.append(f"## {titles[st]} — {len(rows)}")
        lines.append("")
        if not rows:
            lines.append("_нет_\n"); continue
        for r in rows:
            flag = " · 🔻→боковик (сильный сигнал)" if r["downtrend_to_range"] else ""
            lines.append(
                f"- **{r['symbol']}** · баллы **{r['score']}** (spark {r['spark']}, twix {r['twix']}) · "
                f"{r['trend']}{flag}"
            )
            lines.append(
                f"  - канал {r['acc_start']}→{r['acc_end']} ({r['acc_days']} дн), ширина {r['channel_width']}"
            )
            lines.append(
                f"  - границы: низ {fmt_price(r['lower'])} / верх {fmt_price(r['upper'])} · "
                f"цена {fmt_price(r['last_close'])} · до границы {r['dist_to_upper_pct']}%"
            )
        lines.append("")
    out = "/root/Pump_V5/docs/formations_code_scan.md"
    with open(out, "w") as fh:
        fh.write("\n".join(lines))
    # also dump raw json next to it
    with open("/root/Pump_V5/data/experiments/formations_code_scan.json", "w") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print("written ->", out, f"({len(results)} formations)", flush=True)


if __name__ == "__main__":
    res, n = scan()
    write_md(res, n)
