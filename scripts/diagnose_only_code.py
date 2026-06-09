"""Diagnose WHY each 'only-code' formation did NOT appear in the manual scan.

Reuses the manual analyzer's exact functions (daily fetch, score_window, the
same window set + thresholds) but instead of just returning pass/fail, records
the rejection reason per symbol. Output: a markdown explaining each miss.
"""
import sys, json, importlib.util
import numpy as np
import pandas as pd

sys.path.insert(0, "/root/Pump_V5")
spec = importlib.util.spec_from_file_location("mm", "/root/Pump_V5/scripts/scan_manual_method.py")
mm = importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)

code = json.load(open("/root/Pump_V5/data/experiments/formations_code_scan.json"))
manual = json.load(open("/root/Pump_V5/data/experiments/formations_manual_scan.json"))
cs = {r["symbol"] for r in code}; ms = {r["symbol"] for r in manual}
only_code = sorted(cs - ms)
cd = {r["symbol"]: r for r in code}

WINDOWS = (45, 55, 70, 90, 120, 150)


def diagnose(symbol):
    """Return (reason_code, detail) explaining why manual method rejected it."""
    df = mm.daily(symbol)
    if df.empty:
        return "no_data", "нет дневных данных с Bybit"
    if len(df) < mm.MIN_DAYS + 1:
        return "too_short", f"история {len(df)} баров < {mm.MIN_DAYS+1}"
    # Walk the same windows the manual method uses; record best width & best score
    min_width = None; max_score = 0.0
    width_ok_any = False; score_ok_any = False; both_ok = False
    for wd in WINDOWS:
        if len(df) < wd:
            continue
        w = df.tail(wd).reset_index(drop=True)
        hi = float(w["high"].max()); lo = float(w["low"].min())
        width = (hi - lo) / max(lo, 1e-12)
        score, nsp, ntw = mm.score_window(w)
        if min_width is None or width < min_width:
            min_width = width
        max_score = max(max_score, score)
        wok = width <= mm.MAX_WIDTH
        sok = score >= mm.MIN_SCORE
        width_ok_any |= wok
        score_ok_any |= sok
        if wok and sok:
            both_ok = True
    if both_ok:
        return "should_have_matched", f"прошёл бы (min_width={min_width:.3f}, max_score={max_score})"
    # Determine dominant reason
    if not width_ok_any and not score_ok_any:
        return "width_and_score", f"во всех окнах канал шире 50% (min {min_width:.2f}) И балл <3 (max {max_score})"
    if not width_ok_any:
        return "width", f"канал во ВСЕХ окнах шире 50% (минимум {min_width:.2f}) — ручной метод режет по ≤0.50"
    if not score_ok_any:
        return "score", f"балл <3 во всех окнах (макс {max_score}) при подходящей ширине"
    # width_ok in some window, score_ok in some window, but never SAME window
    return "no_common_window", (
        f"узкий канал и ≥3 балла НЕ совпали в одном окне "
        f"(min_width={min_width:.2f}, max_score={max_score}); "
        f"ручной берёт фикс-окна, код — переменные")


REASONS_RU = {
    "width": "🟠 Канал шире 50%",
    "score": "🟡 Балл накопления < 3",
    "width_and_score": "🔴 Канал >50% И балл <3",
    "no_common_window": "🔵 Узость и баллы не совпали в одном окне",
    "should_have_matched": "⚪ Должен был пройти (расхождение порогов/окон)",
    "no_data": "⚫ Нет данных",
    "too_short": "⚫ Короткая история",
}

rows = []
for i, s in enumerate(only_code):
    rc, detail = diagnose(s)
    rows.append({"symbol": s, "reason": rc, "detail": detail, "code": cd[s]})
    print(f"{i+1}/{len(only_code)} {s}: {rc}", flush=True)

# Group by reason
from collections import Counter
counts = Counter(r["reason"] for r in rows)

def fmt_price(x):
    if x >= 1: return f"{x:,.4f}".rstrip("0").rstrip(".")
    return f"{x:.8f}".rstrip("0").rstrip(".")

today = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")
lines = [
    "# Формации, найденные КОДОМ но НЕ ручным методом — с объяснением",
    "",
    f"Дата: **{today}** · вселенная 429 linear-перпов Bybit.",
    f"Код (V5-детектор) нашёл **{len(cs)}** монет, ручной метод — **{len(ms)}**. "
    f"Здесь — **{len(only_code)}** монет, что есть ТОЛЬКО в коде, и почему каждая не вошла в ручной.",
    "",
    "## Почему вообще расходятся",
    "Код-детектор сканирует ПЕРЕМЕННЫЕ окна 45-180 дн и находит самое тугое под-окно под балл. "
    "Ручной метод берёт ФИКСИРОВАННЫЕ окна (45/55/70/90/120/150 дн), заканчивающиеся сегодня, "
    "с жёстким порогом ширины ≤50% и ≥3 балла В ОДНОМ окне. Отсюда код «вытягивает» больше "
    "(в т.ч. широкие/слабые), ручной — строже.",
    "",
    "## Разбивка причин",
]
for rc, n in counts.most_common():
    lines.append(f"- {REASONS_RU.get(rc, rc)} — **{n}**")
lines.append("")

order = ["width", "score", "width_and_score", "no_common_window", "should_have_matched", "no_data", "too_short"]
for rc in order:
    grp = [r for r in rows if r["reason"] == rc]
    if not grp:
        continue
    lines.append(f"## {REASONS_RU.get(rc, rc)} — {len(grp)}")
    lines.append("")
    grp.sort(key=lambda r: -r["code"]["score"])
    for r in grp:
        c = r["code"]
        lines.append(
            f"- **{r['symbol']}** · код: балл {c['score']} ({c['trend']}, ширина {c['channel_width']}, "
            f"окно {c['acc_days']}д) → состояние {c['state']}"
        )
        lines.append(f"  - причина невхода: {r['detail']}")
    lines.append("")

out = "/root/Pump_V5/docs/formations_only_code_explained.md"
open(out, "w").write("\n".join(lines))
json.dump([{k: v for k, v in r.items() if k != "code"} for r in rows],
          open("/root/Pump_V5/data/experiments/only_code_diag.json", "w"),
          ensure_ascii=False, indent=2)
print("\nwritten ->", out)
print("reason counts:", dict(counts))
