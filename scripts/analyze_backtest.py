"""Analyze the post-guard full backtest and compare to the pre-guard baseline.

Pre-guard baseline (documented full run, post-VAH-fix):
  152 entries, win 73.0%, median +1.90%, mean +1.36%, compound x2.32, sum +207%
"""
import pandas as pd, numpy as np, json

P = "/root/Pump_V5/data/experiments/full_trades.parquet"
df = pd.read_parquet(P)
print(f"rows={len(df)}  symbols={df['symbol'].nunique()}")
e = df[df.get("entered") == True].copy()
print(f"ENTERED: {len(e)}")
if not len(e):
    raise SystemExit("no entries")

e["win"] = e["pnl_pct"] > 0
def compound(series, frac=1.0):
    eq = 1.0
    for r in series:
        eq *= (1 + frac * r)
    return eq

print("\n=== HEADLINE (post-guard) ===")
hl = {
    "entries": int(len(e)),
    "win_rate": round(float(e["win"].mean()), 4),
    "median_pnl_pct": round(float(e["pnl_pct"].median()), 4),
    "mean_pnl_pct": round(float(e["pnl_pct"].mean()), 4),
    "sum_pnl_pct": round(float(e["pnl_pct"].sum()), 4),
    "compound_full": round(compound(e["pnl_pct"]), 3),
    "compound_10pct_size": round(compound(e["pnl_pct"], 0.10), 3),
}
print(json.dumps(hl, ensure_ascii=False, indent=2))

print("\n=== BASELINE (pre-guard, documented) ===")
base = {"entries": 152, "win_rate": 0.730, "median_pnl_pct": 0.019,
        "mean_pnl_pct": 0.0136, "compound_full": 2.32, "sum_pnl_pct": 2.07}
print(json.dumps(base, ensure_ascii=False, indent=2))

print("\n=== DELTA ===")
print(f"entries: {base['entries']} -> {hl['entries']}  ({hl['entries']-base['entries']:+d})")
print(f"win_rate: {base['win_rate']:.1%} -> {hl['win_rate']:.1%}  ({(hl['win_rate']-base['win_rate'])*100:+.1f}pp)")
print(f"mean_pnl: {base['mean_pnl_pct']:.2%} -> {hl['mean_pnl_pct']:.2%}")
print(f"compound: x{base['compound_full']} -> x{hl['compound_full']}")

print("\n=== OUTCOMES ===")
print(e["outcome"].value_counts().to_string())

print("\n=== BY MODE ===")
print(e.groupby("mode").agg(n=("pnl_pct","size"), win=("win","mean"), mean=("pnl_pct","mean")).round(4).to_string())

print("\n=== BY WIDTH ===")
e["w_b"] = pd.cut(e["channel_width"], [0,0.2,0.3,0.4,0.5,9], right=False)
print(e.groupby("w_b", observed=True).agg(n=("pnl_pct","size"), win=("win","mean"), mean=("pnl_pct","mean")).round(4).to_string())

print("\n=== BY TREND ===")
print(e.groupby("trend").agg(n=("pnl_pct","size"), win=("win","mean"), mean=("pnl_pct","mean")).round(4).to_string())

print("\n=== PnL distribution ===")
for q in [0.05,0.25,0.5,0.75,0.95]:
    print(f"  p{int(q*100)}: {e['pnl_pct'].quantile(q):+.2%}")
print(f"  best: {e['pnl_pct'].max():+.2%}  worst: {e['pnl_pct'].min():+.2%}")

# outlier sensitivity
top = e.nlargest(10, "pnl_pct")
print(f"\ncompound without top-3: x{compound(e.nsmallest(len(e)-3,'pnl_pct')['pnl_pct']):.2f}")
print(f"compound without top-10: x{compound(e.nsmallest(len(e)-10,'pnl_pct')['pnl_pct']):.2f}")
