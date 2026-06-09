"""Exit-policy comparison on the FULL cached universe (honest numbers).

Runs the real pipeline (detector -> entry -> VAH -> stops) once per policy and
reports aggregate PnL. The detector is the slow part; we run each policy as a
full pass. Only real cached data, no fabricated numbers.

Policies compared:
  A  baseline tight trailing 5%               (current)
  B  partial_vah 50% @ trail + runner to VAH  (book half, ride half)
  C  ATR-adaptive trailing (1.5 x ATR%, 3-15%)
  D  partial_vah + ATR-adaptive trail         (combine B+C)
"""
import sys, os, glob, copy
import pandas as pd
import yaml

sys.path.insert(0, "/root/Pump_V5")
from app.backtest.pipeline import BacktestPipeline

CACHE = os.environ.get(
    "V4CACHE",
    "/root/PumpV4_transfer_20260525_124030/PumpV4_transfer_20260525_124030/Pump_V4/data/cache/market",
)
base = yaml.safe_load(open("/root/Pump_V5/app/config/default.yaml"))
syms = sorted(os.listdir(CACHE))

POLICIES = {
    "A_trail5":      {"policy": "trailing",    "atr": False},
    "B_partial":     {"policy": "partial_vah", "atr": False},
    "C_atr":         {"policy": "trailing",    "atr": True},
    "D_partial_atr": {"policy": "partial_vah", "atr": True},
}

def stats(df):
    ent = df[df.get("entered") == True].copy()
    if ent.empty:
        return dict(n=0)
    p = ent["pnl_pct"] * 100
    return dict(
        n=len(ent),
        win=round((ent["pnl_pct"] > 0).mean() * 100, 1),
        med=round(p.median(), 2),
        mean=round(p.mean(), 2),
        compound=round((1 + ent["pnl_pct"]).prod(), 2),
        vah=int(ent["outcome"].astype(str).str.contains("vah").sum()),
        ran_to_vah=int((ent["outcome"] == "target_vah").sum() + (ent["outcome"] == "partial_then_vah").sum()),
    )

rows = []
for name, pol in POLICIES.items():
    cfg = copy.deepcopy(base)
    cfg["exit"]["policy"] = pol["policy"]
    cfg["exit"]["trailing"]["atr_adaptive"] = pol["atr"]
    pipe = BacktestPipeline(cfg=cfg, cache_root=CACHE)
    df = pipe.run(syms)
    s = stats(df); s["policy"] = name
    rows.append(s)
    print(name, s, flush=True)

print("\n=== SUMMARY (full {} pairs) ===".format(len(syms)))
print(f"{'policy':>14} {'N':>4} {'win%':>6} {'med%':>7} {'mean%':>7} {'comp':>6} {'VAHexit':>8}")
for s in rows:
    if s.get("n", 0) == 0:
        print(f"{s['policy']:>14}  (no entries)"); continue
    print(f"{s['policy']:>14} {s['n']:>4} {s['win']:>5.1f}% {s['med']:>+6.2f}% {s['mean']:>+6.2f}% {s['compound']:>5.2f}x {s['ran_to_vah']:>8}")
pd.DataFrame(rows).to_parquet("/root/Pump_V5/data/experiments/policy_compare.parquet")
print("\nsaved -> data/experiments/policy_compare.parquet")
