"""Trailing-distance sweep — measure the tight-trail vs reach-VAH trade-off
with REAL cached data. Honest numbers only.

For each trailing distance we re-walk the SAME entries (same detector, same
entry, same VAH target, same stop) and only vary the trailing stop width.
"""
import sys, glob, copy
import pandas as pd
import yaml

sys.path.insert(0, "/root/Pump_V5")
from app.backtest.pipeline import BacktestPipeline

CACHE = "/root/PumpV4_transfer_20260525_124030/PumpV4_transfer_20260525_124030/Pump_V4/data/cache/market"

with open("/root/Pump_V5/app/config/default.yaml") as f:
    base = yaml.safe_load(f)

# representative symbol subset for speed (first 120 cached symbols)
syms = sorted(p.split("/")[-1] for p in glob.glob(f"{CACHE}/*") if "/" not in p.split(CACHE+"/")[-1].rstrip("/"))
import os
syms = sorted(os.listdir(CACHE))[:120]

results = []
for dist in [0.05, 0.08, 0.12, 0.20, 0.30]:
    cfg = copy.deepcopy(base)
    cfg["exit"]["trailing"]["distance_pct"] = dist
    cfg["exit"]["trailing"]["atr_adaptive"] = False  # isolate the parameter
    pipe = BacktestPipeline(cfg=cfg, cache_root=CACHE)
    df = pipe.run(syms)
    ent = df[df.get("entered") == True].copy()
    if ent.empty:
        results.append((dist, 0, 0, 0, 0, 0)); continue
    p = ent["pnl_pct"] * 100
    wr = (ent["pnl_pct"] > 0).mean() * 100
    vah_hits = (ent["outcome"] == "target_vah").sum()
    eq = (1 + ent["pnl_pct"]).prod()
    results.append((dist, len(ent), wr, p.median(), p.mean(), vah_hits, eq))

print(f"{'trail%':>7} {'N':>4} {'win%':>6} {'med%':>7} {'mean%':>7} {'VAHhit':>7} {'compound':>9}")
for r in results:
    print(f"{r[0]*100:>6.0f}% {r[1]:>4} {r[2]:>5.1f}% {r[3]:>+6.2f}% {r[4]:>+6.2f}% {r[5]:>7} {r[6]:>8.2f}x")
