import os, sys
import pandas as pd
from app.settings import load_settings
from app.backtest.pipeline import BacktestPipeline

cache = os.environ["V4CACHE"]
limit = int(os.environ.get("LIMIT", "120"))
cfg = load_settings().raw
syms = sorted(os.listdir(cache))[:limit]

pipe = BacktestPipeline(cfg=cfg, cache_root=cache)
df = pipe.run(syms)
out = os.environ.get("OUT", "/root/Pump_V5/data/experiments/t24_trades.parquet")
os.makedirs(os.path.dirname(out), exist_ok=True)
df.to_parquet(out, index=False)

entered = df[df.get("entered") == True] if "entered" in df.columns else pd.DataFrame()
print(f"symbols scanned: {len(syms)}")
print(f"rows: {len(df)} | entered trades: {len(entered)}")
if len(entered):
    e = entered.copy()
    e["win"] = e["pnl_pct"] > 0
    print(f"\nOVERALL: win-rate {e['win'].mean()*100:.1f}% | mean pnl {e['pnl_pct'].mean()*100:.2f}% | median {e['pnl_pct'].median()*100:.2f}%")
    print("\nOutcomes:")
    print(e["outcome"].value_counts().to_string())
    print("\nBy mode:")
    print(e.groupby("mode").agg(n=("pnl_pct","size"), win_rate=("win","mean"), mean_pnl=("pnl_pct","mean")).to_string())
    print("\nBy trend:")
    print(e.groupby("trend").agg(n=("pnl_pct","size"), win_rate=("win","mean"), mean_pnl=("pnl_pct","mean")).to_string())
    print("\nScore buckets (win-rate / mean pnl):")
    e["score_b"] = pd.cut(e["score"], [0,3,4,5,6,99], right=False)
    print(e.groupby("score_b", observed=True).agg(n=("pnl_pct","size"), win_rate=("win","mean"), mean_pnl=("pnl_pct","mean")).to_string())
    print("\nWidth buckets (win-rate / mean pnl):")
    e["w_b"] = pd.cut(e["channel_width"], [0,0.2,0.3,0.4,0.5,9], right=False)
    print(e.groupby("w_b", observed=True).agg(n=("pnl_pct","size"), win_rate=("win","mean"), mean_pnl=("pnl_pct","mean")).to_string())
else:
    print("no entered trades; reasons:")
    if "reason" in df.columns:
        print(df["reason"].value_counts().to_string())
