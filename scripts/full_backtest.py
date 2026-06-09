import os, json, time
import pandas as pd
from app.settings import load_settings
from app.backtest.pipeline import BacktestPipeline

cache = os.environ["V4CACHE"]
cfg = load_settings().raw
syms = sorted(os.listdir(cache))
t0 = time.time()
pipe = BacktestPipeline(cfg=cfg, cache_root=cache)
df = pipe.run(syms)
out = "/root/Pump_V5/data/experiments/full_trades.parquet"
os.makedirs(os.path.dirname(out), exist_ok=True)
df.to_parquet(out, index=False)

entered = df[df.get("entered") == True] if "entered" in df.columns else pd.DataFrame()
rep = {"symbols": len(syms), "elapsed_sec": round(time.time()-t0,1),
       "rows": int(len(df)), "entered": int(len(entered))}
if len(entered):
    e = entered.copy(); e["win"] = e["pnl_pct"] > 0
    rep["win_rate"] = round(float(e["win"].mean()),4)
    rep["mean_pnl_pct"] = round(float(e["pnl_pct"].mean()),4)
    rep["median_pnl_pct"] = round(float(e["pnl_pct"].median()),4)
    rep["sum_pnl_pct"] = round(float(e["pnl_pct"].sum()),4)
print(json.dumps(rep, ensure_ascii=False))
if len(entered):
    print("\nOutcomes:\n" + e["outcome"].value_counts().to_string())
    print("\nBy mode:\n" + e.groupby("mode").agg(n=("pnl_pct","size"),win=("win","mean"),mean=("pnl_pct","mean")).round(4).to_string())
    e["w_b"] = pd.cut(e["channel_width"],[0,0.2,0.3,0.4,0.5,9],right=False)
    print("\nBy width:\n" + e.groupby("w_b",observed=True).agg(n=("pnl_pct","size"),win=("win","mean"),mean=("pnl_pct","mean")).round(4).to_string())
    if "case_id" in e.columns:
        pass
print(f"\ntrades -> {out}")
