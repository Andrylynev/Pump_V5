"""B_partial robustness check — is the 34.98x compound an outlier artifact?
Saves per-trade rows for honest distribution analysis."""
import sys, os, copy, yaml
import pandas as pd, numpy as np
sys.path.insert(0,"/root/Pump_V5")
from app.backtest.pipeline import BacktestPipeline
CACHE=os.environ.get("V4CACHE","/root/PumpV4_transfer_20260525_124030/PumpV4_transfer_20260525_124030/Pump_V4/data/cache/market")
base=yaml.safe_load(open("/root/Pump_V5/app/config/default.yaml"))
cfg=copy.deepcopy(base); cfg["exit"]["policy"]="partial_vah"; cfg["exit"]["trailing"]["atr_adaptive"]=False
df=BacktestPipeline(cfg=cfg,cache_root=CACHE).run(sorted(os.listdir(CACHE)))
ent=df[df.entered==True].copy()
ent.to_parquet("/root/Pump_V5/data/experiments/b_partial_trades.parquet")
p=ent.pnl_pct*100
print("N",len(ent),"win%",round((ent.pnl_pct>0).mean()*100,1))
print("top8 pnl%:",sorted(p.round(1).tolist(),reverse=True)[:8])
print(">50%:",int((p>50).sum())," >100%:",int((p>100).sum()))
s=ent.pnl_pct.sort_values(ascending=False)
print(f"compound all: {(1+ent.pnl_pct).prod():.2f}x   without top-3: {(1+s.iloc[3:]).prod():.2f}x   without top-10: {(1+s.iloc[10:]).prod():.2f}x")
# realistic 10% fixed-fraction sizing, chronological
eq=10000.0
for pnl in ent.sort_values("entry_time").pnl_pct: eq*=(1+0.10*pnl)
print(f"realistic 10%-equity sizing: ${eq:,.0f} from $10k ({eq/10000:.2f}x)")
print("VAH-exit pnl%:",ent[ent.outcome.astype(str).str.contains('vah')].pnl_pct.mul(100).round(1).tolist())
print("outcomes:",ent.outcome.value_counts().to_dict())
