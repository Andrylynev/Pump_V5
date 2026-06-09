import os, glob, sys
import pandas as pd
from app.settings import load_settings
from app.detectors.methodic_detector import MethodicDetector

cache = os.environ["V4CACHE"]
limit = int(os.environ.get("LIMIT", "25"))
cfg = load_settings().raw["detector"]
det = MethodicDetector(detector_cfg=cfg)

syms = sorted(os.listdir(cache))[:limit]
total = 0
hits = []
scanned = 0
for sym in syms:
    parts = sorted(glob.glob(f"{cache}/{sym}/D/*.parquet"))
    if not parts:
        continue
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    if len(df) < 45:
        continue
    scanned += 1
    cands = det.detect_accumulations(sym, df)
    total += len(cands)
    for c in cands[:1]:
        hits.append((sym, c.diagnostics["trend"], round(c.channel_width,3),
                     c.spark_count, c.twix_count, round(c.score,1),
                     str(c.accumulation_start.date()), str(c.accumulation_end.date())))

print(f"scanned {scanned} symbols with >=45 daily bars, found {total} formations (score>=3.0)")
print("sym | trend | width | sparks | twix | score | acc_start | acc_end")
for h in hits[:20]:
    print(" | ".join(str(x) for x in h))
