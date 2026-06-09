"""Pipeline e2e wiring test on synthetic data (no network, no cache)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.backtest.pipeline import BacktestPipeline
from app.settings import load_settings


def _write_cache(tmp_path):
    """Build a synthetic symbol with a 45+ day accumulation + sparks then a breakout."""
    n = 80
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(1)
    base = 100 + rng.normal(0, 0.5, n)
    o = base.copy(); c = base.copy(); h = base + 2; l = base - 2
    vol = np.full(n, 100.0)
    # inject spark pattern: reds then a small-body green with big volume, repeated
    for k in (10, 20, 30):
        o[k-2], c[k-2] = 101, 99   # red
        o[k-1], c[k-1] = 99, 98    # red
        o[k], c[k], h[k], l[k] = 98, 98.3, 99, 97  # small green
        vol[k] = 500
    # breakout after day 60
    for i in range(60, n):
        o[i] = c[i-1]; c[i] = o[i] + 5; h[i] = c[i] + 1; l[i] = o[i] - 1
    daily = pd.DataFrame({"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": vol})
    d = tmp_path / "SYNTHUSDT" / "D"
    d.mkdir(parents=True)
    daily.to_parquet(d / "2024.parquet", index=False)
    return str(tmp_path)


def test_pipeline_runs_end_to_end(tmp_path):
    cache_root = _write_cache(tmp_path)
    cfg = load_settings().raw
    # relax score so the synthetic formation is detectable
    cfg["detector"]["min_accumulation_score"] = 1.0
    pipe = BacktestPipeline(cfg=cfg, cache_root=cache_root)
    df = pipe.run(["SYNTHUSDT"])
    # pipeline must run without raising and return a frame
    assert isinstance(df, pd.DataFrame)
    # columns present even if no entry
    assert "symbol" in df.columns
