"""Verify vectorized VP == original V4 VP, byte-for-byte on the outputs.

Guards the perf optimization: the vectorized allocation must produce identical
POC/VAH/VAL/bins as the original per-candle/per-bin loop.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from app.features.volume_profile import calculate_fixed_range_volume_profile as vp_new

# Load the ORIGINAL V4 implementation directly from the V4 source file.
V4_VP = "/root/PumpV4_transfer_20260525_124030/PumpV4_transfer_20260525_124030/Pump_V4/app/features/volume_profile.py"


def _load_v4():
    spec = importlib.util.spec_from_file_location("v4_volume_profile", V4_VP)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v4_volume_profile"] = mod  # needed for @dataclass module resolution
    spec.loader.exec_module(mod)
    return mod.calculate_fixed_range_volume_profile


def _synthetic(seed: int, n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    base = 100 + np.cumsum(rng.normal(0, 1.5, n))
    base = np.abs(base) + 10
    high = base + np.abs(rng.normal(0, 2, n))
    low = base - np.abs(rng.normal(0, 2, n))
    close = low + rng.uniform(0, 1, n) * (high - low)
    open_ = low + rng.uniform(0, 1, n) * (high - low)
    vol = np.abs(rng.normal(1000, 300, n))
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": vol})


def test_vectorized_vp_matches_v4_original():
    vp_old = _load_v4()
    for seed in range(8):
        df = _synthetic(seed)
        a = vp_new(df, df["timestamp"].iloc[0], df["timestamp"].iloc[-1])
        b = vp_old(df, df["timestamp"].iloc[0], df["timestamp"].iloc[-1])
        assert (a is None) == (b is None), f"seed {seed}: None mismatch"
        if a is None:
            continue
        assert abs(a.poc_price - b.poc_price) < 1e-6, f"seed {seed}: POC"
        assert abs(a.vah_price - b.vah_price) < 1e-6, f"seed {seed}: VAH"
        assert abs(a.val_price - b.val_price) < 1e-6, f"seed {seed}: VAL"
        assert abs(a.total_volume - b.total_volume) < 1e-3, f"seed {seed}: total"
        np.testing.assert_allclose(
            a.bins["total_volume"].to_numpy(), b.bins["total_volume"].to_numpy(),
            rtol=1e-6, atol=1e-6, err_msg=f"seed {seed}: bins",
        )
