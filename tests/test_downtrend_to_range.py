"""Test the downtrend->sideways stronger-signal flag (T1.4)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.detectors.methodic_detector import MethodicDetector
from app.settings import load_settings


def _cfg():
    cfg = load_settings().raw["detector"]
    cfg["min_accumulation_score"] = 1.0  # ease detection for synthetic data
    cfg["min_range_days"] = 45
    return cfg


def _spark_block(o, c, h, l, vol, i):
    """Make bars i-2,i-1 red and i a small green spark with big volume."""
    o[i-2], c[i-2] = 101, 99
    o[i-1], c[i-1] = 99, 98
    o[i], c[i], h[i], l[i] = 97.9, 98.2, 99, 97
    vol[i] = 600


def test_preceding_downtrend_detected():
    # 40 days hard downtrend, then 60 days sideways with sparks.
    n = 100
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    down = np.linspace(200, 110, 40)
    flat = 105 + np.zeros(60)
    closes = np.concatenate([down, flat])
    o = closes.copy(); c = closes.copy(); h = closes + 2; l = closes - 2
    vol = np.full(n, 100.0)
    for i in (50, 60, 70, 80):
        _spark_block(o, c, h, l, vol, i)
    df = pd.DataFrame({"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": vol})

    det = MethodicDetector(detector_cfg=_cfg())
    # window starting at index 41 (the sideways section) should see preceding downtrend
    assert det._preceding_downtrend(df, start_idx=45, lookback=30) is True
    # a window starting deep in the flat zone has no preceding downtrend
    assert det._preceding_downtrend(df, start_idx=95, lookback=30) is False


def test_downtrend_to_range_flag_in_formation():
    n = 110
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    down = np.linspace(200, 110, 45)
    flat = 105 + np.zeros(65)
    closes = np.concatenate([down, flat])
    o = closes.copy(); c = closes.copy(); h = closes + 2; l = closes - 2
    vol = np.full(n, 100.0)
    for i in (55, 65, 75, 85, 95):
        o[i-2], c[i-2] = 106, 104
        o[i-1], c[i-1] = 104, 103
        o[i], c[i], h[i], l[i] = 102.9, 103.2, 104, 102
        vol[i] = 600
    df = pd.DataFrame({"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": vol})
    det = MethodicDetector(detector_cfg=_cfg())
    formations = det.detect_accumulations("SYN", df)
    # at least one sideways formation should carry the downtrend_to_range flag
    flags = [f.diagnostics.get("downtrend_to_range") for f in formations if f.diagnostics.get("trend") == "sideways"]
    assert any(flags)
