"""Unit tests for the methodic detector — encode the TZ canon as executable rules.

These tests are the verification gate for Phase 1. If a future change breaks the
method, these fail.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.detectors.methodic_detector import (
    build_channel,
    detect_sparks_and_twix,
)


def _mk(rows):
    """rows: list of (open, high, low, close, volume)."""
    ts = pd.date_range("2024-01-01", periods=len(rows), freq="D", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])
    df.insert(0, "timestamp", ts)
    return df


# ─────────────────────────── SPARK ───────────────────────────

def test_spark_basic_green_beats_each_of_preceding_reds():
    # 3 red candles (vol 10, 20, 30), then a green small-body candle vol 40.
    # 40 > 30, 40 > 20, 40 > 10 -> spark. No glued green neighbour -> SPARK (1.0).
    rows = [
        (100, 101, 99, 98, 10),   # red
        (98, 99, 96, 95, 20),     # red
        (95, 96, 93, 92, 30),     # red
        (92, 93, 91, 92.3, 40),   # green, small body (0.3/2=0.15) , vol 40
        (92.3, 93, 91, 90, 5),    # red after (so candle above is not glued to a green)
    ]
    res = detect_sparks_and_twix(_mk(rows))
    assert res.spark_count == 1, res
    assert res.twix_count == 0
    assert res.score == 1.0


def test_spark_rejected_if_volume_not_above_each_red():
    # green vol 25 but one preceding red has vol 30 -> fails "exceeds EACH".
    rows = [
        (100, 101, 99, 98, 10),
        (98, 99, 96, 95, 30),     # red vol 30 > green's 25
        (95, 96, 93, 92, 20),
        (92, 93, 91, 92.3, 25),   # green vol 25 -> NOT a spark
        (92.3, 93, 91, 90, 5),
    ]
    res = detect_sparks_and_twix(_mk(rows))
    assert res.spark_count == 0
    assert res.twix_count == 0


def test_spark_rejected_if_body_too_large():
    # green but large body (high price impact) -> not a spark.
    rows = [
        (100, 101, 99, 98, 10),
        (98, 99, 96, 95, 20),
        (92, 100, 91, 99.5, 40),  # body 7.5 / range 9 = 0.83 -> too big
        (99.5, 100, 98, 97, 5),
    ]
    res = detect_sparks_and_twix(_mk(rows))
    assert res.spark_count == 0


def test_spark_red_lookback_capped_at_4():
    # 6 preceding reds with an early red vol 1000 (5th back). With cap=4 it's
    # outside the window, so green vol 50 only needs to beat the 4 nearest reds.
    rows = [
        (200, 201, 199, 198, 1000),  # red, 6th back -> ignored (cap 4)
        (198, 199, 196, 195, 999),   # red, 5th back -> ignored
        (195, 196, 193, 192, 10),    # red 4th
        (192, 193, 190, 189, 20),    # red 3rd
        (189, 190, 187, 186, 30),    # red 2nd
        (186, 187, 184, 183, 40),    # red 1st
        (183, 184, 182, 183.3, 50),  # green small body vol 50 > 40,30,20,10
        (183.3, 184, 182, 181, 5),
    ]
    res = detect_sparks_and_twix(_mk(rows), max_red_lookback=4)
    assert res.spark_count == 1


def test_no_preceding_red_means_no_mark():
    rows = [
        (100, 101, 99, 100.5, 10),  # green
        (100.5, 102, 100, 101.5, 40),  # green small body, but no preceding red
    ]
    res = detect_sparks_and_twix(_mk(rows))
    assert res.spark_count == 0
    assert res.twix_count == 0


# ─────────────────────────── TWIX ───────────────────────────

def test_twix_when_glued_to_adjacent_green():
    # anomalous green at i=3, and i=4 is also green -> glued -> TWIX (0.5).
    rows = [
        (100, 101, 99, 98, 10),     # red
        (98, 99, 96, 95, 20),       # red
        (95, 96, 93, 92, 30),       # red
        (92, 93, 91, 92.3, 50),     # green small body, vol50 > 30,20,10
        (92.3, 94, 92, 93.5, 12),   # green (glued neighbour)
    ]
    res = detect_sparks_and_twix(_mk(rows))
    assert res.twix_count == 1, res
    assert res.spark_count == 0
    assert res.score == 0.5


def test_score_sums_spark_and_twix():
    # one standalone spark (1.0) + one twix (0.5) = 1.5
    rows = [
        # spark block
        (100, 101, 99, 98, 10),     # red
        (98, 99, 96, 95, 20),       # red
        (95, 96, 94, 95.2, 40),     # green small body spark, vol40>20,10
        (95.2, 96, 93, 90, 5),      # red (isolates spark)
        # reds then twix
        (90, 91, 88, 87, 8),        # red
        (87, 88, 85, 84, 12),       # red
        (84, 85, 83, 84.3, 30),     # green small body vol30>12,8
        (84.3, 86, 84, 85.5, 9),    # green glued -> makes prior a twix
    ]
    res = detect_sparks_and_twix(_mk(rows))
    assert res.spark_count == 1
    assert res.twix_count == 1
    assert res.score == 1.5


# ─────────────────────────── CHANNEL ───────────────────────────

def test_channel_sideways_flat_bounds():
    rng = np.random.default_rng(0)
    closes = 100 + rng.normal(0, 1, 60)
    rows = [(c, c + 2, c - 2, c, 100.0) for c in closes]
    ch = build_channel(_mk(rows))
    assert ch.trend == "sideways"
    # flat: upper line constant == max high, lower line constant == min low
    assert np.allclose(ch.upper_line, ch.upper_line[0])
    assert np.allclose(ch.lower_line, ch.lower_line[0])


def test_channel_downtrend_sloped_bounds():
    closes = np.linspace(200, 100, 60)  # clear downtrend
    rows = [(c, c + 2, c - 2, c, 100.0) for c in closes]
    ch = build_channel(_mk(rows))
    assert ch.trend == "downtrend"
    # sloped: upper line should be strictly decreasing on net
    assert ch.upper_line[0] > ch.upper_line[-1]
    assert ch.slope_norm < 0


def test_channel_width_is_range_fraction():
    rows = [(100, 150, 100, 120, 100.0)] + [(120, 150, 100, 120, 100.0)] * 59
    ch = build_channel(_mk(rows))
    # max_high=150, min_low=100 -> width = 50/100 = 0.5
    assert abs(ch.channel_width - 0.5) < 1e-6


# ──────────────── METHOD GUARDS (Никита 2026-06-10 review) ────────────────

from app.detectors.methodic_detector import MethodicDetector

_DET_CFG = {
    "min_range_days": 45, "max_range_days": 180, "scan_step_days": 3,
    "window_step_days": 10, "max_channel_width": 0.50,
    "downtrend_slope_threshold": -0.0005, "uptrend_slope_threshold": 0.0010,
    "touch_tol": 0.02, "min_bound_touches": 2,
    "max_intra_window_run_up": 0.30, "breakdown_consec_closes": 3, "breakdown_tol": 0.0,
    "tail_trend_bars": 10, "tail_downtrend_slope": -0.005,
    "spark": {"max_red_lookback": 4, "small_body_max_ratio": 0.35},
    "spark_weight": 1.0, "twix_weight": 0.5, "min_accumulation_score": 3.0,
    "downtrend_to_range_bonus": False, "downtrend_lookback_days": 30,
}


def test_channel_uptrend_classified_and_excluded():
    # Clear rising market -> "uptrend", which the detector must NOT treat as
    # accumulation (ATOM/COTI/ARKM were rising / broke up, not ranging).
    closes = np.linspace(100, 200, 60)
    rows = [(c, c + 2, c - 2, c, 100.0) for c in closes]
    ch = build_channel(_mk(rows), uptrend_slope_threshold=0.0010)
    assert ch.trend == "uptrend"
    assert ch.slope_norm > 0


def test_count_touches_rejects_single_wick():
    # 59 flat bars near 100, ONE spike low to 50 (a wick that's never revisited).
    rows = [(100, 102, 99, 100, 100.0) for _ in range(59)]
    rows.insert(30, (100, 102, 50, 100, 100.0))  # single deep wick
    ch = build_channel(_mk(rows))
    # min_low=50 was touched exactly once -> lower border "hangs in the air".
    assert ch.lower_touches == 1


def test_max_run_up_detects_completed_pump():
    det = MethodicDetector(detector_cfg=_DET_CFG)
    # flat ~100 then a +40% rally inside the window
    closes = [100.0] * 30 + list(np.linspace(100, 140, 20))
    w = _mk([(c, c + 1, c - 1, c, 100.0) for c in closes])
    assert det._max_run_up_pct(w) > 0.30


def test_broke_down_at_end_three_closes_below_floor():
    det = MethodicDetector(detector_cfg=_DET_CFG)
    n = 50
    lower_line = np.full(n, 100.0)
    closes = [105.0] * 47 + [98.0, 97.0, 96.0]  # 3 closes below the 100 floor
    w = _mk([(c, c + 1, c - 1, c, 100.0) for c in closes])
    assert det._broke_down_at_end(w, lower_line, tol=0.0, consec=3) is True
    # a single prick (1 close below) must NOT count as a breakdown
    closes2 = [105.0] * 49 + [98.0]
    w2 = _mk([(c, c + 1, c - 1, c, 100.0) for c in closes2])
    assert det._broke_down_at_end(w2, lower_line, tol=0.0, consec=3) is False


def test_detect_rejects_broken_down_channel():
    # A valid scoring range that then loses its floor with 3 closes below at the
    # end must NOT be returned as a live formation.
    det = MethodicDetector(detector_cfg=_DET_CFG)
    rng = np.random.default_rng(1)
    base = 100 + rng.normal(0, 1.5, 47)
    rows = []
    # seed enough sparks: red runs then anomalous green
    for k, c in enumerate(base):
        if k % 8 == 0 and k >= 3:
            rows.append((c, c + 0.3, c - 0.3, c + 0.2, 500.0))  # green small-body high-vol
        elif k % 8 in (1, 2, 3):
            rows.append((c, c + 1, c - 2, c - 1.5, 50.0))       # red low-vol
        else:
            rows.append((c, c + 1, c - 1, c, 100.0))
    rows += [(95, 96, 90, 92, 100.0), (92, 93, 88, 89, 100.0), (89, 90, 85, 86, 100.0)]
    df = _mk(rows)
    forms = det.detect_accumulations("TESTBROKE", df)
    # Either no formation, or none whose window ends in the broken-down tail.
    for f in forms:
        assert not f.diagnostics.get("broke_down_at_end", False)
