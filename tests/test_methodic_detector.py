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
