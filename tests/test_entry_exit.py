"""Unit tests for Phase 2 entry/exit — encode the TZ canon as executable rules."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from app.backtest.entry import EntryEvaluator
from app.backtest.exit import (
    ExitConfig,
    compute_pump_target,
    risk_reward,
    simulate_exit,
)
from app.contracts import FormationCandidate


def _cand(upper=110.0, lower=90.0):
    return FormationCandidate(
        case_id="X", symbol="XUSDT", timeframe="1D", branch="1D",
        accumulation_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        accumulation_end=datetime(2024, 2, 15, tzinfo=timezone.utc),
        entry_time=None, upper_bound=upper, lower_bound=lower,
        score=4.0, spark_count=3, twix_count=2, volume_score=4.0, channel_width=0.22,
    )


def _intraday(rows, start="2024-02-15"):
    ts = pd.date_range(start, periods=len(rows), freq="4h", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])
    df.insert(0, "timestamp", ts)
    return df


# ─────────────── ENTRY ───────────────

def test_entry_requires_body_close_above_not_just_wick():
    cfg = {"require_body_close_above": True, "skip_if_extended_pct": 0.10, "extended_pct_atr_adaptive": False}
    ev = EntryEvaluator(entry_cfg=cfg)
    # bar 0: wick pokes above 110 (high 112) but body closes at 109 -> no entry
    # bar 1: body closes at 111 above 110 -> entry
    rows = [
        (108, 112, 107, 109, 100),  # wick poke only
        (109, 113, 108, 111, 120),  # body close above
    ]
    res = ev.evaluate(_cand(), _intraday(rows), "240")
    assert res["entered"] is True
    assert res["entry_idx"] == 1
    assert abs(res["entry_price"] - 111) < 1e-9


def test_entry_skipped_if_price_extended():
    cfg = {"require_body_close_above": True, "skip_if_extended_pct": 0.10, "extended_pct_atr_adaptive": False}
    ev = EntryEvaluator(entry_cfg=cfg)
    # body closes at 125 -> 13.6% above 110 -> > 10% -> skip
    rows = [(120, 126, 119, 125, 100)]
    res = ev.evaluate(_cand(), _intraday(rows), "240")
    assert res["entered"] is False
    assert res["reason"] == "price_extended_skip"


def test_entry_invalidated_by_downward_break():
    cfg = {"require_body_close_above": True, "skip_if_extended_pct": 0.10, "extended_pct_atr_adaptive": False}
    ev = EntryEvaluator(entry_cfg=cfg)
    rows = [(95, 96, 88, 89, 100)]  # low 88 < lower 90 -> break down
    res = ev.evaluate(_cand(), _intraday(rows), "240")
    assert res["entered"] is False
    assert res["reason"] == "channel_break_down"


# ─────────────── RISK / REWARD ───────────────

def test_risk_reward_gate_3to1():
    # entry 100, target 130 (+30%), stop 90 (-10%) -> RR = 3.0
    assert abs(risk_reward(100, 130, 90) - 3.0) < 1e-9
    # tighter stop improves RR
    assert risk_reward(100, 130, 95) > 3.0
    # wide stop fails
    assert risk_reward(100, 130, 80) < 3.0


# ─────────────── EXIT ───────────────

def _exit_cfg(dist=0.05, exit_at_target=True):
    return ExitConfig(
        target_mode="vah", exit_at_target=exit_at_target,
        trailing_enabled=True, trailing_distance_pct=dist, trailing_atr_adaptive=False,
        stop_single="prev_channel_low", stop_martingale="channel_break_down",
    )


def test_exit_takes_profit_at_vah():
    # entry 100, vah 130; price climbs to 131 -> exit at vah 130
    rows = [(100, 100, 100, 100, 1)] + [(c, c + 1, c - 1, c, 1) for c in [110, 120, 131]]
    df = _intraday(rows)
    res = simulate_exit(df, entry_idx=0, entry_price=100, target_vah=130,
                        stop_price=90, channel_lower=90, mode="single", exit_cfg=_exit_cfg(exit_at_target=True))
    assert res["outcome"] == "target_vah"
    assert abs(res["exit_price"] - 130) < 1e-9


def test_exit_single_hits_prev_channel_low_stop():
    # entry 100, stop 90; price drops to 88 -> stop at 90
    rows = [(100, 100, 100, 100, 1), (100, 101, 88, 89, 1)]
    df = _intraday(rows)
    res = simulate_exit(df, entry_idx=0, entry_price=100, target_vah=130,
                        stop_price=90, channel_lower=85, mode="single", exit_cfg=_exit_cfg())
    assert res["outcome"] == "stop_loss"
    assert abs(res["exit_price"] - 90) < 1e-9


def test_exit_martingale_stops_on_channel_break_down():
    # martingale ignores single stop; exits when price exits channel (lower 85)
    rows = [(100, 100, 100, 100, 1), (100, 101, 84, 86, 1)]
    df = _intraday(rows)
    res = simulate_exit(df, entry_idx=0, entry_price=100, target_vah=130,
                        stop_price=90, channel_lower=85, mode="martingale", exit_cfg=_exit_cfg())
    assert res["outcome"] == "channel_break_down"
    assert abs(res["exit_price"] - 85) < 1e-9


def test_exit_trailing_stop_locks_gains():
    # entry 100, dist 5%; climbs to 120 (trail=114) then falls to 113 -> trail hit
    rows = [
        (100, 100, 100, 100, 1),
        (100, 120, 100, 119, 1),   # high 120 -> trailing active, trail=114
        (119, 119, 113, 113, 1),   # low 113 <= 114 -> trailing stop
    ]
    df = _intraday(rows)
    res = simulate_exit(df, entry_idx=0, entry_price=100, target_vah=200,
                        stop_price=90, channel_lower=85, mode="single", exit_cfg=_exit_cfg(dist=0.05))
    assert res["outcome"] == "trailing_stop"
    assert abs(res["exit_price"] - 114) < 1e-6


# ─────────────── PUMP TARGET (VAH) ───────────────

def test_pump_target_vah_from_fixed_range_vp():
    ts = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
    # accumulation around 100, then a run-up; VAH should sit above POC
    closes = [100] * 45 + list(range(101, 116))
    df = pd.DataFrame({
        "timestamp": ts,
        "open": closes, "high": [c + 2 for c in closes],
        "low": [c - 2 for c in closes], "close": closes, "volume": [1000] * 60,
    })
    tgt = compute_pump_target(df, ts[0], ts[-1])
    assert tgt.ok
    assert tgt.vah >= tgt.poc >= tgt.val
