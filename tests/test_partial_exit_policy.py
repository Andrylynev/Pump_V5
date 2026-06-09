"""Partial-VAH exit policy tests (Pump V5).

Verifies the partial_vah policy books a fraction on the first trail hit and
rides the remainder, and that the baseline 'trailing' policy is unchanged.
"""
from __future__ import annotations

import pandas as pd

from app.backtest.exit import ExitConfig, simulate_exit


def _bars(prices):
    """Build a simple intraday frame where each bar's high/low straddle close."""
    rows = []
    t0 = pd.Timestamp("2025-01-01", tz="UTC")
    for i, (lo, hi, cl) in enumerate(prices):
        rows.append({"timestamp": t0 + pd.Timedelta(hours=i), "high": hi, "low": lo, "close": cl})
    return pd.DataFrame(rows)


def test_baseline_trailing_unchanged():
    # rise to +10% then retrace 5% -> trailing stop fires, single full exit
    df = _bars([(100, 100, 100), (105, 110, 108), (104, 110, 105), (100, 105, 100)])
    cfg = ExitConfig(policy="trailing", trailing_distance_pct=0.05, exit_at_target=True)
    r = simulate_exit(df, entry_idx=0, entry_price=100.0, target_vah=150.0,
                      stop_price=90.0, channel_lower=90.0, mode="single", exit_cfg=cfg)
    assert r["outcome"] == "trailing_stop"
    # exited at high(110)*0.95 = 104.5 -> +4.5%
    assert abs(r["pnl_pct"] - 0.045) < 1e-6


def test_partial_vah_books_partial_then_rides_to_vah():
    # rises, retraces to trigger partial, then continues up to VAH
    df = _bars([
        (100, 100, 100),
        (105, 110, 108),   # high 110, active trail
        (104, 110, 104),   # low 104 <= 110*0.95=104.5 -> partial booked here
        (108, 120, 118),
        (140, 155, 150),   # high 155 >= VAH 150 -> runner exits at VAH
    ])
    cfg = ExitConfig(policy="partial_vah", trailing_distance_pct=0.05,
                     partial_fraction=0.5, runner_trail_pct=0.15, exit_at_target=True)
    r = simulate_exit(df, entry_idx=0, entry_price=100.0, target_vah=150.0,
                      stop_price=90.0, channel_lower=90.0, mode="single", exit_cfg=cfg)
    assert r["outcome"] == "partial_then_vah"
    # 0.5 booked at 104.5 (+4.5% -> 0.0225) + 0.5 to VAH 150 (+50% -> 0.25) = 0.2725
    assert abs(r["pnl_pct"] - (0.5 * 0.045 + 0.5 * 0.50)) < 1e-6


def test_partial_vah_runner_stops_out_wider():
    # partial booked, then runner gives back on the wider 15% trail before VAH
    df = _bars([
        (100, 100, 100),
        (105, 110, 108),   # active
        (104, 110, 104),   # partial booked at 104.5
        (108, 120, 118),   # high 120
        (100, 120, 100),   # low 100 <= 120*0.85=102 -> runner trail stop
    ])
    cfg = ExitConfig(policy="partial_vah", trailing_distance_pct=0.05,
                     partial_fraction=0.5, runner_trail_pct=0.15, exit_at_target=True)
    r = simulate_exit(df, entry_idx=0, entry_price=100.0, target_vah=150.0,
                      stop_price=90.0, channel_lower=90.0, mode="single", exit_cfg=cfg)
    assert r["outcome"] == "partial_then_trail"
    # 0.5 at 104.5 (+0.045) + 0.5 at 120*0.85=102 (+0.02) -> 0.5*0.045+0.5*0.02
    assert abs(r["pnl_pct"] - (0.5 * 0.045 + 0.5 * 0.02)) < 1e-6
