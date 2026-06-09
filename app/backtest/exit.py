"""Pump target + exit logic (Pump V5) — strict to the TZ canon.

Target (длина пампа):
  Fixed-range volume profile from the PRIOR HIGH (the high before accumulation
  began) to the CURRENT price. Two volume zones form (upper/lower); the pump
  most often runs to the UPPER zone (VAH). We use VAH as the take-profit target.

Exit:
  * Trailing stop ~5% (distance_pct), optionally ATR-adaptive to the coin's
    normal move.
  * If price reaches VAH -> take profit (exit_at_target).
  * Single entry: hard stop at the previous channel low (pre-rise low).
  * Martingale: stop when price exits the channel downward.

Risk management:
  * Require >= min_risk_reward (default 3:1): upside to VAH vs downside to stop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.features.volume_profile import calculate_fixed_range_volume_profile

EPS = 1e-9


@dataclass
class PumpTarget:
    vah: float
    val: float
    poc: float
    ok: bool
    reason: str = ""


def compute_pump_target(
    daily_df: pd.DataFrame,
    prior_high_time: Any,
    current_time: Any,
    vp_cfg: dict[str, Any] | None = None,
) -> PumpTarget:
    """Fixed-range VP from prior-high time to current time; VAH = pump target."""
    vp_cfg = vp_cfg or {}
    res = calculate_fixed_range_volume_profile(
        daily_df,
        prior_high_time,
        current_time,
        bins_count=int(vp_cfg.get("bins_count", 100)),
        value_area_percent=float(vp_cfg.get("value_area_percent", 0.70)),
        hvn_quantile=float(vp_cfg.get("hvn_quantile", 0.80)),
        lvn_quantile=float(vp_cfg.get("lvn_quantile", 0.20)),
    )
    if res is None:
        return PumpTarget(vah=0.0, val=0.0, poc=0.0, ok=False, reason="vp_unavailable")
    return PumpTarget(vah=float(res.vah_price), val=float(res.val_price), poc=float(res.poc_price), ok=True)


def risk_reward(entry: float, target: float, stop: float) -> float:
    """Upside (to target) divided by downside (to stop). Higher = better."""
    upside = max(target - entry, 0.0) / max(entry, EPS)
    downside = max(entry - stop, 0.0) / max(entry, EPS)
    if downside <= EPS:
        return float("inf")
    return upside / downside


def find_prerise_swing_low(
    intraday_df: pd.DataFrame,
    entry_idx: int,
    lookback_bars: int = 30,
    channel_lower: float = 0.0,
) -> float:
    """The previous swing low before the breakout move (TZ single-entry stop).

    NOT the absolute channel floor — the TZ says stop on the *previous low that
    was before the rise*. We take the minimum low over a short lookback window
    ending at the entry bar, floored at the channel lower bound for safety.
    """
    df = intraday_df.sort_values("timestamp").reset_index(drop=True)
    lo = pd.to_numeric(df["low"], errors="coerce").to_numpy()
    start = max(0, entry_idx - lookback_bars)
    window = lo[start : entry_idx + 1]
    if len(window) == 0:
        return float(channel_lower)
    swing_low = float(window.min())
    # Don't let the stop fall below the channel floor (that's the hard invalidation).
    if channel_lower > 0:
        swing_low = max(swing_low, channel_lower)
    return swing_low


@dataclass
class ExitConfig:
    target_mode: str = "vah"
    exit_at_target: bool = True
    trailing_enabled: bool = True
    trailing_distance_pct: float = 0.05
    trailing_atr_adaptive: bool = True
    stop_single: str = "prev_channel_low"
    stop_martingale: str = "channel_break_down"
    # Exit policy: how we balance the tight trail vs running to the VAH target.
    #   "trailing"     -> single trailing stop (baseline; profit lives on trail)
    #   "partial_vah"  -> book `partial_fraction` of the position on the trailing
    #                     stop, ride the remainder to VAH (or its own wider trail)
    #   (ATR-adaptive trail is orthogonal: set trailing_atr_adaptive=True and the
    #    pipeline passes an ATR-derived distance per symbol.)
    policy: str = "trailing"
    partial_fraction: float = 0.5          # fraction booked on the first trail hit
    runner_trail_pct: float = 0.15         # wider trail for the VAH runner leg

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> "ExitConfig":
        tr = dict(cfg.get("trailing", {}))
        return cls(
            target_mode=str(cfg.get("target_mode", "vah")),
            exit_at_target=bool(cfg.get("exit_at_target", True)),
            trailing_enabled=bool(tr.get("enabled", True)),
            trailing_distance_pct=float(tr.get("distance_pct", 0.05)),
            trailing_atr_adaptive=bool(tr.get("atr_adaptive", True)),
            stop_single=str(cfg.get("stop_single", "prev_channel_low")),
            stop_martingale=str(cfg.get("stop_martingale", "channel_break_down")),
            policy=str(cfg.get("policy", "trailing")),
            partial_fraction=float(cfg.get("partial_fraction", 0.5)),
            runner_trail_pct=float(cfg.get("runner_trail_pct", 0.15)),
        )


def simulate_exit(
    intraday_df: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    target_vah: float,
    stop_price: float,
    channel_lower: float,
    mode: str,  # "single" | "martingale"
    exit_cfg: ExitConfig,
    trailing_distance_pct: float | None = None,
) -> dict[str, Any]:
    """Walk intraday bars from entry forward and resolve the exit.

    Priority each bar (intrabar, conservative):
      1. stop / channel-break-down hit (low <= stop)  -> exit at stop
      2. trailing stop hit (low <= trail)              -> exit at trail
      3. VAH target reached (high >= vah)              -> exit at vah
    Falls back to last close (market_close) if none triggers.
    """
    df = intraday_df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    if entry_idx >= n - 1:
        return {"outcome": "no_post_entry_bars", "exit_price": entry_price, "pnl_pct": 0.0}

    dist = trailing_distance_pct if trailing_distance_pct is not None else exit_cfg.trailing_distance_pct
    high_since = entry_price
    trail_price = 0.0
    trailing_active = False
    # partial_vah policy state
    booked = 0.0           # PnL fraction already realized on the partial leg
    booked_frac = 0.0      # position fraction already closed
    partial_booked = False
    runner_trail = 0.0
    runner_done = False

    h = pd.to_numeric(df["high"], errors="coerce").to_numpy()
    l = pd.to_numeric(df["low"], errors="coerce").to_numpy()
    c = pd.to_numeric(df["close"], errors="coerce").to_numpy()
    tcol = df["timestamp"]

    for i in range(entry_idx + 1, n):
        high_since = max(high_since, float(h[i]))

        # 1. Hard stop / channel-break-down. (applies to full remaining size)
        eff_stop = stop_price if mode == "single" else channel_lower
        if eff_stop > 0 and l[i] <= eff_stop:
            return _finish(
                booked, booked_frac, eff_stop, entry_price,
                "stop_loss" if mode == "single" else "channel_break_down",
                tcol, i,
            )

        # 2. Trailing stop (owns the position until a partial is booked).
        if exit_cfg.trailing_enabled and not partial_booked:
            if not trailing_active and high_since >= entry_price * (1.0 + dist):
                trailing_active = True
            if trailing_active:
                trail_price = max(trail_price, high_since * (1.0 - dist))
                if l[i] <= trail_price:
                    if exit_cfg.policy == "partial_vah" and not partial_booked:
                        # Book `partial_fraction` here; ride the remainder to VAH
                        # under a wider runner trail. The runner re-arms from the
                        # current high so it doesn't insta-stop on this same bar.
                        pf = max(0.0, min(1.0, exit_cfg.partial_fraction))
                        booked += pf * (float(trail_price) - entry_price) / max(entry_price, EPS)
                        booked_frac += pf
                        partial_booked = True
                        rdist = max(exit_cfg.runner_trail_pct, dist)
                        runner_trail = high_since * (1.0 - rdist)
                        continue
                    return _finish(booked, booked_frac, trail_price, entry_price,
                                   "trailing_stop", tcol, i)

        # 2b. Runner leg (partial_vah only): wider trail on the remainder.
        if partial_booked and not runner_done:
            runner_trail = max(runner_trail, high_since * (1.0 - max(exit_cfg.runner_trail_pct, dist)))
            if l[i] <= runner_trail:
                return _finish(booked, booked_frac, runner_trail, entry_price,
                               "partial_then_trail", tcol, i)

        # 3. VAH target.
        if exit_cfg.exit_at_target and target_vah > 0 and h[i] >= target_vah:
            outcome = "partial_then_vah" if partial_booked else "target_vah"
            return _finish(booked, booked_frac, target_vah, entry_price, outcome, tcol, i)

    # Fallback: market close at last bar (remaining size).
    last = n - 1
    outcome = "partial_then_close" if partial_booked else "market_close"
    return _finish(booked, booked_frac, float(c[last]), entry_price, outcome, tcol, last)


def _finish(booked, booked_frac, exit_price, entry_price, outcome, tcol, i):
    """Blend already-booked partial PnL with the remaining-size exit."""
    rem = max(0.0, 1.0 - booked_frac)
    leg = rem * (float(exit_price) - entry_price) / max(entry_price, EPS)
    return {
        "outcome": outcome,
        "exit_price": float(exit_price),
        "exit_time": pd.Timestamp(tcol.iloc[i]).to_pydatetime(),
        "exit_idx": int(i),
        "pnl_pct": float(booked + leg),
    }
