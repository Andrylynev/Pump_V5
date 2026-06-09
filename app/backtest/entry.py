"""Entry evaluation (Pump V5) — strict to the TZ canon.

After accumulation, we watch the breakout of the upper channel bound on the
intraday timeframes (4h, then 1h). Rules:

  * Enter only after a candle BODY closes above the upper bound (закрепление
    телом свечи выше границы) — not just a wick poke.
  * Skip the entry if price already flew far past the breakout level
    (> skip_if_extended_pct, ~10%, optionally ATR-adaptive) to avoid the pullback.
  * A downward break of the lower bound before any upward confirmation
    invalidates the setup (channel broke down).

This module answers ONLY the entry question. Target/stop/exit live in the
simulator. The detector decides the formation; this decides the trigger.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.contracts import FormationCandidate

EPS = 1e-9


def compute_atr_pct(df: pd.DataFrame, period: int = 14) -> float:
    """ATR as a fraction of price (mean close), from intraday OHLC.

    Used to make the "price flew too high" and trailing distance adaptive to the
    coin's normal move, as the TZ allows.
    """
    if df.empty or len(df) < 2:
        return 0.0
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    n = min(period, len(tr))
    atr = float(np.mean(tr[-n:]))
    mean_price = float(np.mean(close)) or EPS
    return atr / mean_price


@dataclass
class EntryEvaluator:
    entry_cfg: dict[str, Any]

    def evaluate(
        self,
        candidate: FormationCandidate,
        intraday_df: pd.DataFrame,
        branch: str,
    ) -> dict[str, Any]:
        """Return an entry decision dict.

        intraday_df: OHLCV on the breakout timeframe (e.g. "240" or "60"),
        already restricted to bars at/after accumulation_end is fine but not
        required (we filter here).
        """
        if intraday_df.empty:
            return {"entered": False, "reason": "no_data", "branch": branch}

        df = intraday_df.sort_values("timestamp").reset_index(drop=True)
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        acc_end = pd.Timestamp(candidate.accumulation_end)
        acc_end = acc_end.tz_localize("UTC") if acc_end.tzinfo is None else acc_end.tz_convert("UTC")

        after = df[ts >= acc_end].copy().reset_index(drop=True)
        if after.empty:
            return {"entered": False, "reason": "no_post_accumulation_bars", "branch": branch}

        upper = float(candidate.upper_bound)
        lower = float(candidate.lower_bound)

        require_body = bool(self.entry_cfg.get("require_body_close_above", True))
        skip_ext = float(self.entry_cfg.get("skip_if_extended_pct", 0.10))
        atr_adaptive = bool(self.entry_cfg.get("extended_pct_atr_adaptive", True))

        # Adapt the "flew too high" threshold to the coin's normal move.
        if atr_adaptive:
            atr_pct = compute_atr_pct(after)
            # normal move ~ a few ATRs; use max of configured floor and ~3xATR.
            skip_ext = max(skip_ext, 3.0 * atr_pct)

        o = pd.to_numeric(after["open"], errors="coerce").to_numpy()
        h = pd.to_numeric(after["high"], errors="coerce").to_numpy()
        l = pd.to_numeric(after["low"], errors="coerce").to_numpy()
        c = pd.to_numeric(after["close"], errors="coerce").to_numpy()

        for i in range(len(after)):
            # Downward break before any confirmed entry -> invalidate.
            if l[i] < lower:
                return {
                    "entered": False,
                    "reason": "channel_break_down",
                    "breakout_direction": "down",
                    "break_time": pd.Timestamp(after["timestamp"].iloc[i]).to_pydatetime(),
                    "channel_upper": upper,
                    "channel_lower": lower,
                    "branch": branch,
                }

            # Body close above upper bound = закрепление телом.
            body_top = max(o[i], c[i])
            confirmed = (c[i] > upper) if require_body else (h[i] > upper)
            confirmed = confirmed and body_top > upper if require_body else confirmed
            if not confirmed:
                continue

            # Skip if price already flew too far past the breakout level.
            extended = (c[i] - upper) / max(upper, EPS)
            if extended > skip_ext:
                return {
                    "entered": False,
                    "reason": "price_extended_skip",
                    "breakout_direction": "up",
                    "extended_pct": float(extended),
                    "skip_threshold_pct": float(skip_ext),
                    "entry_time_candidate": pd.Timestamp(after["timestamp"].iloc[i]).to_pydatetime(),
                    "channel_upper": upper,
                    "channel_lower": lower,
                    "branch": branch,
                }

            return {
                "entered": True,
                "reason": "breakout_body_confirmed",
                "entry_time": pd.Timestamp(after["timestamp"].iloc[i]).to_pydatetime(),
                "entry_price": float(c[i]),
                "breakout_level": upper,
                "breakout_direction": "up",
                "extended_pct": float(extended),
                "channel_upper": upper,
                "channel_lower": lower,
                "branch": branch,
                "entry_idx": int(i),
            }

        return {
            "entered": False,
            "reason": "no_breakout_confirmed",
            "breakout_direction": "none",
            "channel_upper": upper,
            "channel_lower": lower,
            "branch": branch,
        }
