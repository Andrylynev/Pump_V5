"""Methodic accumulation detector (Pump V5) — strict to the TZ canon.

Differences from V4 (which drifted into statistical proxies):
  * Channel = a real price RANGE, not a regression trend line.
      - sideways  -> FLAT horizontal bounds (max-high / min-low of the window)
      - downtrend -> SLOPED bounds (linear fit, bands at the extreme residuals)
  * Spark = a GREEN candle with a small body (low price impact) whose volume
    exceeds EACH of the up-to-4 immediately preceding RED candles individually.
    Worth 1.0 accumulation point.
  * Twix = a spark that has an adjacent (glued) green candle. We score the one
    anomalous candle; the glued candle does not add to the score. For the volume
    reading of the pair we take the larger of the two. Worth 0.5 (weaker signal).
  * Manipulation likely when total accumulation score >= 3.0.

All scoring is done on 1D candles. Breakout/entry is handled separately on 4h/1h.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from app.contracts import FormationCandidate

EPS = 1e-9


@dataclass
class SparkTwixResult:
    spark_count: int
    twix_count: int
    score: float
    spark_idx: list[int]
    twix_idx: list[int]


def _is_green(open_: np.ndarray, close: np.ndarray) -> np.ndarray:
    return close > open_


def _is_red(open_: np.ndarray, close: np.ndarray) -> np.ndarray:
    return close < open_


def detect_sparks_and_twix(
    w: pd.DataFrame,
    max_red_lookback: int = 4,
    small_body_max_ratio: float = 0.35,
    spark_weight: float = 1.0,
    twix_weight: float = 0.5,
) -> SparkTwixResult:
    """Find sparks and twix in a window of 1D candles.

    A candle i qualifies as an *anomalous green* (spark base) when:
      - it is green (close > open),
      - its body is small: |close-open| / (high-low) < small_body_max_ratio,
      - there is at least one red candle in the immediately preceding run
        (up to ``max_red_lookback`` consecutive reds), and
      - its volume is strictly greater than EACH of those preceding red candles.

    The anomalous green is a TWIX (0.5) when it is glued to an adjacent green
    candle (i-1 green or i+1 green); otherwise it is a SPARK (1.0).
    Each candle contributes at most once.
    """
    open_ = w["open"].to_numpy(dtype=float)
    close = w["close"].to_numpy(dtype=float)
    high = w["high"].to_numpy(dtype=float)
    low = w["low"].to_numpy(dtype=float)
    volume = w["volume"].to_numpy(dtype=float)
    n = len(w)

    green = _is_green(open_, close)
    red = _is_red(open_, close)
    rng = np.maximum(high - low, EPS)
    body = np.abs(close - open_)
    small_body = (body / rng) < small_body_max_ratio

    spark_idx: list[int] = []
    twix_idx: list[int] = []

    for i in range(n):
        if not (green[i] and small_body[i]):
            continue

        # Collect the immediately preceding run of consecutive RED candles, max N.
        red_vols: list[float] = []
        j = i - 1
        while j >= 0 and red[j] and len(red_vols) < max_red_lookback:
            red_vols.append(volume[j])
            j -= 1

        if not red_vols:
            # No preceding red series to overcome -> not a methodic mark.
            continue

        # Volume must exceed EACH preceding red individually.
        if not all(volume[i] > rv for rv in red_vols):
            continue

        # Glued green neighbour => twix (weaker). Else => spark.
        glued = (i - 1 >= 0 and green[i - 1]) or (i + 1 < n and green[i + 1])
        if glued:
            twix_idx.append(i)
        else:
            spark_idx.append(i)

    spark_count = len(spark_idx)
    twix_count = len(twix_idx)
    score = spark_count * spark_weight + twix_count * twix_weight
    return SparkTwixResult(
        spark_count=spark_count,
        twix_count=twix_count,
        score=float(score),
        spark_idx=spark_idx,
        twix_idx=twix_idx,
    )


@dataclass
class ChannelResult:
    trend: str  # "sideways" | "downtrend" | "uptrend"
    upper_bound: float  # bound at the last bar
    lower_bound: float  # bound at the last bar
    upper_line: np.ndarray  # per-bar upper bound
    lower_line: np.ndarray  # per-bar lower bound
    channel_width: float  # (max_high - min_low) / min_low
    slope_norm: float  # close slope normalised by mean price
    upper_touches: int = 0  # how many bars touch the upper bound (real resistance)
    lower_touches: int = 0  # how many bars touch the lower bound (real support)


def _count_touches(values: np.ndarray, level: np.ndarray, tol: float) -> int:
    """Count bars whose value comes within ``tol`` (fraction) of the level line.

    A real channel border is one that price tested MORE THAN ONCE. A border
    that is a single wick spike (touched once, then never revisited) is "висит
    в воздухе" — not a tradable support/resistance.
    """
    ref = np.maximum(np.abs(level), EPS)
    return int(np.sum(np.abs(values - level) / ref <= tol))


def build_channel(
    w: pd.DataFrame,
    downtrend_slope_threshold: float = -0.0005,
    uptrend_slope_threshold: float = 0.0010,
    touch_tol: float = 0.02,
) -> ChannelResult:
    """Classify the window trend and build the channel bounds.

    Trend (by normalised close slope):
      * slope <= downtrend_slope_threshold (< 0)  -> "downtrend" (sloped bounds)
      * slope >= uptrend_slope_threshold   (> 0)  -> "uptrend"   (EXCLUDED — a
        rising market is not accumulation; the operator is already marking up)
      * otherwise                                 -> "sideways"  (flat bounds)

    Sideways  -> FLAT bounds at absolute max-high / min-low of the window.
    Downtrend/Uptrend -> SLOPED bounds: linear regression of close, with
                 upper/lower offset to the extreme high/low residuals.

    Also counts how many bars TOUCH each bound (within ``touch_tol``) so callers
    can reject "hanging" channels whose border was a single wick.
    """
    close = w["close"].to_numpy(dtype=float)
    high = w["high"].to_numpy(dtype=float)
    low = w["low"].to_numpy(dtype=float)
    n = len(w)

    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, close, 1)
    mean_price = float(np.mean(close)) or EPS
    slope_norm = float(slope / mean_price)

    max_high = float(high.max())
    min_low = float(low.min())
    channel_width = (max_high - min_low) / max(min_low, EPS)

    if slope_norm <= downtrend_slope_threshold:
        trend = "downtrend"
        trend_line = slope * x + intercept
        upper_off = float(np.max(high - trend_line))
        lower_off = float(np.min(low - trend_line))
        upper_line = trend_line + upper_off
        lower_line = trend_line + lower_off
    elif slope_norm >= uptrend_slope_threshold:
        # Rising market — NOT accumulation. Bounds still computed for diagnostics,
        # but detect_accumulations will reject this trend outright.
        trend = "uptrend"
        trend_line = slope * x + intercept
        upper_off = float(np.max(high - trend_line))
        lower_off = float(np.min(low - trend_line))
        upper_line = trend_line + upper_off
        lower_line = trend_line + lower_off
    else:
        trend = "sideways"
        upper_line = np.full(n, max_high, dtype=float)
        lower_line = np.full(n, min_low, dtype=float)

    upper_touches = _count_touches(high, upper_line, touch_tol)
    lower_touches = _count_touches(low, lower_line, touch_tol)

    return ChannelResult(
        trend=trend,
        upper_bound=float(upper_line[-1]),
        lower_bound=float(lower_line[-1]),
        upper_line=upper_line,
        lower_line=lower_line,
        channel_width=float(channel_width),
        slope_norm=slope_norm,
        upper_touches=upper_touches,
        lower_touches=lower_touches,
    )


@dataclass
class MethodicDetector:
    detector_cfg: dict[str, Any]

    def _precompute_marks(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Precompute per-bar spark/twix base flags once for the whole symbol.

        The spark test for bar i depends only on bar i and its immediately
        preceding run of <= max_red_lookback red candles — it does NOT depend on
        the scan window, except that the preceding reds must exist in the data.
        So we can compute it globally and just count within each window.
        """
        cfg = self.detector_cfg
        spark_cfg = dict(cfg.get("spark", {}))
        max_red = int(spark_cfg.get("max_red_lookback", 4))
        small_body_max = float(spark_cfg.get("small_body_max_ratio", 0.35))

        open_ = df["open"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        volume = df["volume"].to_numpy(dtype=float)
        n = len(df)

        green = close > open_
        red = close < open_
        rng = np.maximum(high - low, EPS)
        small_body = (np.abs(close - open_) / rng) < small_body_max

        is_spark = np.zeros(n, dtype=bool)
        is_twix = np.zeros(n, dtype=bool)
        for i in range(n):
            if not (green[i] and small_body[i]):
                continue
            red_vols: list[float] = []
            j = i - 1
            while j >= 0 and red[j] and len(red_vols) < max_red:
                red_vols.append(volume[j])
                j -= 1
            if not red_vols:
                continue
            if not all(volume[i] > rv for rv in red_vols):
                continue
            glued = (i - 1 >= 0 and green[i - 1]) or (i + 1 < n and green[i + 1])
            if glued:
                is_twix[i] = True
            else:
                is_spark[i] = True

        return {
            "is_spark": is_spark,
            "is_twix": is_twix,
            "spark_cum": np.concatenate([[0], np.cumsum(is_spark)]),
            "twix_cum": np.concatenate([[0], np.cumsum(is_twix)]),
        }

    def _preceding_downtrend(self, df: pd.DataFrame, start_idx: int, lookback: int = 30) -> bool:
        """Was the run immediately BEFORE this window a downtrend?

        Per TZ: a sideways formation that broke a prior downtrend is a NEW, and
        STRONGER, formation (the operator who accumulated through the downtrend
        AND the range must push price higher to profit).
        """
        pre_start = max(0, start_idx - lookback)
        if start_idx - pre_start < 5:
            return False
        pre = df.iloc[pre_start:start_idx]
        if len(pre) < 5:
            return False
        ch = build_channel(pre, downtrend_slope_threshold=float(self.detector_cfg.get("downtrend_slope_threshold", -0.0005)))
        return ch.trend == "downtrend"

    def _max_run_up_pct(self, w: pd.DataFrame) -> float:
        """Largest sustained up-move INSIDE the accumulation window.

        A genuine accumulation range only has small, single anomalous green
        candles that DON'T move price much. If a multi-bar rally already lifted
        price by a large fraction inside the window, the pump has already
        happened (INUSDT 15-Apr, LPTUSDT +37% 10-Apr, ALTUSDT +50% 22-May) — the
        big player has booked profit and there is nothing left to ride. We scan
        every bar pair (i<j) for the max close-to-close gain over a bounded span.
        """
        close = w["close"].to_numpy(dtype=float)
        n = len(close)
        if n < 2:
            return 0.0
        # running min of close up to each bar -> max gain from any earlier trough
        run_min = np.minimum.accumulate(close)
        gains = close / np.maximum(run_min, EPS) - 1.0
        return float(np.max(gains))

    def _broke_down_at_end(
        self, w: pd.DataFrame, lower_line: np.ndarray, tol: float, consec: int
    ) -> bool:
        """Did the channel break DOWN near the end of the window?

        Per TZ a single prick below the border that the operator returns is OK,
        but ``consec`` consecutive daily CLOSES below the lower bound = the floor
        is lost, the range is broken, the formation is dead (LTC 2-Jun, ATOM
        27-May, EGLD 2-Jun, MNT -24% 1-6-Jun, COTI late-May).
        """
        close = w["close"].to_numpy(dtype=float)
        n = len(close)
        if n == 0 or consec <= 0:
            return False
        floor = lower_line * (1.0 - tol)
        below = close < floor
        # count trailing consecutive closes below the floor
        run = 0
        for k in range(n - 1, -1, -1):
            if below[k]:
                run += 1
            else:
                break
        return run >= consec

    def _window_metrics(self, w: pd.DataFrame, marks: dict[str, np.ndarray], start_idx: int, end_idx: int, df: pd.DataFrame | None = None) -> dict[str, Any]:
        cfg = self.detector_cfg
        spark_weight = float(cfg.get("spark_weight", 1.0))
        twix_weight = float(cfg.get("twix_weight", 0.5))

        channel = build_channel(
            w,
            downtrend_slope_threshold=float(cfg.get("downtrend_slope_threshold", -0.0005)),
            uptrend_slope_threshold=float(cfg.get("uptrend_slope_threshold", 0.0010)),
            touch_tol=float(cfg.get("touch_tol", 0.02)),
        )

        # Count marks within [start_idx, end_idx] via cumulative sums (O(1)).
        spark_count = int(marks["spark_cum"][end_idx + 1] - marks["spark_cum"][start_idx])
        twix_count = int(marks["twix_cum"][end_idx + 1] - marks["twix_cum"][start_idx])
        score = spark_count * spark_weight + twix_count * twix_weight

        # Stronger-signal case: sideways range that broke a prior downtrend.
        downtrend_to_range = False
        if df is not None and channel.trend == "sideways" and bool(cfg.get("downtrend_to_range_bonus", True)):
            lookback = int(cfg.get("downtrend_lookback_days", 30))
            downtrend_to_range = self._preceding_downtrend(df, start_idx, lookback)

        prior_high = float(w["high"].iloc[0])

        # NEW guards (TZ canon, calibrated on Никита's 2026-06-10 review):
        #  - max_run_up: a completed pump already inside the window?
        #  - broke_down: the floor already lost at the window end?
        #  - tail_downtrend: is the recent tail of the window itself collapsing
        #    (a range that's breaking DOWN right now, even if the greedy wide
        #    bounds swallowed the dump — GLM 03-27→today w=0.48)?
        max_run_up = self._max_run_up_pct(w)
        broke_down = self._broke_down_at_end(
            w,
            channel.lower_line,
            tol=float(cfg.get("breakdown_tol", 0.0)),
            consec=int(cfg.get("breakdown_consec_closes", 3)),
        )
        tail_n = int(cfg.get("tail_trend_bars", 10))
        tail_downtrend = False
        if len(w) >= tail_n:
            tail_ch = build_channel(
                w.iloc[-tail_n:],
                downtrend_slope_threshold=float(cfg.get("tail_downtrend_slope", -0.005)),
                uptrend_slope_threshold=float(cfg.get("uptrend_slope_threshold", 0.0010)),
            )
            tail_downtrend = tail_ch.trend == "downtrend"

        return {
            "trend": channel.trend,
            "upper_bound": channel.upper_bound,
            "lower_bound": channel.lower_bound,
            "channel_width": channel.channel_width,
            "slope_norm": channel.slope_norm,
            "upper_touches": channel.upper_touches,
            "lower_touches": channel.lower_touches,
            "spark_count": spark_count,
            "twix_count": twix_count,
            "accumulation_score": float(score),
            "downtrend_to_range": bool(downtrend_to_range),
            "prior_high": prior_high,
            "max_run_up_pct": float(max_run_up),
            "broke_down_at_end": bool(broke_down),
            "tail_downtrend": bool(tail_downtrend),
            "n_bars": int(len(w)),
        }

    def detect_accumulations(self, symbol: str, daily_df: pd.DataFrame) -> list[FormationCandidate]:
        cfg = self.detector_cfg
        min_days = int(cfg["min_range_days"])
        max_days = int(cfg["max_range_days"])
        scan_step = int(cfg.get("scan_step_days", 3))
        window_step = int(cfg.get("window_step_days", 10))
        max_width = float(cfg["max_channel_width"])
        min_score = float(cfg["min_accumulation_score"])

        if daily_df.empty or len(daily_df) < min_days:
            return []
        df = daily_df.sort_values("timestamp").reset_index(drop=True)

        window_lengths = list(range(min_days, max_days + 1, max(1, window_step)))
        marks = self._precompute_marks(df)
        raw: list[FormationCandidate] = []
        for end_idx in range(min_days - 1, len(df), max(1, scan_step)):
            # Cheap pre-filter: skip windows that cannot reach the score threshold.
            for wlen in window_lengths:
                if wlen > end_idx + 1:
                    continue
                start_idx = end_idx - wlen + 1
                if start_idx < 0:
                    continue
                # O(1) score check first — skip expensive channel build if too weak.
                sc = float(
                    int(marks["spark_cum"][end_idx + 1] - marks["spark_cum"][start_idx]) * float(cfg.get("spark_weight", 1.0))
                    + int(marks["twix_cum"][end_idx + 1] - marks["twix_cum"][start_idx]) * float(cfg.get("twix_weight", 0.5))
                )
                if sc < min_score:
                    continue
                w = df.iloc[start_idx : end_idx + 1]
                m = self._window_metrics(w, marks, start_idx, end_idx, df=df)

                # Range constraint: up to ~50% (channel_width <= max_width).
                if m["channel_width"] > max_width:
                    continue
                # Trend must be sideways or downtrend. An UPTREND is markup, not
                # accumulation — exclude it (ATOM, COTI, ARKM were rising/broke up).
                if m["trend"] not in ("sideways", "downtrend"):
                    continue
                # Accumulation score threshold (manipulation likely at >= 3).
                if m["accumulation_score"] < min_score:
                    continue
                # GUARD 1 — already pumped: a completed rally inside the window
                # means the move is gone (INUSDT, LPTUSDT, ALTUSDT, FOLKS, PUNDIX).
                max_run_up_limit = float(cfg.get("max_intra_window_run_up", 0.30))
                if m["max_run_up_pct"] > max_run_up_limit:
                    continue
                # GUARD 2 — channel broken DOWN at the end: floor lost, dead
                # formation (LTC, ATOM, EGLD, MNT, COTI, GLM).
                if m["broke_down_at_end"]:
                    continue
                # GUARD 2b — a recent tail that is itself STEEPLY downtrending
                # means the range is collapsing right now (markdown, not
                # accumulation). The steep tail threshold (tail_downtrend_slope,
                # ~10x the gentle formation downtrend threshold) lets a genuine
                # slow downtrend accumulation pass while killing freefalls — GLM
                # w=0.48 (tail slope -0.027), EGLD/COTI dumps.
                if m.get("tail_downtrend"):
                    continue
                # GUARD 3 — bounds must be REAL (tested more than once), not a
                # single wick "hanging in the air" (PUNDIX, EGLD floor 3.61).
                min_touches = int(cfg.get("min_bound_touches", 2))
                if m["lower_touches"] < min_touches or m["upper_touches"] < min_touches:
                    continue

                case_id = f"{symbol}_{w['timestamp'].iloc[0].date()}_{w['timestamp'].iloc[-1].date()}"
                raw.append(
                    FormationCandidate(
                        case_id=case_id,
                        symbol=symbol,
                        timeframe="1D",
                        branch="1D",
                        accumulation_start=w["timestamp"].iloc[0].to_pydatetime(),
                        accumulation_end=w["timestamp"].iloc[-1].to_pydatetime(),
                        entry_time=None,
                        upper_bound=m["upper_bound"],
                        lower_bound=m["lower_bound"],
                        score=m["accumulation_score"],
                        spark_count=m["spark_count"],
                        twix_count=m["twix_count"],
                        volume_score=m["accumulation_score"],
                        channel_width=m["channel_width"],
                        diagnostics=m,
                    )
                )

        if not raw:
            return []

        # Rank: downtrend->range formations first (stronger signal per TZ), then
        # higher score, then narrower channel (strongest pumps from narrow channels).
        raw.sort(
            key=lambda c: (
                1 if c.diagnostics.get("downtrend_to_range") else 0,
                c.score,
                -c.channel_width,
            ),
            reverse=True,
        )
        selected: list[FormationCandidate] = []
        for cand in raw:
            overlap = False
            for ex in selected:
                a1, a2 = cand.accumulation_start, cand.accumulation_end
                b1, b2 = ex.accumulation_start, ex.accumulation_end
                inter = max(timedelta(0), min(a2, b2) - max(a1, b1))
                union = max(a2, b2) - min(a1, b1)
                iou = inter.total_seconds() / max(union.total_seconds(), EPS)
                if iou > 0.60:
                    overlap = True
                    break
            if not overlap:
                selected.append(cand)
            if len(selected) >= 24:
                break
        return selected
