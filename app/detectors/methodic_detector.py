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
    trend: str  # "sideways" | "downtrend"
    upper_bound: float  # bound at the last bar
    lower_bound: float  # bound at the last bar
    upper_line: np.ndarray  # per-bar upper bound
    lower_line: np.ndarray  # per-bar lower bound
    channel_width: float  # (max_high - min_low) / min_low
    slope_norm: float  # close slope normalised by mean price


def build_channel(
    w: pd.DataFrame,
    downtrend_slope_threshold: float = -0.0005,
) -> ChannelResult:
    """Classify the window trend and build the channel bounds.

    Sideways  -> FLAT bounds at absolute max-high / min-low of the window.
    Downtrend -> SLOPED bounds: linear regression of close, with upper/lower
                 offset to the extreme high/low residuals so the channel hugs
                 the actual price envelope.
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
    else:
        trend = "sideways"
        upper_line = np.full(n, max_high, dtype=float)
        lower_line = np.full(n, min_low, dtype=float)

    return ChannelResult(
        trend=trend,
        upper_bound=float(upper_line[-1]),
        lower_bound=float(lower_line[-1]),
        upper_line=upper_line,
        lower_line=lower_line,
        channel_width=float(channel_width),
        slope_norm=slope_norm,
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

    def _window_metrics(self, w: pd.DataFrame, marks: dict[str, np.ndarray], start_idx: int, end_idx: int, df: pd.DataFrame | None = None) -> dict[str, Any]:
        cfg = self.detector_cfg
        spark_weight = float(cfg.get("spark_weight", 1.0))
        twix_weight = float(cfg.get("twix_weight", 0.5))

        channel = build_channel(
            w,
            downtrend_slope_threshold=float(cfg.get("downtrend_slope_threshold", -0.0005)),
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

        return {
            "trend": channel.trend,
            "upper_bound": channel.upper_bound,
            "lower_bound": channel.lower_bound,
            "channel_width": channel.channel_width,
            "slope_norm": channel.slope_norm,
            "spark_count": spark_count,
            "twix_count": twix_count,
            "accumulation_score": float(score),
            "downtrend_to_range": bool(downtrend_to_range),
            "prior_high": prior_high,
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
                # Must be sideways or downtrend (uptrend excluded by construction:
                # sideways uses flat bounds; downtrend has negative slope).
                if m["trend"] not in ("sideways", "downtrend"):
                    continue
                # Accumulation score threshold (manipulation likely at >= 3).
                if m["accumulation_score"] < min_score:
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
