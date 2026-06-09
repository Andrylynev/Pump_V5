"""Chart builders for the Pump V5 UI — testable, framework-agnostic.

Produces a Plotly figure of a coin's candles with the full formation overlaid:
channel bounds, spark/twix markers, entry, VAH target and stop. The Streamlit
app imports build_formation_figure; tests import it directly (no Streamlit needed).
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

from app.detectors.methodic_detector import build_channel
from app.detectors.methodic_detector import MethodicDetector


def build_formation_figure(
    daily_df: pd.DataFrame,
    detector_cfg: dict[str, Any],
    acc_start: Any,
    acc_end: Any,
    entry_price: float | None = None,
    vah: float | None = None,
    stop: float | None = None,
    title: str = "",
) -> go.Figure:
    """Render candles for the window [acc_start .. last] with formation overlay."""
    df = daily_df.sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    a0 = pd.Timestamp(acc_start)
    a0 = a0.tz_localize("UTC") if a0.tzinfo is None else a0.tz_convert("UTC")
    a1 = pd.Timestamp(acc_end)
    a1 = a1.tz_localize("UTC") if a1.tzinfo is None else a1.tz_convert("UTC")

    win = df[(ts >= a0)].copy().reset_index(drop=True)
    acc = df[(ts >= a0) & (ts <= a1)].copy().reset_index(drop=True)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=win["timestamp"], open=win["open"], high=win["high"],
        low=win["low"], close=win["close"], name="OHLC",
    ))

    # Channel bounds over the accumulation window.
    if len(acc) >= 2:
        ch = build_channel(acc, downtrend_slope_threshold=float(detector_cfg.get("downtrend_slope_threshold", -0.0005)))
        fig.add_trace(go.Scatter(x=acc["timestamp"], y=ch.upper_line, name="Верхняя граница",
                                 line=dict(color="#f59e0b", dash="dash")))
        fig.add_trace(go.Scatter(x=acc["timestamp"], y=ch.lower_line, name="Нижняя граница",
                                 line=dict(color="#60a5fa", dash="dash")))

        # Spark / twix markers from the detector precompute.
        det = MethodicDetector(detector_cfg=detector_cfg)
        marks = det._precompute_marks(acc)
        sp = acc[marks["is_spark"]]
        tw = acc[marks["is_twix"]]
        if len(sp):
            fig.add_trace(go.Scatter(x=sp["timestamp"], y=sp["high"] * 1.01, mode="markers",
                                     name="Spark", marker=dict(color="#22c55e", size=10, symbol="triangle-up")))
        if len(tw):
            fig.add_trace(go.Scatter(x=tw["timestamp"], y=tw["high"] * 1.01, mode="markers",
                                     name="Twix", marker=dict(color="#e879f9", size=9, symbol="diamond")))

    # Trade levels.
    if entry_price:
        fig.add_hline(y=float(entry_price), line=dict(color="#e5e7eb", dash="dot"), annotation_text="Вход")
    if vah:
        fig.add_hline(y=float(vah), line=dict(color="#22c55e"), annotation_text="Цель VAH")
    if stop:
        fig.add_hline(y=float(stop), line=dict(color="#ef4444"), annotation_text="Стоп")

    fig.update_layout(
        title=title or "Формация",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=600,
    )
    return fig
