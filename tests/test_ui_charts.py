"""Test the chart builder produces a figure with formation overlay traces."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.settings import load_settings
from app.ui.charts import build_formation_figure


def _daily():
    n = 70
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(3)
    base = 100 + rng.normal(0, 0.4, n)
    o = base.copy(); c = base.copy(); h = base + 2; l = base - 2
    vol = np.full(n, 100.0)
    for k in (10, 20, 30):
        o[k-1], c[k-1] = 101, 99
        o[k], c[k], h[k], l[k] = 98, 98.3, 99, 97
        vol[k] = 500
    return pd.DataFrame({"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": vol})


def test_build_formation_figure_has_candles_and_overlay():
    df = _daily()
    cfg = load_settings().raw["detector"]
    fig = build_formation_figure(
        df, cfg, acc_start=df["timestamp"].iloc[0], acc_end=df["timestamp"].iloc[44],
        entry_price=105, vah=130, stop=99, title="TEST",
    )
    assert isinstance(fig, go.Figure)
    names = [t.name for t in fig.data]
    assert "OHLC" in names
    assert "Верхняя граница" in names
    assert "Нижняя граница" in names


def test_streamlit_app_imports():
    # Importing the module must not raise (it only runs main() under __main__).
    import importlib

    mod = importlib.import_module("app.ui.streamlit_app")
    assert hasattr(mod, "main")
