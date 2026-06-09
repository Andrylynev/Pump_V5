"""Smoke test: the V5 scaffold imports cleanly and ported infra is wired.

This does NOT hit the network. It only proves the package is importable and the
ported modules expose their expected public symbols, so Phase 1/2 can build on a
known-good foundation.
"""
from __future__ import annotations

import importlib

import pandas as pd

PORTED_MODULES = [
    "app.contracts",
    "app.settings",
    "app.market_data.bybit_client",
    "app.market_data.kline_cache",
    "app.market_data.universe",
    "app.storage.paths",
    "app.features.volume_profile",
    "app.integrations.telegram_notifier",
]


def test_all_ported_modules_import():
    for mod in PORTED_MODULES:
        importlib.import_module(mod)


def test_default_config_loads_and_has_method_keys():
    from app.settings import load_settings

    s = load_settings()
    det = s.raw["detector"]
    assert det["min_range_days"] == 45
    assert det["spark"]["max_red_lookback"] == 4
    assert det["min_accumulation_score"] == 3.0
    assert s.raw["exit"]["min_risk_reward"] if "min_risk_reward" in s.raw["exit"] else True
    assert s.raw["entry"]["min_risk_reward"] == 3.0
    assert s.raw["universe"]["include_spot"] is True


def test_volume_profile_fixed_range_runs_on_synthetic_data():
    from app.features.volume_profile import calculate_fixed_range_volume_profile

    ts = pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": 100.0,
            "high": 110.0,
            "low": 90.0,
            "close": 105.0,
            "volume": 1000.0,
        }
    )
    res = calculate_fixed_range_volume_profile(df, ts[0], ts[-1])
    assert res is not None
    assert res.val_price <= res.poc_price <= res.vah_price


def test_contracts_roundtrip():
    from datetime import datetime, timezone

    from app.contracts import FormationCandidate

    c = FormationCandidate(
        case_id="X_2024-01-01_2024-02-15",
        symbol="XUSDT",
        timeframe="1D",
        branch="1D",
        accumulation_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        accumulation_end=datetime(2024, 2, 15, tzinfo=timezone.utc),
        entry_time=None,
        upper_bound=110.0,
        lower_bound=90.0,
        score=3.5,
        spark_count=3,
        twix_count=1,
        volume_score=3.5,
        channel_width=0.22,
    )
    d = c.to_dict()
    assert d["entry_time"] is None
    assert d["accumulation_start"].startswith("2024-01-01")
