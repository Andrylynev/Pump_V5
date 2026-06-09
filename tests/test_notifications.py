"""Tests for the 5 TZ notification types + dispatcher dedup."""
from __future__ import annotations

from app.integrations.notifications import (
    ALL_EVENTS,
    BREAKOUT_CONFIRMED,
    FORMATION_FOUND,
    PRICE_NEAR_UPPER,
    TRADE_CLOSED,
    TRADE_OPENED,
    NotificationDispatcher,
    build_dispatcher,
    format_event,
)


def test_all_five_event_types_format_nonempty():
    payloads = {
        FORMATION_FOUND: {"symbol": "BTCUSDT", "trend": "sideways", "channel_width": 0.25,
                          "score": 4.5, "spark": 3, "twix": 3, "lower_bound": 90, "upper_bound": 110,
                          "acc_start": "2024-01-01", "acc_end": "2024-02-15"},
        PRICE_NEAR_UPPER: {"symbol": "BTCUSDT", "price": 108, "upper_bound": 110, "distance_pct": 0.018},
        BREAKOUT_CONFIRMED: {"symbol": "BTCUSDT", "close": 111, "upper_bound": 110, "branch": "240"},
        TRADE_OPENED: {"symbol": "BTCUSDT", "mode": "single", "entry_price": 111, "vah": 140,
                       "stop": 105, "rr": 3.2, "size_usd": 1000},
        TRADE_CLOSED: {"symbol": "BTCUSDT", "outcome": "target_vah", "exit_price": 140,
                       "pnl_pct": 0.26, "pnl_usd": 260},
    }
    for ev in ALL_EVENTS:
        msg = format_event(ev, payloads[ev])
        assert isinstance(msg, str) and len(msg) > 10
        assert "BTCUSDT" in msg


def test_dispatcher_dedups_same_event():
    d = NotificationDispatcher(notifier=None, enabled=False)
    p = {"symbol": "X", "case_id": "X_2024", "trend": "sideways", "channel_width": 0.2,
         "score": 4, "spark": 2, "twix": 2, "lower_bound": 1, "upper_bound": 2,
         "acc_start": "a", "acc_end": "b"}
    first = d.notify(FORMATION_FOUND, p)
    second = d.notify(FORMATION_FOUND, p)
    assert first is not None
    assert second is None  # deduped


def test_dispatcher_returns_text_without_notifier():
    d = NotificationDispatcher(notifier=None, enabled=False)
    msg = d.notify(TRADE_CLOSED, {"symbol": "Y", "outcome": "trailing_stop",
                                  "exit_price": 5, "pnl_pct": -0.03}, dedup=False)
    assert "Y" in msg and "trailing_stop" in msg


def test_build_dispatcher_disabled_when_no_token():
    d = build_dispatcher({"enabled": True, "bot_token": "", "chat_id": ""})
    assert d.notifier is None
    assert d.enabled is False
