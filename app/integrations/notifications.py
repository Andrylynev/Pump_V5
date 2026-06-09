"""Notification events (Pump V5) — the 5 TZ notification types.

ТЗ требует уведомления в Telegram пяти видов:
  1. formation_found       — найдена формация
  2. price_near_upper      — цена подошла к верхней границе
  3. breakout_confirmed    — цена пробила верхнюю границу и закрепилась
  4. trade_opened          — открытие сделки
  5. trade_closed          — закрытие сделки и её PnL

This module formats each event into a Telegram HTML message and dispatches via
the ported TelegramNotifier. Dispatch is dedup-guarded so the IRL scanner does
not re-spam the same event for the same formation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.integrations.telegram_notifier import TelegramNotifier

# Event type constants.
FORMATION_FOUND = "formation_found"
PRICE_NEAR_UPPER = "price_near_upper"
BREAKOUT_CONFIRMED = "breakout_confirmed"
TRADE_OPENED = "trade_opened"
TRADE_CLOSED = "trade_closed"

ALL_EVENTS = (
    FORMATION_FOUND,
    PRICE_NEAR_UPPER,
    BREAKOUT_CONFIRMED,
    TRADE_OPENED,
    TRADE_CLOSED,
)


def _fmt_price(x: float) -> str:
    ax = abs(float(x))
    if ax >= 1:
        return f"{x:,.4f}"
    if ax >= 0.0001:
        return f"{x:.6f}"
    return f"{x:.10f}"


def _pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def format_event(event_type: str, payload: dict[str, Any]) -> str:
    """Return a Telegram-HTML message string for the event."""
    sym = payload.get("symbol", "?")
    market = payload.get("market", "")
    mtag = f" <i>({market})</i>" if market else ""

    if event_type == FORMATION_FOUND:
        return (
            f"🔍 <b>Найдена формация</b> — <b>{sym}</b>{mtag}\n"
            f"Тренд: {payload.get('trend','?')} · ширина канала: {payload.get('channel_width',0):.2%}\n"
            f"Баллы: <b>{payload.get('score',0):.1f}</b> "
            f"(spark {payload.get('spark',0)} · twix {payload.get('twix',0)})\n"
            f"Канал: {_fmt_price(payload.get('lower_bound',0))} — {_fmt_price(payload.get('upper_bound',0))}\n"
            f"Накопление: {payload.get('acc_start','?')} → {payload.get('acc_end','?')}"
        )
    if event_type == PRICE_NEAR_UPPER:
        return (
            f"📈 <b>Цена подошла к верхней границе</b> — <b>{sym}</b>{mtag}\n"
            f"Цена: {_fmt_price(payload.get('price',0))} · граница: {_fmt_price(payload.get('upper_bound',0))}\n"
            f"До границы: {payload.get('distance_pct',0):.2%}"
        )
    if event_type == BREAKOUT_CONFIRMED:
        return (
            f"🚀 <b>Пробой + закрепление телом</b> — <b>{sym}</b>{mtag}\n"
            f"Закрытие: {_fmt_price(payload.get('close',0))} выше границы {_fmt_price(payload.get('upper_bound',0))}\n"
            f"Таймфрейм: {payload.get('branch','?')}"
        )
    if event_type == TRADE_OPENED:
        return (
            f"✅ <b>Открытие сделки</b> — <b>{sym}</b>{mtag}\n"
            f"Режим: <b>{payload.get('mode','?')}</b> · вход: {_fmt_price(payload.get('entry_price',0))}\n"
            f"Цель (VAH): {_fmt_price(payload.get('vah',0))} · стоп: {_fmt_price(payload.get('stop',0))}\n"
            f"RR: <b>{payload.get('rr',0):.2f}</b> · размер: {payload.get('size_usd',0):.2f} USD"
        )
    if event_type == TRADE_CLOSED:
        pnl_usd = payload.get("pnl_usd")
        pnl_usd_s = f" · {pnl_usd:+.2f} USD" if pnl_usd is not None else ""
        return (
            f"🏁 <b>Закрытие сделки</b> — <b>{sym}</b>{mtag}\n"
            f"Исход: {payload.get('outcome','?')} · выход: {_fmt_price(payload.get('exit_price',0))}\n"
            f"PnL: <b>{_pct(payload.get('pnl_pct',0))}</b>{pnl_usd_s}"
        )
    return f"ℹ️ {event_type} — {sym}"


@dataclass
class NotificationDispatcher:
    notifier: TelegramNotifier | None
    enabled: bool = True
    _sent: set[tuple[str, str]] = field(default_factory=set)

    def _key(self, event_type: str, payload: dict[str, Any]) -> tuple[str, str]:
        # Dedup per (case_id or symbol, event_type) so we don't re-spam.
        ident = str(payload.get("case_id") or payload.get("symbol") or "?")
        return (ident, event_type)

    def notify(self, event_type: str, payload: dict[str, Any], dedup: bool = True) -> str | None:
        """Format + dispatch. Returns the message text (also when notifier is None)."""
        msg = format_event(event_type, payload)
        if dedup:
            key = self._key(event_type, payload)
            if key in self._sent:
                return None
            self._sent.add(key)
        if self.enabled and self.notifier is not None:
            self.notifier.send_message(msg)
        return msg


def build_dispatcher(telegram_cfg: dict[str, Any]) -> NotificationDispatcher:
    enabled = bool(telegram_cfg.get("enabled", False))
    token = str(telegram_cfg.get("bot_token", ""))
    chat = str(telegram_cfg.get("chat_id", ""))
    notifier = TelegramNotifier(bot_token=token, chat_id=chat) if (enabled and token and chat) else None
    return NotificationDispatcher(notifier=notifier, enabled=enabled and notifier is not None)
