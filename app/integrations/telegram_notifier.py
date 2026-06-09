from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx


@dataclass
class TelegramNotifier:
    bot_token: str
    chat_id: str
    timeout_sec: int = 20

    def __post_init__(self) -> None:
        self._client = httpx.Client(timeout=self.timeout_sec)
        self._base = f"https://api.telegram.org/bot{self.bot_token}"

    def close(self) -> None:
        self._client.close()

    def _post_with_retry(self, endpoint: str, payload: dict[str, Any], attempts: int = 5) -> None:
        delay = 0.8
        last_resp: httpx.Response | None = None
        for _ in range(attempts):
            resp = self._client.post(f"{self._base}/{endpoint}", json=payload)
            last_resp = resp
            if resp.status_code != 429:
                resp.raise_for_status()
                return
            retry_after = 0.0
            try:
                body = resp.json()
                retry_after = float(body.get("parameters", {}).get("retry_after", 0) or 0)
            except Exception:
                retry_after = 0.0
            wait_s = max(delay, retry_after + 0.2)
            time.sleep(wait_s)
            delay = min(delay * 1.8, 8.0)
        if last_resp is not None:
            last_resp.raise_for_status()

    def send_message(self, text: str) -> None:
        self._post_with_retry("sendMessage", {"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"})

    def send_photo(self, photo_url: str, caption: str = "") -> None:
        self._post_with_retry(
            "sendPhoto",
            {"chat_id": self.chat_id, "photo": photo_url, "caption": caption[:1024], "parse_mode": "HTML"},
        )


def quickchart_url(symbol: str, branch: str, levels: list[dict[str, float]], last_price: float, tp_price: float) -> str:
    labels = [f"L{int(x['level'])}" for x in levels]
    prices = [round(float(x["price"]), 8) for x in levels]
    sizes = [round(float(x["size_usd"]), 4) for x in levels]

    single_level = len(labels) <= 1

    cfg: dict[str, Any] = {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [
                {
                    "type": "line",
                    "label": "Grid price",
                    "data": prices,
                    "borderColor": "#00bcd4",
                    "backgroundColor": "#00bcd4",
                    "pointRadius": 4,
                    "pointHoverRadius": 5,
                    "borderWidth": 2,
                    "tension": 0.0,
                    "yAxisID": "y",
                },
                {
                    "label": "Size USD",
                    "data": sizes,
                    "backgroundColor": "rgba(76,175,80,0.45)",
                    "borderColor": "#2e7d32",
                    "borderWidth": 1,
                    "yAxisID": "y1",
                    "maxBarThickness": 80,
                    "barThickness": 80 if single_level else None,
                    "categoryPercentage": 0.45,
                    "barPercentage": 0.7,
                    "borderRadius": 6,
                },
            ],
        },
        "options": {
            "plugins": {
                "title": {"display": True, "text": f"{symbol} {branch} martingale", "color": "#e5e7eb"},
                "legend": {"labels": {"color": "#cbd5e1"}},
            },
            "layout": {"padding": {"left": 10, "right": 10, "top": 10, "bottom": 10}},
            "scales": {
                "x": {"offset": True, "ticks": {"color": "#94a3b8"}, "grid": {"color": "rgba(148,163,184,0.15)"}},
                "y": {
                    "position": "left",
                    "title": {"display": True, "text": "Price", "color": "#cbd5e1"},
                    "ticks": {"color": "#94a3b8"},
                    "grid": {"color": "rgba(148,163,184,0.15)"},
                },
                "y1": {
                    "position": "right",
                    "grid": {"drawOnChartArea": False},
                    "title": {"display": True, "text": "USD size", "color": "#cbd5e1"},
                    "ticks": {"color": "#94a3b8"},
                },
            },
        },
    }

    encoded = quote(json.dumps(cfg, separators=(",", ":"), ensure_ascii=False))
    return f"https://quickchart.io/chart?width=1000&height=550&c={encoded}&f=png&v=4&devicePixelRatio=2&bkg=%23111827&key=***"


def quickchart_price_url(
    symbol: str,
    branch: str,
    labels: list[str],
    ohlc: list[dict[str, float]],
    entry_price: float,
    tp_price: float,
) -> str:
    if not labels or not ohlc or len(labels) != len(ohlc):
        return ""

    candle_data = []
    for lbl, row in zip(labels, ohlc):
        candle_data.append({"x": lbl, "o": round(float(row["open"]), 8), "h": round(float(row["high"]), 8), "l": round(float(row["low"]), 8), "c": round(float(row["close"]), 8)})

    entry_line = [{"x": lbl, "y": round(float(entry_price), 8)} for lbl in labels]
    tp_line = [{"x": lbl, "y": round(float(tp_price), 8)} for lbl in labels]

    cfg: dict[str, Any] = {
        "type": "candlestick",
        "data": {
            "datasets": [
                {"label": "OHLC", "data": candle_data, "color": {"up": "#22c55e", "down": "#ef4444", "unchanged": "#94a3b8"}, "borderColor": {"up": "#22c55e", "down": "#ef4444", "unchanged": "#94a3b8"}},
                {"type": "line", "label": "Entry", "data": entry_line, "parsing": {"xAxisKey": "x", "yAxisKey": "y"}, "borderColor": "#f59e0b", "pointRadius": 0, "borderWidth": 1.3, "borderDash": [6, 4]},
                {"type": "line", "label": "TP", "data": tp_line, "parsing": {"xAxisKey": "x", "yAxisKey": "y"}, "borderColor": "#60a5fa", "pointRadius": 0, "borderWidth": 1.3, "borderDash": [6, 4]},
            ]
        },
        "options": {
            "plugins": {"title": {"display": True, "text": f"{symbol} {branch} candles (150D)", "color": "#e5e7eb"}, "legend": {"labels": {"color": "#cbd5e1"}}},
            "scales": {
                "x": {"type": "category", "ticks": {"color": "#94a3b8", "maxTicksLimit": 12}, "grid": {"color": "rgba(148,163,184,0.12)"}},
                "y": {"ticks": {"color": "#94a3b8"}, "title": {"display": True, "text": "Price", "color": "#cbd5e1"}, "grid": {"color": "rgba(148,163,184,0.15)"}},
            },
        },
    }

    encoded = quote(json.dumps(cfg, separators=(",", ":"), ensure_ascii=False))
    return f"https://quickchart.io/chart?width=1400&height=700&c={encoded}&f=png&v=4&devicePixelRatio=2&bkg=%23111827&key=***"


def quickchart_formation_url(
    symbol: str,
    labels: list[str],
    ohlc: list[dict[str, float]],
    upper_line: list[float],
    lower_line: list[float],
    spark_points: list[dict[str, float]],
    twix_points: list[dict[str, float]],
) -> str:
    if not labels or not ohlc or len(labels) != len(ohlc):
        return ""

    candle_data = [
        {
            "x": x,
            "o": round(float(row["open"]), 8),
            "h": round(float(row["high"]), 8),
            "l": round(float(row["low"]), 8),
            "c": round(float(row["close"]), 8),
        }
        for x, row in zip(labels, ohlc)
    ]
    up_points = [{"x": x, "y": round(float(y), 8)} for x, y in zip(labels, upper_line)]
    low_points = [{"x": x, "y": round(float(y), 8)} for x, y in zip(labels, lower_line)]

    # Auto-scale Y around formation candles/channels so chart stays readable.
    lows = [float(x.get("l", 0.0)) for x in candle_data]
    highs = [float(x.get("h", 0.0)) for x in candle_data]
    band_vals = [float(v) for v in upper_line] + [float(v) for v in lower_line]
    y_min_raw = min((lows + band_vals) or [0.0])
    y_max_raw = max((highs + band_vals) or [1.0])
    y_span = max(y_max_raw - y_min_raw, max(abs(y_max_raw), 1.0) * 1e-4)
    y_pad = y_span * 0.12
    y_min = y_min_raw - y_pad
    y_max = y_max_raw + y_pad

    cfg: dict[str, Any] = {
        "type": "candlestick",
        "data": {
            "datasets": [
                {"label": "OHLC", "data": candle_data, "color": {"up": "#22c55e", "down": "#ef4444", "unchanged": "#94a3b8"}, "borderColor": {"up": "#22c55e", "down": "#ef4444", "unchanged": "#94a3b8"}},
                {"label": "Upper", "data": up_points, "parsing": {"xAxisKey": "x", "yAxisKey": "y"}, "borderColor": "#f59e0b", "pointRadius": 0, "borderWidth": 1.2, "borderDash": [6, 4]},
                {"label": "Lower", "data": low_points, "parsing": {"xAxisKey": "x", "yAxisKey": "y"}, "borderColor": "#60a5fa", "pointRadius": 0, "borderWidth": 1.2, "borderDash": [6, 4]},
                {"type": "scatter", "label": "Sparks", "data": spark_points, "parsing": {"xAxisKey": "x", "yAxisKey": "y"}, "pointRadius": 4, "pointBackgroundColor": "#22c55e"},
                {"type": "scatter", "label": "Twix", "data": twix_points, "parsing": {"xAxisKey": "x", "yAxisKey": "y"}, "pointRadius": 4, "pointBackgroundColor": "#e879f9"},
            ]
        },
        "options": {
            "plugins": {"title": {"display": True, "text": f"{symbol} formation", "color": "#e5e7eb"}, "legend": {"labels": {"color": "#cbd5e1"}}},
            "scales": {
                "x": {"type": "category", "ticks": {"color": "#94a3b8", "maxTicksLimit": 12}, "grid": {"color": "rgba(148,163,184,0.12)"}},
                "y": {"min": round(float(y_min), 8), "max": round(float(y_max), 8), "ticks": {"color": "#94a3b8"}, "title": {"display": True, "text": "Price", "color": "#cbd5e1"}, "grid": {"color": "rgba(148,163,184,0.15)"}},
            },
        },
    }

    encoded = quote(json.dumps(cfg, separators=(",", ":"), ensure_ascii=False))
    return f"https://quickchart.io/chart?width=1400&height=700&c={encoded}&f=png&v=4&devicePixelRatio=2&bkg=%23111827&key=***"
