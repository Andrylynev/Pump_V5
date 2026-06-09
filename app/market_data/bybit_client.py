from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx


class BybitAPIError(RuntimeError):
    pass


@dataclass
class BybitClient:
    hosts: list[str]
    timeout_sec: int = 20
    min_request_interval_sec: float = 0.08
    max_retries_per_host: int = 2

    def __post_init__(self) -> None:
        self._last_request_ts = 0.0
        self.last_success_host: str | None = None
        self._client = httpx.Client(timeout=self.timeout_sec)
        self._rate_limit_backoff_sec = 1.0
        self._blocked_hosts: set[str] = set()

    def close(self) -> None:
        self._client.close()

    def _throttle(self) -> None:
        now = time.monotonic()
        delta = now - self._last_request_ts
        if delta < self.min_request_interval_sec:
            time.sleep(self.min_request_interval_sec - delta)
        self._last_request_ts = time.monotonic()

    def _request(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = params or {}
        errors: list[str] = []

        ordered_hosts = []
        if self.last_success_host and self.last_success_host not in self._blocked_hosts:
            ordered_hosts.append(self.last_success_host)
        ordered_hosts.extend([h for h in self.hosts if h not in ordered_hosts and h not in self._blocked_hosts])
        if not ordered_hosts:
            ordered_hosts = list(self.hosts)

        for host in ordered_hosts:
            url = f"{host}{path}"
            for attempt in range(self.max_retries_per_host):
                try:
                    self._throttle()
                    resp = self._client.get(url, params=params)
                    if resp.status_code == 403:
                        errors.append(f"{host}: HTTP 403")
                        self._blocked_hosts.add(host)
                        break
                    if resp.status_code in (429, 503):
                        sleep_s = min(8.0, self._rate_limit_backoff_sec * (2**attempt))
                        time.sleep(sleep_s)
                        errors.append(f"{host}: HTTP {resp.status_code} backoff={sleep_s}s")
                        continue
                    resp.raise_for_status()
                    payload: dict[str, Any] = resp.json()
                    ret_code = int(payload.get("retCode", 0))
                    if ret_code == 10006:
                        sleep_s = min(8.0, self._rate_limit_backoff_sec * (2**attempt))
                        time.sleep(sleep_s)
                        errors.append(f"{host}: rate-limit retCode=10006 backoff={sleep_s}s")
                        continue
                    if ret_code != 0:
                        errors.append(f"{host}: retCode={ret_code} msg={payload.get('retMsg')}")
                        break
                    self.last_success_host = host
                    return payload
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{host}: {exc}")
                    time.sleep(0.15)
            # host-level retry finished, try next host

        raise BybitAPIError("All hosts failed: " + " | ".join(errors))

    def get_server_time(self) -> dict[str, Any]:
        return self._request("/v5/market/time")

    def get_instruments_info(
        self,
        category: str = "linear",
        status: str = "Trading",
        limit: int = 1000,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "category": category,
            "status": status,
            "limit": limit,
        }
        if cursor:
            params["cursor"] = cursor
        return self._request("/v5/market/instruments-info", params=params)

    def get_kline(
        self,
        symbol: str,
        interval: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int = 1000,
        category: str = "linear",
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_ms is not None:
            params["start"] = int(start_ms)
        if end_ms is not None:
            params["end"] = int(end_ms)
        return self._request("/v5/market/kline", params=params)

    def get_open_interest(
        self,
        symbol: str,
        interval_time: str = "1d",
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int = 200,
        cursor: str | None = None,
        category: str = "linear",
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "category": category,
            "symbol": symbol,
            "intervalTime": interval_time,
            "limit": limit,
        }
        if start_ms is not None:
            params["startTime"] = int(start_ms)
        if end_ms is not None:
            params["endTime"] = int(end_ms)
        if cursor:
            params["cursor"] = cursor
        return self._request("/v5/market/open-interest", params=params)

    def get_long_short_ratio(
        self,
        symbol: str,
        period: str = "1d",
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int = 500,
        cursor: str | None = None,
        category: str = "linear",
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "category": category,
            "symbol": symbol,
            "period": period,
            "limit": limit,
        }
        if start_ms is not None:
            params["startTime"] = int(start_ms)
        if end_ms is not None:
            params["endTime"] = int(end_ms)
        if cursor:
            params["cursor"] = cursor
        return self._request("/v5/market/account-ratio", params=params)

    def get_funding_history(
        self,
        symbol: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int = 200,
        category: str = "linear",
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "category": category,
            "symbol": symbol,
            "limit": limit,
        }
        if start_ms is not None:
            params["startTime"] = int(start_ms)
        if end_ms is not None:
            params["endTime"] = int(end_ms)
        return self._request("/v5/market/funding/history", params=params)
