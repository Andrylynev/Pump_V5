from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from app.market_data.bybit_client import BybitClient


UTC = timezone.utc


def dt_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


def interval_to_timedelta(interval: str) -> timedelta:
    if interval == "D":
        return timedelta(days=1)
    return timedelta(minutes=int(interval))


def parse_kline_rows(rows: list[list[str]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    out = pd.DataFrame(
        rows,
        columns=["ts", "open", "high", "low", "close", "volume", "turnover"],
    )
    out["timestamp"] = pd.to_datetime(out["ts"].astype("int64"), unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume", "turnover"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.drop(columns=["ts"]).dropna(subset=["timestamp"]).sort_values("timestamp")
    out = out.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    return out


@dataclass
class MarketCacheBuilder:
    client: BybitClient
    market_cache_root: Path
    features_cache_root: Path

    def _symbol_interval_path(self, symbol: str, interval: str, year: int) -> Path:
        return self.market_cache_root / symbol / interval / f"{year}.parquet"

    def _feature_path(self, feature: str, symbol: str, year: int) -> Path:
        return self.features_cache_root / feature / symbol / f"{year}.parquet"

    @staticmethod
    def _is_time_coverage_ok(df: pd.DataFrame, start_dt: datetime, end_dt: datetime) -> bool:
        if df.empty:
            return False
        min_ts = pd.to_datetime(df["timestamp"], utc=True).min()
        max_ts = pd.to_datetime(df["timestamp"], utc=True).max()
        return bool(
            (min_ts <= start_dt + timedelta(days=2) and max_ts >= end_dt - timedelta(days=2))
            or (len(df) >= 10 and max_ts >= end_dt - timedelta(days=2))
        )

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        start_ms = dt_to_ms(start_dt)
        end_ms = dt_to_ms(end_dt)
        cursor_end = end_ms
        all_rows: list[list[str]] = []
        safety = 0

        while cursor_end >= start_ms and safety < 10000:
            safety += 1
            payload = self.client.get_kline(
                symbol=symbol,
                interval=interval,
                start_ms=start_ms,
                end_ms=cursor_end,
                limit=1000,
            )
            rows: list[list[str]] = payload.get("result", {}).get("list", []) or []
            if not rows:
                break
            all_rows.extend(rows)
            min_ts = min(int(r[0]) for r in rows)
            next_end = min_ts - 1
            if next_end >= cursor_end:
                break
            cursor_end = next_end
            if min_ts <= start_ms:
                break

        df = parse_kline_rows(all_rows)
        if df.empty:
            return df
        df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)].copy()
        df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        return df

    def save_partitioned(self, df: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        out_rows: list[dict[str, Any]] = []
        by_year = dict(tuple(df.groupby(df["timestamp"].dt.year)))
        for year, part in by_year.items():
            path = self._symbol_interval_path(symbol, interval, int(year))
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                old = pd.read_parquet(path)
                part = pd.concat([old, part], ignore_index=True)
                part = part.drop_duplicates("timestamp").sort_values("timestamp")
            part.to_parquet(path, index=False)
            out_rows.append(
                {
                    "kind": "kline",
                    "symbol": symbol,
                    "interval": interval,
                    "year": int(year),
                    "path": str(path),
                    "bars": int(len(part)),
                    "min_ts": part["timestamp"].min(),
                    "max_ts": part["timestamp"].max(),
                    "updated_at": datetime.now(tz=UTC),
                    "host": self.client.last_success_host,
                }
            )
        return pd.DataFrame(out_rows)

    def _persist_feature_rows(self, feature: str, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        rows: list[dict[str, Any]] = []
        by_year = dict(tuple(df.groupby(df["timestamp"].dt.year)))
        for year, part in by_year.items():
            path = self._feature_path(feature, symbol, int(year))
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                old = pd.read_parquet(path)
                part = pd.concat([old, part], ignore_index=True)
                part = part.drop_duplicates("timestamp").sort_values("timestamp")
            part.to_parquet(path, index=False)
            rows.append(
                {
                    "kind": feature,
                    "symbol": symbol,
                    "interval": "1d",
                    "year": int(year),
                    "path": str(path),
                    "bars": int(len(part)),
                    "min_ts": part["timestamp"].min(),
                    "max_ts": part["timestamp"].max(),
                    "updated_at": datetime.now(tz=UTC),
                    "host": self.client.last_success_host,
                }
            )
        return pd.DataFrame(rows)

    def fetch_open_interest(self, symbol: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        start_ms = dt_to_ms(start_dt)
        end_ms = dt_to_ms(end_dt)
        cursor: str | None = None
        rows: list[dict[str, Any]] = []
        safety = 0
        while safety < 1000:
            safety += 1
            payload = self.client.get_open_interest(
                symbol=symbol,
                interval_time="1d",
                start_ms=start_ms,
                end_ms=end_ms,
                limit=200,
                cursor=cursor,
            )
            result = payload.get("result", {})
            items = result.get("list", []) or []
            if not items:
                break
            for x in items:
                rows.append(
                    {
                        "timestamp": pd.to_datetime(int(x["timestamp"]), unit="ms", utc=True),
                        "open_interest": float(x["openInterest"]),
                    }
                )
            cursor = result.get("nextPageCursor") or None
            if not cursor:
                break
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    def fetch_long_short(self, symbol: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        start_ms = dt_to_ms(start_dt)
        end_ms = dt_to_ms(end_dt)
        cursor: str | None = None
        rows: list[dict[str, Any]] = []
        safety = 0
        while safety < 1000:
            safety += 1
            payload = self.client.get_long_short_ratio(
                symbol=symbol,
                period="1d",
                start_ms=start_ms,
                end_ms=end_ms,
                limit=500,
                cursor=cursor,
            )
            result = payload.get("result", {})
            items = result.get("list", []) or []
            if not items:
                break
            for x in items:
                rows.append(
                    {
                        "timestamp": pd.to_datetime(int(x["timestamp"]), unit="ms", utc=True),
                        "buy_ratio": float(x["buyRatio"]),
                        "sell_ratio": float(x["sellRatio"]),
                    }
                )
            cursor = result.get("nextPageCursor") or None
            if not cursor:
                break
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    def fetch_funding(self, symbol: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        start_ms = dt_to_ms(start_dt)
        end_ms = dt_to_ms(end_dt)
        cursor_end = end_ms
        rows: list[dict[str, Any]] = []
        safety = 0
        while cursor_end >= start_ms and safety < 5000:
            safety += 1
            payload = self.client.get_funding_history(
                symbol=symbol,
                start_ms=start_ms,
                end_ms=cursor_end,
                limit=200,
            )
            items = payload.get("result", {}).get("list", []) or []
            if not items:
                break
            for x in items:
                ts = int(x["fundingRateTimestamp"])
                rows.append(
                    {
                        "timestamp": pd.to_datetime(ts, unit="ms", utc=True),
                        "funding_rate": float(x["fundingRate"]),
                    }
                )
            min_ts = min(int(x["fundingRateTimestamp"]) for x in items)
            next_end = min_ts - 1
            if next_end >= cursor_end:
                break
            cursor_end = next_end
            if min_ts <= start_ms:
                break
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    def cache_symbol(
        self,
        symbol: str,
        intervals: Iterable[str],
        start_dt: datetime,
        end_dt: datetime,
        with_features: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        market_manifest_parts: list[pd.DataFrame] = []
        feature_manifest_parts: list[pd.DataFrame] = []

        for interval in intervals:
            cached = self.load_cached_interval(symbol, interval, start_dt, end_dt)
            if self._is_time_coverage_ok(cached, start_dt, end_dt):
                years = sorted(cached["timestamp"].dt.year.unique().tolist())
                for year in years:
                    part = cached[cached["timestamp"].dt.year == year]
                    market_manifest_parts.append(
                        pd.DataFrame(
                            [
                                {
                                    "kind": "kline",
                                    "symbol": symbol,
                                    "interval": interval,
                                    "year": int(year),
                                    "path": str(self._symbol_interval_path(symbol, interval, int(year))),
                                    "bars": int(len(part)),
                                    "min_ts": part["timestamp"].min(),
                                    "max_ts": part["timestamp"].max(),
                                    "updated_at": datetime.now(tz=UTC),
                                    "host": self.client.last_success_host,
                                }
                            ]
                        )
                    )
                continue

            df = self.fetch_klines(symbol=symbol, interval=interval, start_dt=start_dt, end_dt=end_dt)
            if df.empty:
                continue
            market_manifest_parts.append(self.save_partitioned(df, symbol=symbol, interval=interval))

        if with_features:
            oi_cached = self.load_feature_daily("open_interest", symbol, start_dt, end_dt)
            ls_cached = self.load_feature_daily("long_short_ratio", symbol, start_dt, end_dt)
            fr_cached = self.load_feature_daily("funding_rate", symbol, start_dt, end_dt)

            oi = oi_cached if self._is_time_coverage_ok(oi_cached, start_dt, end_dt) else self.fetch_open_interest(symbol=symbol, start_dt=start_dt, end_dt=end_dt)
            ls = ls_cached if self._is_time_coverage_ok(ls_cached, start_dt, end_dt) else self.fetch_long_short(symbol=symbol, start_dt=start_dt, end_dt=end_dt)
            fr = fr_cached if self._is_time_coverage_ok(fr_cached, start_dt, end_dt) else self.fetch_funding(symbol=symbol, start_dt=start_dt, end_dt=end_dt)

            if not oi.empty:
                feature_manifest_parts.append(self._persist_feature_rows("open_interest", symbol, oi))
            if not ls.empty:
                feature_manifest_parts.append(self._persist_feature_rows("long_short_ratio", symbol, ls))
            if not fr.empty:
                feature_manifest_parts.append(self._persist_feature_rows("funding_rate", symbol, fr))

        market_manifest = pd.concat(market_manifest_parts, ignore_index=True) if market_manifest_parts else pd.DataFrame()
        feature_manifest = (
            pd.concat(feature_manifest_parts, ignore_index=True) if feature_manifest_parts else pd.DataFrame()
        )
        return market_manifest, feature_manifest

    def load_cached_interval(self, symbol: str, interval: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        years = list(range(start_dt.year, end_dt.year + 1))
        parts: list[pd.DataFrame] = []
        for year in years:
            path = self._symbol_interval_path(symbol, interval, year)
            if not path.exists():
                continue
            parts.append(pd.read_parquet(path))
        if not parts:
            return pd.DataFrame()
        df = pd.concat(parts, ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)].copy()
        return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    def load_feature_daily(self, feature: str, symbol: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        years = list(range(start_dt.year, end_dt.year + 1))
        parts: list[pd.DataFrame] = []
        for year in years:
            path = self._feature_path(feature, symbol, year)
            if path.exists():
                parts.append(pd.read_parquet(path))
        if not parts:
            return pd.DataFrame()
        df = pd.concat(parts, ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)].copy()
        return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
