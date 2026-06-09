from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.market_data.bybit_client import BybitClient


@dataclass
class UniverseSelector:
    """Resolve the tradable universe across Bybit linear (futures) and spot.

    Per the methodic TZ:
      - Track every futures AND spot pair on Bybit.
      - If a coin is listed on BOTH, keep ONLY the futures entry (futures priority).

    The resolved frame carries a ``market`` column ("linear" | "spot") and a
    ``base_coin`` column so downstream code knows where to route the trade.
    """

    client: BybitClient
    quote_coin: str = "USDT"
    status: str = "Trading"
    linear_contract_type: str = "LinearPerpetual"
    include_spot: bool = True

    def _load_category(self, category: str) -> pd.DataFrame:
        rows: list[dict] = []
        cursor: str | None = None
        while True:
            payload = self.client.get_instruments_info(
                category=category,
                status=self.status,
                limit=1000,
                cursor=cursor,
            )
            result = payload.get("result", {})
            items = result.get("list", []) or []
            rows.extend(items)
            cursor = result.get("nextPageCursor") or None
            if not cursor:
                break
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        for col in ("launchTime", "deliveryTime"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _filter_linear(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        mask = (
            (df.get("quoteCoin") == self.quote_coin)
            & (df.get("contractType") == self.linear_contract_type)
            & (df.get("status") == self.status)
        )
        out = df.loc[mask].copy()
        out["market"] = "linear"
        return out

    def _filter_spot(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        mask = (df.get("quoteCoin") == self.quote_coin) & (df.get("status") == self.status)
        out = df.loc[mask].copy()
        out["market"] = "spot"
        return out

    def load(self) -> pd.DataFrame:
        """Return resolved universe with columns at least: symbol, base_coin, market.

        Futures-priority dedup: a base coin present in both linear and spot keeps
        only the linear row.
        """
        linear = self._filter_linear(self._load_category("linear"))
        spot = self._filter_spot(self._load_category("spot")) if self.include_spot else pd.DataFrame()

        frames = [f for f in (linear, spot) if not f.empty]
        if not frames:
            return pd.DataFrame(columns=["symbol", "base_coin", "market"])

        combined = pd.concat(frames, ignore_index=True)
        if "baseCoin" in combined.columns:
            combined["base_coin"] = combined["baseCoin"].astype(str)
        else:
            combined["base_coin"] = combined["symbol"].astype(str)

        # Futures-priority: rank linear above spot, keep first per base_coin.
        combined["_priority"] = combined["market"].map({"linear": 0, "spot": 1}).fillna(2)
        combined = combined.sort_values(["base_coin", "_priority", "symbol"])
        deduped = combined.drop_duplicates(subset=["base_coin"], keep="first").copy()
        deduped = deduped.drop(columns=["_priority"]).sort_values("symbol").reset_index(drop=True)
        return deduped
