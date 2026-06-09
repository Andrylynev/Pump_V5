from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass
class VolumeProfileResult:
    poc_price: float
    vah_price: float
    val_price: float
    total_volume: float
    bins: pd.DataFrame
    high_volume_nodes: list[float]
    low_volume_nodes: list[float]


def calculate_fixed_range_volume_profile(
    candles: pd.DataFrame,
    start_time: Any,
    end_time: Any,
    bins_count: int = 100,
    value_area_percent: float = 0.70,
    hvn_quantile: float = 0.80,
    lvn_quantile: float = 0.20,
) -> VolumeProfileResult | None:
    req_cols = {"timestamp", "open", "high", "low", "close", "volume"}
    if candles.empty or not req_cols.issubset(set(candles.columns)):
        return None

    ts = pd.to_datetime(candles["timestamp"], utc=True)
    start_ts = pd.Timestamp(start_time)
    end_ts = pd.Timestamp(end_time)
    start_ts = start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts.tz_convert("UTC")
    end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")

    data = candles[(ts >= start_ts) & (ts <= end_ts)].copy()
    if data.empty:
        return None

    bins_count = int(max(10, min(400, bins_count)))
    value_area_percent = float(max(0.5, min(0.95, value_area_percent)))

    price_min = float(data["low"].min())
    price_max = float(data["high"].max())
    if price_max <= price_min:
        return None

    edges = np.linspace(price_min, price_max, bins_count + 1)
    mids = (edges[:-1] + edges[1:]) / 2

    total_volume = np.zeros(bins_count, dtype=float)
    up_volume = np.zeros(bins_count, dtype=float)
    down_volume = np.zeros(bins_count, dtype=float)

    for _, candle in data.iterrows():
        low = float(candle["low"])
        high = float(candle["high"])
        volume = float(candle["volume"])
        if high <= low or volume <= 0:
            continue
        candle_is_up = float(candle["close"]) >= float(candle["open"])
        candle_range = max(high - low, EPS)

        for i in range(bins_count):
            bin_low = edges[i]
            bin_high = edges[i + 1]
            overlap = max(0.0, min(high, bin_high) - max(low, bin_low))
            if overlap <= 0:
                continue
            allocated = volume * (overlap / candle_range)
            total_volume[i] += allocated
            if candle_is_up:
                up_volume[i] += allocated
            else:
                down_volume[i] += allocated

    total = float(total_volume.sum())
    if total <= EPS:
        return None

    poc_idx = int(np.argmax(total_volume))
    poc_price = float(mids[poc_idx])

    target = total * value_area_percent
    selected = {poc_idx}
    selected_vol = float(total_volume[poc_idx])
    left = poc_idx - 1
    right = poc_idx + 1

    while selected_vol < target and (left >= 0 or right < bins_count):
        left_vol = float(total_volume[left]) if left >= 0 else -1.0
        right_vol = float(total_volume[right]) if right < bins_count else -1.0
        if right_vol >= left_vol:
            if right < bins_count:
                selected.add(right)
                selected_vol += max(0.0, right_vol)
            right += 1
        else:
            if left >= 0:
                selected.add(left)
                selected_vol += max(0.0, left_vol)
            left -= 1

    val_idx = min(selected)
    vah_idx = max(selected)

    bins = pd.DataFrame(
        {
            "price_low": edges[:-1],
            "price_high": edges[1:],
            "price_mid": mids,
            "total_volume": total_volume,
            "up_volume": up_volume,
            "down_volume": down_volume,
            "volume_share": total_volume / total,
        }
    )

    hvn_th = float(bins["total_volume"].quantile(hvn_quantile))
    lvn_th = float(bins["total_volume"].quantile(lvn_quantile))
    high_nodes = bins[bins["total_volume"] >= hvn_th]["price_mid"].astype(float).tolist()
    low_nodes = bins[bins["total_volume"] <= lvn_th]["price_mid"].astype(float).tolist()

    return VolumeProfileResult(
        poc_price=poc_price,
        vah_price=float(edges[vah_idx + 1]),
        val_price=float(edges[val_idx]),
        total_volume=total,
        bins=bins,
        high_volume_nodes=high_nodes,
        low_volume_nodes=low_nodes,
    )


def current_price_position(price: float, profile: VolumeProfileResult) -> str:
    p = float(price)
    if p < profile.val_price:
        return "below_val"
    if p > profile.vah_price:
        return "above_vah"
    if p > profile.poc_price:
        return "between_poc_vah"
    return "between_val_poc"
