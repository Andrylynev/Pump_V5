"""Minimal end-to-end backtest pipeline (Pump V5).

Ties detector -> entry -> exit together on cached daily + intraday candles and
produces a trades table. Intentionally lean: enough to (a) measure the method on
real data and (b) derive the single-vs-martingale routing criterion (T2.4) from
numbers rather than a guess.
"""
from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.backtest.entry import EntryEvaluator
from app.backtest.exit import (
    ExitConfig,
    compute_pump_target,
    find_prerise_swing_low,
    risk_reward,
    simulate_exit,
)
from app.detectors.methodic_detector import MethodicDetector

EPS = 1e-9


def _load_interval(cache_root: str, symbol: str, interval: str) -> pd.DataFrame:
    parts = sorted(glob.glob(f"{cache_root}/{symbol}/{interval}/*.parquet"))
    if not parts:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


@dataclass
class BacktestPipeline:
    cfg: dict[str, Any]
    cache_root: str

    def _prior_high_time(self, daily: pd.DataFrame, acc_start: Any, lookback_days: int = 120):
        """Timestamp of the swing HIGH just before accumulation began.

        The fixed-range VP must start here (per TZ) so its upper zone (VAH) lands
        ABOVE the breakout — the span includes the higher-priced bars of the
        decline into the range. Falls back to acc_start if no prior bars.
        """
        ts = pd.to_datetime(daily["timestamp"], utc=True)
        a0 = pd.Timestamp(acc_start)
        a0 = a0.tz_localize("UTC") if a0.tzinfo is None else a0.tz_convert("UTC")
        prior = daily[ts < a0]
        if prior.empty:
            return acc_start
        prior = prior.tail(lookback_days)
        idx = prior["high"].astype(float).idxmax()
        return pd.Timestamp(prior.loc[idx, "timestamp"]).to_pydatetime()

    def run_symbol(self, symbol: str) -> list[dict[str, Any]]:
        det = MethodicDetector(detector_cfg=self.cfg["detector"])
        ev = EntryEvaluator(entry_cfg=self.cfg["entry"])
        exit_cfg = ExitConfig.from_cfg(self.cfg["exit"])
        vp_cfg = self.cfg.get("volume_profile", {})
        min_rr = float(self.cfg["entry"].get("min_risk_reward", 3.0))

        daily = _load_interval(self.cache_root, symbol, "D")
        if daily.empty or len(daily) < int(self.cfg["detector"]["min_range_days"]):
            return []

        # Breakout timeframe: prefer 4h, fall back to 1h, then daily.
        intraday = pd.DataFrame()
        for tf in self.cfg["entry"].get("breakout_timeframes", ["240", "60"]):
            intraday = _load_interval(self.cache_root, symbol, tf)
            if not intraday.empty:
                break
        if intraday.empty:
            intraday = daily.copy()

        formations = det.detect_accumulations(symbol, daily)
        trades: list[dict[str, Any]] = []
        for f in formations:
            decision = ev.evaluate(f, intraday, branch=f.branch)
            if not decision.get("entered"):
                continue
            entry_price = float(decision["entry_price"])
            entry_time = decision["entry_time"]

            # Pump target (длина пампа): fixed-range VP from the PRIOR HIGH (the
            # swing high BEFORE accumulation began) to current. Its upper volume
            # zone (VAH) is the target — it sits above the breakout because the
            # span includes the higher-priced decline into the range. (TZ: «от
            # предыдущего хая до начала накопления … до текущей».)
            prior_high_time = self._prior_high_time(daily, f.accumulation_start)
            tgt = compute_pump_target(daily, prior_high_time, entry_time, vp_cfg)
            # VAH is the target only if it sits above entry; else fall back.
            vah = tgt.vah if (tgt.ok and tgt.vah > entry_price) else entry_price * 1.5

            # Resolve entry index in intraday for the swing-low stop + exit walk.
            idf = intraday.sort_values("timestamp").reset_index(drop=True)
            ts = pd.to_datetime(idf["timestamp"], utc=True)
            entry_idx_arr = idf.index[ts >= pd.Timestamp(entry_time)].tolist()
            if not entry_idx_arr:
                continue
            entry_idx = int(entry_idx_arr[0])

            # Single-entry stop = PREVIOUS swing low before the rise (TZ), NOT the
            # absolute channel floor. Floored at channel lower for safety.
            swing_lookback = int(self.cfg["exit"].get("swing_low_lookback_bars", 30))
            stop_single = find_prerise_swing_low(
                idf, entry_idx, lookback_bars=swing_lookback, channel_lower=float(f.lower_bound)
            )
            rr = risk_reward(entry_price, vah, stop_single)

            # 3:1 risk-management gate.
            if rr < min_rr:
                trades.append({
                    "symbol": symbol, "case_id": f.case_id, "entered": False,
                    "reason": "rr_below_min", "rr": rr, "score": f.score,
                    "channel_width": f.channel_width, "trend": f.diagnostics.get("trend"),
                })
                continue

            # Route single vs martingale — T2.4 data-driven: by CHANNEL WIDTH.
            # Narrow channel = high conviction = single; wide = hedge w/ martingale.
            route_cfg = self.cfg.get("routing", {})
            forced = str(route_cfg.get("mode", "auto"))
            if forced in ("single", "martingale"):
                mode = forced
            else:
                width_thr = float(route_cfg.get("martingale_when_width_above", 0.30))
                mode = "martingale" if f.channel_width >= width_thr else "single"

            ex = simulate_exit(
                idf, entry_idx=entry_idx, entry_price=entry_price,
                target_vah=vah, stop_price=stop_single, channel_lower=float(f.lower_bound),
                mode=mode, exit_cfg=exit_cfg,
            )
            trades.append({
                "symbol": symbol, "case_id": f.case_id, "entered": True,
                "mode": mode, "trend": f.diagnostics.get("trend"),
                "score": f.score, "spark": f.spark_count, "twix": f.twix_count,
                "channel_width": f.channel_width, "rr": rr,
                "entry_time": entry_time, "entry_price": entry_price,
                "vah": vah, "stop": stop_single,
                "exit_time": ex.get("exit_time"), "exit_price": ex["exit_price"],
                "outcome": ex["outcome"], "pnl_pct": ex["pnl_pct"],
            })
        return trades

    def run(self, symbols: list[str], limit: int | None = None) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        syms = symbols[:limit] if limit else symbols
        for s in syms:
            try:
                rows.extend(self.run_symbol(s))
            except Exception as exc:  # keep going; record nothing for failed symbol
                rows.append({"symbol": s, "entered": False, "reason": f"error:{exc}"})
        return pd.DataFrame(rows)
