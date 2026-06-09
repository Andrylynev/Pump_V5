from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class FormationCandidate:
    case_id: str
    symbol: str
    timeframe: str
    branch: str
    accumulation_start: datetime
    accumulation_end: datetime
    entry_time: datetime | None
    upper_bound: float
    lower_bound: float
    score: float
    spark_count: int
    twix_count: int
    volume_score: float
    channel_width: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("accumulation_start", "accumulation_end", "entry_time"):
            value = payload[key]
            payload[key] = value.isoformat() if value is not None else None
        return payload


@dataclass
class BacktestTrade:
    trade_id: str
    case_id: str
    symbol: str
    branch: str
    mode: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    size_usd: float
    pnl_usd: float
    pnl_pct: float
    max_drawdown_pct: float
    bars_held: int
    outcome: str
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["entry_time"] = self.entry_time.isoformat()
        payload["exit_time"] = self.exit_time.isoformat()
        return payload


@dataclass
class ArbiterLabel:
    case_id: str
    case_has_real_formation: bool
    best_algorithm: str
    baseline_good_enough: bool
    accurate_start: bool
    accurate_end: bool
    structure_break: bool
    quality_baseline: int | None
    quality_candidate: int | None
    quality_other: int | None
    comment: str
    updated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["updated_at"] = self.updated_at.isoformat()
        return payload

