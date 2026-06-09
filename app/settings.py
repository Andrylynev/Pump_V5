from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AppSettings:
    raw: dict[str, Any]
    config_path: Path | None = None

    @property
    def config_dir(self) -> Path:
        return self.config_path.parent if self.config_path is not None else Path.cwd()

    @property
    def api_hosts(self) -> list[str]:
        return list(self.raw["api"]["hosts"])

    @property
    def request_timeout_sec(self) -> int:
        return int(self.raw["api"]["request_timeout_sec"])

    @property
    def min_request_interval_sec(self) -> float:
        return float(self.raw["api"]["min_request_interval_sec"])

    @property
    def max_retries_per_host(self) -> int:
        return int(self.raw["api"]["max_retries_per_host"])

    @property
    def universe(self) -> dict[str, Any]:
        return dict(self.raw["universe"])

    @property
    def detector(self) -> dict[str, Any]:
        return dict(self.raw["detector"])

    @property
    def entry(self) -> dict[str, Any]:
        return dict(self.raw["entry"])

    @property
    def martingale(self) -> dict[str, Any]:
        return dict(self.raw["martingale"])

    @property
    def backtest(self) -> dict[str, Any]:
        return dict(self.raw["backtest"])

    @property
    def intervals(self) -> list[str]:
        return list(self.raw["cache"]["intervals"])

    @property
    def paths(self) -> dict[str, str]:
        return dict(self.raw["paths"])


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_settings(config_path: Path | None = None) -> AppSettings:
    root = Path(__file__).resolve().parent
    default_path = root / "config" / "default.yaml"
    with default_path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    if config_path is not None:
        with config_path.open("r", encoding="utf-8") as f:
            user_data = yaml.safe_load(f) or {}
        data = _deep_update(data, user_data)

    return AppSettings(raw=data, config_path=config_path)

