from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectPaths:
    root: Path
    data_root: Path
    market_cache_root: Path
    features_cache_root: Path
    experiments_root: Path

    @property
    def market_manifest(self) -> Path:
        return self.market_cache_root / "manifest.parquet"

    @property
    def features_manifest(self) -> Path:
        return self.features_cache_root / "manifest.parquet"

    @property
    def results_path(self) -> Path:
        return self.experiments_root / "results.parquet"

    @property
    def trades_path(self) -> Path:
        return self.experiments_root / "trades.parquet"

    @property
    def metrics_path(self) -> Path:
        return self.experiments_root / "metrics.parquet"

    @property
    def summary_path(self) -> Path:
        return self.experiments_root / "summary.json"

    @property
    def arbiter_cases_path(self) -> Path:
        return self.experiments_root / "arbiter_cases.parquet"

    @property
    def arbiter_labels_path(self) -> Path:
        return self.experiments_root / "arbiter_labels.parquet"

    @property
    def formation_audit_path(self) -> Path:
        return self.experiments_root / "formation_audit.parquet"

    @property
    def resolved_universe_path(self) -> Path:
        return self.experiments_root / "resolved_universe.json"


def build_paths(root: Path, cfg_paths: dict[str, str]) -> ProjectPaths:
    data_root = root / cfg_paths["data_root"]
    market_cache = root / cfg_paths["market_cache"]
    features_cache = root / cfg_paths["features_cache"]
    experiments = root / cfg_paths["experiments_root"]

    for p in (data_root, market_cache, features_cache, experiments):
        p.mkdir(parents=True, exist_ok=True)

    return ProjectPaths(
        root=root,
        data_root=data_root,
        market_cache_root=market_cache,
        features_cache_root=features_cache,
        experiments_root=experiments,
    )

