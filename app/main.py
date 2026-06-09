"""Single CLI entrypoint for Pump V5.

    python -m app.main cache    --start YYYY-MM-DD --end YYYY-MM-DD [--symbols ...]
    python -m app.main backtest --start YYYY-MM-DD --end YYYY-MM-DD [--limit N] [--cache-root P]
    python -m app.main scan     [--limit N] [--cache-root P]        # IRL formation scan + notify
    python -m app.main universe                                     # resolve spot+futures universe

All actions read app/config/default.yaml (optionally overlaid with --config).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.settings import load_settings
from app.storage.paths import build_paths


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _client_and_cache(settings):
    from app.market_data.bybit_client import BybitClient
    from app.market_data.kline_cache import MarketCacheBuilder

    paths = build_paths(Path.cwd(), settings.paths)
    client = BybitClient(
        hosts=settings.api_hosts,
        timeout_sec=settings.request_timeout_sec,
        min_request_interval_sec=settings.min_request_interval_sec,
        max_retries_per_host=settings.max_retries_per_host,
    )
    builder = MarketCacheBuilder(
        client=client,
        market_cache_root=paths.market_cache_root,
        features_cache_root=paths.features_cache_root,
    )
    return client, builder, paths


def cmd_universe(args, settings) -> int:
    from app.market_data.bybit_client import BybitClient
    from app.market_data.universe import UniverseSelector

    client = BybitClient(hosts=settings.api_hosts, timeout_sec=settings.request_timeout_sec)
    uni = settings.raw["universe"]
    sel = UniverseSelector(
        client=client,
        quote_coin=uni.get("quote_coin", "USDT"),
        status=uni.get("status", "Trading"),
        linear_contract_type=uni.get("linear_contract_type", "LinearPerpetual"),
        include_spot=bool(uni.get("include_spot", True)),
    )
    df = sel.load()
    client.close()
    if df.empty:
        print("universe: 0 symbols (network/API issue?)")
        return 1
    n_lin = int((df["market"] == "linear").sum())
    n_spot = int((df["market"] == "spot").sum())
    print(f"universe: {len(df)} symbols (linear {n_lin} + spot {n_spot}, futures-priority dedup)")
    print(df[["symbol", "market"]].head(20).to_string(index=False))
    return 0


def cmd_cache(args, settings) -> int:
    client, builder, _ = _client_and_cache(settings)
    symbols = args.symbols
    if not symbols:
        from app.market_data.universe import UniverseSelector

        uni = settings.raw["universe"]
        sel = UniverseSelector(
            client=client, quote_coin=uni.get("quote_coin", "USDT"),
            status=uni.get("status", "Trading"),
            linear_contract_type=uni.get("linear_contract_type", "LinearPerpetual"),
            include_spot=bool(uni.get("include_spot", True)),
        )
        u = sel.load()
        symbols = u["symbol"].tolist()[: args.limit] if args.limit else u["symbol"].tolist()
    start, end = _utc(args.start), _utc(args.end)
    intervals = settings.intervals
    ok = 0
    for s in symbols:
        try:
            builder.cache_symbol(s, intervals, start, end, with_features=False)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  cache fail {s}: {exc}", file=sys.stderr)
    client.close()
    print(f"cached {ok}/{len(symbols)} symbols, intervals={intervals}")
    return 0


def cmd_backtest(args, settings) -> int:
    from app.backtest.pipeline import BacktestPipeline

    cache_root = args.cache_root or settings.raw["paths"]["market_cache"]
    cache_path = Path(cache_root)
    if not cache_path.exists():
        print(f"cache root not found: {cache_root}", file=sys.stderr)
        return 1
    symbols = args.symbols or sorted([p.name for p in cache_path.iterdir() if p.is_dir()])
    pipe = BacktestPipeline(cfg=settings.raw, cache_root=str(cache_root))
    df = pipe.run(symbols, limit=args.limit)

    paths = build_paths(Path.cwd(), settings.paths)
    out = paths.trades_path
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    entered = df[df.get("entered") == True] if "entered" in df.columns else pd.DataFrame()
    summary = {
        "symbols": len(symbols) if not args.limit else min(args.limit, len(symbols)),
        "rows": int(len(df)),
        "entered": int(len(entered)),
    }
    if len(entered):
        e = entered.copy()
        e["win"] = e["pnl_pct"] > 0
        summary.update({
            "win_rate": round(float(e["win"].mean()), 4),
            "mean_pnl_pct": round(float(e["pnl_pct"].mean()), 4),
            "median_pnl_pct": round(float(e["pnl_pct"].median()), 4),
        })
    print(json.dumps(summary, ensure_ascii=False))
    print(f"trades -> {out}")
    return 0


def cmd_scan(args, settings) -> int:
    """IRL formation scan: detect formations on cached data and emit notifications."""
    from app.backtest.pipeline import BacktestPipeline
    from app.integrations.notifications import (
        FORMATION_FOUND,
        TRADE_CLOSED,
        TRADE_OPENED,
        build_dispatcher,
    )

    cache_root = args.cache_root or settings.raw["paths"]["market_cache"]
    cache_path = Path(cache_root)
    if not cache_path.exists():
        print(f"cache root not found: {cache_root}", file=sys.stderr)
        return 1
    symbols = args.symbols or sorted([p.name for p in cache_path.iterdir() if p.is_dir()])
    pipe = BacktestPipeline(cfg=settings.raw, cache_root=str(cache_root))
    df = pipe.run(symbols, limit=args.limit)
    dispatcher = build_dispatcher(settings.raw.get("telegram", {}))

    entered = df[df.get("entered") == True] if "entered" in df.columns else pd.DataFrame()
    n_msgs = 0
    for _, r in (entered.iterrows() if len(entered) else iter(())):
        dispatcher.notify(TRADE_OPENED, {
            "symbol": r["symbol"], "case_id": r.get("case_id"), "mode": r.get("mode"),
            "entry_price": r.get("entry_price"), "vah": r.get("vah"),
            "stop": r.get("stop"), "rr": r.get("rr"), "size_usd": 0,
        })
        dispatcher.notify(TRADE_CLOSED, {
            "symbol": r["symbol"], "case_id": r.get("case_id"),
            "outcome": r.get("outcome"), "exit_price": r.get("exit_price"),
            "pnl_pct": r.get("pnl_pct"),
        })
        n_msgs += 2
    print(json.dumps({"formations_entered": int(len(entered)), "notifications": n_msgs,
                      "telegram_enabled": dispatcher.enabled}, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pump_v5")
    p.add_argument("--config", type=str, default=None, help="overlay config yaml")
    sub = p.add_subparsers(dest="command", required=True)

    pu = sub.add_parser("universe"); pu.set_defaults(func=cmd_universe)

    pc = sub.add_parser("cache")
    pc.add_argument("--start", required=True); pc.add_argument("--end", required=True)
    pc.add_argument("--symbols", nargs="*"); pc.add_argument("--limit", type=int, default=0)
    pc.set_defaults(func=cmd_cache)

    pb = sub.add_parser("backtest")
    pb.add_argument("--start", default=None); pb.add_argument("--end", default=None)
    pb.add_argument("--symbols", nargs="*"); pb.add_argument("--limit", type=int, default=0)
    pb.add_argument("--cache-root", default=None)
    pb.set_defaults(func=cmd_backtest)

    ps = sub.add_parser("scan")
    ps.add_argument("--symbols", nargs="*"); ps.add_argument("--limit", type=int, default=0)
    ps.add_argument("--cache-root", default=None)
    ps.set_defaults(func=cmd_scan)

    args = p.parse_args(argv)
    settings = load_settings(Path(args.config) if args.config else None)
    return int(args.func(args, settings))


if __name__ == "__main__":
    raise SystemExit(main())
