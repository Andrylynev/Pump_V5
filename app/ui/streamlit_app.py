"""Pump V5 Streamlit UI — backtest analysis stand + IRL dashboard.

Run:
    streamlit run app/ui/streamlit_app.py -- --cache-root <path>

Two tabs (per TZ):
  * Бэктест — таблица сделок + график монеты с нанесённой формацией.
  * IRL — текущие сделки / актуальные формации / история.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.settings import load_settings  # noqa: E402
from app.ui.charts import build_formation_figure  # noqa: E402


def _cache_root() -> str:
    for i, a in enumerate(sys.argv):
        if a == "--cache-root" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    s = load_settings()
    return s.raw["paths"]["market_cache"]


def _load_daily(cache_root: str, symbol: str) -> pd.DataFrame:
    parts = sorted(glob.glob(f"{cache_root}/{symbol}/D/*.parquet"))
    if not parts:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def main() -> None:
    st.set_page_config(page_title="Pump V5", layout="wide")
    st.title("Pump V5 — анализ формаций и сделок")

    settings = load_settings()
    cache_root = _cache_root()
    det_cfg = settings.raw["detector"]

    trades_path = Path(settings.raw["paths"]["experiments_root"]) / "trades.parquet"
    trades = pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()

    tab_bt, tab_irl = st.tabs(["📊 Бэктест", "📡 IRL дашборд"])

    with tab_bt:
        if trades.empty:
            st.info("Нет trades.parquet — запусти `python -m app.main backtest`.")
        else:
            entered = trades[trades.get("entered") == True] if "entered" in trades.columns else trades
            c1, c2, c3 = st.columns(3)
            if len(entered):
                c1.metric("Сделок", len(entered))
                c2.metric("Win-rate", f"{(entered['pnl_pct'] > 0).mean()*100:.1f}%")
                c3.metric("Медиана PnL", f"{entered['pnl_pct'].median()*100:.2f}%")
            st.dataframe(entered, use_container_width=True, height=300)

            syms = sorted(entered["symbol"].unique().tolist()) if len(entered) else []
            if syms:
                sym = st.selectbox("Монета для графика", syms)
                row = entered[entered["symbol"] == sym].iloc[0]
                daily = _load_daily(cache_root, sym)
                if not daily.empty:
                    case = str(row.get("case_id", ""))
                    # case_id format: SYMBOL_START_END
                    parts = case.split("_")
                    acc_start = parts[-2] if len(parts) >= 2 else daily["timestamp"].iloc[0]
                    acc_end = parts[-1] if len(parts) >= 1 else daily["timestamp"].iloc[-1]
                    fig = build_formation_figure(
                        daily, det_cfg, acc_start, acc_end,
                        entry_price=row.get("entry_price"), vah=row.get("vah"),
                        stop=row.get("stop"), title=f"{sym} — {row.get('outcome','')}",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Нет дневных данных для {sym} в {cache_root}")

    with tab_irl:
        st.subheader("Актуальные формации и сделки")
        if trades.empty:
            st.info("Запусти `python -m app.main scan` для актуальных формаций.")
        else:
            entered = trades[trades.get("entered") == True] if "entered" in trades.columns else trades
            open_like = entered[entered["outcome"].isin(["market_close"])] if "outcome" in entered.columns else pd.DataFrame()
            st.write("**Открытые / активные:**")
            st.dataframe(open_like, use_container_width=True)
            st.write("**История (закрытые):**")
            closed = entered[~entered["outcome"].isin(["market_close"])] if "outcome" in entered.columns else entered
            st.dataframe(closed, use_container_width=True)


if __name__ == "__main__":
    main()
