"""Loader condivisi (cachati) per tutte le pagine della GUI."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import select

from trading_bot.data.storage import ResearchLog, get_session

UNIVERSE = 500
GATE = "PBO<0.4 · OOS attivo≥0.5 · path+≥65%"
# Costo IBKR realistico per le curve mostrate (conto USD): commissione fissa
# ~$0.35/ordine su un capitale reale di riferimento. Stessi default della ricerca.
CAPITAL_BASE = 10_000.0
FIXED_COST_PER_TRADE = 0.35


@st.cache_data(ttl=300)
def load_top5() -> tuple[list[dict], bool]:
    """Top-5 strategie oneste: promosse se esistono, altrimenti le migliori
    testate. Quarantena (pre-fix engine) sempre esclusa."""
    with get_session() as s:
        rows = s.execute(
            select(ResearchLog)
            .where(ResearchLog.universe_size == UNIVERSE,
                   ResearchLog.status.in_(["promoted", "tested"]))
            .order_by(ResearchLog.oos_sharpe.desc())
        ).scalars().all()
    out, seen = [], set()
    any_promoted = any(r.status == "promoted" for r in rows)
    for r in rows:
        if r.hypothesis_name in seen or r.oos_sharpe is None:
            continue
        seen.add(r.hypothesis_name)
        out.append({"name": r.hypothesis_name, "oos": r.oos_sharpe,
                    "pbo": r.pbo, "dsr": r.dsr, "status": r.status,
                    "config": json.loads(r.config_json)})
        if len(out) == 5:
            break
    return out, any_promoted


@st.cache_data(ttl=900, show_spinner="Carico i prezzi…")
def load_chart_panel():
    from trading_bot.data.ingest import load_panel
    from trading_bot.data.universe import get_top_n_by_liquidity
    syms = get_top_n_by_liquidity(200)
    for x in ("SPY", "GLD"):
        if x not in syms:
            syms = [x, *syms]
    return load_panel(syms, start="2005-01-01").dropna(how="all", axis=1)


def build_strategy(cd: dict, name: str):
    from trading_bot.strategies.composer import (ComposedConfig,
                                                 ComposedStrategy, FilterSpec,
                                                 RegimeSpec, SignalSpec)
    sigs = tuple(SignalSpec(feature=x["feature"], params=x.get("params", {}),
                            weight=x.get("weight", 1.0), use_rank=True,
                            negate=x.get("negate", False)) for x in cd["signals"])
    filts = tuple(FilterSpec(feature=f["feature"], params=f.get("params", {}),
                             threshold=f.get("threshold", 0.0))
                  for f in cd.get("filters", []))
    reg = RegimeSpec(**cd["regime"]) if cd.get("regime") else None
    return ComposedStrategy(ComposedConfig(
        name=name, rationale="", signals=sigs, filters=filts,
        regime=reg, top_n=cd.get("top_n", 10),
        gold_weight=cd.get("gold_weight", 0.0),
        gold_mode=cd.get("gold_mode", "defensive")))


@st.cache_data(ttl=900, show_spinner="Backtest top-5 + benchmark…")
def compute_curves():
    """Equity per strategia (oro condizionale già dentro la strategia) +
    benchmark SPY ed equal-weight. Ritorna (curve, rendimenti)."""
    from trading_bot.backtest.engine import BacktestConfig, CrossSectionalBacktester

    panel = load_chart_panel()
    top5, _ = load_top5()

    spy_eq = 100 * (1 + panel["SPY"].pct_change().fillna(0)).cumprod()
    ew_ret = (panel.drop(columns=["SPY", "GLD"], errors="ignore")
              .pct_change().mean(axis=1).fillna(0))
    ew_eq = 100 * (1 + ew_ret).cumprod()

    data = {"SPY": spy_eq, "Equal-weight": ew_eq}
    rets = {}
    for item in top5:
        try:
            r = CrossSectionalBacktester(
                build_strategy(item["config"], item["name"]),
                BacktestConfig(capital_base=CAPITAL_BASE,
                               fixed_cost_per_trade=FIXED_COST_PER_TRADE)).run(panel)
            rets[item["name"]] = r.returns
            data[item["name"]] = 100 * (1 + r.returns).cumprod()
        except Exception:
            continue
    return data, rets


@st.cache_data(ttl=900)
def load_fx_eurusd() -> pd.Series:
    """Serie EUR/USD (USD per 1 EUR) dal 2005. Vuota se non ancora ingerita."""
    from trading_bot.data.ingest import load_fx_eurusd as _load
    return _load("2005-01-01")


def to_eur_curve(eq_usd: pd.Series, fx: pd.Series) -> pd.Series:
    """Converte una curva equity in USD → EUR (conto USD vissuto da un europeo).

    Valore in EUR = valore in USD / (USD per EUR). Rinormalizzata allo stesso
    punto di partenza così le curve USD ed EUR sono confrontabili a vista.
    """
    if fx.empty:
        return eq_usd
    fx_a = fx.reindex(eq_usd.index).ffill().bfill()
    raw = eq_usd / fx_a
    return raw / raw.iloc[0] * eq_usd.iloc[0]


def to_eur_returns(ret_usd: pd.Series, fx: pd.Series) -> pd.Series:
    """Converte i rendimenti giornalieri USD → EUR.

    r_EUR = (1 + r_USD) × (fx_{t-1}/fx_t) − 1: il cambio aggiunge il proprio
    moltiplicatore al rendimento del portafoglio USD.
    """
    if fx.empty:
        return ret_usd
    fx_a = fx.reindex(ret_usd.index).ffill().bfill()
    fx_ret = fx_a.pct_change().fillna(0.0)
    return (1 + ret_usd) / (1 + fx_ret) - 1


def fmt_metrics(ret: pd.Series) -> dict:
    from trading_bot.backtest.metrics import max_drawdown, sharpe, sortino
    eq = (1 + ret).cumprod()
    years = max(len(ret) / 252, 1e-9)
    cagr = eq.iloc[-1] ** (1 / years) - 1
    mdd = max_drawdown(eq)
    return {"sharpe": sharpe(ret), "sortino": sortino(ret), "cagr": cagr,
            "maxdd": mdd, "calmar": (cagr / abs(mdd)) if mdd else 0.0,
            "vol": ret.std() * np.sqrt(252), "hit": float((ret > 0).mean())}


@st.cache_data(ttl=180)
def load_logs() -> pd.DataFrame:
    """Research log completo come DataFrame."""
    with get_session() as s:
        rows = s.execute(select(ResearchLog).order_by(
            ResearchLog.created_at.desc())).scalars().all()
    return pd.DataFrame([{
        "name": r.hypothesis_name, "status": r.status,
        "universe": r.universe_size, "ic": r.ic_prescan,
        "oos": r.oos_sharpe, "pbo": r.pbo, "dsr": r.dsr,
        "created": r.created_at,
        "rationale": r.rationale or "", "skip_reason": r.skip_reason or "",
        "config": r.config_json,
    } for r in rows])


def honest_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Solo universo corrente, quarantene escluse."""
    if df.empty:
        return df
    return df[(df["universe"] == UNIVERSE)
              & (~df["status"].str.endswith("_prepit"))
              & (~df["status"].str.endswith("_enginev1"))]


@st.cache_data(ttl=300)
def load_runs() -> list[dict]:
    from trading_bot.registry import list_runs
    return list_runs()


def daemon_info() -> dict:
    """Stato daemon per le pagine: {attivo, pid, started, uptime_s}."""
    from trading_bot.research.daemon import daemon_pid, daemon_started_at
    pid = daemon_pid()
    if pid is None:
        return {"attivo": False, "pid": None, "started": None, "uptime_s": 0}
    started = daemon_started_at()
    uptime_s = 0
    if started:
        from datetime import datetime, timezone
        uptime_s = int((datetime.now(timezone.utc) - started).total_seconds())
    return {"attivo": True, "pid": pid, "started": started, "uptime_s": uptime_s}


def fmt_uptime(seconds: int) -> str:
    h, rem = divmod(seconds, 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m"
