"""📊 Panoramica — hero metrics, equity top-5 vs benchmark, drawdown,
heatmap mensile, regime di mercato, posizioni, segnali."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import select

from trading_bot.data.storage import ResearchLog, get_session
from trading_bot.gui.data import (GATE, compute_curves, daemon_info,
                                  fmt_metrics, fmt_uptime, load_chart_panel,
                                  load_top5)
from trading_bot.gui.glossary import section_help, spiega
from trading_bot.gui.style import (BLUE, GREEN, PALETTE, RED, TEXT2, cards,
                                   pill, plotly_layout, section)

st.title("◉ TradingBot")

# ── HERO ─────────────────────────────────────────────────────────────────────

dmn = daemon_info()
with get_session() as s:
    last = s.execute(select(ResearchLog).order_by(
        ResearchLog.created_at.desc())).scalars().first()
last_local = None
if last:
    last_local = (last.created_at.replace(tzinfo=timezone.utc).astimezone())
nxt_reb = (pd.Timestamp.today() + pd.offsets.BMonthBegin(1)).date()

acct_val, acct_delta = "—", None
try:
    from trading_bot.live.alpaca import get_alpaca_client
    _c = get_alpaca_client()
    if _c:
        a = _c.get_account()
        acct_val = f"${a.portfolio_value:,.0f}"
        try:
            h = _c.get_portfolio_history(period="1M", timeframe="1D")
            eqs = [v for v in h.get("equity", []) if v]
            if len(eqs) > 1:
                pct = (eqs[-1] / eqs[0] - 1) * 100
                acct_delta = (f"{pct:+.1f}% (1M)", "green" if pct >= 0 else "red")
        except Exception:
            pass
except Exception:
    pass

cards([
    ("Conto paper", acct_val,
     acct_delta[0] if acct_delta else None,
     acct_delta[1] if acct_delta else "dim"),
    ("Research daemon",
     f"Attivo · {fmt_uptime(dmn['uptime_s'])}" if dmn["attivo"] else "Fermo",
     (last_local.strftime("ultimo trial %H:%M") if last_local else "—"),
     "green" if dmn["attivo"] else "red"),
    ("Prossimo rebalance", str(nxt_reb), "mensile + vol-trigger", "dim"),
    ("Gate", "Onesto", GATE, "blue"),
])
spiega("OOS attivo", "PBO", "Vol targeting",
       label="ⓘ cosa significano queste card")

# ── PERFORMANCE ──────────────────────────────────────────────────────────────

section_help("Performance — equity top 5 strategie vs benchmark",
             "Sharpe", "OOS attivo", "CAGR", "PBO")
top5, any_promoted = load_top5()
if not top5:
    st.info("Il loop sta accumulando le prime strategie oneste — la sezione "
            "si popola da sola. Avvia: `tradebot research-daemon`")
else:
    if not any_promoted:
        st.markdown(pill("nessuna promossa ancora — mostro le migliori TESTATE "
                         "(gate onesto in corso)", "gold"),
                    unsafe_allow_html=True)
    data, rets = compute_curves()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["SPY"].index, y=data["SPY"], name="SPY",
                             line=dict(color=TEXT2, width=1.2, dash="dash")))
    fig.add_trace(go.Scatter(x=data["Equal-weight"].index, y=data["Equal-weight"],
                             name="Equal-weight (benchmark gate)",
                             line=dict(color="#5e5ce6", width=1.2, dash="dot")))
    i = 0
    for item in top5:
        nm = item["name"]
        if nm not in data:
            continue
        c = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatter(x=data[nm].index, y=data[nm], name=nm,
                                 line=dict(color=c, width=1.8)))
        i += 1
    fig.update_yaxes(
        type="log", title="valore di $100 investiti",
        tickvals=[100, 200, 500, 1000, 2000, 5000, 10000, 20000],
        ticktext=["$100", "$200", "$500", "$1k", "$2k", "$5k", "$10k", "$20k"],
    )
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.06),
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1A", step="year", stepmode="backward"),
                dict(count=3, label="3A", step="year", stepmode="backward"),
                dict(count=5, label="5A", step="year", stepmode="backward"),
                dict(count=10, label="10A", step="year", stepmode="backward"),
                dict(step="all", label="TUTTO"),
            ],
            bgcolor="#1c1c1e", activecolor="#0a84ff",
            font=dict(color="#f5f5f7"),
        ),
    )
    st.plotly_chart(plotly_layout(fig, 480), width="stretch")

    rows = []
    for item in top5:
        nm = item["name"]
        if nm not in rets:
            continue
        m = fmt_metrics(rets[nm])
        gw = item["config"].get("gold_weight", 0) or 0
        gm = item["config"].get("gold_mode", "defensive")
        rows.append({
            "strategia": nm,
            "stato": "★ promossa" if item["status"] == "promoted" else "testata",
            "oro": (f"{gw:.0%} {'solo in stress' if gm == 'defensive' else 'sempre'}"
                    if gw > 0 else "—"),
            "OOS attivo": round(item["oos"], 2), "PBO": round(item["pbo"], 2),
            "CAGR": f"{m['cagr']:.1%}", "Sharpe": round(m["sharpe"], 2),
            "Calmar": round(m["calmar"], 2), "MaxDD": f"{m['maxdd']:.1%}",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # ── RISCHIO ──────────────────────────────────────────────────────────────
    left, right = st.columns(2)
    with left:
        section_help("Rischio — drawdown (underwater)", "MaxDD", "Calmar")
        fig2 = go.Figure()
        i = 0
        for item in top5[:3]:
            nm = item["name"]
            if nm not in rets:
                continue
            eq = (1 + rets[nm]).cumprod()
            dd = eq / eq.cummax() - 1
            fig2.add_trace(go.Scatter(x=dd.index, y=dd, name=nm,
                                      fill="tozeroy",
                                      line=dict(color=PALETTE[i % 5], width=1)))
            i += 1
        fig2.update_yaxes(tickformat=".0%")
        st.plotly_chart(plotly_layout(fig2, 260), width="stretch")

    with right:
        section("Rendimenti mensili — best strategy")
        best = top5[0]["name"]
        if best in rets:
            mr = rets[best].resample("ME").apply(lambda r: (1 + r).prod() - 1)
            hm = pd.DataFrame({"y": mr.index.year, "m": mr.index.month,
                               "v": mr.values})
            pv = hm.pivot_table(index="y", columns="m", values="v")
            fig3 = go.Figure(go.Heatmap(
                z=pv.values * 100, x=[f"{m:02d}" for m in pv.columns],
                y=pv.index,
                colorscale=[[0, RED], [0.5, "#1c1c1e"], [1, GREEN]],
                zmid=0, text=np.round(pv.values * 100, 1),
                texttemplate="%{text}", textfont=dict(size=10),
                showscale=False))
            fig3.update_yaxes(autorange="reversed")
            st.plotly_chart(plotly_layout(fig3, 260, legend=False),
                            width="stretch")

# ── ESECUZIONE ───────────────────────────────────────────────────────────────

c1, c2, c3 = st.columns([1.2, 1, 1])

with c1:
    section_help("Esecuzione — regime di mercato", "Regime filter")
    try:
        panel = load_chart_panel()
        spy = panel["SPY"].dropna()
        sma200 = spy.rolling(200).mean()
        vol21 = spy.pct_change().rolling(21).std() * np.sqrt(252)
        above = spy.iloc[-1] > sma200.iloc[-1]
        v = float(vol21.iloc[-1])
        cards([
            ("SPY vs SMA200", "RISK-ON" if above else "RISK-OFF",
             f"{(spy.iloc[-1]/sma200.iloc[-1]-1)*100:+.1f}%",
             "green" if above else "red"),
            ("Vol mercato 21g", f"{v:.0%}",
             "trigger settimanale 25%" + (" — ATTIVO" if v > 0.25 else ""),
             "red" if v > 0.25 else "dim"),
        ])
        figr = go.Figure()
        tail = spy.iloc[-504:]
        figr.add_trace(go.Scatter(x=tail.index, y=tail, name="SPY",
                                  line=dict(color=BLUE, width=1.5)))
        figr.add_trace(go.Scatter(x=tail.index, y=sma200.iloc[-504:],
                                  name="SMA200",
                                  line=dict(color=TEXT2, width=1, dash="dash")))
        st.plotly_chart(plotly_layout(figr, 220), width="stretch")
    except Exception as e:
        st.warning(f"Regime non disponibile: {e}")

with c2:
    section("Posizioni correnti")
    try:
        from trading_bot.live.alpaca import get_alpaca_client
        client = get_alpaca_client()
        if client:
            pos = client.get_positions()
            if pos:
                figp = go.Figure(go.Pie(
                    labels=[p.symbol for p in pos],
                    values=[p.market_value for p in pos],
                    hole=0.62, marker=dict(colors=PALETTE * 4),
                    textinfo="label+percent", textfont=dict(size=10)))
                st.plotly_chart(plotly_layout(figp, 230, legend=False),
                                width="stretch")
                st.dataframe(pd.DataFrame([{
                    "sym": p.symbol, "valore": f"${p.market_value:,.0f}",
                    "P&L": f"{p.unrealized_plpc:+.1%}"} for p in pos]),
                    width="stretch", hide_index=True, height=180)
            else:
                st.info("Nessuna posizione aperta.")
        else:
            st.info("Alpaca non configurato (.env).")
    except Exception as e:
        st.warning(f"Posizioni: {e}")

with c3:
    section_help("Segnali (best strategy + overlay)",
                 "Vol targeting", "Crisis alpha")
    if st.button("Genera segnali di oggi", width="stretch"):
        try:
            from trading_bot.live.runner import apply_portfolio_overlays
            from trading_bot.live.signals import generate_signals
            with st.spinner("Calcolo…"):
                sig = generate_signals()
                final = apply_portfolio_overlays(dict(sig.positions))
            st.markdown(pill(f"{sig.strategy_name} — esposizione "
                             f"{sum(final.values()):.0%}", "green"),
                        unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(
                [{"sym": k, "peso": f"{v:.1%}"} for k, v in
                 sorted(final.items(), key=lambda x: -x[1])]),
                width="stretch", hide_index=True, height=300)
        except Exception as e:
            st.error(f"Errore segnali: {e}")
    else:
        st.caption("Pesi della strategia migliore con sleeve oro 20% e "
                   "vol targeting 25% già applicati.")
