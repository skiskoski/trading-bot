"""💼 Live — conto Alpaca: equity, posizioni, ordini, PnL."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trading_bot.gui.glossary import section_help, spiega
from trading_bot.gui.style import (BLUE, GREEN, PALETTE, RED, TEXT2, cards,
                                   pill, plotly_layout, section)

st.title("💼 Live")

try:
    from trading_bot.live.alpaca import get_alpaca_client
    client = get_alpaca_client()
except Exception as e:
    client = None
    st.error(f"Errore client Alpaca: {e}")

if client is None:
    st.markdown(pill("Alpaca non configurato", "red"), unsafe_allow_html=True)
    st.markdown(
        "Per collegare il conto paper:\n"
        "1. Crea le chiavi su [alpaca.markets](https://alpaca.markets)\n"
        "2. Copia `.env.example` in `.env`\n"
        "3. Compila `TRADEBOT_ALPACA_API_KEY` e `TRADEBOT_ALPACA_SECRET_KEY`\n"
        "4. Riavvia la GUI")
    st.stop()

# ── CONTO ────────────────────────────────────────────────────────────────────

try:
    acct = client.get_account()
    mkt_open = False
    try:
        mkt_open = client.is_market_open()
    except Exception:
        pass

    # PnL da portfolio history
    pnl_day, pnl_mtd, pnl_ytd = None, None, None
    eq_series = None
    try:
        h = client.get_portfolio_history(period="1A", timeframe="1D")
        eqs = h.get("equity", []) or []
        ts = h.get("timestamp", []) or []
        pairs = [(t, v) for t, v in zip(ts, eqs) if v]
        if len(pairs) > 1:
            idx = pd.to_datetime([p[0] for p in pairs], unit="s")
            eq_series = pd.Series([p[1] for p in pairs], index=idx)
            pnl_day = eq_series.iloc[-1] / eq_series.iloc[-2] - 1
            this_month = eq_series[eq_series.index.to_period("M")
                                   == eq_series.index[-1].to_period("M")]
            if len(this_month) > 1:
                pnl_mtd = this_month.iloc[-1] / this_month.iloc[0] - 1
            this_year = eq_series[eq_series.index.year
                                  == eq_series.index[-1].year]
            if len(this_year) > 1:
                pnl_ytd = this_year.iloc[-1] / this_year.iloc[0] - 1
    except Exception:
        pass

    def _pnl(v):
        return ("—", "dim") if v is None else (f"{v:+.2%}",
                                               "green" if v >= 0 else "red")

    section("Conto")
    d, m, y = _pnl(pnl_day), _pnl(pnl_mtd), _pnl(pnl_ytd)
    cards([
        ("Equity", f"${acct.portfolio_value:,.0f}",
         "mercato APERTO" if mkt_open else "mercato chiuso",
         "green" if mkt_open else "dim"),
        ("PnL oggi", d[0], None, d[1]),
        ("PnL mese", m[0], None, m[1]),
        ("PnL anno", y[0], None, y[1]),
    ])
    cards([
        ("Cash", f"${acct.cash:,.0f}", None, "dim"),
        ("Buying power", f"${acct.buying_power:,.0f}", None, "dim"),
    ])

    if eq_series is not None:
        section("Equity del conto — ultimo anno")
        fige = go.Figure(go.Scatter(x=eq_series.index, y=eq_series,
                                    line=dict(color=BLUE, width=1.6),
                                    fill="tozeroy",
                                    fillcolor="rgba(10,132,255,.08)"))
        fige.update_yaxes(tickformat="$,.0f",
                          rangemode="tozero" if eq_series.min() > 0 else "normal")
        st.plotly_chart(plotly_layout(fige, 300, legend=False),
                        width="stretch")
except Exception as e:
    st.error(f"Conto non disponibile: {e}")

# ── POSIZIONI ────────────────────────────────────────────────────────────────

l, r = st.columns([1.4, 1])

with l:
    section("Posizioni aperte")
    try:
        pos = client.get_positions()
        if pos:
            st.dataframe(pd.DataFrame([{
                "sym": p.symbol,
                "qty": p.qty,
                "prezzo": f"${p.current_price:,.2f}",
                "valore": f"${p.market_value:,.0f}",
                "P&L $": f"${p.unrealized_pl:+,.0f}",
                "P&L %": f"{p.unrealized_plpc:+.1%}",
            } for p in sorted(pos, key=lambda x: -x.market_value)]),
                width="stretch", hide_index=True, height=380)
        else:
            st.info("Nessuna posizione aperta.")
    except Exception as e:
        st.warning(f"Posizioni: {e}")

with r:
    section("Allocazione")
    try:
        pos = client.get_positions()
        if pos:
            figp = go.Figure(go.Pie(
                labels=[p.symbol for p in pos],
                values=[p.market_value for p in pos],
                hole=0.62, marker=dict(colors=PALETTE * 5),
                textinfo="label+percent", textfont=dict(size=10)))
            st.plotly_chart(plotly_layout(figp, 380, legend=False),
                            width="stretch")
    except Exception:
        pass

# ── ORDINI ───────────────────────────────────────────────────────────────────

section("Ordini")
tab_open, tab_closed = st.tabs(["Aperti", "Chiusi (ultimi)"])
with tab_open:
    try:
        orders = client.get_orders(status="open")
        if orders:
            st.dataframe(pd.DataFrame([{
                "sym": o.symbol, "lato": o.side, "qty": o.qty,
                "stato": o.status, "inviato": o.submitted_at,
            } for o in orders]), width="stretch", hide_index=True)
        else:
            st.info("Nessun ordine pendente.")
    except Exception as e:
        st.warning(f"Ordini: {e}")
with tab_closed:
    try:
        orders = client.get_orders(status="closed")
        if orders:
            st.dataframe(pd.DataFrame([{
                "sym": o.symbol, "lato": o.side, "qty": o.qty,
                "stato": o.status,
                "prezzo medio": (f"${o.filled_avg_price:,.2f}"
                                 if o.filled_avg_price else "—"),
                "inviato": o.submitted_at,
            } for o in orders[:30]]), width="stretch", hide_index=True)
        else:
            st.info("Nessun ordine eseguito di recente.")
    except Exception as e:
        st.warning(f"Ordini: {e}")

st.caption("Il ciclo giornaliero (ingest → segnali → rebalance → notifica) "
           "gira con `tradebot run-daily` — di default in dry-run, "
           "`--live` per ordini reali sul conto paper.")
