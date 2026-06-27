"""🎯 Portafoglio — overlay attivi (vol target, sleeve oro, rebalance
adattivo), vol realizzata vs target, allocazione corrente."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trading_bot.config import settings
from trading_bot.gui.data import compute_curves, load_top5
from trading_bot.gui.glossary import section_help, spiega
from trading_bot.gui.style import (BLUE, GOLD, GREEN, PALETTE, RED, TEXT2,
                                   cards, pill, plotly_layout, section)

st.title("🎯 Portafoglio")

st.caption("La ricerca trova l'alpha grezzo; gli OVERLAY lo rendono "
           "vivibile: vol targeting, sleeve oro anticrisi, rebalance "
           "adattivo. Qui vedi gli overlay al lavoro.")

# ── OVERLAY ATTIVI ───────────────────────────────────────────────────────────

section_help("Overlay attivi", "Vol targeting", "Crisis alpha", "Turnover")
cards([
    ("Vol target", f"{settings.vol_target:.0%}",
     "Moreira-Muir 2017 — growth", "blue"),
    ("Sleeve oro", f"{settings.gold_sleeve_weight:.0%}",
     "TSMOM, crisis alpha", "gold"),
    ("Rebalance", "Mensile + vol-trigger",
     f"trigger {settings.vol_trigger:.0%} annualizzato", "dim"),
    ("Panic scale", f"{settings.panic_scale:.0%}",
     "esposizione equity in panico", "dim"),
])

# ── VOL REALIZZATA VS TARGET ─────────────────────────────────────────────────

top5, _ = load_top5()
if not top5:
    st.info("La sezione si popola quando il loop ha strategie testate.")
    st.stop()

data, rets = compute_curves()
best = top5[0]["name"]

if best in rets:
    ret = rets[best]
    vol_roll = ret.rolling(21).std() * np.sqrt(252)

    section_help(f"Vol realizzata (21g) vs target — {best}", "Vol targeting")
    figv = go.Figure()
    figv.add_trace(go.Scatter(x=vol_roll.index, y=vol_roll,
                              name="vol realizzata 21g",
                              line=dict(color=BLUE, width=1.2)))
    figv.add_hline(y=settings.vol_target,
                   line=dict(color=GOLD, dash="dash"),
                   annotation_text=f"target {settings.vol_target:.0%}",
                   annotation_font_color=GOLD)
    figv.update_yaxes(tickformat=".0%")
    st.plotly_chart(plotly_layout(figv, 300), width="stretch")

    scale = (settings.vol_target /
             max(float(vol_roll.dropna().iloc[-1]), 1e-9))
    scale = min(max(scale, 0.2), 1.0)
    sopra = float(vol_roll.dropna().iloc[-1]) > settings.vol_target
    cards([
        ("Vol realizzata oggi", f"{float(vol_roll.dropna().iloc[-1]):.1%}",
         "sopra target" if sopra else "sotto target",
         "red" if sopra else "green"),
        ("Scala esposizione", f"{scale:.0%}",
         "min(target/realizzata, 100%) — floor 20%", "blue"),
    ])

# ── PROSSIMO REBALANCE ───────────────────────────────────────────────────────

section("Calendario rebalance")
from trading_bot.live.runner import is_rebalance_day, is_volatility_rebalance_day

oggi = pd.Timestamp.today().normalize()
nxt = (oggi + pd.offsets.BMonthBegin(1)).date()
reb_oggi = is_rebalance_day(oggi)
try:
    vol_trig = is_volatility_rebalance_day(oggi)
except Exception:
    vol_trig = False

cards([
    ("Oggi è rebalance day?", "SÌ" if reb_oggi else "No",
     "primo giorno di borsa del mese", "green" if reb_oggi else "dim"),
    ("Vol-trigger oggi?", "ATTIVO" if vol_trig else "No",
     "lunedì + vol > trigger", "red" if vol_trig else "dim"),
    ("Prossimo rebalance mensile", str(nxt), None, "blue"),
])

# ── ALLOCAZIONE CORRENTE ─────────────────────────────────────────────────────

section_help("Allocazione corrente (conto Alpaca)", "Crisis alpha")
try:
    from trading_bot.live.alpaca import get_alpaca_client
    client = get_alpaca_client()
    if client:
        pos = client.get_positions()
        acct = client.get_account()
        if pos:
            tot = sum(p.market_value for p in pos)
            cash = acct.cash
            figp = go.Figure(go.Pie(
                labels=[p.symbol for p in pos] + ["CASH"],
                values=[p.market_value for p in pos] + [max(cash, 0)],
                hole=0.62, marker=dict(colors=PALETTE * 5 + ["#3a3a3c"]),
                textinfo="label+percent", textfont=dict(size=11)))
            st.plotly_chart(plotly_layout(figp, 320, legend=False),
                            width="stretch")
            inv = tot / max(tot + max(cash, 0), 1e-9)
            cards([
                ("Investito", f"{inv:.0%}", f"${tot:,.0f}", "blue"),
                ("Cash", f"{1-inv:.0%}", f"${cash:,.0f}", "dim"),
            ])
        else:
            st.info("Nessuna posizione aperta — il portafoglio è 100% cash.")
    else:
        st.info("Alpaca non configurato: aggiungi le chiavi in `.env` "
                "(vedi `.env.example`).")
except Exception as e:
    st.warning(f"Alpaca: {e}")

# ── PERCHÉ L'ORO ─────────────────────────────────────────────────────────────

section_help("Perché la sleeve oro — performance nei mesi rossi dell'equity",
             "Crisis alpha", "TSMOM")
try:
    from trading_bot.gui.data import load_chart_panel
    panel = load_chart_panel()
    if "GLD" in panel.columns:
        spy_m = panel["SPY"].resample("ME").last().pct_change().dropna()
        gld_m = panel["GLD"].resample("ME").last().pct_change().dropna()
        both = pd.concat([spy_m, gld_m], axis=1, keys=["SPY", "GLD"]).dropna()
        red_months = both[both["SPY"] < -0.03]
        if len(red_months) > 5:
            figc = go.Figure()
            figc.add_trace(go.Bar(x=red_months.index, y=red_months["SPY"],
                                  name="SPY", marker_color=RED))
            figc.add_trace(go.Bar(x=red_months.index, y=red_months["GLD"],
                                  name="GLD", marker_color=GOLD))
            figc.update_yaxes(tickformat=".0%")
            st.plotly_chart(plotly_layout(figc, 280), width="stretch")
            beat = float((red_months["GLD"] > red_months["SPY"]).mean())
            st.caption(f"Nei {len(red_months)} mesi con SPY sotto −3%, l'oro "
                       f"ha fatto meglio dell'equity il {beat:.0%} delle "
                       f"volte. È questo il 'crisis alpha' che compri con la "
                       f"sleeve al {settings.gold_sleeve_weight:.0%}.")
except Exception as e:
    st.warning(f"Analisi oro: {e}")
