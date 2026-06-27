"""📅 Storico — tutti i run persistiti, con filtri e dettaglio."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trading_bot.gui.data import load_runs
from trading_bot.gui.glossary import section_help, spiega
from trading_bot.gui.style import (BLUE, GREEN, RED, TEXT2, cards,
                                   plotly_layout, section)

st.title("📅 Storico run")

runs = load_runs()
if not runs:
    st.info("Nessun run persistito. Ogni backtest (`tradebot run`, daemon, "
            "CPCV) finisce qui.")
    st.stop()

df = pd.DataFrame([{
    "id": r["id"],
    "strategia": r["strategy_name"],
    "dal": r["start_date"], "al": r["end_date"],
    "universo": r["universe_size"],
    "PIT": "sì" if r["use_pit"] else "no",
    "Sharpe": round(r["metrics"].get("Sharpe", float("nan")), 3)
    if r["metrics"].get("Sharpe") is not None else None,
    "CAGR": r["metrics"].get("CAGR"),
    "MaxDD": (round(r["metrics"]["MaxDrawdown"], 3)
              if isinstance(r["metrics"].get("MaxDrawdown"), (int, float))
              else None),
    "PSR": round(r["psr"], 3) if r["psr"] is not None else None,
    "DSR": round(r["dsr"], 3) if r["dsr"] is not None else None,
    "PBO": (round(r["cpcv"]["pbo"], 3) if r.get("cpcv") else None),
    "trial #": r["n_trials_used"],
    "quando": r["created_at"],
} for r in runs])

# ── FILTRI ───────────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
with c1:
    strategie = sorted(df["strategia"].unique())
    sel_strat = st.multiselect("Strategia", strategie, default=[])
with c2:
    solo_pit = st.checkbox("Solo PIT", value=False)
with c3:
    solo_cpcv = st.checkbox("Solo con CPCV", value=False)
with c4:
    min_sharpe = st.number_input("Sharpe min", value=-10.0, step=0.5)

filtered = df.copy()
if sel_strat:
    filtered = filtered[filtered["strategia"].isin(sel_strat)]
if solo_pit:
    filtered = filtered[filtered["PIT"] == "sì"]
if solo_cpcv:
    filtered = filtered[filtered["PBO"].notna()]
filtered = filtered[(filtered["Sharpe"].isna())
                    | (filtered["Sharpe"] >= min_sharpe)]

cards([
    ("Run totali", str(len(df)), None, "dim"),
    ("Filtrati", str(len(filtered)), None, "blue"),
    ("Strategie uniche", str(df["strategia"].nunique()), None, "dim"),
])
spiega("PIT", "DSR", "PBO", label="ⓘ colonne della tabella")

st.dataframe(filtered, width="stretch", hide_index=True, height=420)

# ── DETTAGLIO RUN ────────────────────────────────────────────────────────────

section_help("Dettaglio run", "Sharpe", "MaxDD", "PBO", "DSR")
sel_id = st.selectbox("Run id", filtered["id"].tolist())
if sel_id:
    from trading_bot.registry import get_run
    try:
        r = get_run(int(sel_id))
        m = r["metrics"]
        cards([
            ("Sharpe", f"{m.get('Sharpe', 0):.2f}", None, "blue"),
            ("CAGR", (f"{m['CAGR']:.1%}" if isinstance(m.get("CAGR"), float)
                      else str(m.get("CAGR", "—"))), None, "green"),
            ("MaxDD", (f"{m['MaxDrawdown']:.1%}"
                       if isinstance(m.get("MaxDrawdown"), float)
                       else str(m.get("MaxDrawdown", "—"))), None, "red"),
            ("DSR", f"{r['dsr']:.3f}", f"trial #{r['n_trials_used']}", "gold"),
        ])
        eq = r["equity"]
        if not eq.empty:
            fig = go.Figure(go.Scatter(x=eq.index, y=eq.values,
                                       line=dict(color=BLUE, width=1.5)))
            fig.update_yaxes(title="equity")
            st.plotly_chart(plotly_layout(fig, 300, legend=False),
                            width="stretch")
    except Exception as e:
        st.warning(f"Run non caricabile: {e}")
