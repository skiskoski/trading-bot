"""✅ Validazione — distribuzione path CPCV, IS vs OOS, DSR vs trial,
stato holdout."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trading_bot.config import settings
from trading_bot.gui.data import load_runs
from trading_bot.gui.glossary import section_help, spiega
from trading_bot.gui.style import (BLUE, GOLD, GREEN, RED, TEXT2, cards,
                                   pill, plotly_layout, section)

st.title("✅ Validazione")

st.caption("Tre linee di difesa contro l'overfitting: CPCV (45 path), "
           "Deflated Sharpe (multiple testing), holdout finale mai toccato.")

runs = load_runs()
runs_cpcv = [r for r in runs if r.get("cpcv")]

# ── CPCV ─────────────────────────────────────────────────────────────────────

section_help("CPCV — distribuzione dei path out-of-sample", "CPCV", "PBO")
if not runs_cpcv:
    st.info("Nessun run con CPCV ancora. I trial del daemon lo includono "
            "automaticamente.")
else:
    options = {f"#{r['id']} {r['strategy_name']} "
               f"({r['created_at']:%Y-%m-%d %H:%M})": r for r in runs_cpcv[:80]}
    sel = st.selectbox("Run", list(options.keys()), index=0)
    r = options[sel]
    cp = r["cpcv"]

    oos = np.array(cp["oos_sharpe_distribution"], dtype=float)
    iss = np.array(cp["is_sharpe_distribution"], dtype=float)
    med_is = float(np.median(iss)) if len(iss) else 0.0

    cards([
        ("Path OOS", str(cp["n_paths"]), f"k={cp['k']}", "blue"),
        ("PBO", f"{cp['pbo']:.3f}", "gate < 0.4 · neutro ≈ 0.5",
         "green" if cp["pbo"] < 0.4 else
         ("gold" if cp["pbo"] <= 0.6 else "red")),
        ("OOS Sharpe medio", f"{cp['mean_oos_sharpe']:.3f}",
         f"± {cp['std_oos_sharpe']:.3f}", "dim"),
        ("Path positivi", f"{cp['fraction_positive']:.0%}", "gate ≥ 65%",
         "green" if cp["fraction_positive"] >= 0.65 else "gold"),
    ])

    l, rr = st.columns(2)
    with l:
        figh = go.Figure()
        figh.add_trace(go.Histogram(x=oos, nbinsx=20, name="OOS",
                                    marker=dict(color=BLUE, opacity=0.85)))
        figh.add_vline(x=med_is, line=dict(color=GOLD, dash="dash"),
                       annotation_text="mediana IS",
                       annotation_font_color=GOLD)
        figh.add_vline(x=0, line=dict(color=TEXT2, dash="dot"))
        figh.update_xaxes(title="Sharpe OOS per path")
        st.plotly_chart(plotly_layout(figh, 300, legend=False),
                        width="stretch")
        st.caption("PBO = frazione di path a SINISTRA della linea oro. "
                   "Distribuzione tutta a destra dello zero = edge robusto.")

    with rr:
        figb = go.Figure()
        figb.add_trace(go.Box(y=iss, name="IS", marker_color=TEXT2))
        figb.add_trace(go.Box(y=oos, name="OOS", marker_color=BLUE))
        figb.update_yaxes(title="Sharpe")
        st.plotly_chart(plotly_layout(figb, 300, legend=False),
                        width="stretch")
        st.caption("IS e OOS allo stesso livello = niente memoria del "
                   "passato. IS molto sopra OOS = overfitting.")

# ── DSR VS TRIAL ─────────────────────────────────────────────────────────────

section_help("DSR vs numero di trial — il costo del cercare tanto",
             "DSR", "Sharpe")
pts = [(r["n_trials_used"], r["dsr"], r["strategy_name"], r["created_at"])
       for r in runs if r.get("dsr") is not None and r.get("n_trials_used")]
if pts:
    df_d = pd.DataFrame(pts, columns=["trials", "dsr", "name", "created"])
    df_new = df_d[df_d["dsr"] > 0]
    figd = go.Figure()
    figd.add_trace(go.Scatter(
        x=df_new["trials"], y=df_new["dsr"], mode="markers",
        text=df_new["name"],
        marker=dict(color=BLUE, size=8, opacity=0.75),
        hovertemplate="%{text}<br>trial #%{x} DSR=%{y:.3f}<extra></extra>"))
    figd.add_hline(y=0.95, line=dict(color=GREEN, dash="dash"),
                   annotation_text="evidenza forte 0.95",
                   annotation_font_color=GREEN)
    figd.update_xaxes(title="trial cumulativi al momento del run")
    figd.update_yaxes(title="DSR", range=[-0.05, 1.05])
    st.plotly_chart(plotly_layout(figd, 320, legend=False), width="stretch")
    n_zero = int((df_d["dsr"] == 0).sum())
    if n_zero:
        st.caption(f"{n_zero} run storici con DSR=0 spurio (bug σ_SR fisso, "
                   f"corretto) esclusi dal grafico.")
else:
    st.info("Nessun run persistito ancora.")

# ── WALK-FORWARD ─────────────────────────────────────────────────────────────

section_help("Walk-forward", "Walk-forward")
st.markdown("Il walk-forward gira on-demand da terminale — rispetta l'ordine "
            "temporale e simula l'esperienza reale:")
st.code("tradebot validate momentum_12_1", language="bash")

# ── HOLDOUT ──────────────────────────────────────────────────────────────────

section_help("Holdout — l'esame finale", "Holdout")
cards([
    ("Periodo holdout", f"dal {settings.holdout_start}",
     "MAI usato in ricerca", "gold"),
    ("Stato", "Intatto", "si consuma con un solo uso", "green"),
])
st.markdown(pill("⚠ Eseguilo UNA volta sola, quando hai deciso di andare "
                 "live: ogni sguardo lo consuma.", "gold"),
            unsafe_allow_html=True)
st.code("tradebot holdout-test", language="bash")
