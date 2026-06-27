"""📈 Strategie — dettaglio per strategia: composizione segnali, equity vs
benchmark, drawdown, heatmap mensile, metriche complete."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trading_bot.gui.data import (compute_curves, fmt_metrics, load_top5)
from trading_bot.gui.glossary import section_help, spiega
from trading_bot.gui.style import (BLUE, GREEN, RED, TEXT2, cards, pill,
                                   plotly_layout, section)

st.title("📈 Strategie")

top5, any_promoted = load_top5()
if not top5:
    st.info("Nessuna strategia ancora — avvia `tradebot research-daemon` e "
            "torna tra qualche ora.")
    st.stop()

names = [t["name"] for t in top5]
sel = st.selectbox("Strategia", names, index=0)
item = next(t for t in top5 if t["name"] == sel)
cfg = item["config"]

# ── IDENTITÀ ─────────────────────────────────────────────────────────────────

stato = ("★ promossa" if item["status"] == "promoted" else "testata")
st.markdown(
    pill(stato, "green" if item["status"] == "promoted" else "gold")
    + " " + pill(f"top {cfg.get('top_n', 10)} titoli", "blue"),
    unsafe_allow_html=True)

section_help("Composizione — segnali, filtri, regime",
             "Momentum 12-1", "Regime filter", "TSMOM")

sig_rows = [{
    "segnale": s["feature"],
    "parametri": ", ".join(f"{k}={v}" for k, v in s.get("params", {}).items()) or "—",
    "peso": s.get("weight", 1.0),
    "direzione": "inverso" if s.get("negate") else "diretto",
} for s in cfg.get("signals", [])]
st.dataframe(pd.DataFrame(sig_rows), width="stretch", hide_index=True)

extra = []
for f in cfg.get("filters", []):
    extra.append(f"filtro: {f['feature']} "
                 f"({', '.join(f'{k}={v}' for k, v in f.get('params', {}).items())})")
if cfg.get("regime"):
    rg = cfg["regime"]
    extra.append(f"regime: {rg['feature']} su {rg['symbol']}")
gw = cfg.get("gold_weight", 0) or 0
if gw > 0:
    extra.append(f"oro {gw:.0%} ({'solo in stress' if cfg.get('gold_mode') == 'defensive' else 'sempre'})")
if extra:
    st.caption(" · ".join(extra))

# ── PERFORMANCE ──────────────────────────────────────────────────────────────

data, rets = compute_curves()
if sel not in rets:
    st.warning("Backtest non disponibile per questa strategia (dati mancanti).")
    st.stop()

ret = rets[sel]
m = fmt_metrics(ret)

section_help("Performance", "Sharpe", "CAGR", "Calmar", "MaxDD")
cards([
    ("CAGR", f"{m['cagr']:.1%}", None, "green" if m["cagr"] > 0 else "red"),
    ("Sharpe", f"{m['sharpe']:.2f}", None, "blue"),
    ("Calmar", f"{m['calmar']:.2f}", None, "dim"),
    ("MaxDD", f"{m['maxdd']:.1%}", None, "red"),
])
cards([
    ("OOS attivo", f"{item['oos']:.2f}", "gate ≥ 0.5",
     "green" if item["oos"] >= 0.5 else "gold"),
    ("PBO", f"{item['pbo']:.2f}", "gate < 0.4",
     "green" if item["pbo"] < 0.4 else "gold"),
    ("Vol annua", f"{m['vol']:.1%}", "target 25%", "dim"),
    ("Hit rate", f"{m['hit']:.1%}", "giorni positivi", "dim"),
])
spiega("OOS attivo", "PBO", "DSR", label="ⓘ metriche del gate")

fig = go.Figure()
fig.add_trace(go.Scatter(x=data["SPY"].index, y=data["SPY"], name="SPY",
                         line=dict(color=TEXT2, width=1.2, dash="dash")))
fig.add_trace(go.Scatter(x=data[sel].index, y=data[sel], name=sel,
                         line=dict(color=BLUE, width=1.8)))
fig.update_yaxes(type="log", title="valore di $100 investiti")
st.plotly_chart(plotly_layout(fig, 380), width="stretch")

# ── RISCHIO ──────────────────────────────────────────────────────────────────

l, r = st.columns(2)
with l:
    section_help("Rischio — drawdown", "MaxDD")
    eq = (1 + ret).cumprod()
    dd = eq / eq.cummax() - 1
    figd = go.Figure(go.Scatter(x=dd.index, y=dd, fill="tozeroy",
                                line=dict(color=RED, width=1)))
    figd.update_yaxes(tickformat=".0%")
    st.plotly_chart(plotly_layout(figd, 260, legend=False), width="stretch")

with r:
    section("Rendimenti mensili")
    mr = ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    hm = pd.DataFrame({"y": mr.index.year, "m": mr.index.month, "v": mr.values})
    pv = hm.pivot_table(index="y", columns="m", values="v")
    figh = go.Figure(go.Heatmap(
        z=pv.values * 100, x=[f"{mm:02d}" for mm in pv.columns], y=pv.index,
        colorscale=[[0, RED], [0.5, "#1c1c1e"], [1, GREEN]],
        zmid=0, text=np.round(pv.values * 100, 1),
        texttemplate="%{text}", textfont=dict(size=9), showscale=False))
    figh.update_yaxes(autorange="reversed")
    st.plotly_chart(plotly_layout(figh, 260, legend=False), width="stretch")

# ── ROLLING ──────────────────────────────────────────────────────────────────

section_help("Stabilità — Sharpe rolling 12 mesi", "Sharpe")
roll = ret.rolling(252).mean() / ret.rolling(252).std() * np.sqrt(252)
figr = go.Figure(go.Scatter(x=roll.index, y=roll,
                            line=dict(color=GREEN, width=1.4)))
figr.add_hline(y=0, line=dict(color=TEXT2, dash="dot"))
st.plotly_chart(plotly_layout(figr, 240, legend=False), width="stretch")
st.caption("Un edge sano resta sopra zero nella maggior parte delle finestre; "
           "lunghi periodi sotto zero = edge episodico o regime sbagliato.")
