"""Market Storm — regime/risk overlay dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trading_bot.gui.style import BLUE, GOLD, GREEN, RED, TEXT2, cards, plotly_layout
from trading_bot.risk.market_storm import load_storm_reports

st.title("🌩 Meteo Mercato")
st.caption("Market Storm è un risk overlay: misura quanto rischio accettare, non genera alpha.")

reports = load_storm_reports()
if not reports:
    st.info("Nessun report Market Storm salvato. Genera il primo da terminale:")
    st.code("tradebot storm-report momentum_12_1 --top 300", language="bash")
    st.stop()

latest_by_strategy = {}
for report in reports:
    name = report.get("strategy_name", "market")
    if name not in latest_by_strategy:
        latest_by_strategy[name] = report

rows = []
for name, report in latest_by_strategy.items():
    cur = report.get("current", {})
    overlay = report.get("overlay")
    failed = []
    if overlay:
        failed = [k for k, ok in overlay.get("gates", {}).items() if not ok]
    rows.append({
        "Strategia": name,
        "Regime": cur.get("regime", "n/a"),
        "Score": cur.get("storm_score"),
        "Azione": cur.get("recommended_action", "n/a"),
        "Exposure": cur.get("exposure_scale"),
        "Overlay": report.get("decision", "n/a"),
        "Gate falliti": ", ".join(failed) if failed else "nessuno",
        "Comando": f"tradebot storm-report {name}" if name != "market" else "tradebot market-storm",
    })

overview = pd.DataFrame(rows)
current = reports[0].get("current", {})
regime = current.get("regime", "n/a")
score = float(current.get("storm_score", 0.0))
color = "green" if regime == "calm" else "gold" if regime == "unstable" else "red"

cards([
    ("Regime attuale", regime, current.get("recommended_action", "n/a"), color),
    ("Storm score", f"{score:.0f}/100", "0=calm · 100=panic", color),
    ("Exposure", f"{float(current.get('exposure_scale', 1.0)):.0%}", "overlay risk", "blue"),
    ("Report", str(len(reports)), "storico salvato", "dim"),
])

st.dataframe(overview, width="stretch", hide_index=True)

selected_strategy = st.selectbox("Report da ispezionare", list(latest_by_strategy.keys()))
report = latest_by_strategy[selected_strategy]
current = report.get("current", {})
st.code(
    f"tradebot storm-report {selected_strategy} --top 300"
    if selected_strategy != "market" else "tradebot market-storm --top 300",
    language="bash",
)

components = current.get("components", {})
if components:
    comp_df = pd.DataFrame([
        {"Componente": key, "Score": value}
        for key, value in sorted(components.items(), key=lambda kv: kv[1], reverse=True)
    ])
    fig = go.Figure(go.Bar(
        x=comp_df["Componente"],
        y=comp_df["Score"],
        marker_color=[
            RED if v >= 75 else GOLD if v >= 50 else BLUE if v >= 25 else GREEN
            for v in comp_df["Score"]
        ],
    ))
    fig.update_yaxes(title="Stress score", range=[0, 100])
    st.plotly_chart(plotly_layout(fig, 320, legend=False), width="stretch")
    st.dataframe(comp_df, width="stretch", hide_index=True)

tail = report.get("history_tail", [])
if tail:
    hist = pd.DataFrame([
        {
            "dt": row.get("dt"),
            "storm_score": row.get("storm_score"),
            "regime": row.get("regime"),
            "exposure": row.get("exposure_scale"),
        }
        for row in tail
    ])
    hist["dt"] = pd.to_datetime(hist["dt"], errors="coerce")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["dt"], y=hist["storm_score"], mode="lines",
        name="Storm score", line=dict(color=BLUE, width=2),
    ))
    fig.add_hline(y=25, line=dict(color=TEXT2, dash="dot"))
    fig.add_hline(y=50, line=dict(color=GOLD, dash="dot"))
    fig.add_hline(y=75, line=dict(color=RED, dash="dot"))
    fig.update_yaxes(title="Score", range=[0, 100])
    st.plotly_chart(plotly_layout(fig, 340, legend=False), width="stretch")

overlay = report.get("overlay")
if overlay:
    st.subheader("Overlay validation")
    gates = overlay.get("gates", {})
    passed = sum(bool(v) for v in gates.values())
    cards([
        ("Decisione", "PASS" if overlay.get("passed") else "FAIL",
         report.get("decision", "n/a"), "green" if overlay.get("passed") else "red"),
        ("Gate", f"{passed}/{max(1, len(gates))}", "overlay", "blue"),
    ])
    metrics = []
    base = overlay.get("base_metrics", {})
    over = overlay.get("overlay_metrics", {})
    delta = overlay.get("deltas", {})
    for key in base:
        metrics.append({
            "Metric": key,
            "Base": base.get(key),
            "Overlay": over.get(key),
            "Delta": delta.get(key),
        })
    st.dataframe(pd.DataFrame(metrics), width="stretch", hide_index=True)

    worst = overlay.get("worst_months", [])
    if worst:
        st.subheader("Peggiori mesi SPY")
        st.dataframe(pd.DataFrame(worst), width="stretch", hide_index=True)

with st.expander("Storico report", expanded=False):
    history = pd.DataFrame([
        {
            "Data": r.get("generated_at", "")[:19],
            "Strategia": r.get("strategy_name", "market"),
            "Regime": r.get("current", {}).get("regime", "n/a"),
            "Score": r.get("current", {}).get("storm_score"),
            "Decisione": r.get("decision", "n/a"),
            "File": r.get("_path", "").split("/")[-1],
        }
        for r in reports[:50]
    ])
    st.dataframe(history, width="stretch", hide_index=True)
