"""Statistical-health page: trial counter, DSR distribution, IC distribution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from trading_bot.registry import list_runs, trial_count

st.set_page_config(page_title="Statistical health", page_icon="🩺", layout="wide")

st.title("🩺 Statistical health")

st.caption(
    "Aggregate honesty diagnostics across every run. High DSR concentration "
    "near zero is *fine* and expected — it means we're not over-claiming "
    "statistical significance. The danger sign is DSR ≥ 0.95 on many runs "
    "without a corresponding economic story."
)

n_total = trial_count()
runs = list_runs()
st.metric("Cumulative trial counter (N)", n_total)
st.metric("Distinct runs in registry", len(runs))

if not runs:
    st.info("No runs yet.")
    st.stop()

df = pd.DataFrame(
    [
        {
            "strategy": r["strategy_name"],
            "Sharpe": r["metrics"].get("Sharpe", float("nan")),
            "DSR": r["dsr"],
            "PSR": r["psr"],
            "IC_mean": r["metrics"].get("IC_mean", float("nan")),
            "CAGR": r["metrics"].get("CAGR", float("nan")),
            "MaxDD": r["metrics"].get("MaxDrawdown", float("nan")),
        }
        for r in runs
    ]
)

col1, col2 = st.columns(2)
with col1:
    fig = px.histogram(
        df, x="Sharpe", nbins=20, title="Sharpe distribution across runs",
        marginal="rug",
    )
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.histogram(
        df, x="DSR", nbins=20, title="Deflated Sharpe distribution",
        marginal="rug",
    )
    fig.add_vline(x=0.95, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    fig = px.histogram(
        df, x="IC_mean", nbins=20, title="IC distribution",
        marginal="rug",
    )
    fig.add_vline(x=0.02, line_dash="dot", line_color="green",
                  annotation_text="0.02 = 'good' daily signal")
    st.plotly_chart(fig, use_container_width=True)
with col4:
    fig = px.scatter(
        df, x="Sharpe", y="DSR",
        hover_data=["strategy", "IC_mean", "CAGR"],
        title="Sharpe vs DSR (which runs survive multiple testing?)",
    )
    fig.add_hline(y=0.95, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Reality check warning")
n_above_dsr = int((df["DSR"] >= 0.95).sum())
n_above_sharpe = int((df["Sharpe"] >= 1.0).sum())
if n_above_dsr == 0 and n_above_sharpe > 3:
    st.warning(
        f"{n_above_sharpe} runs show Sharpe ≥ 1.0 but **none** pass DSR ≥ 0.95. "
        "Classic multiple-testing trap: with a large search space, some configs "
        "produce high Sharpe by luck alone. Do NOT promote any of these to "
        "paper trading without independent OOS confirmation."
    )
elif n_above_dsr > 0:
    st.success(
        f"{n_above_dsr} runs pass DSR ≥ 0.95 with the current trial count N={n_total}. "
        "These are candidates for paper-trading promotion."
    )
else:
    st.info("Neutral state — keep researching, keep tracking N honestly.")
