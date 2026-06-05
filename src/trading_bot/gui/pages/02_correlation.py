"""Correlation heatmap between strategy returns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trading_bot.registry import latest_run_per_strategy

st.set_page_config(page_title="Correlation", page_icon="🔗", layout="wide")

st.title("🔗 Strategy correlation matrix")
st.caption(
    "Pairwise correlation of daily returns across strategies. Low/negative "
    "values mean genuine diversification — exactly what an Asness-Moskowitz-"
    "Pedersen-style multi-sleeve portfolio needs."
)

runs = latest_run_per_strategy()
if len(runs) < 2:
    st.info("Need at least 2 strategies with runs to compute correlations.")
    st.stop()

# Build a wide returns DataFrame
panel = pd.DataFrame()
for r in runs:
    ret = r["returns"]
    if ret.empty:
        continue
    panel[r["strategy_name"]] = ret

if panel.shape[1] < 2:
    st.info("Need at least 2 non-empty return series.")
    st.stop()

corr = panel.corr()

fig = go.Figure(
    data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        zmin=-1,
        zmax=1,
        colorscale="RdBu_r",
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}",
        hovertemplate="%{x} × %{y}: %{z:.3f}<extra></extra>",
    )
)
fig.update_layout(
    height=600,
    margin=dict(l=20, r=20, t=20, b=20),
    yaxis=dict(autorange="reversed"),
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Diagnostics")

# Average pairwise correlation (off-diagonal)
mask = ~np.eye(corr.shape[0], dtype=bool)
avg_corr = float(corr.values[mask].mean())
max_corr = float(corr.values[mask].max())
min_corr = float(corr.values[mask].min())

c1, c2, c3 = st.columns(3)
c1.metric("Mean pairwise ρ", f"{avg_corr:.3f}")
c2.metric("Max pairwise ρ", f"{max_corr:.3f}")
c3.metric("Min pairwise ρ", f"{min_corr:.3f}")

st.caption(
    "Mean ρ < 0.5 is the rough threshold above which the diversification "
    "benefit becomes weak. Look for pairs with negative ρ as the building "
    "blocks of a low-variance composite portfolio."
)
