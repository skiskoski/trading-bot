"""Streamlit GUI — Overview page.

Run via:
    tradebot gui
or:
    streamlit run src/trading_bot/gui/app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trading_bot.registry import latest_run_per_strategy, trial_count

st.set_page_config(
    page_title="Trading bot — Overview",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Strategy overview")
st.caption(
    "All strategies persisted in the registry. The equity curves below are "
    "the latest run for each strategy. Use the toggles to compare."
)

runs = latest_run_per_strategy()

if not runs:
    st.info(
        "No runs persisted yet. From the terminal:\n\n"
        "```\n"
        "tradebot run momentum_12_1\n"
        "tradebot run-all\n"
        "```"
    )
    st.stop()

# Sidebar — strategy toggles, date range, options
st.sidebar.header("Filters")
all_names = [r["strategy_name"] for r in runs]
selected = st.sidebar.multiselect(
    "Strategies", options=all_names, default=all_names
)
normalize = st.sidebar.checkbox("Normalise to 100 at start", value=True)
log_y = st.sidebar.checkbox("Log y-axis", value=False)

st.sidebar.markdown("---")
st.sidebar.metric("Honest trial counter", trial_count())
st.sidebar.caption(
    "Total backtest runs performed against this database — used to deflate the "
    "Sharpe Ratio for multiple testing."
)

# Equity curve chart
fig = go.Figure()
palette = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

for i, r in enumerate(runs):
    if r["strategy_name"] not in selected:
        continue
    eq = r["equity"].copy()
    if eq.empty:
        continue
    if normalize:
        eq = eq / eq.iloc[0] * 100.0
    fig.add_trace(
        go.Scatter(
            x=eq.index,
            y=eq.values,
            mode="lines",
            name=r["strategy_name"],
            line=dict(color=palette[i % len(palette)], width=2),
            hovertemplate=f"<b>{r['strategy_name']}</b><br>%{{x|%Y-%m-%d}}: %{{y:.2f}}<extra></extra>",
        )
    )

fig.update_layout(
    height=520,
    margin=dict(l=20, r=20, t=10, b=20),
    yaxis_type="log" if log_y else "linear",
    yaxis_title="Equity" + (" (base 100)" if normalize else ""),
    xaxis_title=None,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, use_container_width=True)

# Metrics table
st.subheader("Metrics")

rows = []
for r in runs:
    if r["strategy_name"] not in selected:
        continue
    m = r["metrics"]
    rows.append(
        {
            "Strategy": r["strategy_name"],
            "CAGR": m.get("CAGR", 0),
            "Sharpe": m.get("Sharpe", 0),
            "Sortino": m.get("Sortino", 0),
            "MaxDD": m.get("MaxDrawdown", 0),
            "HitRate": m.get("HitRate", 0),
            "IC mean": m.get("IC_mean", 0),
            "IC IR": m.get("IC_IR", 0),
            "PSR": r["psr"],
            "DSR": r["dsr"],
            "N trials": r["n_trials_used"],
            "Universe": r["universe_size"],
            "PIT": "✓" if r["use_pit"] else "✗",
        }
    )

df = pd.DataFrame(rows)
if not df.empty:
    # Pretty formatting
    fmt = {
        "CAGR": "{:.2%}",
        "MaxDD": "{:.2%}",
        "HitRate": "{:.2%}",
        "Sharpe": "{:.3f}",
        "Sortino": "{:.3f}",
        "IC mean": "{:.4f}",
        "IC IR": "{:.3f}",
        "PSR": "{:.3f}",
        "DSR": "{:.3f}",
    }
    st.dataframe(
        df.style.format(fmt).background_gradient(
            subset=["Sharpe", "Sortino", "PSR", "DSR"], cmap="RdYlGn", vmin=-1, vmax=1.5
        ),
        use_container_width=True,
    )

st.markdown("---")
st.caption(
    "DSR ≥ 0.95 → the observed Sharpe survives multiple testing penalty. "
    "Strategies marked **promoted** in the registry pass this gate; otherwise "
    "they are *tested* but not yet considered statistically significant."
)
