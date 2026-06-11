"""Strategy detail page: composition + equity + drawdown + walk-forward."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from trading_bot.registry import latest_run_per_strategy, list_strategies

st.set_page_config(page_title="Strategy detail", page_icon="🔍", layout="wide")

st.title("🔍 Strategy detail")

strategies = list_strategies()
if not strategies:
    st.info("No strategies registered yet. Run `tradebot run-all` from the terminal.")
    st.stop()

names = [s["name"] for s in strategies]
choice = st.selectbox("Strategy", names)
strat = next(s for s in strategies if s["name"] == choice)
runs = latest_run_per_strategy()
run = next((r for r in runs if r["strategy_name"] == choice), None)

# Composition card
st.subheader("Composition")
cfg = strat["config"]

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("**Signals**")
    sig_rows = []
    for s in cfg["signals"]:
        sig_rows.append(
            {
                "Feature": s["feature"],
                "Params": str(s["params"]),
                "Weight": s["weight"],
                "Rank?": "✓" if s["use_rank"] else "✗",
                "Negate?": "✓" if s["negate"] else "✗",
            }
        )
    st.dataframe(pd.DataFrame(sig_rows), use_container_width=True, hide_index=True)

    st.markdown("**Filters**")
    if cfg["filters"]:
        st.dataframe(
            pd.DataFrame(
                [
                    {"Feature": f["feature"], "Params": str(f["params"]), "Threshold": f["threshold"]}
                    for f in cfg["filters"]
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No filters")

    st.markdown("**Regime gate**")
    if cfg["regime"]:
        r = cfg["regime"]
        st.code(
            f"{r['symbol']}.{r['feature']}({r['params']}) > {r['threshold']}", language="text"
        )
    else:
        st.caption("Always on")

with col2:
    st.markdown("**Pre-registered hypothesis**")
    st.write(strat["rationale"])
    st.markdown(f"**Status:** `{strat['status']}`")
    st.markdown(f"**Top N:** {cfg['top_n']}")

st.markdown("---")

if run is None:
    st.warning("No run for this strategy yet. Run `tradebot run " + choice + "`.")
    st.stop()

# Equity + drawdown
st.subheader("Performance")

eq = run["equity"]
running_max = eq.cummax()
dd = eq / running_max - 1.0

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
    row_heights=[0.7, 0.3],
    subplot_titles=("Equity", "Drawdown"),
)
fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Equity", line=dict(color="#1f77b4")), row=1, col=1)
fig.add_trace(
    go.Scatter(
        x=dd.index, y=dd.values * 100,
        name="DD %", fill="tozeroy", line=dict(color="#d62728"),
    ),
    row=2, col=1,
)
fig.update_layout(height=550, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
fig.update_yaxes(title_text="Equity", row=1, col=1)
fig.update_yaxes(title_text="DD %", row=2, col=1)
st.plotly_chart(fig, use_container_width=True)

# Metrics box
cols = st.columns(6)
m = run["metrics"]
cols[0].metric("CAGR", f"{m.get('CAGR', 0)*100:.2f}%")
cols[1].metric("Sharpe", f"{m.get('Sharpe', 0):.3f}")
cols[2].metric("MaxDD", f"{m.get('MaxDrawdown', 0)*100:.2f}%")
cols[3].metric("IC", f"{m.get('IC_mean', 0):.4f}")
cols[4].metric("PSR", f"{run['psr']:.3f}")
cols[5].metric("DSR", f"{run['dsr']:.3f}", help=f"N trials = {run['n_trials_used']}")

st.caption(
    f"Backtested {run['start_date']} → {run['end_date']} on the top-{run['universe_size']} "
    f"liquidity universe, PIT={'on' if run['use_pit'] else 'off'}."
)
