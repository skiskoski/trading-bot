"""Portfolio page — multi-sleeve combination with crisis alpha analysis.

Shows why gold is not evaluated on standalone Sharpe but on its portfolio
contribution: low/negative correlation with equity during drawdowns provides
crisis alpha and lowers the combined MaxDrawdown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from trading_bot.registry import get_run, list_runs

st.set_page_config(page_title="Portfolio", page_icon="🏛️", layout="wide")

st.title("🏛️ Portfolio — sleeve combination")
st.caption(
    "Gold (TSMOM) is not evaluated for its standalone Sharpe. Its role is "
    "**crisis alpha**: negative/low correlation with equity during drawdowns "
    "reduces combined MaxDD and improves risk-adjusted returns. "
    "*(Baur-Lucey 2010, Moskowitz-Ooi-Pedersen 2012)*"
)

# ── Load runs from registry ─────────────────────────────────────────────────
runs = list_runs()
run_map: dict[str, dict] = {}
for r in runs:
    name = r["strategy_name"]
    if name not in run_map:
        full = get_run(r["id"])
        run_map[name] = full

equity_strategies = [n for n in run_map if n != "tsmom_gold"]
gold_available = "tsmom_gold" in run_map

if not equity_strategies:
    st.info("No equity strategy runs found. Run `tradebot run-all` first.")
    st.stop()

if not gold_available:
    st.warning(
        "No TSMOM gold run found. Run:\n\n"
        "```\ntradebot run tsmom_gold --start 2005-01-01\n```"
    )

# ── Sidebar controls ────────────────────────────────────────────────────────
st.sidebar.header("Sleeve configuration")
core_choice = st.sidebar.selectbox("Core equity sleeve", equity_strategies,
                                    index=0 if "momentum_12_1" not in equity_strategies
                                    else equity_strategies.index("momentum_12_1"))
gold_weight = st.sidebar.slider("Gold weight (%)", 0, 40, 20, step=5) / 100.0
core_weight = 1.0 - gold_weight
rolling_window = st.sidebar.slider("Rolling correlation window (days)", 30, 252, 66, step=22)

# ── Build combined returns ───────────────────────────────────────────────────
from trading_bot.portfolio.combiner import PortfolioConfig, SleeveSpec, combine_sleeves, crisis_alpha_table

core_run = run_map[core_choice]
core_rets = core_run["returns"]
core_equity = core_run["equity"]

if gold_available:
    gold_run = run_map["tsmom_gold"]
    gold_rets = gold_run["returns"]
else:
    gold_rets = pd.Series(dtype=float)

cfg = PortfolioConfig(
    sleeves=[
        SleeveSpec(core_choice, core_weight),
        SleeveSpec("tsmom_gold",  gold_weight),
    ] if gold_available and gold_weight > 0 else [SleeveSpec(core_choice, 1.0)],
    name="portfolio",
)

sleeve_map = {core_choice: core_rets}
if gold_available and gold_weight > 0:
    sleeve_map["tsmom_gold"] = gold_rets

combined = combine_sleeves(sleeve_map, cfg)

# ── Section 1: Equity curves comparison ────────────────────────────────────
st.subheader("Equity curves")

fig = go.Figure()
# Core standalone
eq_core_norm = core_equity / core_equity.iloc[0] * 100
fig.add_trace(go.Scatter(x=eq_core_norm.index, y=eq_core_norm.values,
                          name=f"Core only ({core_choice})",
                          line=dict(color="#1f77b4", width=2)))
# Gold standalone
if gold_available and gold_weight > 0:
    eq_gold_norm = gold_run["equity"]
    eq_gold_norm = eq_gold_norm / eq_gold_norm.iloc[0] * 100
    # Align to same start
    common_start = max(eq_core_norm.index[0], eq_gold_norm.index[0])
    eq_gold_norm = eq_gold_norm.loc[common_start:]
    fig.add_trace(go.Scatter(x=eq_gold_norm.index, y=eq_gold_norm.values,
                              name="Gold TSMOM standalone",
                              line=dict(color="#ffd700", width=1.5, dash="dot")))
# Combined
eq_comb_norm = combined.equity / combined.equity.iloc[0] * 100
fig.add_trace(go.Scatter(x=eq_comb_norm.index, y=eq_comb_norm.values,
                          name=f"Combined ({int(core_weight*100)}% core + {int(gold_weight*100)}% gold)",
                          line=dict(color="#2ca02c", width=2.5)))

fig.update_layout(height=420, hovermode="x unified",
                  yaxis_title="Equity (base 100)",
                  legend=dict(orientation="h", yanchor="bottom", y=1.02))
st.plotly_chart(fig, use_container_width=True)

# ── Section 2: Metrics comparison ───────────────────────────────────────────
st.subheader("Metrics comparison")

from trading_bot.backtest.metrics import cagr, max_drawdown, sharpe, sortino

rows = []
# Core only
rows.append({
    "Portfolio": f"Core only ({core_choice})",
    "CAGR": cagr(core_equity),
    "Sharpe": sharpe(core_rets),
    "Sortino": sortino(core_rets),
    "MaxDD": max_drawdown(core_equity),
})
if gold_available and gold_weight > 0:
    # Gold only
    rows.append({
        "Portfolio": "Gold TSMOM standalone",
        "CAGR": cagr(gold_run["equity"]),
        "Sharpe": sharpe(gold_rets),
        "Sortino": sortino(gold_rets),
        "MaxDD": max_drawdown(gold_run["equity"]),
    })
    # Combined
    rows.append({
        "Portfolio": f"Combined {int(core_weight*100)}/{int(gold_weight*100)}",
        "CAGR": combined.metrics["CAGR"],
        "Sharpe": combined.metrics["Sharpe"],
        "Sortino": combined.metrics["Sortino"],
        "MaxDD": combined.metrics["MaxDrawdown"],
    })

df_metrics = pd.DataFrame(rows)
fmt = {"CAGR": "{:.2%}", "MaxDD": "{:.2%}",
       "Sharpe": "{:.3f}", "Sortino": "{:.3f}"}
st.dataframe(
    df_metrics.style.format(fmt)
    .background_gradient(subset=["Sharpe", "Sortino"], cmap="RdYlGn", vmin=0, vmax=1.5)
    .background_gradient(subset=["MaxDD"], cmap="RdYlGn_r", vmin=-0.6, vmax=0),
    use_container_width=True,
)

# ── Section 3: Drawdown comparison ─────────────────────────────────────────
st.subheader("Drawdown — core vs combined")
st.caption("The shaded gold area shows where gold allocation reduced the drawdown.")

dd_core = core_equity / core_equity.cummax() - 1
dd_comb = combined.equity / combined.equity.cummax() - 1
# Align
common_idx = dd_core.index.intersection(dd_comb.index)
dd_core = dd_core.loc[common_idx]
dd_comb = dd_comb.loc[common_idx]

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=dd_core.index, y=dd_core.values * 100,
    name=f"Core only", line=dict(color="#d62728", width=1.5),
    fill="tozeroy", fillcolor="rgba(214,39,40,0.15)",
))
fig_dd.add_trace(go.Scatter(
    x=dd_comb.index, y=dd_comb.values * 100,
    name=f"Combined", line=dict(color="#2ca02c", width=2),
    fill="tozeroy", fillcolor="rgba(44,160,44,0.15)",
))
fig_dd.update_layout(
    height=320, yaxis_title="Drawdown (%)", hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig_dd, use_container_width=True)

# ── Section 4: Rolling correlation ─────────────────────────────────────────
if gold_available and gold_weight > 0:
    st.subheader(f"Rolling {rolling_window}d correlation: core ↔ gold")
    st.caption(
        "Near zero or negative during equity crises = gold acts as safe haven. "
        "Positive in calm bull markets is acceptable — the benefit concentrates "
        "exactly when you need it most."
    )
    common = core_rets.index.intersection(gold_rets.index)
    rolling_corr = (
        core_rets.loc[common]
        .rolling(rolling_window, min_periods=rolling_window // 2)
        .corr(gold_rets.loc[common])
    )
    fig_corr = go.Figure()
    fig_corr.add_trace(go.Scatter(
        x=rolling_corr.index, y=rolling_corr.values,
        name=f"Rolling {rolling_window}d corr",
        line=dict(color="#9467bd", width=1.5),
        fill="tozeroy", fillcolor="rgba(148,103,189,0.15)",
    ))
    fig_corr.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_corr.add_hline(y=-0.3, line_dash="dot", line_color="green",
                       annotation_text="hedge zone")
    fig_corr.update_layout(
        height=280, yaxis_title="Correlation", yaxis_range=[-1, 1],
        hovermode="x unified",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # ── Section 5: Crisis alpha table ──────────────────────────────────────
    st.subheader("🛡️ Crisis alpha — worst equity months vs gold")
    st.caption(
        "In the worst equity months, gold tends to be flat or positive — "
        "this is the safe-haven effect. The table shows the 10 worst months "
        "for the core equity sleeve and gold's return in the same month."
    )
    cat_df = crisis_alpha_table(
        {core_choice: core_rets, "tsmom_gold": gold_rets},
        equity_sleeve=core_choice,
        gold_sleeve="tsmom_gold",
        n_worst=10,
    )
    if not cat_df.empty:
        eq_col = f"{core_choice} return"
        gold_col = "tsmom_gold return"
        cat_df[eq_col] = cat_df[eq_col].map("{:.2%}".format)
        cat_df[gold_col] = cat_df[gold_col].apply(
            lambda x: f"{x:.2%}" if x == x else "n/a"
        )
        st.dataframe(cat_df, use_container_width=True, hide_index=True)

        helped = (cat_df["Gold helped?"] == "✅").sum()
        st.metric("Gold outperformed core in worst months",
                  f"{helped}/10",
                  delta=f"{'safe-haven confirmed' if helped >= 6 else 'mixed evidence'}",
                  delta_color="normal" if helped >= 6 else "off")
