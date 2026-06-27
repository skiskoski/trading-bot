"""Statistical-health page: trial counter, DSR distribution, IC distribution, CPCV."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
            "has_cpcv": r["cpcv"] is not None,
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

st.subheader("Reality check")
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

# ── CPCV section ────────────────────────────────────────────────────────────
st.divider()
st.subheader("🔬 CPCV — Combinatorial Purged Cross-Validation")
st.caption(
    "For each strategy that has been CPCV-validated, the chart below shows "
    "the full distribution of OOS Sharpe ratios across all C(k,2) test paths. "
    "PBO (Probability of Backtest Overfitting) < 0.5 is required for promotion."
)

cpcv_runs = [r for r in runs if r.get("cpcv") is not None]

if not cpcv_runs:
    st.info(
        "No CPCV results yet. Run:\n\n"
        "```\ntradebot validate-cpcv momentum_12_1 --start 2005-01-01\n```\n\n"
        "or add `--cpcv` to `tradebot run-all`."
    )
else:
    for r in cpcv_runs:
        cpcv = r["cpcv"]
        strat_name = r["strategy_name"]
        oos = cpcv.get("oos_sharpe_distribution", [])
        is_dist = cpcv.get("is_sharpe_distribution", [])
        pbo = cpcv.get("pbo", float("nan"))
        mean_oos = cpcv.get("mean_oos_sharpe", float("nan"))
        median_oos = cpcv.get("median_oos_sharpe", float("nan"))
        n_paths = cpcv.get("n_paths", len(oos))
        frac_pos = cpcv.get("fraction_positive", float("nan"))

        pbo_ok = pbo < 0.5
        badge = "✅ PASS" if pbo_ok else "❌ FAIL"
        color = "green" if pbo_ok else "red"

        with st.expander(f"{strat_name}  —  PBO={pbo:.3f} {badge}", expanded=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("OOS Sharpe (mean)", f"{mean_oos:.3f}")
            m2.metric("OOS Sharpe (median)", f"{median_oos:.3f}")
            m3.metric("Fraction positive paths", f"{frac_pos*100:.1f}%")
            m4.metric("PBO", f"{pbo:.3f}", delta="PASS" if pbo_ok else "FAIL",
                      delta_color="normal" if pbo_ok else "inverse")

            fig = go.Figure()
            if oos:
                fig.add_trace(go.Histogram(
                    x=oos, name="OOS Sharpe", nbinsx=max(10, n_paths // 2),
                    marker_color="steelblue", opacity=0.75,
                ))
            if is_dist:
                fig.add_trace(go.Histogram(
                    x=is_dist, name="IS Sharpe", nbinsx=max(10, len(is_dist) // 2),
                    marker_color="orange", opacity=0.55,
                ))
            if oos:
                median_is = float(np.median(is_dist)) if is_dist else 0.0
                fig.add_vline(x=median_is, line_dash="dot", line_color="orange",
                              annotation_text=f"IS median={median_is:.2f}")
                fig.add_vline(x=0, line_dash="dash", line_color="red",
                              annotation_text="0")
            fig.update_layout(
                title=f"OOS vs IS Sharpe distribution — {n_paths} paths  (k={cpcv.get('k',8)})",
                xaxis_title="Sharpe (annualised)",
                yaxis_title="Count",
                barmode="overlay",
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig, use_container_width=True)

            if pbo_ok:
                st.success(
                    f"PBO={pbo:.3f} < 0.5 — OOS performance is NOT dominated by in-sample luck. "
                    f"{frac_pos*100:.0f}% of the {n_paths} OOS paths are positive."
                )
            else:
                st.error(
                    f"PBO={pbo:.3f} ≥ 0.5 — the in-sample optimum is more likely to "
                    "underperform OOS than not. Do NOT promote to paper trading."
                )
