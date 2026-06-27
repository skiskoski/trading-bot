"""Research loop page — live view of the autonomous hypothesis generator."""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import select

from trading_bot.data.storage import ResearchLog, get_session
from trading_bot.registry import trial_count
from trading_bot.research.scorecard import get_all_scores

st.set_page_config(page_title="Research loop", page_icon="🔬", layout="wide")

st.title("🔬 Autonomous research loop")
st.caption(
    "The research loop generates new strategy hypotheses guided by a Bayesian "
    "scorecard, filters them via IC pre-scan, validates with CPCV, and updates "
    "its priors after each run. Only strategies that pass **PBO < 0.5 + DSR ≥ 0.95** "
    "are marked as promoted."
)

# ── Header metrics ──────────────────────────────────────────────────────────
with get_session() as session:
    logs_all = session.execute(
        select(ResearchLog).order_by(ResearchLog.created_at.desc())
    ).scalars().all()

n_total = trial_count()
n_tested = sum(1 for l in logs_all if l.status in ("tested", "promoted"))
n_skipped = sum(1 for l in logs_all if l.status == "skipped")
n_promoted = sum(1 for l in logs_all if l.status == "promoted")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total trials (DB)", n_total)
c2.metric("Hypotheses generated", len(logs_all))
c3.metric("Fully tested (CPCV)", n_tested)
c4.metric("IC pre-filtered", n_skipped)
c5.metric("Promoted ★", n_promoted,
          delta="target met" if n_promoted >= 3 else f"{3 - n_promoted} to go",
          delta_color="normal" if n_promoted >= 3 else "off")

if not logs_all:
    st.info(
        "No research runs yet. Start the loop from the terminal:\n\n"
        "```bash\ntradebot research --rounds 5 --candidates 8\n```"
    )
    st.stop()

# ── Build dataframe ─────────────────────────────────────────────────────────
rows = []
for log in logs_all:
    try:
        cfg = json.loads(log.config_json)
        signals = [s.get("feature", "?") for s in cfg.get("signals", [])]
    except Exception:
        signals = []
    rows.append({
        "round": log.round_id,
        "name": log.hypothesis_name,
        "signals": " + ".join(signals),
        "status": log.status,
        "IC_prescan": log.ic_prescan,
        "OOS_Sharpe": log.oos_sharpe,
        "PBO": log.pbo,
        "DSR": log.dsr,
        "created": log.created_at,
    })
df = pd.DataFrame(rows)

# ── Section 1: Research log table ─────────────────────────────────────────
st.subheader("Hypothesis log")

status_filter = st.multiselect(
    "Filter by status",
    options=["promoted", "tested", "skipped", "error", "pending"],
    default=["promoted", "tested", "skipped"],
)
df_view = df[df["status"].isin(status_filter)] if status_filter else df

def _color_status(val):
    colors = {"promoted": "#2ca02c", "tested": "#ff7f0e",
              "skipped": "#aaaaaa", "error": "#d62728", "pending": "#7f7f7f"}
    return f"color: {colors.get(val, 'black')}"

fmt = {}
for col in ["IC_prescan", "OOS_Sharpe", "PBO", "DSR"]:
    fmt[col] = "{:.3f}"

st.dataframe(
    df_view.style.applymap(_color_status, subset=["status"]).format(fmt, na_rep="—"),
    use_container_width=True,
    hide_index=True,
)

# ── Section 2: OOS Sharpe evolution over rounds ────────────────────────────
st.subheader("OOS Sharpe evolution across rounds")
st.caption("Each dot is one tested hypothesis. Red line = PBO gate; stars = promoted.")

tested_df = df[df["OOS_Sharpe"].notna()].copy()
if not tested_df.empty:
    fig = go.Figure()
    for status, color in [("tested", "#ff7f0e"), ("promoted", "#2ca02c"), ("error", "#d62728")]:
        sub = tested_df[tested_df["status"] == status]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["round"], y=sub["OOS_Sharpe"],
            mode="markers",
            marker=dict(size=10 if status == "promoted" else 7,
                        color=color,
                        symbol="star" if status == "promoted" else "circle"),
            name=status,
            text=sub["name"],
            hovertemplate="<b>%{text}</b><br>Round %{x}<br>OOS Sharpe %{y:.3f}<extra></extra>",
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    # Add trend line if enough points
    if len(tested_df) >= 4:
        import numpy as np
        z = np.polyfit(tested_df["round"].values, tested_df["OOS_Sharpe"].fillna(0).values, 1)
        x_line = sorted(tested_df["round"].unique())
        y_line = [z[0] * x + z[1] for x in x_line]
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines",
                                  line=dict(dash="dot", color="steelblue", width=1.5),
                                  name="trend"))
    fig.update_layout(height=360, xaxis_title="Round", yaxis_title="OOS Sharpe",
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

# ── Section 3: IC pre-scan funnel ─────────────────────────────────────────
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Funnel: IC pre-scan → CPCV")
    funnel_data = {
        "Generated": len(df),
        "IC passed": n_tested + n_promoted,
        "CPCV passed (PBO<0.5)": sum(
            1 for l in logs_all
            if l.pbo is not None and l.pbo < 0.5
        ),
        "DSR promoted": n_promoted,
    }
    fig_f = go.Figure(go.Funnel(
        y=list(funnel_data.keys()),
        x=list(funnel_data.values()),
        marker=dict(color=["#1f77b4", "#ff7f0e", "#2ca02c", "#ffd700"]),
    ))
    fig_f.update_layout(height=300)
    st.plotly_chart(fig_f, use_container_width=True)

with col_b:
    st.subheader("PBO distribution (tested)")
    pbo_vals = [l.pbo for l in logs_all if l.pbo is not None]
    if pbo_vals:
        fig_pbo = px.histogram(pbo_vals, nbins=15, title="",
                                labels={"value": "PBO", "count": "Count"})
        fig_pbo.add_vline(x=0.5, line_dash="dash", line_color="red",
                           annotation_text="gate 0.5")
        fig_pbo.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_pbo, use_container_width=True)
    else:
        st.info("No CPCV results yet.")

# ── Section 4: Feature scorecard ──────────────────────────────────────────
st.subheader("🧠 Bayesian feature scorecard")
st.caption(
    "Features are scored by: mean OOS Sharpe × (1 − mean PBO) × √(times used) × promotion bonus. "
    "Higher score = higher sampling weight in the next round."
)

scores = get_all_scores()
if scores:
    df_sc = pd.DataFrame(scores)
    df_sc["feature"] = df_sc["key"].str.split(":").str[0]
    df_sc["params"] = df_sc["key"].str.split(":", n=1).str[1]

    fig_sc = px.bar(
        df_sc.head(15),
        x="score", y="key",
        orientation="h",
        color="mean_oos_sharpe",
        color_continuous_scale="RdYlGn",
        hover_data=["times_used", "times_promoted", "mean_oos_sharpe", "mean_pbo"],
        title="Top 15 features by score",
        labels={"key": "Feature (params)", "score": "Score"},
    )
    fig_sc.update_layout(height=420, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_sc, use_container_width=True)
else:
    st.info("Scorecard empty — run at least one research round first.")

# ── Section 5: How to run ─────────────────────────────────────────────────
with st.expander("How to run / resume the research loop"):
    st.code("""
# Start (or resume) the research loop
tradebot research --rounds 10 --candidates 8 --cpcv-k 6

# View status without running
tradebot research-status

# Options:
#  --ic-threshold 0.02   minimum IC for IC pre-scan (default 0.02)
#  --cpcv-k 8            more folds = slower but more rigorous (28 paths vs 15)
#  --target 3            stop after N promotions
#  --rounds 20           total rounds
    """, language="bash")
    st.caption(
        "The loop is **resumable**: it skips already-tested configurations. "
        "The trial counter grows with every run (honesty) so DSR threshold rises. "
        "Start with `--cpcv-k 6` for speed, use `--cpcv-k 8` to confirm finalists."
    )
