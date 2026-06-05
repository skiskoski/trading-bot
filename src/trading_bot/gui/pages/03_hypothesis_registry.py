"""Hypothesis registry: every strategy ever registered, with rationale + status."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from trading_bot.registry import latest_run_per_strategy, list_strategies, trial_count

st.set_page_config(page_title="Hypothesis registry", page_icon="📚", layout="wide")

st.title("📚 Hypothesis registry")

st.caption(
    "Every strategy ever registered. Pre-registration enforces a non-trivial "
    "rationale before any backtest is run — anti-p-hacking discipline."
)

st.metric("Cumulative honest trial counter", trial_count())

strategies = list_strategies()
runs_by_name = {r["strategy_name"]: r for r in latest_run_per_strategy()}

if not strategies:
    st.info("Nothing registered yet.")
    st.stop()

rows = []
for s in strategies:
    r = runs_by_name.get(s["name"])
    rows.append(
        {
            "Name": s["name"],
            "Status": s["status"],
            "Rationale": s["rationale"][:160] + ("..." if len(s["rationale"]) > 160 else ""),
            "Latest Sharpe": r["metrics"].get("Sharpe", float("nan")) if r else float("nan"),
            "Latest DSR": r["dsr"] if r else float("nan"),
            "N trials": r["n_trials_used"] if r else 0,
            "Registered": s["created_at"],
        }
    )

df = pd.DataFrame(rows)
st.dataframe(
    df.style.format(
        {"Latest Sharpe": "{:.3f}", "Latest DSR": "{:.3f}"}
    ),
    use_container_width=True,
    hide_index=True,
)

# Status breakdown
st.subheader("Status breakdown")
counts = df["Status"].value_counts().to_dict()
cols = st.columns(len(counts) if counts else 1)
for i, (status, n) in enumerate(counts.items()):
    cols[i].metric(status, n)
