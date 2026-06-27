"""✅ Validazione — distribuzione path CPCV, IS vs OOS, DSR vs trial,
stato holdout."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trading_bot.config import settings
from trading_bot.gui.data import load_runs
from trading_bot.gui.glossary import section_help
from trading_bot.gui.style import BLUE, GOLD, GREEN, RED, TEXT2, cards, pill, plotly_layout
from trading_bot.validation.robustness import load_reports

st.title("✅ Validazione")

st.caption("Tre linee di difesa contro l'overfitting: CPCV (45 path), "
           "Deflated Sharpe (multiple testing), holdout finale mai toccato.")

runs = load_runs()
runs_cpcv = [r for r in runs if r.get("cpcv")]

# ── CPCV ─────────────────────────────────────────────────────────────────────

section_help("CPCV — distribuzione dei path out-of-sample", "CPCV", "PBO")
if not runs_cpcv:
    st.info("Nessun run con CPCV ancora. I trial del daemon lo includono "
            "automaticamente.")
else:
    options = {f"#{r['id']} {r['strategy_name']} "
               f"({r['created_at']:%Y-%m-%d %H:%M})": r for r in runs_cpcv[:80]}
    sel = st.selectbox("Run", list(options.keys()), index=0)
    r = options[sel]
    cp = r["cpcv"]

    oos = np.array(cp["oos_sharpe_distribution"], dtype=float)
    iss = np.array(cp["is_sharpe_distribution"], dtype=float)
    med_is = float(np.median(iss)) if len(iss) else 0.0

    cards([
        ("Path OOS", str(cp["n_paths"]), f"k={cp['k']}", "blue"),
        ("PBO", f"{cp['pbo']:.3f}", "gate < 0.4 · neutro ≈ 0.5",
         "green" if cp["pbo"] < 0.4 else
         ("gold" if cp["pbo"] <= 0.6 else "red")),
        ("OOS Sharpe medio", f"{cp['mean_oos_sharpe']:.3f}",
         f"± {cp['std_oos_sharpe']:.3f}", "dim"),
        ("Path positivi", f"{cp['fraction_positive']:.0%}", "gate ≥ 65%",
         "green" if cp["fraction_positive"] >= 0.65 else "gold"),
    ])

    left, rr = st.columns(2)
    with left:
        figh = go.Figure()
        figh.add_trace(go.Histogram(x=oos, nbinsx=20, name="OOS",
                                    marker=dict(color=BLUE, opacity=0.85)))
        figh.add_vline(x=med_is, line=dict(color=GOLD, dash="dash"),
                       annotation_text="mediana IS",
                       annotation_font_color=GOLD)
        figh.add_vline(x=0, line=dict(color=TEXT2, dash="dot"))
        figh.update_xaxes(title="Sharpe OOS per path")
        st.plotly_chart(plotly_layout(figh, 300, legend=False),
                        width="stretch")
        st.caption("PBO = frazione di path a SINISTRA della linea oro. "
                   "Distribuzione tutta a destra dello zero = edge robusto.")

    with rr:
        figb = go.Figure()
        figb.add_trace(go.Box(y=iss, name="IS", marker_color=TEXT2))
        figb.add_trace(go.Box(y=oos, name="OOS", marker_color=BLUE))
        figb.update_yaxes(title="Sharpe")
        st.plotly_chart(plotly_layout(figb, 300, legend=False),
                        width="stretch")
        st.caption("IS e OOS allo stesso livello = niente memoria del "
                   "passato. IS molto sopra OOS = overfitting.")

# ── DSR VS TRIAL ─────────────────────────────────────────────────────────────

section_help("DSR vs numero di trial — il costo del cercare tanto",
             "DSR", "Sharpe")
pts = [(r["n_trials_used"], r["dsr"], r["strategy_name"], r["created_at"])
       for r in runs if r.get("dsr") is not None and r.get("n_trials_used")]
if pts:
    df_d = pd.DataFrame(pts, columns=["trials", "dsr", "name", "created"])
    df_new = df_d[df_d["dsr"] > 0]
    figd = go.Figure()
    figd.add_trace(go.Scatter(
        x=df_new["trials"], y=df_new["dsr"], mode="markers",
        text=df_new["name"],
        marker=dict(color=BLUE, size=8, opacity=0.75),
        hovertemplate="%{text}<br>trial #%{x} DSR=%{y:.3f}<extra></extra>"))
    figd.add_hline(y=0.95, line=dict(color=GREEN, dash="dash"),
                   annotation_text="evidenza forte 0.95",
                   annotation_font_color=GREEN)
    figd.update_xaxes(title="trial cumulativi al momento del run")
    figd.update_yaxes(title="DSR", range=[-0.05, 1.05])
    st.plotly_chart(plotly_layout(figd, 320, legend=False), width="stretch")
    n_zero = int((df_d["dsr"] == 0).sum())
    if n_zero:
        st.caption(f"{n_zero} run storici con DSR=0 spurio (bug σ_SR fisso, "
                   f"corretto) esclusi dal grafico.")
else:
    st.info("Nessun run persistito ancora.")

# ── WALK-FORWARD ─────────────────────────────────────────────────────────────

section_help("Walk-forward", "Walk-forward")
st.markdown("Il walk-forward gira on-demand da terminale — rispetta l'ordine "
            "temporale e simula l'esperienza reale:")
st.code("tradebot validate momentum_12_1", language="bash")

# ── ROBUSTNESS REPORTS ───────────────────────────────────────────────────────

section_help("Robustness report — parametri, Monte Carlo, walk-forward",
             "Walk-forward", "DSR", "PBO")
reports = load_reports()
if not reports:
    st.info("Nessun robustness report salvato. Genera il primo da terminale:")
    st.code("tradebot robustness-report momentum_12_1 --mc-trials 1000", language="bash")
else:
    latest_by_strategy = {}
    for report in reports:
        name = report.get("strategy_name", "strategy")
        if name not in latest_by_strategy:
            latest_by_strategy[name] = report

    overview_rows = []
    for name, report in latest_by_strategy.items():
        gates = report.get("hard_gates", {})
        failed = [gate for gate, ok in gates.items() if not ok]
        generated = pd.to_datetime(report.get("generated_at"), errors="coerce")
        age_days = None
        if pd.notna(generated):
            age_days = max(0, int((pd.Timestamp.now(tz="UTC") - generated).days))
        decision = report.get("decision", "n/a")
        overview_rows.append({
            "Strategia": name,
            "Decisione": decision,
            "Gate": f"{sum(bool(v) for v in gates.values())}/{max(1, len(gates))}",
            "Falliti": ", ".join(failed) if failed else "nessuno",
            "Eta giorni": age_days,
            "Comando": (
                f"tradebot validate-strategy {name}"
                if failed else f"tradebot robustness-report {name} --mc-trials 1000"
            ),
        })

    section_help("Centro operativo robustness", "Walk-forward", "PBO", "DSR")
    overview = pd.DataFrame(overview_rows)
    n_promote = int(overview["Decisione"].str.startswith("promote", na=False).sum())
    n_reject = int((overview["Decisione"] == "reject_or_research_more").sum())
    stale = int((overview["Eta giorni"].fillna(999) > 14).sum())
    cards([
        ("Strategie validate", str(len(overview)), "ultimo report per strategia", "blue"),
        ("Paper-review", str(n_promote), "tutti i gate passano", "green"),
        ("Da lavorare", str(n_reject), "almeno un gate fallisce", "red" if n_reject else "dim"),
        ("Report vecchi", str(stale), ">14 giorni", "gold" if stale else "dim"),
    ])
    st.dataframe(overview, width="stretch", hide_index=True)
    command_by_strategy = {
        row["Strategia"]: row["Comando"] for row in overview_rows
    }

    selected_strategy = st.selectbox(
        "Strategia da ispezionare",
        list(latest_by_strategy.keys()),
        index=0,
    )
    latest = latest_by_strategy[selected_strategy]
    st.code(command_by_strategy[selected_strategy], language="bash")

    history = [
        {
            "Data": r.get("generated_at", "")[:19],
            "Strategia": r.get("strategy_name", "strategy"),
            "Decisione": r.get("decision", "n/a"),
            "Gate passati": (
                f"{sum(bool(v) for v in r.get('hard_gates', {}).values())}/"
                f"{max(1, len(r.get('hard_gates', {})))}"
            ),
            "File": r.get("_path", "").split("/")[-1],
        }
        for r in reports[:50]
    ]
    with st.expander("Storico robustness report", expanded=False):
        st.dataframe(pd.DataFrame(history), width="stretch", hide_index=True)

    labels = {
        f"{r.get('generated_at', '')[:19]} · {r.get('strategy_name', 'strategy')} · "
        f"{r.get('decision', 'n/a')}": r
        for r in reports[:30]
    }
    default_label = next(
        (
            label for label, report in labels.items()
            if report.get("_path") == latest.get("_path")
        ),
        next(iter(labels)),
    )
    sel_report = st.selectbox(
        "Robustness report",
        list(labels.keys()),
        index=list(labels.keys()).index(default_label),
    )
    rep = labels[sel_report]
    gates = rep.get("hard_gates", {})
    passed = sum(1 for ok in gates.values() if ok)
    total = max(1, len(gates))
    cards([
        ("Decisione", rep.get("decision", "n/a"), rep.get("strategy_name", ""), "green"
         if rep.get("decision", "").startswith("promote") else "red"),
        ("Gate passati", f"{passed}/{total}", "hard gates", "green"
         if passed == total else "gold"),
        ("Report", rep.get("_path", "n/a").split("/")[-1], "JSON salvato", "blue"),
    ])

    gate_df = pd.DataFrame([
        {"Gate": k, "Stato": "PASS" if v else "FAIL"}
        for k, v in gates.items()
    ])
    if not gate_df.empty:
        st.dataframe(gate_df, width="stretch", hide_index=True)

    ps = rep.get("parameter_stability")
    mc = rep.get("monte_carlo")
    wf = rep.get("walk_forward")

    if ps:
        st.subheader("Parameter stability")
        cards([
            ("Plateau", f"{ps.get('plateau_fraction', 0):.0%}",
             f"gap {ps.get('best_neighbor_gap', 0):.2f}", "green"
             if ps.get("passed") else "red"),
            ("Best metric", f"{ps.get('best_metric', 0):.2f}",
             ps.get("metric", "metric"), "blue"),
            ("Median", f"{ps.get('median_metric', 0):.2f}", "grid", "dim"),
        ])
        rows = ps.get("runs", [])
        if rows:
            flat = []
            for row in rows:
                item = {**row.get("params", {})}
                item.update({
                    "Sharpe": row.get("sharpe"),
                    "IC": row.get("ic_mean"),
                    "MaxDD": row.get("max_drawdown"),
                })
                flat.append(item)
            dfp = pd.DataFrame(flat)
            param_cols = [c for c in dfp.columns if c not in {"Sharpe", "IC", "MaxDD"}]
            if len(param_cols) >= 2:
                x_col, y_col = param_cols[0], param_cols[1]
                pivot = dfp.pivot_table(index=y_col, columns=x_col, values="Sharpe", aggfunc="mean")
                fig = go.Figure(data=go.Heatmap(
                    z=pivot.values,
                    x=[str(c) for c in pivot.columns],
                    y=[str(i) for i in pivot.index],
                    colorscale="RdYlGn",
                    colorbar=dict(title="Sharpe"),
                ))
                fig.update_xaxes(title=x_col)
                fig.update_yaxes(title=y_col)
                st.plotly_chart(plotly_layout(fig, 340, legend=False), width="stretch")
            st.dataframe(dfp.sort_values("Sharpe", ascending=False).head(30),
                         width="stretch", hide_index=True)

    if mc:
        st.subheader("Monte Carlo")
        cards([
            ("Sharpe p05", f"{mc.get('sharpe_p05', 0):.2f}",
             "worst 5%", "green" if mc.get("sharpe_p05", 0) >= 0 else "red"),
            ("MaxDD p95", f"{mc.get('max_drawdown_p95', 0):.1%}",
             "stress", "gold"),
            ("Loss prob.", f"{mc.get('loss_probability', 0):.1%}",
             f"{mc.get('trials', 0):,} trial", "green"
             if mc.get("passed") else "red"),
        ])
        figm = go.Figure()
        figm.add_trace(go.Bar(
            x=["p05", "p50", "p95"],
            y=[mc.get("terminal_return_p05", 0),
               mc.get("terminal_return_p50", 0),
               mc.get("terminal_return_p95", 0)],
            marker_color=[RED, BLUE, GREEN],
        ))
        figm.update_yaxes(title="Terminal return", tickformat=".0%")
        st.plotly_chart(plotly_layout(figm, 260, legend=False), width="stretch")

    if wf:
        st.subheader("Walk-forward")
        cards([
            ("Worst fold", f"{wf.get('worst_sharpe', 0):.2f}",
             "Sharpe", "green" if wf.get("passed") else "red"),
            ("Fold positivi", f"{wf.get('fraction_profitable', 0):.0%}",
             "rolling + anchored", "blue"),
        ])
        wf_rows = []
        for mode in ("rolling", "anchored"):
            for row in wf.get(f"{mode}_folds", []):
                wf_rows.append({"mode": mode, **row})
        if wf_rows:
            st.dataframe(pd.DataFrame(wf_rows), width="stretch", hide_index=True)

# ── HOLDOUT ──────────────────────────────────────────────────────────────────

section_help("Holdout — l'esame finale", "Holdout")
cards([
    ("Periodo holdout", f"dal {settings.holdout_start}",
     "MAI usato in ricerca", "gold"),
    ("Stato", "Intatto", "si consuma con un solo uso", "green"),
])
st.markdown(pill("⚠ Eseguilo UNA volta sola, quando hai deciso di andare "
                 "live: ogni sguardo lo consuma.", "gold"),
            unsafe_allow_html=True)
st.code("tradebot holdout-test", language="bash")
