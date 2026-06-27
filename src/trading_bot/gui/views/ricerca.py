"""🔬 Ricerca — il laboratorio del bot: stato daemon, log live, funnel,
distribuzioni PBO/DSR, scorecard, archetipi, promozioni."""

from __future__ import annotations

import json
import re
from datetime import timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trading_bot.gui.data import (UNIVERSE, daemon_info, fmt_uptime,
                                  honest_logs, load_logs)
from trading_bot.gui.glossary import section_help, spiega
from trading_bot.gui.style import (BLUE, GOLD, GREEN, PALETTE, RED, TEXT2,
                                   cards, pill, plotly_layout, section)

st.title("🔬 Ricerca")

# ── DAEMON ───────────────────────────────────────────────────────────────────

section("Daemon di ricerca")


@st.fragment(run_every="5s")
def daemon_panel() -> None:
    from trading_bot.research.daemon import LOG_FILE

    dmn = daemon_info()
    df_all = load_logs()
    last_trial = "—"
    if not df_all.empty:
        ts = pd.Timestamp(df_all["created"].iloc[0]).to_pydatetime()
        last_trial = (ts.replace(tzinfo=timezone.utc).astimezone()
                      .strftime("%H:%M:%S"))

    if dmn["attivo"]:
        n_sess = "—"
        if dmn["started"] is not None and not df_all.empty:
            started_naive = dmn["started"].replace(tzinfo=None)
            n_sess = str(int((df_all["created"] >= started_naive).sum()))
        cards([
            ("Stato", "● Attivo", f"pid {dmn['pid']}", "green"),
            ("Uptime", fmt_uptime(dmn["uptime_s"]), None, "dim"),
            ("Trial in sessione", n_sess, None, "blue"),
            ("Ultimo trial", last_trial, None, "dim"),
        ])
    else:
        cards([
            ("Stato", "○ Fermo", "avvia da terminale", "red"),
            ("Ultimo trial", last_trial, None, "dim"),
        ])

    try:
        lines = LOG_FILE.read_text(encoding="utf-8",
                                   errors="replace").splitlines()[-20:]
        st.code("\n".join(lines) or "(log vuoto)", language="log")
    except FileNotFoundError:
        st.info("Nessun log ancora — il daemon non è mai stato avviato.")


daemon_panel()

c1, c2 = st.columns(2)
with c1:
    st.caption("Avvia (gira finché non lo fermi, sopravvive alla chiusura "
               "del terminale):")
    st.code("tradebot research-daemon", language="bash")
with c2:
    st.caption("Ferma (chiusura pulita: completa il trial in corso):")
    st.code("tradebot research-stop", language="bash")

# ── FUNNEL ───────────────────────────────────────────────────────────────────

df = load_logs()
if df.empty:
    st.info("Nessun trial. Avvia: `tradebot research-daemon`")
    st.stop()

cur = honest_logs(df)
quarantined = len(df) - len(cur)

n_promo = int((cur["status"] == "promoted").sum())
n_tested = int(cur["status"].isin(["tested", "promoted"]).sum())
n_skip = int((cur["status"] == "skipped").sum())
rate = f"{n_promo / max(n_tested, 1):.0%}"

section_help("Funnel di ricerca", "IC", "CPCV", "PBO")
cards([
    ("Trial onesti", str(len(cur)), f"{quarantined} in quarantena (pre-fix)", "dim"),
    ("Skippate (IC<0.02)", str(n_skip), "pre-scan veloce", "dim"),
    ("Testate CPCV", str(n_tested), "45 path per strategia", "blue"),
    ("Promosse ★", str(n_promo), f"tasso {rate}", "green" if n_promo else "gold"),
])

st.markdown(pill("Gate: PBO<0.4 ∧ OOS ATTIVO ≥0.5 (vs equal-weight) ∧ "
                 "≥65% path positivi — il beta non conta", "blue"),
            unsafe_allow_html=True)

l, r = st.columns(2)
with l:
    stages = ["generate", "IC pre-scan", "CPCV", "promosse"]
    vals = [len(cur), len(cur) - n_skip, n_tested, n_promo]
    figf = go.Figure(go.Funnel(
        y=stages, x=vals, textinfo="value+percent initial",
        marker=dict(color=[TEXT2, BLUE, "#5e5ce6", GREEN])))
    st.plotly_chart(plotly_layout(figf, 280, legend=False), width="stretch")

with r:
    section_help("Distribuzione IC pre-scan", "IC")
    ics = cur["ic"].dropna()
    if len(ics) > 3:
        figh = go.Figure(go.Histogram(x=ics, nbinsx=30,
                                      marker=dict(color=BLUE, opacity=0.8)))
        figh.add_vline(x=0.02, line=dict(color=GOLD, dash="dash"),
                       annotation_text="soglia 0.02",
                       annotation_font_color=GOLD)
        st.plotly_chart(plotly_layout(figh, 280, legend=False), width="stretch")
    else:
        st.info("Pochi dati IC ancora.")

# ── DISTRIBUZIONI PBO / DSR ──────────────────────────────────────────────────

l2, r2 = st.columns(2)
tested_df = cur[cur["status"].isin(["tested", "promoted"])]

with l2:
    section_help("Distribuzione PBO (strategie testate)", "PBO")
    pbos = tested_df["pbo"].dropna()
    if len(pbos) > 3:
        figp = go.Figure(go.Histogram(x=pbos, nbinsx=24,
                                      marker=dict(color="#5e5ce6", opacity=0.85)))
        figp.add_vline(x=0.4, line=dict(color=GREEN, dash="dash"),
                       annotation_text="gate 0.4", annotation_font_color=GREEN)
        figp.add_vline(x=0.5, line=dict(color=TEXT2, dash="dot"),
                       annotation_text="neutro 0.5",
                       annotation_font_color=TEXT2)
        st.plotly_chart(plotly_layout(figp, 260, legend=False), width="stretch")
    else:
        st.info("Servono più strategie testate.")

with r2:
    section_help("Distribuzione DSR (strategie testate)", "DSR")
    dsrs = tested_df["dsr"].dropna()
    dsrs = dsrs[dsrs > 0]  # i vecchi run pre-fix hanno DSR=0 spurio
    if len(dsrs) > 3:
        figd = go.Figure(go.Histogram(x=dsrs, nbinsx=24,
                                      marker=dict(color=GOLD, opacity=0.85)))
        figd.add_vline(x=0.95, line=dict(color=GREEN, dash="dash"),
                       annotation_text="evidenza forte 0.95",
                       annotation_font_color=GREEN)
        st.plotly_chart(plotly_layout(figd, 260, legend=False), width="stretch")
    else:
        st.info("DSR popolato dai trial nuovi (post-fix σ_SR empirico) — "
                "i vecchi run hanno DSR=0 spurio e sono esclusi.")

# ── MAPPA GATE + SCORECARD ───────────────────────────────────────────────────

l3, r3 = st.columns(2)
with l3:
    section_help("Mappa OOS attivo vs PBO (zona verde = gate)",
                 "OOS attivo", "PBO")
    t = tested_df.dropna(subset=["oos", "pbo"])
    if not t.empty:
        figs = go.Figure()
        figs.add_shape(type="rect", x0=0.5, x1=max(t["oos"].max() * 1.1, 1),
                       y0=0, y1=0.4, fillcolor="rgba(48,209,88,.08)",
                       line=dict(color=GREEN, width=1, dash="dot"))
        colors = [GREEN if s == "promoted" else TEXT2 for s in t["status"]]
        figs.add_trace(go.Scatter(
            x=t["oos"], y=t["pbo"], mode="markers", text=t["name"],
            marker=dict(color=colors, size=9, opacity=0.8),
            hovertemplate="%{text}<br>OOS=%{x:.2f} PBO=%{y:.2f}<extra></extra>"))
        figs.update_xaxes(title="OOS Sharpe attivo")
        figs.update_yaxes(title="PBO")
        st.plotly_chart(plotly_layout(figs, 300, legend=False), width="stretch")
    else:
        st.info("Nessuna strategia testata ancora.")

with r3:
    section("Scorecard feature — top 10 (UCB1)")
    try:
        from trading_bot.research.scorecard import get_all_scores
        scores = get_all_scores(universe=UNIVERSE)[:10]
        if scores:
            names = [s["key"].split(":")[0] for s in scores][::-1]
            vals = [s["score"] for s in scores][::-1]
            figb = go.Figure(go.Bar(x=vals, y=names, orientation="h",
                                    marker=dict(color=PALETTE * 2)))
            figb.update_layout(margin=dict(l=160))
            st.plotly_chart(plotly_layout(figb, 300, legend=False),
                            width="stretch")
        else:
            st.info("La scorecard si riempie man mano che il loop impara.")
    except Exception as e:
        st.warning(f"Scorecard: {e}")

# ── ARCHETIPI + SEGNALI ──────────────────────────────────────────────────────

l4, r4 = st.columns(2)
with l4:
    section("Mix archetipi testati")

    def arch_of(row) -> str:
        m = re.match(r"\[(\w+)\]", row["rationale"])
        if m:
            return m.group(1)
        nm = row["name"]
        for k in ("pure_f", "defens", "regime", "factor"):
            if k in nm:
                return {"pure_f": "pure_factor", "defens": "defensive",
                        "regime": "regime_adaptive", "factor": "factor_blend"}[k]
        return "ucb1_parametric"

    arch = cur.apply(arch_of, axis=1).value_counts()
    figa = go.Figure(go.Pie(labels=arch.index, values=arch.values, hole=0.6,
                            marker=dict(colors=PALETTE * 2),
                            textinfo="label+percent", textfont=dict(size=11)))
    st.plotly_chart(plotly_layout(figa, 280, legend=False), width="stretch")

with r4:
    section("Segnali più presenti nelle strategie testate")
    feats: dict[str, int] = {}
    for cfg_s in tested_df["config"]:
        try:
            for sig in json.loads(cfg_s).get("signals", []):
                feats[sig["feature"]] = feats.get(sig["feature"], 0) + 1
        except Exception:
            pass
    if feats:
        top = sorted(feats.items(), key=lambda x: -x[1])[:10][::-1]
        figc = go.Figure(go.Bar(x=[v for _, v in top], y=[k for k, _ in top],
                                orientation="h",
                                marker=dict(color=BLUE, opacity=0.85)))
        figc.update_layout(margin=dict(l=160))
        st.plotly_chart(plotly_layout(figc, 280, legend=False), width="stretch")
    else:
        st.info("Ancora nessuna strategia testata.")

# ── PROMOZIONI + ULTIMI TRIAL ────────────────────────────────────────────────

section("Promozioni (oro = parametro della strategia)")
promo = cur[cur["status"] == "promoted"].copy()
if promo.empty:
    st.markdown(pill("Nessuna promozione col gate onesto — è il sistema che "
                     "rifiuta il beta spacciato per alpha. Il loop continua.",
                     "gold"), unsafe_allow_html=True)
else:
    def gold_param(cfg_s: str) -> str:
        try:
            c = json.loads(cfg_s)
            gw = c.get("gold_weight", 0) or 0
            if gw > 0:
                gm = c.get("gold_mode", "defensive")
                return f"{gw:.0%} " + ("solo in stress" if gm == "defensive"
                                       else "sempre")
        except Exception:
            pass
        return "—"
    promo["oro"] = promo["config"].map(gold_param)
    st.dataframe(promo[["name", "oos", "pbo", "dsr", "oro", "created"]]
                 .rename(columns={"oos": "OOS attivo"}),
                 width="stretch", hide_index=True)

section("Ultimi 20 trial")
st.dataframe(
    cur.head(20)[["name", "status", "ic", "oos", "pbo", "dsr", "created"]]
    .round(3),
    width="stretch", hide_index=True, height=420)
