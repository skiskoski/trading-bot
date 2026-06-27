"""ℹ️ Guida — come funziona il bot + glossario quant completo."""

from __future__ import annotations

import streamlit as st

from trading_bot.gui.glossary import GLOSSARY
from trading_bot.gui.style import cards, pill, section

st.title("ℹ️ Guida")

# ── COME FUNZIONA ────────────────────────────────────────────────────────────

section("Come funziona il bot")
st.markdown("""
**Filosofia: separare la ricerca del segnale dalla gestione del rischio.**

1. **Ricerca** — il daemon genera ipotesi di strategia (combinazioni di
   segnali documentati in letteratura), le pre-filtra con l'IC e le valida
   con CPCV a 45 path. Solo chi passa il *gate onesto* viene promossa.
2. **Gate onesto** — `PBO < 0.4` ∧ `OOS Sharpe attivo ≥ 0.5` ∧ `≥65% path
   positivi`. "Attivo" = al netto del benchmark equal-weight: il beta di
   mercato non conta come skill.
3. **Overlay di portafoglio** — sopra la strategia promossa: vol targeting
   25%, sleeve oro TSMOM 20% (crisis alpha), rebalance mensile + trigger
   di volatilità.
4. **Esecuzione** — ciclo giornaliero su Alpaca paper: ingest → segnali →
   (se rebalance day) ordini → notifica Telegram.
""")

cards([
    ("Universo", "S&P 1500", "PIT, anti-survivorship", "blue"),
    ("Validazione", "CPCV 45 path", "+ DSR + holdout", "green"),
    ("Overlay", "Vol 25% · Oro 20%", "rebalance adattivo", "gold"),
    ("Esecuzione", "Alpaca paper", "ciclo giornaliero", "dim"),
])

section("Il flusso di una strategia")
st.markdown("""
```
ipotesi (segnali + filtri + regime)
   │  IC pre-scan ≥ 0.02          ── scarta il rumore in 20 secondi
   ▼
backtest completo + CPCV k=10     ── 45 scenari out-of-sample
   │  gate: PBO<0.4 ∧ OOS≥0.5 ∧ path+≥65%
   ▼
★ promossa  ──  overlay (vol target + oro)  ──  paper trading
   │
   ▼
holdout test (una volta sola)  ──  live
```
""")

st.markdown(pill("Nessuna promozione per giorni È un buon segno: il gate "
                 "sta rifiutando beta travestito da alpha.", "gold"),
            unsafe_allow_html=True)

# ── GLOSSARIO ────────────────────────────────────────────────────────────────

section("Glossario")
st.caption("Le stesse spiegazioni che trovi nei popover ⓘ in giro per la "
           "GUI, tutte in un posto.")

for term, e in GLOSSARY.items():
    with st.expander(term):
        st.markdown(e["def"])
        st.markdown(f"**Come si legge:** {e['lettura']}")
        st.markdown(f"**Perché conta qui:** {e['perche']}")

# ── COMANDI ──────────────────────────────────────────────────────────────────

section("Comandi essenziali")
st.code("""# ricerca autonoma
tradebot research-daemon      # avvia (gira finché non lo fermi)
tradebot research-status      # stato daemon + scorecard
tradebot research-stop        # ferma pulito

# dati
tradebot ingest-all           # aggiorna i prezzi (salta gli aggiornati)

# validazione
tradebot run momentum_12_1 --cpcv
tradebot validate momentum_12_1
tradebot holdout-test         # SOLO una volta, prima del live

# live
tradebot run-daily            # dry-run (default)
tradebot run-daily --live     # ordini reali sul paper
""", language="bash")
st.caption("Cheat-sheet completo in COMANDI.md nella root del progetto.")
