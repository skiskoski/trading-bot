"""Glossario quant in italiano + popover ⓘ riusabili.

Ogni voce: definizione (2 frasi), come si legge (buono/sospetto/pessimo),
perché conta per QUESTO bot. Usato dai popover di ogni pagina e dalla
pagina Guida.
"""

from __future__ import annotations

import streamlit as st

from trading_bot.gui.style import section

GLOSSARY: dict[str, dict[str, str]] = {
    "Sharpe": {
        "def": "Rendimento extra per unità di rischio: media dei rendimenti "
               "diviso la loro volatilità, annualizzato. È IL numero per "
               "confrontare strategie diverse sulla stessa scala.",
        "lettura": "✅ >1 ottimo per un portafoglio azionario · 🟡 0.5–1 "
                   "decente · ❌ <0.5 non ripaga il rischio. Occhio: un "
                   "backtest con Sharpe >2 su azioni è quasi sempre overfit.",
        "perche": "Il gate di promozione usa lo Sharpe ATTIVO (vs equal-"
                  "weight), così il beta di mercato non conta come skill.",
    },
    "OOS attivo": {
        "def": "Sharpe calcolato SOLO su dati out-of-sample (mai visti in "
               "ottimizzazione) e al netto del benchmark equal-weight. "
               "È la stima più onesta dell'edge reale.",
        "lettura": "✅ ≥0.5 supera il gate · 🟡 0.2–0.5 c'è qualcosa ma "
                   "fragile · ❌ ≤0 il segnale non esiste fuori campione.",
        "perche": "Il bot promuove solo strategie con OOS attivo ≥0.5 su 45 "
                  "path CPCV: se l'edge non sopravvive fuori campione, in "
                  "live brucerebbe soldi veri.",
    },
    "PBO": {
        "def": "Probability of Backtest Overfitting: frazione dei path CPCV "
               "in cui lo Sharpe out-of-sample finisce sotto la mediana "
               "in-sample. Misura quanto il risultato dipende dall'aver "
               "'memorizzato' il passato.",
        "lettura": "In questa variante ~0.5 è NEUTRO (per una strategia "
                   "onesta IS e OOS si equivalgono) · ✅ <0.4 l'OOS batte "
                   "l'IS, edge robusto · ❌ >0.6 l'edge evapora fuori "
                   "campione: overfitting.",
        "perche": "Il famoso 'PBO sempre 0.6' era discretizzazione: con 15 "
                  "path i valori possibili erano 16. Ora k=10 → 45 path, "
                  "granularità 1/45.",
    },
    "DSR": {
        "def": "Deflated Sharpe Ratio (Bailey & López de Prado): probabilità "
               "che lo Sharpe osservato sia vero skill e non il massimo di N "
               "tentativi fortunati. Più strategie provi, più la soglia sale.",
        "lettura": "✅ >0.95 evidenza forte · 🟡 0.5–0.95 promettente ma non "
                   "conclusivo · ❌ ~0 indistinguibile dal caso (o N trial "
                   "altissimo).",
        "perche": "Il bot ha provato centinaia di strategie: senza questa "
                  "correzione, il 'migliore' sarebbe quasi certamente solo "
                  "il più fortunato. σ_SR viene stimato dai trial reali.",
    },
    "CPCV": {
        "def": "Combinatorial Purged Cross-Validation (De Prado): divide la "
               "storia in k=10 blocchi e testa su tutte le 45 coppie di "
               "blocchi out-of-sample, con purge anti-leakage tra train e test.",
        "lettura": "Non un numero ma una DISTRIBUZIONE di 45 Sharpe OOS: "
                   "guarda mediana, dispersione e % di path positivi "
                   "(gate: ≥65%).",
        "perche": "Un singolo backtest è un solo film del passato; 45 path "
                  "dicono se la strategia regge in scenari diversi o ha "
                  "funzionato solo in un periodo fortunato.",
    },
    "Walk-forward": {
        "def": "Validazione 'come se vivessi nel passato': ottimizzi su una "
               "finestra, testi sulla successiva, scorri avanti e ripeti. "
               "Ogni decisione usa solo dati disponibili a quel momento.",
        "lettura": "✅ performance OOS stabile tra le finestre · ❌ un paio "
                   "di finestre buone e il resto piatto = edge episodico.",
        "perche": "Complementare al CPCV: rispetta l'ordine temporale, "
                  "quindi simula meglio l'esperienza reale di trading.",
    },
    "IC": {
        "def": "Information Coefficient: correlazione tra il ranking dei "
               "segnali oggi e i rendimenti realizzati domani. Misura se il "
               "segnale 'indovina' l'ordine dei titoli.",
        "lettura": "✅ >0.05 forte per dati daily · 🟡 0.02–0.05 tipico di "
                   "un edge reale · ❌ ~0 il segnale è rumore.",
        "perche": "Il pre-scan IC (soglia 0.02) scarta in pochi secondi i "
                  "candidati senza potere predittivo, risparmiando un CPCV "
                  "completo da minuti.",
    },
    "MaxDD": {
        "def": "Maximum Drawdown: la peggior discesa dal massimo storico al "
               "minimo successivo. Il dolore peggiore che avresti provato "
               "restando investito.",
        "lettura": "✅ <20% gestibile · 🟡 20–35% serve stomaco · ❌ >50% "
                   "quasi nessuno resta investito (e il recupero richiede "
                   "+100%).",
        "perche": "Con €25k e orizzonte lungo puoi permetterti drawdown "
                  "medi, ma il vol targeting al 25% esiste proprio per "
                  "tagliare le code peggiori.",
    },
    "Calmar": {
        "def": "CAGR diviso MaxDD: quanto rendimento annuo ottieni per ogni "
               "punto di drawdown sofferto.",
        "lettura": "✅ >1 eccellente su orizzonti lunghi · 🟡 0.5–1 buono · "
                   "❌ <0.3 troppo dolore per il rendimento.",
        "perche": "Premia le strategie che crescono SENZA crolli — più "
                  "rilevante dello Sharpe se l'obiettivo è restare investiti "
                  "per anni.",
    },
    "CAGR": {
        "def": "Compound Annual Growth Rate: il tasso di crescita annuo "
               "composto. 'Se ogni anno facesse X%, dopo N anni avrei "
               "esattamente questo capitale'.",
        "lettura": "Confrontalo sempre col benchmark: l'S&P 500 storicamente "
                   "fa ~10% nominale. Un CAGR alto con MaxDD enorme è una "
                   "trappola.",
        "perche": "È il numero che decide quanto vale il tuo portafoglio tra "
                  "10 anni — ma da solo non dice nulla sul rischio corso.",
    },
    "Vol targeting": {
        "def": "Scala l'esposizione per mantenere la volatilità del "
               "portafoglio vicino a un target (qui 25% annuo): riduce in "
               "tempesta, torna piena in mare calmo (Moreira-Muir 2017).",
        "lettura": "Vol realizzata sotto target → esposizione 100%; sopra → "
                   "scala di target/realizzata (floor 20%).",
        "perche": "Il 25% è growth-oriented: nel sweep 2015-2026 era lo "
                  "Sharpe-ottimo mantenendo l'84% del CAGR e tagliando il "
                  "MaxDD.",
    },
    "Crisis alpha": {
        "def": "Rendimento positivo proprio quando l'azionario crolla. "
               "L'oro TSMOM tende a salire (o non scendere) nelle crisi "
               "equity: correlazione bassa o negativa nei drawdown.",
        "lettura": "Guarda la performance della sleeve oro NEI mesi rossi "
                   "dell'equity: è lì che si guadagna il suo posto.",
        "perche": "La sleeve oro al 20% non è lì per il rendimento medio ma "
                  "per comprare 'assicurazione' che paga nei momenti "
                  "peggiori del core azionario.",
    },
    "TSMOM": {
        "def": "Time-Series Momentum: compra un asset se il SUO rendimento "
               "recente è positivo, esci (o short) se negativo. Trend-"
               "following puro, un asset alla volta.",
        "lettura": "Funziona da decenni su quasi tutte le asset class "
                   "(Moskowitz-Ooi-Pedersen 2012); soffre nei mercati "
                   "laterali a sega.",
        "perche": "La sleeve oro usa TSMOM: sta nel metallo solo quando il "
                  "trend è amico, cash altrimenti — meglio del buy&hold "
                  "dell'oro.",
    },
    "Momentum 12-1": {
        "def": "Momentum cross-sectional: compra i titoli col rendimento "
               "migliore negli ultimi 12 mesi SALTANDO l'ultimo (che tende "
               "a invertire). Il fattore più documentato in letteratura.",
        "lettura": "Edge storico ~misurabile ma con crash periodici "
                   "(2009): il regime filter serve a quello.",
        "perche": "È il core del portafoglio: Jegadeesh-Titman 1993, "
                  "replicato su 200+ anni e decine di mercati.",
    },
    "Regime filter": {
        "def": "Interruttore di contesto: esponi solo quando il mercato è "
               "in salute (es. SPY sopra la SMA200), riduci o esci quando "
               "il trend di fondo è rotto.",
        "lettura": "RISK-ON = piena esposizione · RISK-OFF = ridotta/zero. "
                   "Costa qualche falso allarme, evita i disastri lunghi.",
        "perche": "Il momentum crasha proprio nei rimbalzi post-panico: il "
                  "filtro taglia le code peggiori di quelle fasi.",
    },
    "Turnover": {
        "def": "Quanto del portafoglio viene movimentato a ogni ribilancio. "
               "100% = vendi tutto e ricompri tutto.",
        "lettura": "✅ <30%/mese contenuto · ❌ alto turnover = costi e "
                   "slippage che mangiano l'edge su un conto piccolo.",
        "perche": "Con €25k i costi pesano: ribilancio mensile (più trigger "
                  "vol) esiste per tenere il turnover sotto controllo.",
    },
    "Holdout": {
        "def": "Periodo finale di dati MAI toccato durante la ricerca "
               "(qui: dal 2024). Si usa UNA volta sola, come esame finale "
               "prima di andare live.",
        "lettura": "✅ performance in linea col CPCV → conferma · ❌ crollo "
                   "nel holdout → la strategia era overfit nonostante tutto.",
        "perche": "Ogni sguardo al holdout lo 'consuma': se lo guardi dieci "
                  "volte, diventa training set e non prova più niente.",
    },
    "PIT": {
        "def": "Point-In-Time universe: il backtest usa solo i titoli che "
               "erano nell'indice IN QUELLA DATA, inclusi quelli poi falliti "
               "o rimossi.",
        "lettura": "Senza PIT il backtest compra i sopravvissuti di oggi nel "
                   "passato: +1-2% di CAGR fantasma (survivorship bias).",
        "perche": "Il bot scarica anche i ticker delisted proprio per "
                  "questo: i numeri che vedi non barano sui morti.",
    },
}


def spiega(*terms: str, label: str = "ⓘ spiegazione") -> None:
    """Popover con le voci di glossario richieste."""
    valid = [t for t in terms if t in GLOSSARY]
    if not valid:
        return
    with st.popover(label):
        for t in valid:
            e = GLOSSARY[t]
            st.markdown(f"**{t}**")
            st.markdown(e["def"])
            st.markdown(f"*Come si legge:* {e['lettura']}")
            st.markdown(f"*Perché conta qui:* {e['perche']}")
            st.divider()


def section_help(title: str, *terms: str) -> None:
    """Header di sezione con popover ⓘ a destra."""
    c1, c2 = st.columns([0.88, 0.12])
    with c1:
        section(title)
    with c2:
        spiega(*terms, label="ⓘ")
