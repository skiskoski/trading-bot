# COMANDI — cheat-sheet TradingBot

Tutti i comandi presuppongono:

```bash
cd ~/trading-bot && source .venv/bin/activate
```

Per qualsiasi comando: `tradebot <comando> --help` mostra le opzioni.

---

## Setup iniziale (una volta sola)

```bash
# 1. Crea il database
tradebot init-db

# 2. Scarica i costituenti S&P 1500 (~1500 ticker)
tradebot fetch-universe

# 3. Variazioni storiche dell'indice (anti-survivorship bias)
tradebot fetch-changes

# 4. Scarica tutti i prezzi (prima volta: 20-40 min)
tradebot ingest-all

# 5. Prezzi dei ticker delisted (mai più nell'indice)
tradebot ingest-historic

# 6. Verifica copertura
tradebot universe-stats
```

Tutto in serie, copia-incolla unico:

```bash
cd ~/trading-bot && source .venv/bin/activate && tradebot init-db && tradebot fetch-universe && tradebot fetch-changes && tradebot ingest-all && tradebot ingest-historic && tradebot universe-stats
```

---

## Uso quotidiano

```bash
# Aggiorna i prezzi (salta i ticker già aggiornati)
tradebot ingest-all

# Apri la GUI (multi-sezione, sidebar a sinistra)
tradebot gui

# GUI su porta diversa
tradebot gui --port 8502

# Aggiornamento + GUI in un colpo solo
cd ~/trading-bot && source .venv/bin/activate && tradebot ingest-all && tradebot gui
```

---

## Ricerca autonoma (daemon)

Il daemon gira **finché non lo fermi tu** — ricicla batch di round
all'infinito, sopravvive alla chiusura del terminale.

```bash
# AVVIA il daemon (torna subito al prompt)
tradebot research-daemon

# AVVIA con 2 strategie in parallelo (~1.5x più veloce su multi-core)
tradebot research-daemon --workers 2

# SEGUI il log in tempo reale con colori (Ctrl+C per smettere, NON ferma il daemon)
tradebot research-follow

# STATO: pid, uptime, trial in sessione, scorecard
tradebot research-status

# FERMA il daemon (chiusura pulita: completa il trial in corso e persiste)
tradebot research-stop

# Stop con più pazienza (se sta macinando un CPCV lungo)
tradebot research-stop --timeout 180
```

Opzioni utili del daemon:

```bash
# Universo più piccolo, più candidati per round, 2 worker paralleli
tradebot research-daemon --top 200 --candidates 12 --workers 2

# Solo strategie a segnale singolo
tradebot research-daemon --n-signals 1
```

Perché è lento? Ogni trial che supera l'IC prescan richiede 45 backtest CPCV su 20
anni di dati. Su un laptop, un trial dura 1-4 minuti. Non è bloccato — sta lavorando.
Con `--workers 2` si testano due strategie in parallelo: speedup reale ~1.4-1.8x.

Run di ricerca singolo (vecchio comportamento, 5 round e si ferma):

```bash
tradebot research
```

---

## Backtest e strategie

```bash
# Catalogo strategie
tradebot strategies

# Backtest singolo (con validazione CPCV: 45 path OOS)
tradebot run momentum_12_1 --cpcv

# Tutte le strategie del catalogo
tradebot run-all

# Portafoglio core equity + sleeve oro TSMOM
tradebot run-portfolio

# CPCV su una strategia del catalogo
tradebot validate-cpcv momentum_12_1

# Walk-forward + Deflated Sharpe
tradebot validate momentum_12_1
```

---

## Robustness validation pre-paper

Questi comandi non autorizzano capitale reale: servono a decidere se una
strategia resta in ricerca o puo passare a paper-review.

```bash
# Pipeline completa: backtest netto costi, CPCV, stability, Monte Carlo, walk-forward
tradebot validate-strategy momentum_12_1

# Report completo senza CPCV extra
tradebot robustness-report momentum_12_1 --mc-trials 1000

# Solo stabilita parametri / plateau
tradebot parameter-stability momentum_12_1 --grid momentum_core --mc-trials-per-set 100

# Solo Monte Carlo a blocchi sui rendimenti della strategia
tradebot monte-carlo momentum_12_1 --trials 2000

# Solo walk-forward rolling + anchored
tradebot walk-forward momentum_12_1 --folds 8
```

Output salvati in:

```bash
data/validation_reports/*_robustness.json
```

Hard gate MVP: PIT attivo, CPCV PBO < 0.4, OOS Sharpe >= 0.5, path positivi
>=65%, plateau parametri non isolato, Monte Carlo per ogni parameter set nei
report completi, Monte Carlo finale con Sharpe p05 >= 0, loss probability <=35%,
drawdown p95 <=35%, walk-forward con >=65% fold positivi e worst fold Sharpe
>= -0.25.

---

## Market Storm / Meteo Mercato

Risk overlay quantitativo: misura il regime di mercato e suggerisce quanta
esposizione accettare. Non e una strategia alfa e non autorizza capitale reale.

```bash
# Stato attuale del mercato: score, regime, componenti, azione consigliata
tradebot market-storm --top 300

# Storico score/regime, con CSV opzionale
tradebot market-storm-history --start 2005-01-01 --top 300 \
  --output data/market_storm_reports/storm_history.csv

# Confronta una strategia base vs overlay Market Storm
tradebot validate-storm-overlay momentum_12_1 --top 300

# Genera report JSON leggibile dalla GUI Meteo Mercato
tradebot storm-report momentum_12_1 --top 300
```

Output salvati in:

```bash
data/market_storm_reports/*_storm_report.json
```

Regimi MVP:

```text
calm     -> exposure 1.00 -> normal
unstable -> exposure 0.75 -> reduce_exposure
storm    -> exposure 0.50 -> defensive_only
panic    -> exposure 0.25 -> block_new_entries
```

Il gate dell'overlay guarda drawdown, worst month, CAGR, volatilita e mesi
peggiori SPY. L'obiettivo e ridurre i casi brutti, non massimizzare lo Sharpe
in-sample.

---

## Validazione finale e live

```bash
# Holdout test — ATTENZIONE: consuma il holdout, solo una volta per strategia
tradebot holdout-test

# Ciclo giornaliero in simulazione (sicuro, default)
tradebot run-daily

# Ciclo giornaliero con ordini reali su Alpaca paper
tradebot run-daily --live

# Stato conto Alpaca
tradebot alpaca-status

# Solo segnali, senza ordini
tradebot generate-signals

# Scheduler giornaliero macOS (launchd)
tradebot setup-scheduler
```

Nota: `generate-signals` e `run-daily` sono fail-closed. Se nessuna strategia
ha passato il gate corrente di promozione/validazione, il bot non sceglie una
strategia vecchia dal registry e non invia falsi messaggi Telegram "Signals".

---

## File utili

| Percorso | Cosa contiene |
|---|---|
| `~/.trading_bot/research.pid` | PID del daemon di ricerca (se attivo) |
| `~/.trading_bot/research.log` | Log rotante del daemon (10MB × 5) |
| `data/trading_bot.db` | Database SQLite (prezzi, run, research log) |
| `.env` | Chiavi Alpaca / Telegram (vedi `.env.example`) |
