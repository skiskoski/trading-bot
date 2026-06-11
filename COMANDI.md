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

# STATO: pid, uptime, trial in sessione, scorecard
tradebot research-status

# SEGUI il log in tempo reale (Ctrl+C per smettere di guardare, NON lo ferma)
tail -f ~/.trading_bot/research.log

# FERMA il daemon (chiusura pulita: completa il trial in corso e persiste)
tradebot research-stop

# Stop con più pazienza (se sta macinando un CPCV lungo)
tradebot research-stop --timeout 180
```

Opzioni utili del daemon:

```bash
# Universo più piccolo, più candidati per round
tradebot research-daemon --top 200 --candidates 12

# Solo strategie a segnale singolo
tradebot research-daemon --n-signals 1
```

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

---

## File utili

| Percorso | Cosa contiene |
|---|---|
| `~/.trading_bot/research.pid` | PID del daemon di ricerca (se attivo) |
| `~/.trading_bot/research.log` | Log rotante del daemon (10MB × 5) |
| `data/trading_bot.db` | Database SQLite (prezzi, run, research log) |
| `.env` | Chiavi Alpaca / Telegram (vedi `.env.example`) |
