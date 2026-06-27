# TradingBot — Mappa Completa dei Processi

> Documento generato il 2026-06-26 da Claude Code tramite ispezione live del codebase.
> Sorgente: `/Users/simone/trading-bot/src/trading_bot/`
> Aggiornare quando si aggiungono moduli o si cambiano interfacce rilevanti.

---

## Database SQLite — 8 tabelle

| Tabella | Chiave | Campi principali | Scopo |
|---|---|---|---|
| `tickers` | symbol UNIQUE | symbol, name, sector, sub_industry, index_name, added_at | Universo S&P 1500 |
| `candles` | (symbol, dt) UNIQUE | open, high, low, close, adj_close, volume | OHLCV giornaliero |
| `index_changes` | id | dt, added, removed, reason, index_name | Ricostruzione PIT membership |
| `strategies` | name UNIQUE | name, rationale (≥20 ch), config_json, status, created_at | Registry strategie |
| `runs` | id | strategy_id/name, start/end_date, universe_size, use_pit, metrics_json, equity_json, returns_json, psr, dsr, n_trials_used, cpcv_json, created_at | Backtest log |
| `feature_scores` | feature_key UNIQUE | times_used, times_promoted, sum_oos_sharpe, sum_pbo, score, updated_at | Scorecard Bayesiana feature |
| `research_log` | id | round_id, hypothesis_name, config_json, rationale, ic_prescan, oos_sharpe, pbo, dsr, status, skip_reason, arm_fingerprint, universe_size, created_at | Trial research loop |
| `trial_counter` | id | total_trials, updated_at | Correzione multiple-testing globale (DSR) |

---

## Layer 1 — Data

### `data/universe.py` — Universo S&P 1500
| Funzione | Input | Output | DB |
|---|---|---|---|
| `fetch_sp500/400/600_constituents()` | Wikipedia HTML | `list[UniverseEntry]` | — |
| `fetch_all_constituents()` | — | union dedup (500 > 400 > 600) | — |
| `persist_universe(entries)` | `list[UniverseEntry]` | int rows upserted | scrive `tickers` |
| `get_top_n_by_liquidity(n, min_adv_usd, lookback_days)` | filtri liquidità | `list[str]` symbols ranked by ADV | legge `candles` |

### `data/provider.py` — Wrapper yfinance
| Funzione | Input | Output |
|---|---|---|
| `download_ohlcv(symbols, start, end)` | list symbols, date range | `pd.DataFrame` long (date, symbol, OHLCV) |
| `last_stored_date(symbol)` | symbol | `date \| None` |

### `data/ingest.py` — Persistenza
| Funzione | Input | Output | DB |
|---|---|---|---|
| `ingest_symbols(symbols, start, end, chunk_size=50)` | symbols, date range | int rows inserted | scrive `candles` |
| `incremental_update(symbols, default_start)` | symbols | int rows inserted | legge/scrive `candles` |
| `ingest_fx(start)` | start date | int rows | scrive `candles` (EUR/USD pseudo-ticker) |
| `load_fx_eurusd(start, end)` | date range | `pd.Series` | legge `candles` |
| `load_panel(symbols, start, end)` | symbols, date range | `pd.DataFrame` WIDE (date × symbol) | legge `candles` |

### `data/pit_universe.py` — Point-in-Time (no look-ahead)
| Funzione | Input | Output | DB |
|---|---|---|---|
| `fetch_index_changes(indices)` | indici target | `pd.DataFrame` (dt, added, removed, reason) | scrape Wikipedia |
| `persist_index_changes(df)` | DataFrame changes | int rows | scrive `index_changes` (wipe+reinsert) |
| `ever_in_index(current_universe)` | current symbols | `set[str]` (current ∪ ever removed) | legge `index_changes` |
| `build_membership_panel(dates, current_universe, symbols)` | date index | `pd.DataFrame` bool (date × symbol) | legge `index_changes` |

---

## Layer 2 — Features (29)

**File:** `features/price.py`, `features/cross_section.py`

**Principio:** funzioni pure, nessuna scrittura DB. Input = `(prices: pd.Series, asof: Timestamp, **params)`. Output = `float`.

| Categoria | Feature | Params default |
|---|---|---|
| **Momentum (6)** | `momentum` | lookback=252, skip=21 (12-1) |
| | `rate_of_change` | lookback=63 |
| | `price_acceleration` | short=21, long=63 |
| | `ema_ratio` | fast=21, slow=63 |
| | `macd_histogram` | fast=12, slow=26, signal=9 |
| | `relative_strength` | lookback=126 |
| **Reversal (4)** | `short_term_reversal` | lookback=5 |
| | `rsi` | period=2 (Connors) |
| | `bollinger_position` | window=20, n_std=2.0 |
| | `mean_reversion_score` | lookback=252, short=21 |
| **Risk (3)** | `volatility` | lookback=63 |
| | `downside_vol` | lookback=63 |
| | `vol_trend` | fast=21, slow=63 |
| **Structural (4)** | `sma_distance` | window=200 |
| | `above_sma` | window=200 (→ bool) |
| | `drawdown` | lookback=252 |
| | `autocorr_returns` | lag=1 |
| **Quality/State (12)** | `kalman_trend`, `hurst_exponent`, `return_entropy`, `quality_score`, `beta_spy`, `amihud_illiquidity`, `volume_momentum`, `seasonal_month_return`, … | vari |

**Trasformazioni cross-sezione:**
- `cross_section_rank(s)` → [0, 1] rank robusto outlier
- `cross_section_zscore(s)` → z-score standard
- `top_decile(s, decile=0.1)` → bool top 10%

---

## Layer 3 — Strategy Engine

### `strategies/composer.py` — Strategia dichiarativa
**Config:**
- `SignalSpec(feature, params, weight=1.0, use_rank=True, negate=False)`
- `FilterSpec(feature, params, threshold=0.0)`
- `RegimeSpec(symbol, feature, params, threshold)`
- `ComposedConfig(signals, filters, regime, top_n, gold_weight, gold_mode)`

**`ComposedStrategy.rank(prices, asof) → pd.Series`:**
1. Valuta tutti i signal → score per symbol
2. Applica `cross_section_rank` se `use_rank=True`
3. Combina linealmente ponderato
4. Applica filtri (AND logico)
5. Esclude regime asset e GLD

**`ComposedStrategy.weights(prices, asof) → pd.Series`:**
1. Regime gate (equity sleeve): se SPY < SMA200 → all-cash
2. Top-N equal weight da rank
3. Gold sleeve: peso proporzionale a drawdown (se `gold_mode="defensive"`) o fisso
4. GLD solo se sopra SMA200

### `strategies/momentum.py` — Jegadeesh-Titman 1993
- 12-1 momentum + filtri SMA100 + no-gap
- Regime gate: SPY > SMA200

### `strategies/tsmom.py` — TSMOM singolo asset (GLD)
- Trend-following su oro

---

## Layer 4 — Backtest Engine

### `backtest/engine.py` — CrossSectionalBacktester

**Config chiave:**
- `initial_capital=100_000`, `cost_bps=5.0` (1-way), `fixed_cost_per_trade` (IBKR)
- `vol_target`, `vol_lookback=63` (Moreira-Muir)
- `adaptive_rebalance=False`, `vol_trigger=0.25`
- `circuit_breaker_dd=0.15`, `circuit_breaker_scale=0.5`
- `panic_filter=False`, `panic_scale=0.5` (Daniel-Moskowitz)
- `hold_band_mult=2.0` (Novy-Marx-Velikov buy/hold band)

**`run(prices) → BacktestResult`:**
```
prices panel
  → PIT membership mask
  → strategy.weights() per rebalance date
  → shift +1 giorno (no look-ahead)
  → circuit breaker (drawdown > 15% → scale 0.5)
  → panic filter (SPY < SMA200 AND alta vol → scale)
  → vol targeting (scale = clip(vol_target / realized_vol, 0.2, 1.0))
  → transaction cost (turnover × cost_bps + n_orders × fixed_cost / equity)
  → equity curve = (1 + returns).cumprod() × initial_capital
```

### `backtest/metrics.py`
- `cagr(equity)`, `sharpe(returns, rf)`, `sortino(returns)`, `max_drawdown(equity)`
- `information_coefficient(predictions, forward_returns)` → Spearman per data
- `hit_rate(returns)`, `summary(equity, returns, ic)` → dict completo

---

## Layer 5 — Validation Pipeline

### `validation/cpcv.py` — Combinatorial Purged CV
**Config:** k=10 folds, n_test=2, purge_days=21 → C(10,2) = **45 path**

**`run_cpcv(...) → CPCVResult`:**
1. Split in k fold uguali
2. Per ogni combo C(k, n_test):
   - IS Sharpe su train purged (±21d da ogni boundary test)
   - OOS Sharpe su test dates
3. **PBO** = fraction OOS Sharpe < median IS Sharpe (> 0.5 → overfit likely)
4. Output: distribuzione OOS Sharpe, PBO, mean/std/median/fraction_positive

### `validation/deflated_sharpe.py` — Multiple-testing correction
| Funzione | Formula |
|---|---|
| `probabilistic_sharpe_ratio(returns, benchmark_sr)` | P(true_Sharpe > benchmark), account per skew e kurtosi |
| `expected_max_sharpe(n_trials)` | Bailey-López de Prado approximation |
| `deflated_sharpe_ratio(returns, n_trials, sigma_sr)` | benchmark = E[max] × σ_SR; DSR > 0.95 → non overfit |

**σ_SR** = deviazione std cross-sezionale degli Sharpe di tutti i run → `estimate_sigma_sr()` in `registry.py`.

### `validation/robustness.py` + `walk_forward.py` + `holdout.py`
- `run_parameter_stability()` — grid di parametri → stabilità plateau
- `run_monte_carlo(returns, n_trials=1000)` — moving-block bootstrap
- `run_walk_forward(strategy, prices, n_folds=8)` — rolling + anchored
- `run_holdout_test(strategy_name, holdout_start="2024-01-01")` — final OOS (consuma holdout)

### `risk/market_storm.py` — Regime overlay
**9 componenti (score 0-100 ciascuno):**
1. `spy_vol` — volatilità realizzata 21d SPY
2. `vol_spike` — fast vol / slow vol
3. `spy_drawdown` — distanza da peak
4. `correlation` — correlazione media cross-section
5. `breadth` — % titoli sotto SMA200
6. `downside_breadth` — % ritorni negativi
7. `dispersion` — vol dispersion cross-section
8. `sector_stress` — danno a livello settore
9. `credit` — proxy distress HYG

**Regime:** calm (<25) → unstable (25-50) → storm (50-75) → panic (75+)
**Output:** `exposure_scale` per regime → scala le posizioni live

---

## Layer 6 — Research Loop Autonomo

### `research/loop.py` — Main loop

**Flusso per round:**
```
analyze_runs() → AnalysisReport
  ↓
generate candidates:
  - UCB1 arms (exploitation, 50%): top feature_scores
  - archetype (exploration, 50%): structural diversity
  ↓
IC prescan (fast filter, ultimi 3 anni):
  - IC < 0.02 → skip (log research_log status="skipped")
  ↓
full backtest + CPCV (k=10, 45 path):
  - increment_trial_count(1)
  - persist_run()
  - update feature_scores (Bayesian)
  - log research_log status="tested" o "promoted"
  ↓
Promozione gate:
  OOS Sharpe ≥ 0.5 AND PBO < 0.4 AND DSR ≥ 0.95 AND fraction_positive ≥ 65%
  → status="promoted" in strategies
```

### `research/generator.py` — Generazione ipotesi
- Single-signal: itera FEATURES, varia params (max 3 combo), aggiunge filter+regime
- Dual-signal: combina coppie, synergy score, varia weights (50/50, 60/40, 70/30)
- Novelty check: fingerprint delle config già testate → no duplicati
- Ranking: 50% expected_score + 30% novelty + 20% economic_strength

### `research/analyzer.py` — Analisi storico
- Best/worst features per avg OOS Sharpe
- Combinazioni feature non ancora testate
- Promotion rate, best top_n

### `research/scorecard.py` — Bayesian scorecard
- Prior: `score = sum_oos_sharpe / (times_used + 1)`
- Post-trial update: `sum_oos_sharpe += oos_sharpe`, `sum_pbo += pbo`

### `research/daemon.py` — Background daemon
- Subprocess headless, SIGTERM safe, PID tracking
- `start_daemon()`, `stop_daemon()`, `daemon_pid()`, `daemon_started_at()`

---

## Layer 7 — Portfolio

### `portfolio/combiner.py`
**Input:** `dict[str, pd.Series]` returns per sleeve (equity, gold)
**Output:** `PortfolioResult`
- Equity curve pesata (vol. target 25%, gold sleeve 20%)
- Rolling 66d correlation tra sleeves
- Crisis alpha table: N worst months equity → gold return in quegli stessi mesi
- Per-sleeve metrics: CAGR, Sharpe, MaxDD

---

## Layer 8 — Live Trading

### `live/signals.py`
- `generate_signals(strategy_name, universe_size=500) → SignalOutput`
- Carica best promoted strategy dal registry
- Output: `positions` dict (symbol → weight), `regime_active`, `ranked_universe`

### `live/runner.py`
```
run_daily(dry_run=True):
  1. ingest_symbols (dati di oggi)
  2. generate_signals()
  3. is_rebalance_day()? (1° trading day mese) + is_volatility_rebalance_day()? (Monday, alta vol)
  4. apply_portfolio_overlays(core_positions)
  5. submit_order() per ogni posizione target (se non dry_run)
  6. drawdown check → alert Telegram se > 15%
  7. Telegram notify (esito + positions)
```

### `live/alpaca.py` — AlpacaClient (paper only)
- Guard: base_url ≠ live endpoint
- `get_account()`, `get_positions()`, `submit_order(symbol, qty, side)`, `close_all_positions()`

### `live/notifier.py`
- `TelegramNotifier.send_message()`, `send_order_summary()`

---

## Layer 9 — CLI (`cli.py`) — 50+ comandi

| Gruppo | Comandi chiave |
|---|---|
| **Data** | `init-db`, `fetch-universe`, `ingest`, `ingest-all`, `ingest-fx`, `fetch-changes`, `ingest-historic` |
| **Strategy** | `strategies`, `run NAME`, `run-all`, `backtest NAME` |
| **Research** | `research`, `research-daemon`, `research-stop`, `research-follow`, `research-status` |
| **Validation** | `validate-cpcv`, `validate-storm-overlay`, `validate-strategy`, `parameter-stability`, `monte-carlo`, `walk-forward`, `robustness-report` |
| **Portfolio** | `run-portfolio`, `market-storm`, `market-storm-history` |
| **Live** | `generate-signals`, `run-daily [--dry-run\|--live]`, `holdout-test`, `alpaca-status`, `setup-scheduler` |
| **GUI** | `gui` (Streamlit, port 8501) |

---

## Layer 10 — GUI Streamlit (9 view)

| View | Contenuto |
|---|---|
| `panoramica` | Overview metriche aggregate, equity top 5 vs SPY |
| `ricerca` | Progress research loop, funnel IC prescan → CPCV → promozione |
| `strategie` | Catalog + run metrics, scatter OOS/PBO con gate |
| `portafoglio` | Sleeve combination, crisis alpha table, rolling correlation |
| `validazione` | CPCV results, distribuzione OOS Sharpe, robustness gate |
| `meteo` | Market Storm gauge, componenti, regime attuale |
| `live` | Segnali giornalieri, Alpaca account, ordini recenti |
| `storico` | Run history, equity curves, heatmap mensile ritorni |
| `guida` | Documentazione interna, glossario (Sharpe, DSR, PBO, ecc.) |

**Tutte le view sono read-only sul DB.** Nessuna scrittura dalla GUI.

---

## Flussi Principali

### Setup iniziale
```
fetch-universe → tickers DB
ingest-all (yfinance) → candles DB
fetch-changes → index_changes DB
ingest-historic → candles (survivorship bias fix)
ingest-fx → candles (EUR/USD)
```

### Research Loop (autonomo, giornaliero/notturno)
```
analyze_runs() → generate candidates (UCB1 + archetype)
  → IC prescan (fast filter) → full backtest + CPCV (45 path)
  → persist_run() → update scorecard → promote se DSR≥0.95, PBO<0.4
```

### Ciclo Live (giornaliero, schedulato via launchd)
```
ingest prezzi oggi → generate_signals() → is_rebalance_day()?
  → apply overlays → submit orders (Alpaca paper)
  → drawdown check → Telegram notify
```

### Validation Pipeline (prima del deploy)
```
CPCV (45 path, PBO < 0.4)
  → Monte Carlo 1000 → Walk-Forward 8 fold
  → DSR ≥ 0.95 (con trial_counter globale)
  → promuovi / scarta
```

---

## Audit Findings nel Codice

| ID | Descrizione | Status |
|---|---|---|
| M3 | Partial coverage in rank: require full signal coverage | fixed |
| M4 | Alpaca live endpoint guard: normalize URL, block live | fixed |
| M5 | Vol targeting turnover charge: Δscale × gross_exposure × bps | fixed |
| M6 | Sharpe con rf=0: scaling consistente interno | fixed |
| C1 | Look-ahead in weight shift: shift +1 giorno, non +1 period | fixed |
| C2 | Absolute Sharpe su beta: usare active Sharpe vs EW benchmark | noted |
| A1 | Purge CPCV: purge attorno a ogni fold test, non span merged | fixed |
