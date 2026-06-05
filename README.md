# trading-bot

Quantitative trading bot for US equities (paper trading). Built on a
research foundation in `docs/` that compares 5 strategic options via the
Fundamental Law of Active Management and selects a hierarchical
multi-sleeve allocation.

The architecture lets you declare a new strategy in a ~10-line dict
(feature + filter + regime + sizing), runs it through an event-driven
cross-sectional backtester with anti-look-ahead, persists the run, and
visualises everything in a Streamlit GUI with built-in statistical
honesty checks (PIT membership, Information Coefficient, Deflated
Sharpe Ratio with honest trial counter).

## Quickstart

```bash
# 1. install (Python 3.11+)
pip install -e .[dev]

# 2. initialize and ingest data (~2 min for ~1.7M candles)
tradebot init-db
tradebot fetch-universe         # S&P 500 constituents from Wikipedia
tradebot fetch-changes          # historical add/remove for PIT
tradebot ingest-all --start 2015-01-01
tradebot ingest-historic --start 2015-01-01

# 3. run the catalog of strategies
tradebot strategies             # list catalog
tradebot run momentum_12_1      # run a single strategy
tradebot run-all                # run every catalog strategy

# 4. launch the GUI
tradebot gui                    # opens http://localhost:8501
```

## Strategy catalog (v0)

| Name | Family | Rationale |
|---|---|---|
| `momentum_12_1` | Cross-sectional momentum | Jegadeesh-Titman 1993 / Clenow 2015 |
| `rsi2_reversal` | Short-term mean reversion | Connors-Alvarez 2009 / Lehmann 1990 |
| `low_volatility` | Defensive | Frazzini-Pedersen BAB / AFP QMJ |
| `combo_mom_lowvol` | Composite | 60/40 momentum + low-vol (AMP 2013) |

Adding a new strategy is a ~10-line entry in `src/trading_bot/strategies/catalog.py`.

## Layout

```
src/trading_bot/
├── config.py
├── cli.py
├── registry.py             # Strategy/Run/TrialCounter persistence + DSR
├── data/
│   ├── storage.py          # ORM models (Ticker, Candle, IndexChange, …)
│   ├── universe.py         # Wikipedia S&P 500 scraper
│   ├── pit_universe.py     # Point-in-time membership reconstruction
│   ├── provider.py         # yfinance wrapper
│   └── ingest.py           # incremental download + panel loader
├── features/
│   ├── price.py            # 9 cross-sectional features (no look-ahead)
│   ├── cross_section.py    # rank / zscore / decile transforms
│   └── leakage.py          # leakage-check harness
├── strategies/
│   ├── base.py             # abstract BaseStrategy
│   ├── composer.py         # ComposedStrategy + Signal/Filter/Regime specs
│   ├── catalog.py          # pre-built strategies
│   └── momentum.py         # legacy hand-coded version (still used in tests)
├── backtest/
│   ├── engine.py           # cross-sectional, monthly rebalance, PIT-aware
│   └── metrics.py          # Sharpe, Sortino, MaxDD, IC
├── validation/
│   ├── walk_forward.py     # rolling/anchored fold analyser
│   └── deflated_sharpe.py  # PSR + DSR per Bailey-López de Prado
├── gui/
│   ├── app.py              # Streamlit Overview page
│   └── pages/
│       ├── 01_strategy_detail.py
│       ├── 02_correlation.py
│       ├── 03_hypothesis_registry.py
│       └── 04_statistical_health.py
└── utils/logging.py
```

## Statistical honesty (what's already enforced)

1. **PIT universe** — every backtest uses S&P 500 membership as it was at
   each rebalance date, not as it is today. Eliminates the worst
   survivorship bias. See `data/pit_universe.py`.
2. **No-look-ahead by construction** — features take `prices` and `asof`,
   compute on `prices.loc[:asof]`. The `features/leakage.py` harness
   verifies this empirically by recomputing with future masked.
3. **Honest trial counter** — every `persist_run` increments a global
   counter in SQLite, including failed runs. The Deflated Sharpe Ratio
   uses this real N, not the wishful "we only tried one config" number.
4. **Pre-registration** — `register_strategy` refuses any config without
   a substantive `rationale` string (≥ 20 chars). Anti-p-hacking.
5. **DSR gate** — a run only becomes `status='promoted'` if
   DSR ≥ 0.95 *and* Sharpe > 0.5. Otherwise it stays `tested`.
6. **Realistic costs** — backtester charges turnover-based bps at every
   rebalance. Defaults to 5 bps one-way.

## What's not yet enforced (next priorities)

- **CPCV** (Combinatorial Purged Cross-Validation) — more powerful than
  walk-forward, gives a distribution of OOS Sharpes
- **PBO** (Probability of Backtest Overfitting) via CSCV
- **Bonferroni / BH FDR** correction once we have many strategies
- **Hypothesis generator** for full auto-discovery (currently the
  catalog is hand-authored)
- **Optuna TPE** parameter search with DSR gate

## Testing

```bash
pytest -q
```

Covers: storage uniqueness, momentum ranking, regime filter,
no-look-ahead invariant, walk-forward consistency, PSR monotonicity,
DSR≤PSR, point-in-time membership reconstruction, leakage on every
feature, catalog loadability, composer top-N equal weight.

## Research basis

- `docs/research-strategies-2026-06.md` — deep-research on quant
  strategies for US equities (~30 academic citations).
- `docs/asset-allocation-decision-2026-06.md` — quantitative comparison
  of 5 strategic options and the resulting hierarchical roadmap.
