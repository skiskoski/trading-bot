# trading-bot

Quantitative trading bot for US equities (paper trading). The strategy mix is
driven by the research in `docs/`:

- **CORE** — cross-sectional 12-1 momentum on ~100 S&P 500 names, with stock
  and SPY-regime filters (Jegadeesh-Titman / Clenow).
- v2 will add a Quality + Low-Vol sleeve and an RSI(2) mean reversion sleeve.
- v3 will add a TSMOM sleeve on gold (GLD/IAU) for crisis-alpha diversification.

Bias toward statistical honesty: Information Coefficient is the primary
metric for parameter tuning, not Sharpe.

## Quickstart

```bash
# 1. install (Python 3.11+)
pip install -e .[dev]

# 2. initialize SQLite + load S&P 500 constituents from Wikipedia
tradebot init-db
tradebot fetch-universe

# 3. download adjusted OHLCV for top-N tickers (incremental)
tradebot ingest --top 100 --start 2015-01-01

# 4. run the momentum backtest
tradebot backtest momentum --start 2018-01-01 --top 100 --output data/equity.csv
```

## Layout

```
src/trading_bot/
├── config.py           # pydantic settings (env vars: TRADEBOT_*)
├── cli.py              # Typer CLI entrypoint
├── data/
│   ├── storage.py      # SQLAlchemy models + SQLite engine
│   ├── universe.py     # Wikipedia S&P 500 scraper
│   ├── provider.py     # yfinance wrapper (adjusted OHLCV)
│   └── ingest.py       # incremental download + persist + panel loader
├── strategies/
│   ├── base.py         # abstract BaseStrategy (rank, weights)
│   └── momentum.py     # cross-sectional 12-1 momentum + filters
├── backtest/
│   ├── engine.py       # event-driven monthly-rebalance backtester
│   └── metrics.py      # Sharpe, Sortino, MaxDD, Information Coefficient
└── utils/
    └── logging.py      # Rich-based logging
```

## Testing

```bash
pytest -q
```

The current suite covers storage uniqueness, momentum ranking math, regime
filter, no-look-ahead invariant, and backtest equity/metrics consistency.

## Research basis

- `docs/research-strategies-2026-06.md` — deep-research report on quant
  strategies for US equities (~30 academic citations).
- `docs/asset-allocation-decision-2026-06.md` — quantitative comparison of 5
  strategic options via the Fundamental Law of Active Management (IR = IC ·
  √Breadth) and the hierarchical multi-sleeve decision.

## Roadmap

- **v1 (current)**: data layer + momentum sleeve + cross-sectional backtester
  with IC computation
- **v2**: Quality + Low-Vol sleeve, RSI(2) mean-reversion sleeve, portfolio
  combiner
- **v3**: TSMOM gold sleeve, Faber 200d SMA regime overlay
- **v4**: CPCV + Deflated Sharpe + PBO validation gate
- **v5**: paper trading via Alpaca, Telegram notifications, Streamlit dashboard
