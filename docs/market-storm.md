# Market Storm / Meteo Mercato

Market Storm e un regime/risk overlay quantitativo. Non cerca alpha e non
piazza ordini: misura il contesto di rischio e suggerisce quanta esposizione
accettare in research/paper-review.

## Concetto

- Strategie = dove cercare edge.
- Market Storm = quanto rischio accettare oggi.

Il mercato viene trattato come sistema dinamico:

- volatilita SPY = pressione del sistema
- drawdown SPY = stress del mercato principale
- correlazione media = contagio tra asset
- breadth = salute interna del mercato
- dispersione cross-sectional = instabilita tra titoli
- settore/credit stress = warning secondari

## Output

Ogni giorno il modulo produce:

- `storm_score` 0-100
- `regime`: `calm`, `unstable`, `storm`, `panic`
- `recommended_action`: `normal`, `reduce_exposure`, `defensive_only`,
  `block_new_entries`
- `exposure_scale`: 1.00, 0.75, 0.50, 0.25
- component scores e rationale leggibile

## Comandi

```bash
tradebot market-storm --top 300
tradebot market-storm-history --start 2005-01-01 --top 300 \
  --output data/market_storm_reports/storm_history.csv
tradebot validate-storm-overlay momentum_12_1 --top 300
tradebot storm-report momentum_12_1 --top 300
```

I report JSON vengono salvati in:

```bash
data/market_storm_reports/*_storm_report.json
```

## Validazione overlay

L'overlay scala i rendimenti della strategia usando il regime noto il giorno
precedente: niente lookahead. Il gate MVP passa solo se:

- max drawdown non peggiora
- worst month migliora o non peggiora
- CAGR non viene distrutto
- volatilita non aumenta

Il confronto include i peggiori mesi SPY per vedere se l'overlay aiuta davvero
nei periodi brutti.

## GUI

La pagina `Meteo Mercato` legge i report gia salvati e mostra:

- stato corrente
- component scores
- storm score nel tempo
- confronto base vs overlay
- tabella peggiori mesi SPY
- comando CLI consigliato

La GUI non deve lanciare calcoli pesanti: il lavoro serio gira da CLI.

## Vincoli

- Nessuna API esterna.
- Nessuna secret/API key.
- Nessun capitale reale.
- Nessuna dipendenza deep learning.
- Nessuna visualizzazione scenografica scollegata dalle metriche.
