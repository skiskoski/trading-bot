# Robustness validation layer

Decisione: costruire come gate pre-paper incrementale. Non e un modulo di live
trading e non introduce secret/API key.

## Perche esiste

Trading Bot usa gia IC, CPCV/PBO, Deflated Sharpe, PIT universe e costi
realistici. Il nuovo layer copre un rischio diverso: la strategia non deve
funzionare solo nel singolo punto di parametro che e stato scelto dal backtest.

## Pipeline

1. Ipotesi economica preregistrata nel catalogo o nel research log.
2. IC prescan.
3. Backtest netto costi e PIT universe.
4. CPCV/PBO.
5. Parameter stability su griglia locale.
6. Monte Carlo a blocchi su ogni parameter set della griglia.
7. Cluster/heatmap per cercare plateau, non spike.
8. Monte Carlo finale a blocchi sui rendimenti della configurazione scelta.
9. Walk-forward rolling e anchored.
10. Report JSON salvato in `data/validation_reports`.
11. Solo paper-review se tutti i gate passano.

## Comandi

```bash
tradebot validate-strategy momentum_12_1
tradebot robustness-report momentum_12_1 --mc-trials 1000
tradebot parameter-stability momentum_12_1 --grid momentum_core --mc-trials-per-set 100
tradebot monte-carlo momentum_12_1 --trials 2000
tradebot walk-forward momentum_12_1 --folds 8
```

## Hard gates MVP

- CPCV: `PBO < 0.4`, `mean_oos_sharpe >= 0.5`, `fraction_positive >= 0.65`.
- Parameter stability: plateau fraction >= 50% e best-neighbor gap <= 0.35.
- Monte Carlo: Sharpe p05 >= 0, loss probability <= 35%, drawdown p95 <= 35%.
- Walk-forward: fold positivi >= 65%, worst fold Sharpe >= -0.25.

## Rischi espliciti

- Overfitting e data snooping: mitigati da trial counter, DSR, CPCV e griglie
  preregistrate.
- Survivorship/lookahead bias: PIT universe e backtester con pesi shiftati.
- Costi, slippage e capacity: validare sempre netto costi; il gate non deve
  passare una strategia che funziona solo gross.
- Falso comfort Monte Carlo: usare block/moving bootstrap, non shuffle ingenuo.
- Falso comfort Sharpe: il report mostra IC, worst fold, drawdown e dispersione.

## Prossimi step

- Persistenza opzionale in tabella `validation_reports`.
- Stationary bootstrap Politis-Romano oltre al moving-block bootstrap.
- Reality Check / SPA su famiglie di parameter set.
- Gate del daemon: usare robustness solo sui candidati gia sopravvissuti a CPCV,
  non su ogni trial economico.
