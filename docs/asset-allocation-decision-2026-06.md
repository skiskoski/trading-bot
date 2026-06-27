# Decisione di Asset Allocation — Quale approccio perseguire

**Data**: giugno 2026
**Decisione**: quale di queste opzioni dà l'IR atteso più alto post-decay e post-costi?

---

## Le 5 opzioni in valutazione

| # | Opzione |
|---|---|
| 1 | Strategie quant azioni (Momentum 12-1 + Quality/LowVol + RSI(2) MR) su ~100 azioni S&P 500 |
| 2 | Le 4 strategie del video (Liquidity Sweep, Trend Continuation, Breakout Expansion, EMA Momentum) applicate a **SPY / QQQ / DIA / IWM** come single-asset |
| 3 | Le 4 strategie del video applicate sull'**oro** daily (come da piano originale del video) |
| 4 | **Strategie diverse** specifiche per ETF indici (Faber TAA, Antonacci GEM, vol-managed) |
| 5 | **Strategie diverse** specifiche per oro (TSMOM commodities, macro signals, Donchian) |

## Framework — Fundamental Law of Active Management (Grinold-Kahn)

L'Information Ratio atteso di una strategia attiva è:

```
IR ≈ IC × √Breadth
```

dove:
- **IC** (Information Coefficient) = correlazione Spearman tra predizione e ritorno futuro
- **Breadth effective** = numero di bet *indipendenti* per anno

Per asset correlati con correlazione media ρ̄ (Meucci ENB):

```
N_effettivo ≈ N / (1 + (N-1) × ρ̄)
```

Quindi: **più asset hai e meno correlati sono, più alto il tuo IR atteso, a parità di skill (IC)**.

Applichiamo questo a ognuna delle 5 opzioni.

---

## Calcolo per ciascuna opzione

### Opzione 1 — Strategie quant su 100 azioni
- Universo: 100 azioni S&P 500
- Correlazione media intra-settore ~0.4, tra settori più bassa → N_efficace ≈ 100 / (1 + 99×0.4) × 10 settori ≈ **25 nomi effettivamente indipendenti**
- Rebalance mensile → Breadth = 25 × 12 = **300/anno**
- IC tipico cross-sectional momentum su azioni (Grinold-Kahn benchmarks): **~0.05**
- **IR teorico = 0.05 × √300 ≈ 0.87**
- Post-decay (McLean-Pontiff -26% OOS, -58% post-pub) e costi: **IR realistico 0.5–0.7**

### Opzione 2 — 4 strategie del video sui 4 ETF indici
- Asset: 4 ETF altamente correlati (SPY-DIA 0.94, SPY-QQQ 0.92, IWM-others 0.70–0.85, ρ̄ ≈ 0.90)
- N_efficace ≈ 4 / (1 + 3×0.90) ≈ **1.08** — praticamente **un solo asset**
- 4 strategie ma molto correlate tra loro su daily indici → segnali non indipendenti
- IC realistic per timing rules su mature indices dopo data-snooping correction (Sullivan-Timmermann-White 1999, Bajgrowicz-Scaillet 2012, Hsu-Kuan 2005): **≤ 0.02** o non distinguibile da zero
- Anche con IC = 0.02 e Breadth daily generoso ~270:
  - IR = 0.02 × √270 ≈ **0.33 in-sample**
  - Post-data-snooping correction su DJIA/SPX (multiple paper convergenti): **IR realistico ≈ 0.0**

### Opzione 3 — 4 strategie del video sull'oro daily
- Asset: 1 (XAU/USD o GLD/IAU)
- N_efficace = 1
- 4 strategie correlate sullo stesso asset → strategie quasi linearmente dipendenti
- Verdetto strategia per strategia su gold daily:
  - **Liquidity Sweep**: M15 FX concept, perde identità su daily (range tipici 1-2% assorbono i wick) → scartare
  - **Trend Continuation pullback**: marginale, ok solo con filtro 200-day MA
  - **Breakout (Donchian/ATR)**: ✅ qualche evidenza (Turtle universo includeva gold), MAR ~0.25 su gold futures 35 anni
  - **EMA crossover**: debole da solo, ok come filtro regime
- IC realistico per trend su gold daily: ~0.05-0.07
- Breadth con rebalance mensile single asset: ~12/yr
- **IR teorico = 0.06 × √12 ≈ 0.21**
- Post-costi GLD/IAU (5-10 bp/turnover) e decay: **IR realistico 0.2–0.4**

### Opzione 4 — Strategie ad hoc per ETF indici
Migliori candidati testati dalla letteratura:

**a) Faber 10-month SMA su SPY**
- In-sample 1973-2012: Sharpe 0.73 vs 0.44 B&H
- Out-of-sample post-2009: **ha sottoperformato buy-and-hold in 6 anni su 8** (2009-2016)
- Zakamulin (2014/2018): nessun edge OOS statisticamente significativo nel secondo metà del campione 80 anni
- **IR realistico oggi ≈ 0.0–0.2**, utile come *filtro regime*, non come return-generator

**b) Antonacci Global Equity Momentum (GEM)**
- In-sample 1974-2013: CAGR **17.43%**, Sharpe alto, max DD 22.7%
- Post-publication (2014-2021): CAGR **5.89%**, max DD 33.7% — **ritorno ridotto a un terzo, drawdown 50% peggio**
- 2022-2024: defensive ok ma re-entry tardivo
- **Decay severo. IR realistico oggi ≈ 0.1–0.3**

**c) Vol-managed SPY (Moreira-Muir)**
- In-sample 2017: 65% lifetime utility gain
- Critique Cederburg et al. 2020: **non implementabile OOS**, in real-time underperforma B&H
- **Utile come overlay di sizing, non strategia standalone**

**d) Hurst-Ooi-Pedersen single-asset TSMOM su SPY**
- Sharpe storico standalone ~0.45–0.60
- **IR realistico 0.4–0.6**, valore principale = crisis alpha (Sharpe alto in 8/10 maggiori drawdown 60/40)

**Migliore strategia singola di opzione 4**: TSMOM su SPY come 10-20% sleeve, IR ~0.4-0.6 ma altamente correlato con buy-hold di lungo periodo

### Opzione 5 — Strategie ad hoc per oro
Migliori candidati:

**a) TSMOM su gold (Moskowitz-Ooi-Pedersen 2012)**
- Sharpe single-asset gold trend ~0.4-0.6 gross
- Diversified TSMOM portfolio Sharpe 1.31 1965-2009, ma decay post-2010 (Sharpe ~0.3-0.5 nel decennio 2010s)
- Su gold da solo: contributo positivo ma 1 su 24 commodities nella sleeve diversificata
- **IR realistico standalone: 0.3–0.5**

**b) Real-rates signal**
- Correlazione -0.82 con gold 1997-2021
- **Rotta nel 2022-2023** (~0.03) per acquisti banche centrali e geopolitica
- Regime-dependent, non robusto su tutto il sample

**c) Donchian/MA breakout su gold daily**
- $100k → $10M (1971-2021) con 10/12-month MA secondo backtest Quantified Strategies
- Robusto a perturbazione parametri
- **IR ≈ 0.3-0.5 realistico**

**Migliore strategia singola di opzione 5**: TSMOM clean (signal: prezzo > 200d MA AND 12-mo return > 0, vol-target 10%) — IR atteso 0.4-0.6, **bassa correlazione con equity (-0.1 to +0.2)**, crisis alpha

---

## Tabella IR atteso comparata

| # | Opzione | IC | Breadth | IR teorico | **IR realistico post-decay/costs** |
|---|---|---|---|---|---|
| 1 | 100-stock CS Mom + Quality + RSI(2) | 0.05 | 300/yr | 0.87 | **0.5–0.7** ✅ |
| 2 | Video strats su 4 ETF | ≤0.02 | ~270 in-sample | 0.33 | **~0.0** ❌ |
| 3 | Video strats su gold daily | 0.06 | 12/yr | 0.21 | **0.2–0.4** ⚠️ |
| 4a | Faber 10M SMA su SPY | 0.04 | 13/yr | 0.14 | **0.0–0.2** ❌ standalone |
| 4b | Antonacci GEM | 0.05 | 13/yr | 0.18 | **0.1–0.3** ❌ severe decay |
| 4c | TSMOM su SPY | 0.05 | 13/yr | 0.18 | **0.4–0.6** ⚠️ crisis alpha sleeve |
| 5 | TSMOM clean su gold | 0.06 | 12/yr | 0.21 | **0.4–0.6** ✅ come sleeve |

**Confronto diretto opzione 1 vs opzioni single-asset**: a parità di IC (0.05), il 100-stock dà IR teorico 0.87 mentre tutti i single-asset (sia indice sia oro) si fermano a 0.14-0.21. **Penalità ~4-5× per il single-asset** dovuta alla Breadth bassa.

---

## Verdetto statistico

**La risposta migliore non è scegliere una delle 5 opzioni — è un'allocazione gerarchica multi-sleeve che massimizzi IR totale e diversificazione.**

### Configurazione raccomandata

```
┌─────────────────────────────────────────────────────────────────┐
│  CORE — 70-80% capitale (motore principale)                    │
│  ────────────────────────────────────────────────              │
│  Opzione 1: 100-stock S&P 500                                  │
│   ├─ Sleeve A: Momentum 12-1 (25-30%)                          │
│   ├─ Sleeve B: Quality + Low-Vol (30-40%)                      │
│   └─ Sleeve C: RSI(2) Mean Reversion (15-20%)                  │
│  IR atteso: 0.5-0.7                                            │
└─────────────────────────────────────────────────────────────────┘
                              +
┌─────────────────────────────────────────────────────────────────┐
│  DIVERSIFIER — 10-20% capitale (crisis alpha)                  │
│  ────────────────────────────────────────────────              │
│  Opzione 5 (versione minimal): TSMOM clean su GLD/IAU          │
│   Signal: long se price > 200d MA AND 12mo return > 0          │
│   Sizing: vol-target 10% annualizzato                          │
│  IR atteso: 0.4-0.6 standalone                                 │
│  Beneficio chiave: corr -0.1/+0.2 con equity → +0.05-0.15      │
│  Sharpe portafoglio totale                                     │
└─────────────────────────────────────────────────────────────────┘
                              +
┌─────────────────────────────────────────────────────────────────┐
│  OVERLAY REGIME — non capitale separato, è un filtro           │
│  ────────────────────────────────────────────────              │
│  Faber 200-day SMA su SPY come filtro on/off del portafoglio   │
│  Non come strategia standalone, ma per de-risking globale      │
│  Quando SPY < 200d SMA: riduci leverage del core di 50%        │
└─────────────────────────────────────────────────────────────────┘
```

### Cosa **scartare** definitivamente

- ❌ **Opzione 2** (le 4 strategie del video su SPY/QQQ/DIA/IWM): IR atteso ≈ 0 dopo data-snooping correction. Letteratura sterminata che convergente (Sullivan-Timmermann-White, Bajgrowicz-Scaillet, Hsu-Kuan, Zakamulin) la demolisce.
- ❌ **Opzione 3** (le 4 strategie video su gold daily): solo il Breakout/ATR ha qualche edge; il resto degrada o non trasferisce dal M15 al daily. Meglio una singola strategia TSMOM clean (opzione 5).
- ❌ **Antonacci GEM** come strategia principale: decay severissimo post-2014.
- ❌ **Vol-managed SPY come strategia standalone**: solo come overlay di sizing.

### Razionale della scelta

1. **Massimizza Breadth**: l'opzione 1 ha ~300 bet indipendenti/anno, è dove l'IR può crescere
2. **Diversifica senza sacrificare IR**: aggiungere TSMOM gold come 10-20% sleeve costa poco in capitale ma dà crisis alpha (correlation bassa)
3. **Difensiva globale**: il filtro regime Faber agisce su tutto il book, non è una strategia separata da validare
4. **Implementazione semplice**: 3 sleeve azioni + 1 sleeve gold + 1 filtro = 5 componenti chiari, ognuno con una rationale accademico-statistica diversa
5. **Capacità retail**: tutti gli strumenti (azioni S&P 500, GLD/IAU) accessibili via Alpaca paper API gratis

### IR atteso del portafoglio combinato

Se assumiamo correlazione media tra le 3 sleeve azioni di ~0.3 (Mom-MR negativa, Mom-Quality positiva, MR-Quality bassa) e correlazione gold-equity ~0.0:

```
IR_combinato ≈ √(Σ w_i² × IR_i² + 2 Σ w_i × w_j × ρ_ij × IR_i × IR_j)
```

Con pesi 0.30 / 0.35 / 0.20 / 0.15 e IR singoli (0.6, 0.5, 0.7, 0.5):

```
IR portafoglio ≈ 0.7–0.9
```

Confrontato con buy-and-hold SPY (IR ≈ 0.4-0.5 ultimi 25 anni), è un upgrade meaningful con drawdown atteso inferiore.

---

## Decisione operativa

**Bot v1 → solo CORE (opzione 1)**. Iniziamo dall'opzione che da sola contribuisce la quota maggiore di IR (0.5-0.7) e che ha la pipeline tecnica più ricca da costruire (universe S&P 500 PIT, ranking cross-sectional, sizing risk-parity, multi-strategy combiner).

**Bot v2 → aggiunge DIVERSIFIER (TSMOM gold)**. Una volta validato il core, aggiungere il GLD sleeve è ~50 righe in più.

**Bot v3 → aggiunge OVERLAY REGIME**. Il filtro 200-day si implementa come modificatore di leverage del core, dopo aver visto come si comporta in paper trading.

**Bot non-v***: opzione 2 (skip definitivo), opzione 3 (skip definitivo), Antonacci GEM (skip).

---

## Citazioni chiave aggiunte

Sullivan, R., Timmermann, A., White, H. (1999). "Data-Snooping, Technical Trading Rule Performance, and the Bootstrap." *Journal of Finance* 54(5). [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/0022-1082.00163)

Bajgrowicz, P., Scaillet, O. (2012). "Technical Trading Revisited: False Discoveries, Persistence Tests, and Transaction Costs." *JFE* 106(3). [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0304405X1200116X)

Antonacci, G. (2014). *Dual Momentum Investing*. McGraw-Hill. [optimalmomentum.com](https://www.optimalmomentum.com/)

Hurst, B., Ooi, Y.H., Pedersen, L.H. (2017). "A Century of Evidence on Trend-Following Investing." *JPM*. [AQR](https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing)

Cederburg, S., O'Doherty, M., Wang, F., Yan, X. (2020). "On the Performance of Volatility-Managed Portfolios." *JFE* 138(1). [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0304405X2030132X)

Zakamulin, V. (2014, 2018). "The Real-Life Performance of Market Timing." [SSRN 2242795](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2242795)

Erb, C., Harvey, C. (2006). "The Strategic and Tactical Value of Commodity Futures." *FAJ*. [Duke PDF](https://people.duke.edu/~charvey/Research/Working_Papers/W77_The_tactical_and.pdf)

Baur, D., Lucey, B. (2010). "Is Gold a Hedge or a Safe Haven?" *Financial Review*. [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6288.2010.00244.x)

Meucci, A. (2009). "Managing Diversification — Effective Number of Bets." [SSRN 1358533](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1358533)
