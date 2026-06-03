# Research Report — Strategie quantitative per azioni US

**Data**: giugno 2026
**Scope**: bot retail che scansiona ~100 azioni US liquide, timeframe daily, paper trading
**Metodo**: deep-research fan-out (5 angoli) + verifica adversarial + sintesi citata

---

## TL;DR — Cosa fare nel bot

1. **Scarta "Liquidity Sweep"** così com'è descritto nel video — non ha letteratura accademica su azioni daily, è un concetto FX/intraday.
2. **Trend Continuation, Breakout, EMA Momentum** funzionano *solo* come componenti di sistemi più larghi (cross-sectional momentum, regime filter), non come setup single-symbol.
3. **Costruisci il bot attorno a 3 sleeve a bassa correlazione**:
   - **Momentum 12-1 cross-sectional** (long-only, regime filter Faber)
   - **Quality + Low-Vol** (ballast difensivo)
   - **Short-term mean reversion RSI(2)** stile Connors (long-only, su nomi liquidi)
4. **Skippare in v1**: PEAD, Pairs Trading (decadute, dati costosi, alpha trascurabile per retail).
5. **Validazione obbligatoria**: CPCV (purged + embargo), Deflated Sharpe, dati point-in-time. Senza questi tre, qualsiasi backtest mente.
6. **Aspettati alpha decadenti**: McLean-Pontiff documenta -26% OOS e -58% post-publication su 97 anomalie. Targetti realistici: IC 0.03-0.06, Sharpe 0.4-0.8 live (non 1.5+ del backtest).
7. **TradingView non serve**. yfinance/Polygon/Alpaca + pandas-ta replicano tutto.

---

## 1. Verdetto sulle 4 strategie del video

Sono nate per XAU/USD M15 (intraday forex). Su azioni daily il quadro è molto diverso.

### 1.1 Liquidity Sweep ❌
**Verdetto**: scartare. È un concetto di microstruttura/order-flow FX intraday (Osler 2003 sul clustering di stop attorno a round number nel forex). **Zero paper peer-reviewed** la testano su azioni daily. Sul daily diventa un "false breakout reversal" — pattern molto noto ma che non sopravvive ai test anti-data-snooping di Sullivan-Timmermann-White (1999) e Bajgrowicz-Scaillet (2012).

### 1.2 Trend Continuation (pullback su EMA) ⚠️ → ✅ se riformulato
**Verdetto**: funziona *come idea*, ma solo se applicata cross-sectional. La versione single-symbol "pullback all'EMA in trend" non ha edge dimostrato dopo costi. Diventa robusta quando si trasforma in **cross-sectional momentum** (Jegadeesh-Titman 1993): ranking dei nomi per ritorno passato, long il top decile.

Riferimento pratico: Han, Yang, Zhou (2013, *JFQA*) — MA timing applicato a portafogli ordinati per volatilità batte la buy&hold con alpha FF-3 robusti, soprattutto sulle decili high-vol.

### 1.3 Breakout Expansion (ATR/Donchian/Turtle) ⚠️
**Verdetto**: edge debole post-1990 su azioni mature. Il paper seminal di Brock-Lakonishok-LeBaron (1992) trovava edge in-sample sul DJIA 1897-1986; Sullivan-Timmermann-White (1999) ha bootstrappato la stessa famiglia e l'edge **collassa nel campione out-of-sample 1987-1996**. Bajgrowicz-Scaillet (2012) ha testato 7.846 regole con FDR: dopo costi, niente di sistematicamente significativo. Edge residuo solo su Russell 2000/NASDAQ small cap (Hsu-Kuan 2005).

### 1.4 EMA Momentum (crossover) ⚠️
**Verdetto**: marginale come segnale di entry, utile come **regime filter**. Faber (2007) ha mostrato che SMA 10-mesi su S&P 500 riduce drawdown drasticamente in-sample, ma 6 dei primi 8 anni post-pubblicazione (2005-2013) ha sottoperformato buy-and-hold. Single-symbol EMA crossover su daily produce molti whipsaw e non sopravvive ai test di data-snooping.

**Sintesi del video**: il pitch del video confonde tecniche FX/intraday con quant equity. Le metriche mostrate (WR 64%, PF 1.92, Sharpe 1.41) sono **plausibili in backtest** ma probabilmente sopravvalutate per overfitting (LHS su 80 combinazioni × 4 strategie = N effettivo di trial alto), bias di sopravvivenza nel campione, e ottimizzazione del campione totale senza un OOS finale veramente intoccato.

---

## 2. Strategie raccomandate per il bot (long-only, ~100 azioni)

### 2.1 Sleeve A — Cross-Sectional Momentum 12-1 (25-30% del capitale)

**Letteratura**:
- Jegadeesh & Titman (1993), *J. of Finance* 48(1): ~1.31% mensile per portafoglio long-short 12-1, formazione 12-mesi / holding 3-mesi, t-stat > 3 [[paper](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1993.tb04702.x)]
- Jegadeesh & Titman (2001): replica out-of-sample 1990-1998, magnitudini simili
- Asness, Moskowitz, Pedersen (2013) "Value and Momentum Everywhere": value/momentum correlazione **-0.5 / -0.6**, ottime in combo
- Daniel & Moskowitz (2016) "Momentum Crashes": -45.6% mar-apr 2009, -88.5% lug-ago 1932
- Barroso & Santa-Clara (2015): vol-managed momentum porta Sharpe da 0.53 → 0.97

**Implementazione retail (Clenow, *Stocks on the Move*, 2015)**:
1. Ranking signal: ritorno annualizzato exp-regression × R² sui 90 giorni, oppure semplice **ritorno passato 12 mesi escluso ultimo mese (12-1)**
2. Regime filter: aprire long **solo se SPY > SMA 200**
3. Filtro per nome: escludere se il nome è sotto la sua SMA 100 o ha avuto gap > 15% negli ultimi 90 giorni
4. Sizing risk-parity: `shares = AccountValue * 0.001 / ATR_20`
5. Top 10-20 ranked (decile equivalente su 100 nomi)
6. Rebal settimanale

**Aspettative realistiche**: CAGR S&P 500 + 1-5% (non l'1%/mese del paper originale, decaduto), beta ~0.5-0.7, max DD 20-30%, IC ranking ~0.03-0.06.

### 2.2 Sleeve B — Quality + Low-Volatility (30-40% del capitale)

**Letteratura**:
- Frazzini & Pedersen (2014) "Betting Against Beta", *JFE*: Sharpe US 0.78 (1926-2012), 2× value, 1.4× momentum [[paper](https://pages.stern.nyu.edu/~lpederse/papers/BettingAgainstBeta.pdf)]
- Asness, Frazzini, Pedersen (2019) "Quality Minus Junk": rendimenti positivi in 23/24 paesi
- Baker, Bradley, Wurgler (2011) "Benchmarks as Limits to Arbitrage": persiste perché istituzionali con mandate vincolati non possono levereggiare low-beta

**Implementazione**:
- Quality screen: ranking per ROE, gross profitability (Novy-Marx 2013), low accruals, low leverage, low earnings volatility
- Low-vol screen: ranking per volatilità 252-day; long bottom quintile
- Holding: mesi (non giorni), basso turnover, costi bassi
- Serve dati fondamentali (Sharadar SF1 a pagamento, o yfinance per prototipo)

**Aspettative**: QMJ ~4-6% annuo con basso residual risk, decadenza post-pubblicazione ma rimane robusto internazionalmente. Questo sleeve è il "zavorra" (ballast) del portafoglio.

### 2.3 Sleeve C — Short-Term Mean Reversion (20-25% del capitale)

**Letteratura**:
- De Bondt & Thaler (1985) sulla overreaction long-term
- Lehmann (1990), Jegadeesh (1990): reversal 1-settimana/1-mese, ~2%/mese gross
- Lo & MacKinlay (1990): molto del profitto è cross-autocovariance (large lead small)
- Khandani & Lo (2007/2009): quant-quake agosto 2007, libri di mean-reversion squartati (-8 / -12 σ)
- Connors & Alvarez (2009) *Short Term Trading Strategies That Work*: RSI(2)

**Implementazione**:
- Long-only: se prezzo > SMA 200 (uptrend) AND RSI(2) < 10 → buy
- Exit: close > SMA 5 o RSI > 70
- Universo: nomi liquidi (ADV > $10M), top S&P 500
- Limit orders dentro lo spread per ridurre slippage

**Aspettative**: Sharpe gross 0.4-0.7 oggi (era 1.5+ negli anni '90, decadenza forte). Win rate 70-78% in backtest tipici, ma profit factor scende a 1.2-1.5 dopo costi realistici.

### 2.4 Overlay — Volatility-Managed Sizing (Moreira-Muir 2017)

Scalare l'esposizione di Momentum e Quality (ma **NON** Mean Reversion) inversamente alla volatilità realizzata 21-giorni:

```
position_size = base_weight * target_vol / realized_vol_21d
```

Cap a 1.5×. Cederburg et al. (2020) ha criticato la versione cross-anomaly, ma l'effetto sulla parte momentum/quality regge ed è il modo principale di sopprimere i crash di momentum.

### 2.5 Cosa scartare in v1

- **PEAD** (Bernard-Thomas 1989): spread top-bottom decile sceso da ~18%/yr (anni '80) a ~3% o meno oggi su large cap. Decimalization 2001 + Reg NMS 2005 + SOX hanno chiuso la finestra. Serve dati earnings + consensus (costosi).
- **Pairs trading** (Gatev-Goetzmann-Rouwenhorst 2006): Sharpe ~0.6-1.0 pre-2002, dimezzato o peggio post-2002 — HFT arbitraggia in millisecondi.
- **Liquidity Sweep**: vedi §1.1.

### 2.6 Matrice di correlazione attesa tra sleeve

| | Momentum | Mean Rev | Quality/Low-Vol |
|---|---|---|---|
| Momentum | 1.00 | −0.35 | +0.29 |
| Mean Rev | −0.35 | 1.00 | −0.10 |
| Quality | +0.29 | −0.10 | 1.00 |

La combo Momentum + Mean Reversion + Quality offre la migliore diversificazione tra sleeve liquidi implementabili da retail.

---

## 3. Validazione statistica onesta — la parte cruciale

L'utente ha esplicitamente chiesto strategie "con correlazione ≠ 0 anche se non subito profittevoli". La metrica giusta è l'**Information Coefficient**.

### 3.1 Information Coefficient (IC)

**Definizione**: Spearman (rank) correlation tra ranking predetto al tempo *t* e ritorno futuro al tempo *t+h*. Calcolato per ogni periodo, poi mediato.

**Soglie pratiche** ([FE Training](https://www.fe.training/free-resources/portfolio-management/information-coefficient-ic/), [Grinold & Kahn 1999]):
- IC ≈ 0 → nessuna skill
- **IC 0.02-0.05 → tipico "buon" segnale daily/weekly su equity**
- IC 0.05-0.08 → forte; monthly IC 0.05-0.06 è molto forte su US equity
- IC > 0.10 → sospetto, verificare leakage

**Fundamental Law of Active Management** (Grinold): `IR ≈ IC × √Breadth`. Con 100 nomi e ranking mensile, breadth ~12×100=1200 nell'anno. IC 0.04 → IR atteso ~0.04 × √1200 ≈ 1.4. Realistico.

**Per il tuning iterativo**: l'IC è il segnale giusto per dire "questa strategia ha edge anche se non è ancora profittevole dopo costi". Si calcola IC su walk-forward, poi si raffinano parametri ottimizzando IC (non Sharpe), e solo alla fine si aggiungono costi e si verifica Sharpe.

### 3.2 Deflated Sharpe Ratio (DSR)

Bailey & López de Prado (2014) [[paper](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)]:

```
SR_0 = √V[SR] · ( (1-γ) Φ⁻¹(1-1/N) + γ Φ⁻¹(1-1/(N·e)) )
DSR = PSR(SR_0)
```

- N = numero **vero** di trial (parametri × feature sets × seed)
- DSR > 0.95 → lo Sharpe osservato sopravvive al multiple testing

**Critico**: l'errore tipico è dichiarare N=1 quando in realtà sono state provate 1000 combinazioni — solo le 10 che hanno funzionato sono state "ricordate". Sii onesto sul conteggio.

### 3.3 CPCV — Combinatorial Purged Cross-Validation

López de Prado (2018) [[book](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)]:

Il k-fold standard rompe su time series perché le label finanziarie spesso usano forward window sovrapposti. Soluzione:
1. Dividi i dati in N gruppi ordinati
2. Scegli k gruppi come test (totale folds = C(N,k), molti più path che k-fold)
3. **Purge**: scarta dal training le righe la cui label horizon si sovrappone con la test window
4. **Embargo**: scarta righe nelle h periodi dopo ogni test block (h = 1-5% di T)

L'output è una distribuzione di Sharpe OOS — input perfetto per il DSR.

### 3.4 Altri test rilevanti

- **PBO (Probability of Backtest Overfitting)** via CSCV — Bailey, Borwein, López de Prado, Zhu (2014). Soglia: < 0.30. [[paper](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)]
- **White's Reality Check** (2000), Hansen SPA (2005), Romano-Wolf StepM (2005) — test joint per famiglie di strategie
- **Harvey, Liu, Zhu (2016)** "...and the Cross-Section of Expected Returns": dopo correzione multiple testing, una nuova anomalia richiede **|t| > 3.0** (non 2.0). Solo 9 su 313 predictor pubblicati sopravvivono.
- **Stationary block bootstrap** (Politis-Romano 1994), block length ~T^(1/3)
- **Probabilistic Sharpe Ratio** ≥ 0.95 per la singola configurazione scelta

### 3.5 Checklist obbligatoria pre-promozione a paper trading

Da applicare in ordine; fallire uno → non promuovere:

1. ☐ Dati point-in-time, survivorship-bias free, con delisting returns
2. ☐ Ipotesi economica pre-registrata (scritta prima del backtest)
3. ☐ CPCV con ≥ 6 path; embargo ≥ 1% di T
4. ☐ Walk-forward 5-10 finestre, sia anchored sia rolling; reportare Sharpe del *worst fold*
5. ☐ IC ≥ 0.02 (daily/weekly) o ≥ 0.04 (monthly), con IC IR ≥ 0.5
6. ☐ PSR ≥ 0.95
7. ☐ DSR ≥ 0.95 (N onesto)
8. ☐ PBO < 0.30 via CSCV
9. ☐ t-stat ≥ 3.0 post correzione multiple testing
10. ☐ Block-bootstrap p-value < 0.05 su Sharpe vs randomized-return null
11. ☐ Min track length sufficiente (Bailey-López de Prado MinTRL)
12. ☐ Costi realistici applicati; rerun di tutto
13. ☐ Harvey-Liu haircut (non flat 50%) per sizing live

### 3.6 Se implementi solo 3 cose, fai queste

1. **CPCV con purging + embargo** — risolve il singolo errore più devastante (label leakage)
2. **Deflated Sharpe Ratio** con N onesto — il test più economico che punisce "ho provato 1000 grid, riporto il migliore"
3. **Dati point-in-time + Harvey-Liu Sharpe haircut per sizing live** — garantisce realismo

Tutto il resto è additive insurance.

---

## 4. Decadimento atteso live vs backtest

- **McLean & Pontiff (2016)** *J. of Finance* 71(1): media -26% OOS, **-58% post-publication** su 97 anomalie [[paper](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12365)]
- **Harvey & Liu** "Backtesting": haircut non lineare, dipende da numero e correlazione dei trial. Per N=50 trial, haircut ~50% sul marginale, ~8% sul migliore
- **Quantpedia** su 355 strategie: median -44% Sharpe degradation IS→OOS
- **Folklore "halve the Sharpe"**: troppo lenient per Sharpe < 0.4, troppo harsh per Sharpe > 1.0

**Implicazione**: se il backtest mostra Sharpe 1.5, aspettati ~0.7-0.9 live. Se mostra Sharpe 0.6, aspettati ~0.3-0.4 (forse pareggio dopo costi). Non sizing aggressivo su backtest.

---

## 5. Universe selection (~100 azioni)

**Raccomandazione**: top 100 per ADV dello S&P 500 attuale, **ma con storico point-in-time** per il backtest.

- **S&P 500**: market cap minimo ~$14.6B, requisito di profittevolezza
- **Russell 1000**: più inclusivo, ricostituzione annuale a giugno
- **Survivorship bias**: ~75% dei nomi che tradavano 10 anni fa non sono nei dataset current-only

**Dati PIT gratuiti**: scrape revisioni Wikipedia / "Selected Changes" table — accurato a ~2000. Vedi [Robot Wealth guide](https://robotwealth.com/how-to-get-historical-spx-constituents-data-for-free/) e [Teddy Koker tutorial](https://teddykoker.com/2019/05/creating-a-survivorship-bias-free-sp-500-dataset-with-python/).

**A pagamento** se serio: Sharadar SF1/SEP (~$50-150/mo via Nasdaq Data Link), Norgate, CRSP (accademico).

---

## 6. Stack tecnologico raccomandato

### Dati
1. **yfinance** — prototipo veloce, ~25 anni daily, **gratis**. Attenzione: ha survivorship bias se interroghi ticker correnti
2. **Polygon free** — 5 req/min, daily unlimited
3. **Alpaca** — bundled col broker se vai in paper poi live
4. **Tiingo** (~$10/mo) — best price/value EOD + fundamentals quando serio
5. **Sharadar SF1** (via Nasdaq Data Link) — fundamentals PIT, ~$50-150/mo

### Backtest engine
- **vectorbt** — fastest (NumPy/Numba), perfetto per parameter sweep e cross-sectional su 100 nomi → **scelto per la fase di ricerca**
- **QuantConnect Lean** — event-driven, multi-asset, realistic fills → **scelto per validazione finale prima del paper live**
- **backtesting.py**: single asset only, no
- **zipline-reloaded**: lento, equity-only, complicato

### Indicatori
- **pandas-ta** replica praticamente tutti gli indicatori di TradingView

### Ottimizzazione parametri
- **Optuna** (TPE) — meglio di LHS / random / grid in 9/12 study comparativi
- LHS (come nel video) è valido come fallback, ma TPE batte LHS quando N > 50 trial
- Sempre paired con PBO/DSR come gate

### Notifiche e dashboard
- Console + Telegram (python-telegram-bot)
- Streamlit per dashboard interattiva
- **Non serve TradingView** (vedi §7)

---

## 7. TradingView — verdetto

**Non serve.** Ecco perché:

| Cosa offre TV | Possiamo replicarlo? |
|---|---|
| Dati OHLCV | Sì, yfinance/Polygon/Alpaca (stesso dato che TV legge) |
| Indicatori | Sì, pandas-ta — sono formule deterministiche su OHLCV |
| Grafici | Sì, mplfinance/plotly → PNG analizzabili anche da AI vision |
| Alert | Sì, scheduler + notifiche custom |
| Pine Script | No diretto, ma raramente serve veramente |

**Cosa esiste come integrazione**:
- **Webhook TradingView → bot**: Pine `alert()` → POST JSON al tuo server. Funziona (solo porta 80/443, timeout 3s, 2FA richiesta). Utile se vuoi disegnare segnali in Pine e farli triggerare il bot, ma sposta logica fuori dal Python.
- **`tvDatafeed` (non ufficiale)**: scraping della private API. **Contro TOS** di TradingView, account ban risk, breaks su UI changes. **Non usare in produzione**.
- **No chart-image API ufficiale**.

**Per "vedere i grafici" da AI**: il bot genera PNG (mplfinance) per ogni segnale → te li manda su Telegram → tu li carichi qui → io li analizzo come immagini. Questo è il flusso pulito.

Per la dashboard utente: se vuoi il look-and-feel di TradingView, embed gratuito del **TradingView Widget** (free, no API key) dentro Streamlit. Solo display, non automation.

---

## 8. Costi realistici per backtest

Per essere onesti, applica nel simulatore:

| Voce | Valore tipico |
|---|---|
| Commissioni broker (Alpaca/IBKR/Robinhood) | $0 |
| SEC Section 31 fee | $0.00/$M dal 14/05/2025 |
| FINRA TAF | $0.000195/share sui sell, cap $9.79/trade |
| **Half-spread SPY** | ~0.16 bp |
| **Half-spread large cap (>$50B)** | ~5-10 bp |
| **Half-spread mid cap ($1-10B)** | ~20-50 bp |
| **Slippage / market impact** | Square-root law: ~ σ × √(Q/ADV); retail < 0.1% ADV → < 5 bp |
| **Borrow cost short large cap** | 0.25-2% annuo |
| **Borrow HTB** | 5-50% annuo |

**Round-trip cost da assumere in backtest** (long-only):
- SPY/QQQ/AAPL-class: 1-3 bp
- Large cap S&P 500: 10-25 bp
- Mid cap $1-10B: 50-110 bp

Senza modellare questi numeri, il backtest mente di 10-30 bp per trade.

---

## 9. Anti-overfitting — pratico

### LHS (Latin Hypercube Sampling) come nel video
**Verdetto**: valido, migliore di random uniform per stessa N, ma **non adattivo**. Perde contro TPE oltre ~50 trial.

### Bayesian optimization (Optuna TPE)
**Recommended default**. Studio comparativo (MDPI 2026): TPE raggiunge 90% della performance ottima con 13-17% del budget di trial, vince 9/12 strategy-asset pair.

### Quanti trial prima di overfittare?
Non c'è numero magico. Usa **PBO (CSCV)**: se PBO > 0.5 hai overfittato, se < 0.3 sei ok. Empiricamente, su daily equity, dopo ~30-50 trial su un campione di 5-10 anni il rischio cresce significativamente.

### Bailey, Borwein, López de Prado, Zhu (2014) "Pseudo-Mathematics and Financial Charlatanism" [[PDF](https://www.davidhbailey.com/dhbpapers/backtest-pseudo.pdf)]
Lettura obbligatoria. Il messaggio: alto Sharpe in-sample è banalmente raggiungibile provando abbastanza configurazioni; gli analisti raramente dichiarano il numero di trial; standard hold-out è "inaffidabile" — usa CSCV.

---

## 10. Must-read & libri di riferimento

### Paper indispensabili
- **Jegadeesh & Titman (1993)** — momentum [[link](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1993.tb04702.x)]
- **Moskowitz, Ooi, Pedersen (2012)** — time-series momentum [[link](https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum)]
- **Carhart (1997)** — 4-factor model [[link](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1997.tb03808.x)]
- **Asness, Moskowitz, Pedersen (2013)** — value & momentum everywhere [[link](https://onlinelibrary.wiley.com/doi/10.1111/jofi.12021)]
- **Daniel & Moskowitz (2016)** — momentum crashes [[link](https://www.sciencedirect.com/science/article/pii/S0304405X16301490)]
- **Barroso & Santa-Clara (2015)** — vol-managed momentum
- **Frazzini & Pedersen (2014)** — Betting Against Beta [[PDF](https://pages.stern.nyu.edu/~lpederse/papers/BettingAgainstBeta.pdf)]
- **Asness, Frazzini, Pedersen (2019)** — Quality Minus Junk
- **Bernard & Thomas (1989)** — PEAD
- **Gatev, Goetzmann, Rouwenhorst (2006)** — Pairs Trading [[link](https://academic.oup.com/rfs/article-abstract/19/3/797/1646694)]
- **Bailey & López de Prado (2014)** — Deflated Sharpe Ratio [[PDF](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)]
- **Bailey, Borwein, López de Prado, Zhu (2014)** — Probability of Backtest Overfitting [[PDF](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)]
- **Harvey, Liu, Zhu (2016)** — "...and the Cross-Section of Expected Returns" [[NBER](https://www.nber.org/papers/w20592)]
- **Arnott, Harvey, Markowitz (2019)** — Backtesting Protocol [[Duke PDF](https://people.duke.edu/~charvey/Research/Published_Papers/P138_A_backtesting_protocol.pdf)]
- **McLean & Pontiff (2016)** — alpha decay [[link](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12365)]
- **Brock, Lakonishok, LeBaron (1992)** + **Sullivan, Timmermann, White (1999)** — la storia del data-snooping su technical rules
- **Han, Yang, Zhou (2013)** — MA timing cross-sectional su decili di vol

### Libri
- **López de Prado (2018)** *Advances in Financial Machine Learning*, Wiley — bibbia per CPCV, meta-labeling, fractional differentiation
- **Clenow (2015)** *Stocks on the Move* — implementazione retail di momentum su azioni
- **Grinold & Kahn (1999)** *Active Portfolio Management* — IC, IR, Fundamental Law
- **Connors & Alvarez (2009)** *Short Term Trading Strategies That Work* — RSI(2), 2-period RSI
- **Chan (2013)** *Algorithmic Trading: Winning Strategies and Their Rationale*

---

## 11. Onesta riassuntiva sul video sorgente

Il video presenta un sistema impressionante sul piano visivo (PostgreSQL, walk-forward, Monte Carlo, deploy 24/7) ma con red flag importanti per chi vuole replicarlo:

1. **Strategie progettate per XAU/USD M15** (forex/oro intraday) presentate come trasferibili — non lo sono direttamente per azioni daily.
2. **Metriche backtest molto pulite** (WR 64%, PF 1.92, Sharpe 1.41 sulle migliori) ma:
   - Nessuna citazione del Deflated Sharpe applicato
   - 80 combinations × 4 strategie × multipli timeframe = N effettivo alto → DSR probabilmente negativo
   - Il "self-optimizing every 6 hours" è esattamente quello che gonfia N a livelli pericolosi
3. **Monte Carlo "shuffle trades"**: shuffling dei trade testa solo l'autocorrelazione, non l'edge sui parametri. Per quello serve permutation/bootstrap sui ritorni di mercato (block bootstrap), non sui trade già selezionati.
4. **0 lines I wrote**: il rischio principale di "lascio scrivere tutto a Claude" è che l'AI può produrre codice che backtesta perfettamente ma con bug di look-ahead bias o data leakage difficili da vedere — proprio la categoria di errori che il video non discute mai esplicitamente.

Niente di tutto questo significa che il sistema non funzioni: significa che il pitch è ottimistico sul piano statistico, e per il nostro bot dobbiamo essere più rigorosi.
