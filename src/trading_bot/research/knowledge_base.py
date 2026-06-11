"""Academic knowledge base + exhaustive parameter search space.

Each feature has:
- Full parameter grid (dense — all values worth exploring)
- Academic citation
- Economic intuition
- Synergy/anti-synergy with other features

Search space size estimate (24 features, up to 4 signals,
4-signal combos synergy-screened to ~3 800):
  total well into the millions of unique configurations — effectively
  inexhaustible; the UCB1 bandit prioritises, the gates decide.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FeatureSpec:
    name: str
    citation: str
    intuition: str
    param_grid: dict[str, list]
    family: str = "unknown"          # momentum / reversal / risk / structural
    synergy_with: list[str] = field(default_factory=list)
    anti_synergy_with: list[str] = field(default_factory=list)
    can_negate: bool = False
    # Canonical ranking direction (audit A2): True = the raw feature is
    # 'higher = worse' and MUST be negated before ranking. Generators apply
    # this automatically so no strategy ever buys the worst by mistake.
    negate_default: bool = False


# ── Feature knowledge base (24 features) ───────────────────────────────────

FEATURES: dict[str, FeatureSpec] = {

    # ── Momentum family ───────────────────────────────────────────────────
    "momentum": FeatureSpec(
        name="momentum", family="momentum",
        citation="Jegadeesh-Titman (1993 JF), Carhart (1997 JF), "
                 "Asness-Moskowitz-Pedersen (2013 JF)",
        intuition="Past 3-12 month winners continue to outperform. "
                  "Driven by investor underreaction and slow information diffusion.",
        param_grid={
            "lookback": [21, 42, 63, 84, 105, 126, 168, 189, 210, 252, 315, 378],
            "skip":     [0, 1, 5, 10, 15, 21, 42],
        },
        synergy_with=["drawdown",
                      "above_sma", "relative_strength"],
        anti_synergy_with=["rsi", "short_term_reversal", "mean_reversion_score"],
        can_negate=False,
    ),

    "rate_of_change": FeatureSpec(
        name="rate_of_change", family="momentum",
        citation="Murphy (1999) 'Technical Analysis of Financial Markets', "
                 "Brock-Lakonishok-LeBaron (1992 JF)",
        intuition="Pure price momentum without skip. Simpler than 12-1 but "
                  "captures the same trend-following effect.",
        param_grid={
            "lookback": [10, 21, 42, 63, 84, 105, 126, 168, 252],
        },
        synergy_with=["ema_ratio", "macd_histogram", "above_sma"],
        anti_synergy_with=["short_term_reversal", "rsi"],
        can_negate=False,
    ),

    "price_acceleration": FeatureSpec(
        name="price_acceleration", family="momentum",
        citation="Grinblatt-Moskowitz (2004 JFE), "
                 "Novy-Marx (2012 JFE) 'Is momentum really momentum?'",
        intuition="Stocks where momentum is accelerating (short > long return) "
                  "continue to outperform. Captures the second derivative of price.",
        param_grid={
            "short": [5, 10, 21, 42],
            "long":  [42, 63, 126, 252],
        },
        synergy_with=["momentum", "macd_histogram"],
        anti_synergy_with=["mean_reversion_score"],
        can_negate=False,
    ),

    "ema_ratio": FeatureSpec(
        name="ema_ratio", family="momentum",
        citation="Elder (1993) 'Trading for a Living', "
                 "Appel (1979) originator of MACD concept",
        intuition="EMA crossover: fast EMA above slow EMA = confirmed uptrend. "
                  "Exponential weighting gives more weight to recent prices.",
        param_grid={
            "fast": [5, 8, 10, 13, 21, 34],
            "slow": [21, 34, 55, 89, 144, 200],
        },
        synergy_with=["macd_histogram", "momentum", "above_sma"],
        anti_synergy_with=["short_term_reversal", "bollinger_position"],
        can_negate=False,
    ),

    "macd_histogram": FeatureSpec(
        name="macd_histogram", family="momentum",
        citation="Appel (1979), Murphy (1999), "
                 "Brock-Lakonishok-LeBaron (1992 JF)",
        intuition="MACD histogram measures momentum of momentum. "
                  "Positive and growing = strong trend confirmation.",
        param_grid={
            "fast":   [8, 12, 16],
            "slow":   [17, 26, 34],
            "signal": [7, 9, 12],
        },
        synergy_with=["ema_ratio", "rate_of_change", "momentum"],
        anti_synergy_with=["rsi", "bollinger_position"],
        can_negate=False,
    ),

    "relative_strength": FeatureSpec(
        name="relative_strength", family="momentum",
        citation="Levy (1967) 'Relative Strength as a Criterion for Investment', "
                 "Jegadeesh-Titman (1993)",
        intuition="Stock return minus cross-sectional median. Pure relative "
                  "outperformance independent of market direction.",
        param_grid={
            "lookback": [21, 42, 63, 126, 252],
        },
        synergy_with=["momentum", "above_sma"],
        anti_synergy_with=["mean_reversion_score"],
        can_negate=False,
    ),

    # ── Reversal family ───────────────────────────────────────────────────
    "short_term_reversal": FeatureSpec(
        name="short_term_reversal", family="reversal",
        citation="Jegadeesh (1990 JF), Lehmann (1990 JF), "
                 "Da-Liu-Schaumburg (2014 RFS)",
        intuition="Weekly losers outperform weekly winners the next week. "
                  "Microstructure noise and liquidity provision costs.",
        param_grid={
            "lookback": [2, 3, 5, 7, 10, 15],
        },
        synergy_with=["above_sma", "mean_reversion_score"],
        anti_synergy_with=["momentum", "rate_of_change", "macd_histogram"],
        can_negate=False,
    ),

    "rsi": FeatureSpec(
        name="rsi", family="reversal",
        citation="Wilder (1978) 'New Concepts in Technical Trading Systems', "
                 "Connors-Alvarez (2009)",
        intuition="RSI(2) < 10 on uptrending stock = extreme short-term "
                  "oversold. High-probability mean reversion within 1-5 days.",
        param_grid={
            "period": [2, 3, 4, 5, 6, 7, 10, 14],
        },
        synergy_with=["above_sma", "bollinger_position"],
        anti_synergy_with=["momentum", "rate_of_change", "macd_histogram"],
        can_negate=True,
    ),

    "bollinger_position": FeatureSpec(
        name="bollinger_position", family="reversal",
        citation="Bollinger (1992) 'Bollinger on Bollinger Bands', "
                 "Andersen-Bondarenko (2014)",
        intuition="Position within Bollinger bands: near lower band = "
                  "statistically oversold vs recent history → reversion expected.",
        param_grid={
            "window": [10, 15, 20, 30, 50],
            "n_std":  [1.5, 2.0, 2.5],
        },
        synergy_with=["rsi", "mean_reversion_score"],
        anti_synergy_with=["momentum", "ema_ratio"],
        can_negate=True,  # negate: low bollinger_position = buy signal
    ),

    "mean_reversion_score": FeatureSpec(
        name="mean_reversion_score", family="reversal",
        citation="Lo-MacKinlay (1990 RFS), "
                 "Poterba-Summers (1988 JFE)",
        intuition="Z-score of recent return vs historical distribution. "
                  "Extreme negative z-score = oversold vs own history.",
        param_grid={
            "lookback": [63, 126, 252],
            "short":    [5, 10, 21],
        },
        synergy_with=["rsi", "bollinger_position", "short_term_reversal"],
        anti_synergy_with=["momentum", "rate_of_change"],
        can_negate=False,
    ),

    # ── Risk family ───────────────────────────────────────────────────────
    "volatility": FeatureSpec(
        name="volatility", family="risk",
        citation="Ang-Hodrick-Xing-Zhang (2006 JF), "
                 "Baker-Bradley-Wurgler (2011 FAJ)",
        intuition="High idiosyncratic vol predicts lower future returns (AHXZ). "
                  "Used negated: prefer low-vol stocks.",
        param_grid={
            "lookback": [21, 42, 63, 126, 252],
        },
        synergy_with=["beta_spy"],
        anti_synergy_with=[],
        can_negate=True,
        negate_default=True,
    ),


    "downside_vol": FeatureSpec(
        name="downside_vol", family="risk",
        citation="Sortino-Satchell (2001), "
                 "Ang-Chen-Xing (2006 JF) 'Downside Risk'",
        intuition="Downside semi-deviation captures left-tail risk better than "
                  "full volatility. Low downside vol = asymmetric risk profile.",
        param_grid={
            "lookback": [21, 63, 126, 252],
        },
        synergy_with=["beta_spy"],
        anti_synergy_with=[],
        can_negate=True,
        negate_default=True,  # negate = prefer low downside vol
    ),

    "vol_trend": FeatureSpec(
        name="vol_trend", family="risk",
        citation="Moreira-Muir (2017 JF) 'Volatility-Managed Portfolios'",
        intuition="Rising vol (short > long window) = regime change signal. "
                  "Stocks with contracting vol tend to outperform.",
        param_grid={
            "short": [5, 10, 21],
            "long":  [42, 63, 126],
        },
        synergy_with=["volatility"],
        anti_synergy_with=[],
        can_negate=True,
        negate_default=True,  # negate = prefer contracting vol
    ),

    "beta_spy": FeatureSpec(
        name="beta_spy", family="risk",
        citation="Frazzini-Pedersen (2014 JFE), "
                 "Black (1972) 'Capital Market Equilibrium with Restricted Borrowing'",
        intuition="Low-beta stocks earn higher risk-adjusted returns (BAB). "
                  "Leverage-constrained investors overpay for high-beta stocks.",
        param_grid={
            "lookback": [63, 126, 252],
        },
        synergy_with=["downside_vol"],
        anti_synergy_with=[],
        can_negate=True,
        negate_default=True,  # negate = prefer low beta
    ),

    # ── Structural family ─────────────────────────────────────────────────
    "sma_distance": FeatureSpec(
        name="sma_distance", family="structural",
        citation="Faber (2007) 'Quantitative Approach to TAA', "
                 "Clenow (2015) 'Stocks on the Move'",
        intuition="Distance from SMA measures trend strength. "
                  "Large positive = strong confirmed uptrend.",
        param_grid={
            "window": [20, 50, 100, 150, 200, 250],
        },
        synergy_with=["momentum", "above_sma", "ema_ratio"],
        anti_synergy_with=["rsi", "bollinger_position"],
        can_negate=False,
    ),

    "drawdown": FeatureSpec(
        name="drawdown", family="structural",
        citation="George-Hwang (2004 JF) '52-week high and momentum', "
                 "Grinblatt-Han (2005 JFE) 'Prospect theory and momentum'",
        intuition="Stocks near their peak have positive momentum as investors "
                  "anchor to recent highs. Stocks far from peak are disposition-effect-depressed.",
        param_grid={
            "lookback": [63, 126, 189, 252, 315, 378],
        },
        synergy_with=["momentum", "above_sma"],
        anti_synergy_with=[],
        can_negate=True,
    ),


    "autocorr_returns": FeatureSpec(
        name="autocorr_returns", family="structural",
        citation="Lo-MacKinlay (1988 RFS), "
                 "Moskowitz-Ooi-Pedersen (2012 JFE)",
        intuition="Positive autocorrelation = momentum persistence. "
                  "Negative = mean reversion. Directly measures predictability.",
        param_grid={
            "lookback": [63, 126, 252],
            "lag":      [1, 5, 21],
        },
        synergy_with=["momentum", "rate_of_change"],
        anti_synergy_with=["short_term_reversal"],
        can_negate=False,
    ),

    "relative_strength": FeatureSpec(
        name="relative_strength", family="momentum",
        citation="Levy (1967), Jegadeesh-Titman (1993)",
        intuition="Pure relative outperformance vs cross-section.",
        param_grid={"lookback": [21, 42, 63, 126, 252]},
        synergy_with=["momentum"],
        anti_synergy_with=[],
        can_negate=False,
    ),

    # ── Quality / Robustness family ───────────────────────────────────────────
    "quality_score": FeatureSpec(
        name="quality_score", family="quality",
        citation="Novy-Marx (2013 JFE), Asness-Frazzini-Pedersen (2019 JF) — QMJ factor",
        intuition="Low-vol, low-drawdown stocks outperform junk stocks across all regimes. "
                  "Quality acts as a stabilizer: removes fragile stocks before ranking, "
                  "especially valuable during market stress when momentum fails.",
        param_grid={"lookback": [126, 252, 378]},
        synergy_with=["momentum", "trend_quality", "drawdown"],
        anti_synergy_with=["rsi", "bollinger_position"],
        can_negate=False,
    ),

    "trend_quality": FeatureSpec(
        name="trend_quality", family="quality",
        citation="Hurst-Ooi-Pedersen (2017) — trend following consistency",
        intuition="Fraction of up-months over lookback. Stocks that rise steadily "
                  "month by month are more robust than those with a single spike. "
                  "More regime-stable than raw momentum.",
        param_grid={"lookback": [63, 126, 252]},
        synergy_with=["momentum", "quality_score", "above_sma"],
        anti_synergy_with=["short_term_reversal", "mean_reversion_score"],
        can_negate=False,
    ),

    # ── Information-theoretic / state-space family (physics & signal proc.) ──
    "hurst_exponent": FeatureSpec(
        name="hurst_exponent", family="quality",
        citation="Hurst (1951), Lo (1991 Econometrica) 'Long-Term Memory in Stock "
                 "Market Prices', Gatheral-Jaisson-Rosenbaum (2018 QF) 'Volatility "
                 "is Rough'",
        intuition="R/S-based persistence of daily returns. H>0.5 = trends persist, "
                  "H<0.5 = mean-reverting. Measures HOW a stock trends (structural "
                  "persistence) rather than how much it moved — complements raw "
                  "momentum and filters out one-jump movers.",
        param_grid={"lookback": [126, 252, 378]},
        synergy_with=["momentum", "trend_quality", "kalman_trend", "autocorr_returns"],
        anti_synergy_with=["mean_reversion_score", "short_term_reversal"],
        can_negate=True,   # negated = prefer mean-reverting stocks
    ),

    "return_entropy": FeatureSpec(
        name="return_entropy", family="quality",
        citation="Shannon (1948), Maasoumi-Racine (2002 JoE), Singha (2025, "
                 "arXiv:2512.15720) 'Hidden Order in Trades Predicts the Size of "
                 "Price Moves'",
        intuition="1 − normalized Shannon entropy of the daily-return histogram. "
                  "High value = structured, repeatable return behavior; low = chaos. "
                  "Entropy predicts MAGNITUDE of moves, not direction (NASA/SPY "
                  "study: entropy < 5th pctile → 2.89× larger moves). As a daily "
                  "cross-sectional score it prefers stable, predictable stocks.",
        param_grid={"lookback": [63, 126, 252], "bins": [10, 12, 16]},
        synergy_with=["quality_score", "trend_quality"],
        anti_synergy_with=["volatility"],
        can_negate=False,
    ),

    "kalman_trend": FeatureSpec(
        name="kalman_trend", family="momentum",
        citation="Kalman (1960), Harvey (1989) 'Forecasting, structural time series "
                 "models and the Kalman filter'",
        intuition="Annualized slope from a local-linear-trend Kalman filter on log "
                  "price. Lag-free trend estimate: weighs each price against forecast "
                  "uncertainty, reacting to genuine trend changes while ignoring "
                  "noise. Strictly more adaptive than SMA distance or EMA ratio.",
        param_grid={"lookback": [126, 252], "q_ratio": [0.01, 0.05, 0.1]},
        synergy_with=["hurst_exponent", "momentum", "quality_score"],
        anti_synergy_with=["sma_distance", "ema_ratio"],  # redundant trend measures
        can_negate=False,
    ),
    # ── Volume / liquidity family (NEW information: il volume era nel DB,
    #    mai usato — ortogonale alle trasformazioni di prezzo) ─────────────
    "amihud_illiquidity": FeatureSpec(
        name="amihud_illiquidity", family="liquidity",
        citation="Amihud (2002 JFM) 'Illiquidity and stock returns'",
        intuition="mean(|ret|/dollar volume). Illiquid stocks earn a premium "
                  "as compensation; gates decide if it survives trading costs.",
        param_grid={"lookback": [21, 63, 126]},
        synergy_with=["volatility", "quality_score"],
        anti_synergy_with=[],
        can_negate=True,
    ),
    "volume_momentum": FeatureSpec(
        name="volume_momentum", family="liquidity",
        citation="Gervais-Kaniel-Mingelgrin (2001 JF) 'High-volume return premium'",
        intuition="Abnormal volume = attention. Stocks with unusually high "
                  "volume outperform in the following weeks.",
        param_grid={"short": [10, 21], "long": [63, 126]},
        synergy_with=["momentum", "rate_of_change", "price_volume_corr"],
        anti_synergy_with=[],
        can_negate=True,
    ),
    "price_volume_corr": FeatureSpec(
        name="price_volume_corr", family="liquidity",
        citation="Karpoff (1987 JFQA) price-volume relation",
        intuition="Corr(return, volume change): positive = volume confirms "
                  "the move (accumulation), negative = fading participation.",
        param_grid={"lookback": [42, 63, 126]},
        synergy_with=["volume_momentum", "momentum"],
        anti_synergy_with=[],
        can_negate=True,
    ),
    # ── Seasonality family ────────────────────────────────────────────────
    "seasonal_month_return": FeatureSpec(
        name="seasonal_month_return", family="seasonality",
        citation="Heston-Sadka (2008 JFE) 'Seasonality in the cross-section'",
        intuition="Stocks that historically perform well in the current "
                  "calendar month keep doing so. Pure calendar information, "
                  "orthogonal to price trends.",
        param_grid={"years": [3, 5, 8]},
        synergy_with=["momentum", "quality_score"],
        anti_synergy_with=[],
        can_negate=False,
    ),
}

# ── Filter knowledge base ───────────────────────────────────────────────────
FILTERS: dict[str, dict] = {
    # AUDIT de-bias: prima esisteva SOLO above_sma → ogni strategia era
    # forzata in una cornice trend-following. Ora il filtro è un grado di
    # libertà esplorato come gli altri (incluso NESSUN filtro).
    "above_sma": {
        "param_grid": {"window": [50, 100, 150, 200, 250]},
        "threshold": 0.5,
        "citation": "Clenow (2015), Faber (2007)",
        "intuition": "Trade only stocks in confirmed uptrend. Avoids value traps.",
    },
    "quality_score": {
        "param_grid": {"lookback": [126, 252]},
        "threshold": 0.25,
        "citation": "Novy-Marx (2013), Asness-Frazzini-Pedersen (2019) QMJ",
        "intuition": "Trade only non-fragile stocks (top ~75% quality). "
                     "Direction-neutral: works for reversal and defensive too.",
    },
    "none": {
        "param_grid": {},
        "threshold": 0.0,
        "citation": "—",
        "intuition": "No eligibility filter — pure signal ranking.",
    },
}

# ── Regime knowledge base ───────────────────────────────────────────────────
REGIMES: dict[str, dict] = {
    "SPY_sma_200": {
        "symbol": "SPY", "feature": "sma_distance", "params": {"window": 200},
        "threshold": 0.0,
        "citation": "Faber (2007)",
        "intuition": "Go flat when broad market is in confirmed downtrend.",
    },
    "SPY_sma_100": {
        "symbol": "SPY", "feature": "sma_distance", "params": {"window": 100},
        "threshold": 0.0,
        "citation": "Clenow (2015), Antonacci (2014)",
        "intuition": "Faster regime filter.",
    },
    "HYG_credit": {
        "symbol": "HYG", "feature": "sma_distance", "params": {"window": 100},
        "threshold": 0.0,
        "citation": "Asness et al.; credit leads equity in risk-off",
        "intuition": "High-yield credit below trend = funding stress: "
                     "go flat before equity follows.",
    },
    "no_regime": None,  # allow running without a regime filter
}

# ── Synergy matrix ──────────────────────────────────────────────────────────
SYNERGY_MATRIX: dict[tuple[str, str], float] = {
    ("momentum",          "volatility"):      0.35,
    ("momentum",          "drawdown"):            0.30,
    ("momentum",          "drawdown"):   0.30,
    ("momentum",          "relative_strength"):   0.20,
    ("momentum",          "above_sma"):           0.25,
    ("momentum",          "autocorr_returns"):    0.20,
    ("momentum",          "ema_ratio"):           0.15,
    ("momentum",          "rate_of_change"):      0.10,
    ("rsi",               "above_sma"):           0.40,
    ("rsi",               "bollinger_position"):  0.25,
    ("rsi",               "mean_reversion_score"):0.30,
    (   "beta_spy"):            0.25,
    (   "downside_vol"):        0.20,
    ("drawdown",          "drawdown"):   0.35,
    ("ema_ratio",         "macd_histogram"):      0.20,
    ("price_acceleration","momentum"):            0.25,
    ("bollinger_position","mean_reversion_score"):0.30,
    # Quality family synergies
    ("quality_score",     "momentum"):            0.40,
    ("quality_score",     "volatility"):      0.35,
    ("quality_score",     "trend_quality"):       0.45,
    ("quality_score",     "drawdown"):            0.30,
    ("trend_quality",     "momentum"):            0.35,
    ("trend_quality",     "above_sma"):           0.30,
    ("trend_quality",     "relative_strength"):   0.25,
    # Information-theoretic / state-space synergies
    ("hurst_exponent",    "momentum"):            0.40,
    ("hurst_exponent",    "kalman_trend"):        0.45,
    ("hurst_exponent",    "trend_quality"):       0.35,
    ("hurst_exponent",    "autocorr_returns"):    0.25,
    ("return_entropy",    "volatility"):      0.35,
    ("return_entropy",    "quality_score"):       0.40,
    ("return_entropy",    "trend_quality"):       0.30,
    ("kalman_trend",      "momentum"):            0.30,
    ("kalman_trend",      "quality_score"):       0.35,
    ("kalman_trend",      "volatility"):      0.30,
    # Anti-synergies
    ("momentum",          "rsi"):                -0.20,
    ("momentum",          "short_term_reversal"):-0.15,
    ("momentum",          "mean_reversion_score"):-0.25,
    ("rsi",               "macd_histogram"):     -0.15,
    ("kalman_trend",      "sma_distance"):       -0.25,  # redundant trend measures
    ("kalman_trend",      "ema_ratio"):          -0.20,
    ("hurst_exponent",    "mean_reversion_score"):-0.30,
}


def get_synergy(a: str, b: str) -> float:
    return SYNERGY_MATRIX.get((a, b), SYNERGY_MATRIX.get((b, a), 0.0))


def build_rationale(
    signals: list[tuple[str, dict]],
    filters: list[tuple[str, dict]],
    regime: dict | None,
    top_n: int,
) -> str:
    parts = []
    for feat, params in signals:
        spec = FEATURES.get(feat)
        if spec:
            ps = ", ".join(f"{k}={v}" for k, v in params.items())
            parts.append(f"{feat}({ps}): {spec.intuition[:80]} [{spec.citation[:60]}]")
    for feat, params in filters:
        info = FILTERS.get(feat, {})
        ps = ", ".join(f"{k}={v}" for k, v in params.items())
        parts.append(f"Filter {feat}({ps}): {info.get('intuition','')[:60]}")
    if regime:
        parts.append(
            f"Regime: {regime['symbol']} {regime['feature']} > {regime['threshold']}"
        )
    parts.append(f"Top-N={top_n}")
    return " | ".join(parts)
