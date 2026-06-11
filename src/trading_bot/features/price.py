"""Price-based cross-sectional features.

Each feature is a pure function: it takes a wide ``prices`` DataFrame
(date index × symbol columns, adjusted close) and an ``asof`` timestamp,
plus keyword parameters, and returns a Series indexed by symbol with the
feature value at ``asof``.

Functions must NEVER look beyond ``asof``. The leakage-check harness in
``trading_bot.features.leakage`` verifies this automatically.

Feature catalogue (29 in REGISTRY, 26 searchable):
  Momentum:    momentum, rate_of_change, price_acceleration, ema_ratio,
               macd_histogram, relative_strength
  Reversal:    short_term_reversal, rsi, bollinger_position, mean_reversion_score
  Risk:        volatility, downside_vol, vol_trend, beta_spy
  Structural:  sma_distance, above_sma, drawdown, autocorr_returns
  Quality:     quality_score, trend_quality, hurst_exponent, return_entropy
  State-space: kalman_trend
  Liquidity:   amihud_illiquidity, volume_momentum, price_volume_corr (NEW)
  Seasonality: seasonal_month_return (NEW)
  Other:       tsmom_signal, max_gap, realized_vol (legacy)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _history(prices: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    return prices.loc[:asof]


# ── Momentum family ────────────────────────────────────────────────────────

def momentum(
    prices: pd.DataFrame, asof: pd.Timestamp,
    lookback: int = 252, skip: int = 21,
) -> pd.Series:
    """Past-return [asof-lookback : asof-skip]. Default = 12-1 month momentum.
    Jegadeesh-Titman 1993, Carhart 1997."""
    hist = _history(prices, asof)
    if len(hist) < lookback + 1:
        return pd.Series(dtype=float)
    return (hist.iloc[-(skip + 1)] / hist.iloc[-(lookback + 1)] - 1.0).rename(
        f"mom_{lookback}_{skip}"
    )


def rate_of_change(
    prices: pd.DataFrame, asof: pd.Timestamp, lookback: int = 63,
) -> pd.Series:
    """Rate of Change: (P_t / P_{t-n} - 1). Pure price momentum without skip.
    Murphy (1999) 'Technical Analysis of Financial Markets'."""
    hist = _history(prices, asof)
    if len(hist) < lookback + 1:
        return pd.Series(dtype=float)
    return (hist.iloc[-1] / hist.iloc[-(lookback + 1)] - 1.0).rename(f"roc_{lookback}")


def price_acceleration(
    prices: pd.DataFrame, asof: pd.Timestamp,
    short: int = 21, long: int = 63,
) -> pd.Series:
    """Momentum of momentum: short-term return minus longer-term return.
    Captures whether trend is accelerating or decelerating.
    Grinblatt-Moskowitz 2004."""
    hist = _history(prices, asof)
    if len(hist) < long + 1:
        return pd.Series(dtype=float)
    r_short = hist.iloc[-1] / hist.iloc[-(short + 1)] - 1.0
    r_long  = hist.iloc[-1] / hist.iloc[-(long + 1)] - 1.0
    return (r_short - r_long).rename(f"accel_{short}_{long}")


def ema_ratio(
    prices: pd.DataFrame, asof: pd.Timestamp,
    fast: int = 21, slow: int = 63,
) -> pd.Series:
    """EMA(fast) / EMA(slow) - 1. Positive = fast EMA above slow = uptrend.
    Classic EMA crossover signal (Elder 1993)."""
    hist = _history(prices, asof)
    if len(hist) < slow + 10:
        return pd.Series(dtype=float)
    ema_f = hist.ewm(span=fast, adjust=False).mean().iloc[-1]
    ema_s = hist.ewm(span=slow, adjust=False).mean().iloc[-1]
    return (ema_f / ema_s.replace(0, np.nan) - 1.0).rename(f"ema_{fast}_{slow}")


def macd_histogram(
    prices: pd.DataFrame, asof: pd.Timestamp,
    fast: int = 12, slow: int = 26, signal: int = 9,
) -> pd.Series:
    """MACD histogram: (MACD line) - (signal line). Positive = bullish momentum.
    Appel (1979); widely used momentum oscillator."""
    hist = _history(prices, asof)
    if len(hist) < slow + signal + 5:
        return pd.Series(dtype=float)
    ema_f = hist.ewm(span=fast, adjust=False).mean()
    ema_s = hist.ewm(span=slow, adjust=False).mean()
    macd_line = ema_f - ema_s
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()
    return (macd_line - sig_line).iloc[-1].rename(f"macd_{fast}_{slow}_{signal}")


def relative_strength(
    prices: pd.DataFrame, asof: pd.Timestamp, lookback: int = 126,
) -> pd.Series:
    """Each stock's return relative to cross-sectional median. Captures
    relative outperformance independent of absolute level.
    Levy 1967 'Relative Strength as a Criterion for Investment Selection'."""
    hist = _history(prices, asof)
    if len(hist) < lookback + 1:
        return pd.Series(dtype=float)
    rets = hist.iloc[-1] / hist.iloc[-(lookback + 1)] - 1.0
    return (rets - rets.median()).rename(f"relstr_{lookback}")


# ── Reversal / Mean-reversion family ──────────────────────────────────────

def short_term_reversal(
    prices: pd.DataFrame, asof: pd.Timestamp, lookback: int = 5,
) -> pd.Series:
    """Contrarian: recent losers outperform recent winners short-term.
    Jegadeesh 1990, Lehmann 1990."""
    hist = _history(prices, asof)
    if len(hist) < lookback + 1:
        return pd.Series(dtype=float)
    return -(hist.iloc[-1] / hist.iloc[-(lookback + 1)] - 1.0).rename(f"rev_{lookback}")


def rsi(
    prices: pd.DataFrame, asof: pd.Timestamp, period: int = 2,
) -> pd.Series:
    """Wilder RSI. RSI(2) < 10 on uptrending stock = Connors mean-reversion signal.
    Connors-Alvarez 2009."""
    hist = _history(prices, asof)
    if len(hist) < period + 2:
        return pd.Series(dtype=float)
    delta = hist.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).iloc[-1].rename(f"rsi_{period}")


def bollinger_position(
    prices: pd.DataFrame, asof: pd.Timestamp,
    window: int = 20, n_std: float = 2.0,
) -> pd.Series:
    """Position within Bollinger bands: 0 = lower band, 1 = upper band.
    Values < 0.2 = oversold (mean-reversion buy), > 0.8 = overbought.
    Bollinger 1992."""
    hist = _history(prices, asof)
    if len(hist) < window + 5:
        return pd.Series(dtype=float)
    window_data = hist.iloc[-window:]
    mid  = window_data.mean()
    std  = window_data.std()
    lower = mid - n_std * std
    upper = mid + n_std * std
    pos = (hist.iloc[-1] - lower) / (upper - lower + 1e-10)
    return pos.clip(0, 1).rename(f"bbpos_{window}")


def mean_reversion_score(
    prices: pd.DataFrame, asof: pd.Timestamp,
    lookback: int = 252, short: int = 21,
) -> pd.Series:
    """Z-score of short-term return vs long-term distribution.
    Negative = oversold vs history → potential reversion.
    Lo-MacKinlay 1990."""
    hist = _history(prices, asof)
    if len(hist) < lookback + 1:
        return pd.Series(dtype=float)
    daily_rets = hist.pct_change().dropna()
    recent_ret = daily_rets.iloc[-short:].mean()
    hist_mean  = daily_rets.iloc[-lookback:-short].mean()
    hist_std   = daily_rets.iloc[-lookback:-short].std()
    z = -(recent_ret - hist_mean) / (hist_std + 1e-10)  # negated: oversold = positive
    return z.rename(f"mr_{lookback}_{short}")


# ── Risk / Volatility family ───────────────────────────────────────────────

def volatility(
    prices: pd.DataFrame, asof: pd.Timestamp, lookback: int = 252,
) -> pd.Series:
    """Annualised realised volatility of daily log-returns.
    Ang-Hodrick-Xing-Zhang 2006."""
    hist = _history(prices, asof).iloc[-(lookback + 1):]
    if len(hist) < 30:
        return pd.Series(dtype=float)
    return (np.log(hist).diff().dropna().std() * np.sqrt(252)).rename(f"vol_{lookback}")



def realized_vol(
    prices: pd.DataFrame, asof: pd.Timestamp, lookback: int = 66,
) -> pd.Series:
    """Short-window realised vol (for vol-targeting sizing)."""
    hist = _history(prices, asof).iloc[-(lookback + 1):]
    if len(hist) < 10:
        return pd.Series(dtype=float)
    return (np.log(hist).diff().dropna().std() * np.sqrt(252)).rename(f"rvol_{lookback}")


def downside_vol(
    prices: pd.DataFrame, asof: pd.Timestamp, lookback: int = 126,
) -> pd.Series:
    """Annualised downside (semi-)deviation. Better risk measure than full vol
    for left-tail-sensitive investors. Sortino-Satchell 2001."""
    hist = _history(prices, asof).iloc[-(lookback + 1):]
    if len(hist) < 20:
        return pd.Series(dtype=float)
    rets = np.log(hist).diff().dropna()
    down = rets.copy()
    down[down > 0] = 0.0
    return (down.std() * np.sqrt(252)).rename(f"dsvol_{lookback}")


def vol_trend(
    prices: pd.DataFrame, asof: pd.Timestamp,
    short: int = 21, long: int = 126,
) -> pd.Series:
    """Ratio of short-term vol to long-term vol. > 1 = vol expanding (risk-off),
    < 1 = vol contracting (calm trend). Moreira-Muir 2017."""
    hist = _history(prices, asof)
    if len(hist) < long + 10:
        return pd.Series(dtype=float)
    vol_s = np.log(hist).diff().dropna().iloc[-short:].std()
    vol_l = np.log(hist).diff().dropna().iloc[-long:].std()
    return (vol_s / (vol_l + 1e-10)).rename(f"voltrend_{short}_{long}")


def beta_spy(
    prices: pd.DataFrame, asof: pd.Timestamp,
    lookback: int = 252, spy_col: str = "SPY",
) -> pd.Series:
    """Rolling beta to SPY. Low-beta stocks tend to outperform on risk-adjusted
    basis (Frazzini-Pedersen 2014 BAB). Returns NaN for SPY itself."""
    hist = _history(prices, asof).iloc[-(lookback + 1):]
    if len(hist) < 60 or spy_col not in hist.columns:
        return pd.Series(dtype=float)
    rets = hist.pct_change().dropna()
    spy_rets = rets[spy_col]
    betas = {}
    for col in rets.columns:
        if col == spy_col:
            continue
        cov = np.cov(rets[col].values, spy_rets.values)
        betas[col] = cov[0, 1] / (cov[1, 1] + 1e-10)
    return pd.Series(betas).rename(f"beta_{lookback}")


# ── Structural family ──────────────────────────────────────────────────────

def sma_distance(
    prices: pd.DataFrame, asof: pd.Timestamp, window: int = 200,
) -> pd.Series:
    """% distance from N-day SMA. Positive = above SMA = uptrend."""
    hist = _history(prices, asof)
    if len(hist) < window:
        return pd.Series(dtype=float)
    sma = hist.iloc[-window:].mean()
    return (hist.iloc[-1] / sma - 1.0).rename(f"sma_dist_{window}")


def above_sma(
    prices: pd.DataFrame, asof: pd.Timestamp, window: int = 100,
) -> pd.Series:
    """Boolean: price > N-day SMA (1.0 / 0.0)."""
    return (sma_distance(prices, asof, window) > 0).astype(float).rename(f"above_sma_{window}")


def drawdown(
    prices: pd.DataFrame, asof: pd.Timestamp, lookback: int = 252,
) -> pd.Series:
    """Current drawdown from rolling high. George-Hwang 2004."""
    hist = _history(prices, asof).iloc[-lookback:]
    if len(hist) < 30:
        return pd.Series(dtype=float)
    return (hist.iloc[-1] / hist.max() - 1.0).rename(f"dd_{lookback}")



def autocorr_returns(
    prices: pd.DataFrame, asof: pd.Timestamp,
    lookback: int = 126, lag: int = 1,
) -> pd.Series:
    """Lag-1 autocorrelation of daily returns. Positive = momentum persistence,
    Negative = mean reversion. Lo-MacKinlay 1988 'Stock Market Prices do not
    Follow Random Walks'."""
    hist = _history(prices, asof).iloc[-(lookback + lag + 1):]
    if len(hist) < 30:
        return pd.Series(dtype=float)
    rets = hist.pct_change().dropna()
    result = {}
    for col in rets.columns:
        s = rets[col].dropna()
        if len(s) < 20:
            result[col] = np.nan
        else:
            result[col] = float(s.autocorr(lag=lag))
    return pd.Series(result).rename(f"autocorr_{lookback}_{lag}")


# ── Max gap / TSMOM (unchanged) ────────────────────────────────────────────

def max_gap(
    prices: pd.DataFrame, asof: pd.Timestamp, lookback: int = 90,
) -> pd.Series:
    """Largest absolute one-day return in lookback window (outlier filter)."""
    hist = _history(prices, asof).iloc[-(lookback + 1):]
    if len(hist) < 2:
        return pd.Series(dtype=float)
    return hist.pct_change().abs().max().rename(f"max_gap_{lookback}")


def tsmom_signal(
    prices: pd.DataFrame, asof: pd.Timestamp,
    sma_window: int = 200, mom_lookback: int = 252,
) -> pd.Series:
    """TSMOM: 1.0 if price > SMA(sma_window) AND 12-mo return > 0.
    Moskowitz-Ooi-Pedersen 2012."""
    hist = _history(prices, asof)
    if len(hist) < max(sma_window, mom_lookback + 1):
        return pd.Series(dtype=float)
    last  = hist.iloc[-1]
    sma   = hist.iloc[-sma_window:].mean()
    above = last > sma
    pos_m = last > hist.iloc[-(mom_lookback + 1)]
    return (above & pos_m).astype(float).rename(f"tsmom_{sma_window}_{mom_lookback}")


# ── Quality / robustness features ────────────────────────────────────────────

def quality_score(
    prices: pd.DataFrame, asof: pd.Timestamp,
    lookback: int = 252,
) -> pd.Series:
    """Composite quality score: penalizes high-vol, deep-drawdown stocks.

    Score = 1 - (normalized_vol * 0.5 + normalized_drawdown * 0.5)
    Higher is better (less risky, less damaged stock).
    Use as a filter: keep only stocks above median quality.

    Designed to remove fragile stocks during market stress — works
    across regimes where pure momentum fails.
    """
    hist = _history(prices, asof)
    if len(hist) < lookback // 2:
        return pd.Series(dtype=float)
    window = hist.iloc[-lookback:]
    ret = window.pct_change().dropna()
    if len(ret) < 20:
        return pd.Series(dtype=float)

    # Annualized volatility (lower = better quality)
    vol = ret.std() * np.sqrt(252)

    # Max drawdown over lookback (less negative = better quality)
    cum = (1 + ret).cumprod()
    roll_max = cum.cummax()
    dd = ((cum - roll_max) / roll_max).min().abs()  # positive number, bigger = worse

    # Normalize each to [0,1] cross-sectionally, then invert so higher = better
    def _norm(s: pd.Series) -> pd.Series:
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else pd.Series(0.5, index=s.index)

    quality = 1.0 - (_norm(vol) * 0.5 + _norm(dd) * 0.5)
    return quality.rename(f"quality_{lookback}")


def trend_quality(
    prices: pd.DataFrame, asof: pd.Timestamp,
    lookback: int = 126,
) -> pd.Series:
    """Trend consistency: fraction of up-months over lookback.

    A stock that goes up steadily month after month (high trend quality)
    is preferred over one that had a single big jump. More robust than raw
    momentum in volatile regimes.

    Range [0, 1]. Higher = more consistent uptrend.
    """
    hist = _history(prices, asof)
    if len(hist) < lookback + 5:
        return pd.Series(dtype=float)
    window = hist.iloc[-lookback:]
    monthly = window.resample("ME").last().pct_change().dropna()
    if len(monthly) < 3:
        return pd.Series(dtype=float)
    frac_up = (monthly > 0).mean()
    return frac_up.rename(f"trend_quality_{lookback}")


def hurst_exponent(
    prices: pd.DataFrame, asof: pd.Timestamp,
    lookback: int = 252,
) -> pd.Series:
    """Hurst exponent of daily log-returns via rescaled-range (R/S) regression.

    H > 0.5 = persistent (trending), H < 0.5 = anti-persistent (mean-reverting),
    H ≈ 0.5 = random walk. Cross-sectionally, high-H stocks sustain trends —
    a structural complement to momentum that measures HOW a stock trends,
    not how much it moved.

    Literature: Hurst (1951), Lo (1991 Econometrica) 'Long-Term Memory in
    Stock Market Prices', Gatheral-Jaisson-Rosenbaum (2018 QF) 'Volatility
    is Rough' (H~0.1 for realized vol — roughness is a real, measurable
    property of financial series).
    """
    hist = _history(prices, asof).iloc[-(lookback + 1):]
    if len(hist) < 64:
        return pd.Series(dtype=float)
    rets = np.log(hist).diff().dropna()
    n = len(rets)
    # Sub-window sizes for the log-log regression (powers of ~2 up to n/2)
    sizes = [s for s in (8, 16, 32, 64, 128) if s <= n // 2]
    if len(sizes) < 3:
        return pd.Series(dtype=float)

    result = {}
    for col in rets.columns:
        x = rets[col].to_numpy()
        if np.isnan(x).any() or np.std(x) == 0:
            result[col] = np.nan
            continue
        log_rs = []
        for s in sizes:
            n_chunks = len(x) // s
            rs_vals = []
            for i in range(n_chunks):
                chunk = x[i * s:(i + 1) * s]
                dev = np.cumsum(chunk - chunk.mean())
                r = dev.max() - dev.min()
                sd = chunk.std()
                if sd > 0:
                    rs_vals.append(r / sd)
            if rs_vals:
                log_rs.append(np.log(np.mean(rs_vals)))
            else:
                log_rs.append(np.nan)
        log_n = np.log(sizes)
        valid = ~np.isnan(log_rs)
        if valid.sum() < 3:
            result[col] = np.nan
        else:
            h, _ = np.polyfit(log_n[valid], np.array(log_rs)[valid], 1)
            result[col] = float(np.clip(h, 0.0, 1.0))
    return pd.Series(result).rename(f"hurst_{lookback}")


def return_entropy(
    prices: pd.DataFrame, asof: pd.Timestamp,
    lookback: int = 126, bins: int = 12,
) -> pd.Series:
    """Return predictability = 1 − normalized Shannon entropy of daily returns.

    Shannon entropy of the return histogram measures how 'structured' the
    return distribution is. LOW entropy = concentrated, repeatable behavior
    (strong trends or stable patterns); HIGH entropy = chaos. This feature
    returns 1 − H/H_max so HIGHER = more structure/predictability.

    Concept: entropy predicts the MAGNITUDE of future moves, not direction
    (Singha 2025, arXiv:2512.15720: order-flow entropy below 5th pctile →
    2.89× larger subsequent moves on SPY). At daily frequency this acts as
    a regime/stability score: prefer stocks whose recent returns show
    structure rather than noise. Shannon (1948), Maasoumi-Racine (2002 JoE).
    """
    hist = _history(prices, asof).iloc[-(lookback + 1):]
    if len(hist) < 40:
        return pd.Series(dtype=float)
    rets = hist.pct_change().dropna()
    h_max = np.log(bins)

    result = {}
    for col in rets.columns:
        x = rets[col].dropna().to_numpy()
        if len(x) < 30 or np.std(x) == 0:
            result[col] = np.nan
            continue
        counts, _ = np.histogram(x, bins=bins)
        p = counts / counts.sum()
        p = p[p > 0]
        h = -(p * np.log(p)).sum()
        result[col] = float(1.0 - h / h_max)   # 1 = max structure, 0 = max chaos
    return pd.Series(result).rename(f"entropy_{lookback}_{bins}")


def kalman_trend(
    prices: pd.DataFrame, asof: pd.Timestamp,
    lookback: int = 252, q_ratio: float = 0.05,
) -> pd.Series:
    """Annualized trend slope from a local-linear-trend Kalman filter.

    Runs a 2-state (level + slope) Kalman filter on log prices and returns
    the final slope estimate × 252 (annualized log-return trend). Unlike
    SMA distance, the Kalman estimate is lag-free: it weighs each new price
    against forecast uncertainty (gain K), so it reacts to genuine trend
    changes while ignoring noise.

    q_ratio sets process/measurement noise balance: higher = faster
    adaptation (noisier), lower = smoother (laggier).

    Literature: Kalman (1960), Harvey (1989) 'Forecasting, structural time
    series models and the Kalman filter'. Same algorithm as Apollo guidance.
    """
    hist = _history(prices, asof).iloc[-(lookback + 1):]
    if len(hist) < 60:
        return pd.Series(dtype=float)
    log_p = np.log(hist.ffill())
    cols = log_p.columns
    y = log_p.to_numpy()                     # (T, N)
    T, N = y.shape

    # Per-column measurement noise R = variance of daily log returns
    rets = np.diff(y, axis=0)
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            R = np.nanvar(rets, axis=0)      # (N,)
    R = np.where(R <= 0, 1e-8, R)
    Q = R * q_ratio                          # process noise scale

    # State: [level, slope] per column. Vectorized across columns.
    level = y[0].copy()
    slope = np.zeros(N)
    # Covariance per column: P = [[p11, p12], [p12, p22]]
    p11 = R.copy(); p12 = np.zeros(N); p22 = R.copy()

    for t in range(1, T):
        # Predict: level += slope; P = F P F' + Q
        level = level + slope
        p11_n = p11 + 2 * p12 + p22 + Q * 0.25
        p12_n = p12 + p22 + Q * 0.5
        p22_n = p22 + Q
        p11, p12, p22 = p11_n, p12_n, p22_n
        # Update with measurement y[t] (skip NaN columns)
        obs = y[t]
        valid = ~np.isnan(obs)
        innov = np.where(valid, obs - level, 0.0)
        s = p11 + R                          # innovation variance
        k1 = np.where(valid, p11 / s, 0.0)   # gain for level
        k2 = np.where(valid, p12 / s, 0.0)   # gain for slope
        level = level + k1 * innov
        slope = slope + k2 * innov
        p11_u = p11 * (1 - k1)
        p12_u = p12 * (1 - k1)
        p22_u = p22 - k2 * p12
        p11, p12, p22 = p11_u, p12_u, p22_u

    return pd.Series(slope * 252, index=cols).rename(f"kalman_{lookback}")




# ── Volume / liquidity family (NEW information source — volume column was
#    sitting unused in the candles table) ────────────────────────────────────

_VOLUME_PANEL: "pd.DataFrame | None" = None
_VOLUME_TRIED = False


def _volume_panel() -> "pd.DataFrame | None":
    """Lazy-load the full dollar-volume panel from the candles table.

    Loaded once per process (~70MB for 1500 syms × 20y). Returns None when
    the DB is unavailable (e.g. synthetic-data tests) — volume features then
    return an empty Series and the composer treats the symbol as NaN.
    """
    global _VOLUME_PANEL, _VOLUME_TRIED
    if _VOLUME_TRIED:
        return _VOLUME_PANEL
    _VOLUME_TRIED = True
    try:
        from sqlalchemy import text
        from trading_bot.data.storage import get_session
        with get_session() as sess:
            df = pd.read_sql(
                text("SELECT symbol, dt, close * volume AS dv FROM candles"),
                sess.bind, parse_dates=["dt"])
        _VOLUME_PANEL = df.pivot(index="dt", columns="symbol", values="dv")
    except Exception:
        _VOLUME_PANEL = None
    return _VOLUME_PANEL


def _dollar_volume(cols, asof: pd.Timestamp, n: int) -> "pd.DataFrame | None":
    vp = _volume_panel()
    if vp is None:
        return None
    common = [c for c in cols if c in vp.columns]
    if len(common) < max(5, len(cols) // 4):
        return None
    return vp.loc[:asof, common].iloc[-n:]


def amihud_illiquidity(
    prices: pd.DataFrame, asof: pd.Timestamp, lookback: int = 63,
) -> pd.Series:
    """Amihud (2002) illiquidity: mean(|daily return| / dollar volume) × 1e6.

    HIGHER = more illiquid. Illiquid stocks earn a return premium as
    compensation (Amihud 2002 JFM) — but they also cost more to trade;
    the gates decide whether the premium survives our cost model.
    """
    hist = _history(prices, asof).iloc[-(lookback + 1):]
    if len(hist) < 30:
        return pd.Series(dtype=float)
    dv = _dollar_volume(hist.columns, asof, lookback)
    if dv is None:
        return pd.Series(dtype=float)
    rets = hist.pct_change().abs().iloc[1:]
    aligned = rets.reindex(index=dv.index, columns=dv.columns)
    illiq = (aligned / dv.replace(0, np.nan)).mean() * 1e6
    return illiq.dropna().rename(f"amihud_{lookback}")


def volume_momentum(
    prices: pd.DataFrame, asof: pd.Timestamp,
    short: int = 21, long: int = 126,
) -> pd.Series:
    """Abnormal volume: avg dollar volume short window / long window − 1.

    HIGHER = unusually high attention. High-volume stocks earn higher
    subsequent returns (Gervais-Kaniel-Mingelgrin 2001 JF, the
    'high-volume return premium')."""
    dv = _dollar_volume(prices.columns, asof, long + 5)
    if dv is None or len(dv) < long:
        return pd.Series(dtype=float)
    ratio = dv.iloc[-short:].mean() / dv.iloc[-long:].mean().replace(0, np.nan) - 1.0
    return ratio.dropna().rename(f"volmom_{short}_{long}")


def price_volume_corr(
    prices: pd.DataFrame, asof: pd.Timestamp, lookback: int = 63,
) -> pd.Series:
    """Correlation(daily return, dollar-volume change) over the lookback.

    Positive = volume confirms price moves (accumulation); negative =
    moves on fading volume. Karpoff (1987) price-volume relation."""
    hist = _history(prices, asof).iloc[-(lookback + 2):]
    if len(hist) < 30:
        return pd.Series(dtype=float)
    dv = _dollar_volume(hist.columns, asof, lookback + 2)
    if dv is None or len(dv) < 30:
        return pd.Series(dtype=float)
    r = hist.pct_change().iloc[1:]
    v = dv.pct_change().iloc[1:]
    out = {}
    for c in v.columns:
        a = r[c].reindex(v.index)
        b = v[c]
        m = a.notna() & b.notna()
        out[c] = float(a[m].corr(b[m])) if m.sum() > 20 else np.nan
    return pd.Series(out).dropna().rename(f"pvcorr_{lookback}")


# ── Seasonality family (Heston-Sadka 2008) ──────────────────────────────────

def seasonal_month_return(
    prices: pd.DataFrame, asof: pd.Timestamp, years: int = 5,
) -> pd.Series:
    """Average return of the CURRENT calendar month over the past N years.

    Cross-sectional seasonality: stocks that historically perform well in a
    given month keep doing so (Heston-Sadka 2008 JFE 'Seasonality in the
    cross-section of stock returns'). Uses only data ≤ asof."""
    hist = _history(prices, asof)
    if len(hist) < 300:
        return pd.Series(dtype=float)
    monthly = hist.resample("ME").last().pct_change()
    target_month = asof.month
    same = monthly[monthly.index.month == target_month]
    # exclude the current (partial) month, keep last N completed same-months
    same = same[same.index < asof.normalize().replace(day=1)]
    same = same.iloc[-years:]
    if len(same) < 3:
        return pd.Series(dtype=float)
    return same.mean().dropna().rename(f"seasonal_m{target_month}_{years}y")


# ── Registry (all features available to the composer + research loop) ──────

REGISTRY: dict[str, callable] = {
    # Momentum
    "momentum":            momentum,
    "rate_of_change":      rate_of_change,
    "price_acceleration":  price_acceleration,
    "ema_ratio":           ema_ratio,
    "macd_histogram":      macd_histogram,
    "relative_strength":   relative_strength,
    # Reversal
    "short_term_reversal": short_term_reversal,
    "rsi":                 rsi,
    "bollinger_position":  bollinger_position,
    "mean_reversion_score": mean_reversion_score,
    # Risk
    "volatility":          volatility,
    "downside_vol":        downside_vol,
    "vol_trend":           vol_trend,
    "beta_spy":            beta_spy,
    # Structural
    "sma_distance":        sma_distance,
    "above_sma":           above_sma,
    "drawdown":            drawdown,
    "autocorr_returns":    autocorr_returns,
    # Quality / robustness
    "quality_score":       quality_score,
    "trend_quality":       trend_quality,
    # Volume / liquidity (Amihud 2002, Gervais et al. 2001, Karpoff 1987)
    "amihud_illiquidity":  amihud_illiquidity,
    "volume_momentum":     volume_momentum,
    "price_volume_corr":   price_volume_corr,
    # Seasonality (Heston-Sadka 2008)
    "seasonal_month_return": seasonal_month_return,
    # Information-theoretic / state-space (from physics & signal processing)
    "hurst_exponent":      hurst_exponent,
    "return_entropy":      return_entropy,
    "kalman_trend":        kalman_trend,
    # Other
    "max_gap":             max_gap,
    "tsmom_signal":        tsmom_signal,
}
