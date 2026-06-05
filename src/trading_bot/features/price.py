"""Price-based cross-sectional features.

Each feature is a pure function: it takes a wide ``prices`` DataFrame
(date index × symbol columns, adjusted close) and an ``asof`` timestamp,
plus keyword parameters, and returns a Series indexed by symbol with the
feature value at ``asof``.

Functions must never look beyond ``asof``. The leakage-check harness in
``trading_bot.features.leakage`` verifies this by recomputing the feature
with future data masked and asserting equality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _history(prices: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    return prices.loc[:asof]


def momentum(prices: pd.DataFrame, asof: pd.Timestamp, lookback: int = 252, skip: int = 21) -> pd.Series:
    """Past-return over [asof-lookback, asof-skip]. Default = 12-1 month momentum."""
    hist = _history(prices, asof)
    if len(hist) < lookback + 1:
        return pd.Series(dtype=float)
    end_prices = hist.iloc[-(skip + 1)]
    start_prices = hist.iloc[-(lookback + 1)]
    return (end_prices / start_prices - 1.0).rename(f"mom_{lookback}_{skip}")


def short_term_reversal(prices: pd.DataFrame, asof: pd.Timestamp, lookback: int = 5) -> pd.Series:
    """Past short-term return. Used negatively → contrarian signal."""
    hist = _history(prices, asof)
    if len(hist) < lookback + 1:
        return pd.Series(dtype=float)
    end_prices = hist.iloc[-1]
    start_prices = hist.iloc[-(lookback + 1)]
    return -((end_prices / start_prices) - 1.0).rename(f"rev_{lookback}")


def volatility(prices: pd.DataFrame, asof: pd.Timestamp, lookback: int = 252) -> pd.Series:
    """Annualised realised volatility of daily log returns."""
    hist = _history(prices, asof).iloc[-(lookback + 1) :]
    if len(hist) < 30:
        return pd.Series(dtype=float)
    rets = np.log(hist).diff().dropna()
    return (rets.std() * np.sqrt(252)).rename(f"vol_{lookback}")


def low_volatility(prices: pd.DataFrame, asof: pd.Timestamp, lookback: int = 252) -> pd.Series:
    """Inverse volatility (high score = low vol)."""
    vol = volatility(prices, asof, lookback)
    return (-vol).rename(f"low_vol_{lookback}")


def rsi(prices: pd.DataFrame, asof: pd.Timestamp, period: int = 2) -> pd.Series:
    """Wilder RSI on adjusted close. RSI(2) is the Connors short-term reversal signal."""
    hist = _history(prices, asof)
    if len(hist) < period + 2:
        return pd.Series(dtype=float)
    delta = hist.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_series = 100 - 100 / (1 + rs)
    return rsi_series.iloc[-1].rename(f"rsi_{period}")


def sma_distance(prices: pd.DataFrame, asof: pd.Timestamp, window: int = 200) -> pd.Series:
    """Percent distance of current price from its ``window``-day SMA."""
    hist = _history(prices, asof)
    if len(hist) < window:
        return pd.Series(dtype=float)
    sma = hist.iloc[-window:].mean()
    last = hist.iloc[-1]
    return (last / sma - 1.0).rename(f"sma_dist_{window}")


def above_sma(prices: pd.DataFrame, asof: pd.Timestamp, window: int = 100) -> pd.Series:
    """Boolean: price > N-day SMA (returned as float 1.0 / 0.0 for ranking math)."""
    dist = sma_distance(prices, asof, window)
    return (dist > 0).astype(float).rename(f"above_sma_{window}")


def max_gap(prices: pd.DataFrame, asof: pd.Timestamp, lookback: int = 90) -> pd.Series:
    """Largest absolute one-day return over the lookback (used as outlier filter)."""
    hist = _history(prices, asof).iloc[-(lookback + 1) :]
    if len(hist) < 2:
        return pd.Series(dtype=float)
    daily = hist.pct_change().abs()
    return daily.max().rename(f"max_gap_{lookback}")


def drawdown(prices: pd.DataFrame, asof: pd.Timestamp, lookback: int = 252) -> pd.Series:
    """Current drawdown from rolling ``lookback`` high (negative number; 0 = at high)."""
    hist = _history(prices, asof).iloc[-lookback:]
    if len(hist) < 30:
        return pd.Series(dtype=float)
    return (hist.iloc[-1] / hist.max() - 1.0).rename(f"dd_{lookback}")


REGISTRY: dict[str, callable] = {
    "momentum": momentum,
    "short_term_reversal": short_term_reversal,
    "volatility": volatility,
    "low_volatility": low_volatility,
    "rsi": rsi,
    "sma_distance": sma_distance,
    "above_sma": above_sma,
    "max_gap": max_gap,
    "drawdown": drawdown,
}
