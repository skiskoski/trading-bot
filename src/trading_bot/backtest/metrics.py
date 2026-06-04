import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    if len(equity) < 2:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0]
    years = len(equity) / periods_per_year
    if years <= 0:
        return 0.0
    return total_return ** (1 / years) - 1


def sharpe(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    excess = returns - risk_free / periods_per_year
    if excess.std() == 0 or len(excess) < 2:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))


def sortino(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < 0]
    if downside.std() == 0 or len(downside) < 2:
        return 0.0
    return float(excess.mean() / downside.std() * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    dd = equity / running_max - 1
    return float(dd.min())


def information_coefficient(
    predictions: pd.DataFrame, forward_returns: pd.DataFrame
) -> pd.Series:
    """Rank IC per period.

    Both inputs are DataFrames with rebalance dates as the index and symbols as columns.
    Returns a Series of Spearman rank correlations, one per period.
    """
    common_index = predictions.index.intersection(forward_returns.index)
    ics: list[float] = []
    dates: list[pd.Timestamp] = []
    for dt in common_index:
        pred = predictions.loc[dt].dropna()
        ret = forward_returns.loc[dt].dropna()
        common = pred.index.intersection(ret.index)
        if len(common) < 5:
            continue
        rho, _ = spearmanr(pred.loc[common].values, ret.loc[common].values)
        if np.isnan(rho):
            continue
        ics.append(float(rho))
        dates.append(dt)
    return pd.Series(ics, index=pd.DatetimeIndex(dates), name="IC")


def hit_rate(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).mean())


def summary(equity: pd.Series, returns: pd.Series, ic: pd.Series | None = None) -> dict:
    out = {
        "CAGR": cagr(equity),
        "Sharpe": sharpe(returns),
        "Sortino": sortino(returns),
        "MaxDrawdown": max_drawdown(equity),
        "HitRate": hit_rate(returns),
        "Periods": int(len(returns)),
    }
    if ic is not None and len(ic) > 0:
        out["IC_mean"] = float(ic.mean())
        out["IC_std"] = float(ic.std())
        out["IC_IR"] = float(ic.mean() / ic.std()) if ic.std() > 0 else 0.0
        out["IC_count"] = int(len(ic))
    return out
