from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trading_bot.strategies.base import BaseStrategy, StrategyConfig


@dataclass(frozen=True)
class MomentumConfig(StrategyConfig):
    name: str = "momentum_12_1"
    lookback_days: int = 252
    skip_days: int = 21
    top_n: int = 10
    stock_sma_window: int = 100
    max_gap_lookback: int = 90
    max_gap_pct: float = 0.15
    regime_symbol: str = "SPY"
    regime_sma_window: int = 200


class MomentumStrategy(BaseStrategy):
    """Cross-sectional 12-1 momentum with stock and regime filters.

    Signal:
        formation_return = price[asof - skip] / price[asof - lookback] - 1

    Filters applied at rebalance:
        - stock must be above its own ``stock_sma_window`` SMA
        - no single-day gap > ``max_gap_pct`` in the last ``max_gap_lookback`` days

    Regime filter (applied externally to the weights):
        - new longs only when ``regime_symbol`` is above its ``regime_sma_window`` SMA
    """

    def __init__(self, config: MomentumConfig | None = None) -> None:
        super().__init__(config or MomentumConfig())
        self.cfg: MomentumConfig = self.config  # type: ignore[assignment]

    def rank(self, prices: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
        # Exclude the regime symbol from the rankable universe
        tradable = prices.drop(columns=[self.cfg.regime_symbol], errors="ignore")
        history = tradable.loc[:asof]
        if len(history) < self.cfg.lookback_days + 1:
            return pd.Series(dtype=float)

        # 12-1 return: from t-lookback to t-skip
        end_prices = history.iloc[-(self.cfg.skip_days + 1)]
        start_prices = history.iloc[-(self.cfg.lookback_days + 1)]
        formation_return = end_prices / start_prices - 1.0

        # Stock filter: above 100-day SMA
        sma_stock = history.iloc[-self.cfg.stock_sma_window :].mean()
        last_price = history.iloc[-1]
        above_sma = last_price > sma_stock

        # Stock filter: no big gap in the last 90 days
        recent = history.iloc[-self.cfg.max_gap_lookback :]
        daily_change = recent.pct_change().abs()
        no_big_gap = daily_change.max() <= self.cfg.max_gap_pct

        keep = above_sma & no_big_gap
        score = formation_return.where(keep)
        return score.dropna()

    def weights(self, prices: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
        scores = self.rank(prices, asof)
        if scores.empty:
            return pd.Series(dtype=float)

        # Regime filter: only go long if SPY > 200d SMA
        if self.cfg.regime_symbol in prices.columns:
            spy_history = prices[self.cfg.regime_symbol].loc[:asof].dropna()
            if len(spy_history) >= self.cfg.regime_sma_window:
                spy_sma = spy_history.iloc[-self.cfg.regime_sma_window :].mean()
                if spy_history.iloc[-1] < spy_sma:
                    return pd.Series(dtype=float)  # all cash

        # Long top N, equal weight
        top = scores.sort_values(ascending=False).head(self.cfg.top_n)
        if top.empty:
            return pd.Series(dtype=float)
        w = pd.Series(1.0 / len(top), index=top.index)
        return w
