from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trading_bot.backtest.metrics import summary
from trading_bot.strategies.base import BaseStrategy
from trading_bot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    rebalance_freq: str = "ME"  # pandas month-end
    cost_bps: float = 5.0  # one-way transaction cost in bps
    risk_free_rate: float = 0.04


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    weights_history: pd.DataFrame
    forward_returns: pd.DataFrame
    rank_scores: pd.DataFrame
    metrics: dict = field(default_factory=dict)
    ic: pd.Series | None = None


class CrossSectionalBacktester:
    """Event-driven, monthly-rebalanced cross-sectional backtester.

    - At each rebalance date t, strategy uses prices up to and including t (closing).
    - Target weights are realized at the *next* trading day's close to avoid
      look-ahead.
    - Between rebalances, weights drift with returns (no daily reweighting).
    - Transaction costs charged on turnover at each rebalance.
    """

    def __init__(self, strategy: BaseStrategy, config: BacktestConfig | None = None) -> None:
        self.strategy = strategy
        self.config = config or BacktestConfig()

    def run(self, prices: pd.DataFrame) -> BacktestResult:
        prices = prices.sort_index().ffill()
        # Daily returns from adjusted closes
        daily_returns = prices.pct_change().fillna(0.0)

        rebalance_dates = pd.date_range(
            start=prices.index[0], end=prices.index[-1], freq=self.config.rebalance_freq
        )
        rebalance_dates = [d for d in rebalance_dates if d in prices.index]
        if not rebalance_dates:
            # Fallback: pick first business day of each month within the index
            month_first = (
                pd.Series(prices.index, index=prices.index)
                .groupby(prices.index.to_period("M"))
                .first()
            )
            rebalance_dates = list(month_first.values)

        weights_history: dict[pd.Timestamp, pd.Series] = {}
        rank_history: dict[pd.Timestamp, pd.Series] = {}

        for dt in rebalance_dates:
            try:
                w = self.strategy.weights(prices, dt)
                r = self.strategy.rank(prices, dt)
            except Exception as e:
                logger.warning(f"strategy failed at {dt}: {e}")
                continue
            if not w.empty:
                weights_history[dt] = w
            if not r.empty:
                rank_history[dt] = r

        if not weights_history:
            logger.warning("No weights produced — strategy never fired")
            empty_equity = pd.Series(self.config.initial_capital, index=prices.index)
            return BacktestResult(
                equity=empty_equity,
                returns=pd.Series(0.0, index=prices.index),
                weights_history=pd.DataFrame(),
                forward_returns=pd.DataFrame(),
                rank_scores=pd.DataFrame(),
            )

        weights_df = pd.DataFrame(weights_history).T.reindex(columns=prices.columns).fillna(0.0)
        ranks_df = pd.DataFrame(rank_history).T.reindex(columns=prices.columns)

        # Shift weights by one day to avoid look-ahead bias
        held_weights = weights_df.shift(1).reindex(prices.index).ffill().fillna(0.0)

        # Daily portfolio return
        portfolio_returns = (held_weights * daily_returns).sum(axis=1)

        # Transaction cost at each rebalance
        cost_per_bp = self.config.cost_bps / 10_000.0
        turnovers = (
            weights_df.fillna(0.0).diff().abs().sum(axis=1).shift(1).reindex(prices.index).fillna(0.0)
        )
        cost = turnovers * cost_per_bp
        portfolio_returns = portfolio_returns - cost

        equity = (1 + portfolio_returns).cumprod() * self.config.initial_capital

        # Forward returns at each rebalance date for IC computation
        forward_returns = self._forward_returns_at(weights_df.index, prices)

        # IC: rank predictions (formation returns) vs forward returns
        ic = self._compute_ic(ranks_df, forward_returns)

        metrics = summary(equity, portfolio_returns, ic)

        return BacktestResult(
            equity=equity,
            returns=portfolio_returns,
            weights_history=weights_df,
            forward_returns=forward_returns,
            rank_scores=ranks_df,
            metrics=metrics,
            ic=ic,
        )

    @staticmethod
    def _forward_returns_at(rebalance_dates: pd.Index, prices: pd.DataFrame) -> pd.DataFrame:
        """One-period (rebalance-to-rebalance) forward return per symbol at each date."""
        snapshots = prices.reindex(rebalance_dates).ffill()
        fwd = snapshots.pct_change().shift(-1)
        return fwd

    @staticmethod
    def _compute_ic(ranks: pd.DataFrame, forward: pd.DataFrame) -> pd.Series:
        from trading_bot.backtest.metrics import information_coefficient

        if ranks.empty or forward.empty:
            return pd.Series(dtype=float)
        return information_coefficient(ranks, forward)
