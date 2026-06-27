from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    name: str


class BaseStrategy(ABC):
    """A cross-sectional strategy that emits target weights at each rebalance date.

    Subclasses implement ``rank()`` returning a ranking score (higher = more attractive)
    per symbol at each rebalance date. ``weights()`` converts the ranks to target
    portfolio weights (long-only, sum<=1).
    """

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    @abstractmethod
    def rank(self, prices: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
        """Return a ranking score per symbol using only data with index <= asof."""

    @abstractmethod
    def weights(self, prices: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
        """Return target weights per symbol at asof. Long-only, sums to <=1."""
