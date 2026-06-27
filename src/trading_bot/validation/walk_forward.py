from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_bot.backtest.engine import BacktestConfig, BacktestResult, CrossSectionalBacktester
from trading_bot.backtest.metrics import max_drawdown, sharpe
from trading_bot.strategies.base import BaseStrategy
from trading_bot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WalkForwardConfig:
    n_folds: int = 6
    test_years: float = 1.0
    anchored: bool = False  # if False, rolling window (fixed-size train)


@dataclass
class FoldResult:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    sharpe: float
    max_dd: float
    ic_mean: float
    ic_count: int
    n_periods: int


@dataclass
class WalkForwardResult:
    folds: list[FoldResult]
    summary: dict


def run_walk_forward(
    strategy: BaseStrategy,
    prices: pd.DataFrame,
    bt_config: BacktestConfig | None = None,
    wf_config: WalkForwardConfig | None = None,
) -> WalkForwardResult:
    """Run a strategy across N rolling/anchored test windows and report per-fold metrics.

    The strategy has no fitted parameters here (momentum 12-1 is fixed) so the
    "train" portion only serves to provide history for the strategy's lookbacks.
    """
    cfg = wf_config or WalkForwardConfig()
    bt_cfg = bt_config or BacktestConfig()
    prices = prices.sort_index()
    total_days = (prices.index[-1] - prices.index[0]).days
    test_days = int(cfg.test_years * 365)

    if total_days < test_days * (cfg.n_folds + 1):
        logger.warning(
            f"Insufficient history: {total_days} days for "
            f"{cfg.n_folds + 1} × {test_days} day windows"
        )

    end_date = prices.index[-1]
    starts: list[pd.Timestamp] = []
    for i in range(cfg.n_folds):
        # Each fold's test window ends at end_date - i*test_days
        test_end = end_date - pd.Timedelta(days=i * test_days)
        test_start = test_end - pd.Timedelta(days=test_days)
        if test_start <= prices.index[0]:
            break
        starts.append(test_start)
    starts = sorted(starts)

    folds: list[FoldResult] = []
    for idx, test_start in enumerate(starts):
        test_end = min(test_start + pd.Timedelta(days=test_days), end_date)
        train_start = prices.index[0] if cfg.anchored else test_start - pd.Timedelta(days=730)
        train_start = max(train_start, prices.index[0])

        # Restrict prices to [train_start, test_end] so the strategy only sees
        # history up to the test period. The backtester itself handles
        # no-look-ahead within the test window.
        sub_prices = prices.loc[train_start:test_end]
        bt = CrossSectionalBacktester(strategy, bt_cfg)
        # Run a full backtest, then slice the test-period returns
        full_result = bt.run(sub_prices)
        test_mask = (full_result.returns.index >= test_start) & (
            full_result.returns.index <= test_end
        )
        test_returns = full_result.returns[test_mask]
        test_equity = (1 + test_returns).cumprod() * bt_cfg.initial_capital
        test_ic = (
            full_result.ic[
                (full_result.ic.index >= test_start) & (full_result.ic.index <= test_end)
            ]
            if full_result.ic is not None and not full_result.ic.empty
            else pd.Series(dtype=float)
        )

        folds.append(
            FoldResult(
                fold_id=idx,
                train_start=train_start,
                train_end=test_start,
                test_start=test_start,
                test_end=test_end,
                sharpe=sharpe(test_returns),
                max_dd=max_drawdown(test_equity) if len(test_equity) > 0 else 0.0,
                ic_mean=float(test_ic.mean()) if len(test_ic) > 0 else float("nan"),
                ic_count=int(len(test_ic)),
                n_periods=int(len(test_returns)),
            )
        )

    sharpes = [f.sharpe for f in folds]
    ics = [f.ic_mean for f in folds if not np.isnan(f.ic_mean)]
    summary = {
        "n_folds": len(folds),
        "sharpe_mean": float(np.mean(sharpes)) if sharpes else 0.0,
        "sharpe_std": float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0,
        "sharpe_min": float(np.min(sharpes)) if sharpes else 0.0,
        "sharpe_max": float(np.max(sharpes)) if sharpes else 0.0,
        "ic_mean": float(np.mean(ics)) if ics else 0.0,
        "fraction_profitable": float(np.mean([1.0 if s > 0 else 0.0 for s in sharpes]))
        if sharpes
        else 0.0,
    }
    return WalkForwardResult(folds=folds, summary=summary)
