"""Combinatorial Purged Cross-Validation (CPCV).

Implements De Prado (2018) "Advances in Financial Machine Learning", Chapter 12.

Key idea: split T observations into k groups. For each combination of (k-2)
training groups, backtest on the remaining 2 test groups. This produces
C(k, 2) distinct OOS equity paths that can be stitched into a *distribution*
of OOS Sharpe ratios — far richer than a single walk-forward estimate.

PBO (Probability of Backtest Overfitting) is then:
    PBO = fraction of CPCV paths whose OOS Sharpe < median IS Sharpe

A PBO > 0.5 means the in-sample optimum is more likely to underperform
out-of-sample than not — i.e. the strategy is likely overfit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from trading_bot.backtest.engine import BacktestConfig, CrossSectionalBacktester
from trading_bot.backtest.metrics import sharpe
from trading_bot.strategies.base import BaseStrategy
from trading_bot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CPCVConfig:
    # k=10 → C(10,2)=45 path OOS: il PBO ha granularità 1/45 invece di 1/15.
    # Con k=6 i valori possibili erano solo 16 e 9/15=0.60 ricorreva di
    # continuo — il "PBO sempre 0.6" era discretizzazione, non un bug.
    k: int = 10         # number of folds
    n_test: int = 2     # folds held out as test in each combo (k-n_test used for train)
    purge_days: int = 21  # embargo between train and test to prevent leakage


@dataclass
class CPCVPath:
    """A single CPCV test path (one combination of test folds)."""
    combo_id: int
    train_fold_ids: list[int]
    test_fold_ids: list[int]
    oos_returns: pd.Series
    oos_sharpe: float
    is_sharpe: float   # in-sample Sharpe on the training folds of this combo


@dataclass
class CPCVResult:
    paths: list[CPCVPath]
    oos_sharpe_distribution: list[float]
    is_sharpe_distribution: list[float]
    pbo: float              # Probability of Backtest Overfitting
    mean_oos_sharpe: float
    std_oos_sharpe: float
    median_oos_sharpe: float
    fraction_positive: float
    k: int
    n_test: int
    n_paths: int


def run_cpcv(
    strategy: BaseStrategy,
    prices: pd.DataFrame,
    bt_config: BacktestConfig | None = None,
    cpcv_config: CPCVConfig | None = None,
    benchmark_returns: pd.Series | None = None,
) -> CPCVResult:
    """Run CPCV on *strategy* over *prices*.

    If ``benchmark_returns`` is given (daily returns of e.g. the equal-weight
    universe), all Sharpe ratios are ACTIVE: sharpe(strategy − benchmark).
    This removes the long-only beta component that lets skill-free strategies
    pass an absolute-Sharpe floor in bull-heavy samples (audit finding C2).

    Returns a CPCVResult with a distribution of OOS Sharpe ratios and PBO.
    """
    cfg = cpcv_config or CPCVConfig()
    bt_cfg = bt_config or BacktestConfig()
    prices = prices.sort_index()

    if len(prices) < cfg.k * 20:
        raise ValueError(
            f"Need at least {cfg.k * 20} price rows for {cfg.k} folds; "
            f"got {len(prices)}."
        )

    # ── 1. Split price index into k equal groups ───────────────────────────
    groups = _split_into_groups(prices.index, cfg.k)
    logger.info(
        f"CPCV k={cfg.k}, n_test={cfg.n_test}: "
        f"{math.comb(cfg.k, cfg.n_test)} paths × "
        f"{cfg.n_test} test folds each"
    )

    # ── 2. For each C(k, n_test) combination run a backtest ────────────────
    # Run the full backtest once — used to extract OOS returns for every combo.
    # Each test-fold window is extracted from this single run so we honour the
    # no-look-ahead guarantee (the backtester already shifts weights by one day).
    full_result = _safe_backtest(strategy, prices, bt_cfg)
    if full_result is None:
        raise RuntimeError("Full-sample backtest failed — check data and strategy config")

    paths: list[CPCVPath] = []
    all_combos = list(combinations(range(cfg.k), cfg.n_test))

    # Daily returns used everywhere: strategy minus benchmark (active) when
    # a benchmark is provided, otherwise raw strategy returns.
    base_returns = full_result.returns
    if benchmark_returns is not None:
        bench = benchmark_returns.reindex(base_returns.index).fillna(0.0)
        base_returns = base_returns - bench

    purge = pd.Timedelta(days=cfg.purge_days)

    for combo_id, test_fold_ids in enumerate(all_combos):
        train_fold_ids = [i for i in range(cfg.k) if i not in test_fold_ids]

        train_dates = _merge_groups(groups, train_fold_ids)
        test_dates = _merge_groups(groups, test_fold_ids)

        # Purge ±purge_days around EACH test fold's own boundaries — NOT the
        # merged span. With non-contiguous test folds the old span-purge wiped
        # out every training fold lying between them (audit finding A1).
        exclusion_windows = [
            (groups[i][0] - purge, groups[i][-1] + purge) for i in test_fold_ids
        ]
        train_dates_purged = pd.DatetimeIndex([
            d for d in train_dates
            if not any(lo <= d <= hi for lo, hi in exclusion_windows)
        ])
        if len(train_dates_purged) < 60 or len(test_dates) < 21:
            logger.debug(f"Combo {combo_id}: insufficient dates after purge, skipping")
            continue

        # IS Sharpe from the SAME full-run daily returns restricted to the
        # purged training dates. Running a separate backtest on discontiguous
        # dates created ghost returns across the splices (audit finding A1).
        is_ret = base_returns[base_returns.index.isin(train_dates_purged)]
        is_sr = sharpe(is_ret) if len(is_ret) > 20 else float("nan")

        # OOS: same full-run returns restricted to the test dates.
        oos_ret = base_returns[base_returns.index.isin(test_dates)]
        oos_sr = sharpe(oos_ret) if len(oos_ret) > 20 else float("nan")

        if np.isnan(oos_sr):
            continue

        paths.append(
            CPCVPath(
                combo_id=combo_id,
                train_fold_ids=list(train_fold_ids),
                test_fold_ids=list(test_fold_ids),
                oos_returns=oos_ret,
                oos_sharpe=oos_sr,
                is_sharpe=is_sr,
            )
        )
        logger.debug(
            f"  combo {combo_id:3d}: IS Sharpe={is_sr:.3f}  OOS Sharpe={oos_sr:.3f}"
        )

    if not paths:
        raise RuntimeError("CPCV produced no valid paths — check data length / purge_days")

    oos_sharpes = [p.oos_sharpe for p in paths]
    is_sharpes = [p.is_sharpe for p in paths if not np.isnan(p.is_sharpe)]

    median_is = float(np.median(is_sharpes)) if is_sharpes else 0.0
    pbo = float(np.mean([1.0 if s < median_is else 0.0 for s in oos_sharpes]))

    return CPCVResult(
        paths=paths,
        oos_sharpe_distribution=oos_sharpes,
        is_sharpe_distribution=is_sharpes,
        pbo=pbo,
        mean_oos_sharpe=float(np.mean(oos_sharpes)),
        std_oos_sharpe=float(np.std(oos_sharpes, ddof=1)) if len(oos_sharpes) > 1 else 0.0,
        median_oos_sharpe=float(np.median(oos_sharpes)),
        fraction_positive=float(np.mean([1.0 if s > 0 else 0.0 for s in oos_sharpes])),
        k=cfg.k,
        n_test=cfg.n_test,
        n_paths=len(paths),
    )


# ── Helpers ────────────────────────────────────────────────────────────────

def _split_into_groups(index: pd.DatetimeIndex, k: int) -> list[pd.DatetimeIndex]:
    n = len(index)
    size = n // k
    groups = []
    for i in range(k):
        start = i * size
        end = (i + 1) * size if i < k - 1 else n
        groups.append(index[start:end])
    return groups


def _merge_groups(groups: list[pd.DatetimeIndex], ids: list[int]) -> pd.DatetimeIndex:
    merged = pd.DatetimeIndex([])
    for i in ids:
        merged = merged.append(groups[i])
    return merged.sort_values()


def _safe_backtest(
    strategy: BaseStrategy,
    prices: pd.DataFrame,
    bt_cfg: BacktestConfig,
):
    try:
        bt = CrossSectionalBacktester(strategy, bt_cfg)
        return bt.run(prices)
    except Exception as e:
        logger.debug(f"Backtest failed: {e}")
        return None
