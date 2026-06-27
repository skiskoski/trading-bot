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
    cost_bps: float = 5.0  # one-way transaction cost in bps (slippage + spread)
    # Commissione FISSA per ordine (es. IBKR ~$0.35/ordine). A capitali piccoli
    # domina: su una posizione da €1.000, €0.35 = 3.5 bps; su €100 = 35 bps.
    # Modellata solo se capital_base è impostato — altrimenti il backtest resta
    # scale-invariant (comportamento storico invariato).
    #   capital_base       = capitale reale deployato (valuta del conto, es. USD)
    #   fixed_cost_per_trade = costo fisso per singolo ordine, stessa valuta
    capital_base: float | None = None
    fixed_cost_per_trade: float = 0.0
    # NOTE (audit M6): Sharpe is computed with rf=0 everywhere for internal
    # consistency; gates are calibrated on rf=0. No risk_free_rate knob —
    # it was dead config that suggested otherwise.
    # Optional point-in-time membership panel: DataFrame[date, symbol] -> bool
    # If provided, at each rebalance the strategy is restricted to symbols whose
    # most recent membership value at-or-before that date is True.
    membership: pd.DataFrame | None = None
    # Circuit breaker: if portfolio drawdown (from peak) exceeds this threshold,
    # scale all position weights by circuit_breaker_scale until recovery.
    # Set to 1.0 to disable.
    circuit_breaker_dd: float = 0.15       # trigger at -15% drawdown
    circuit_breaker_scale: float = 0.5     # scale positions to 50%
    # Volatility targeting (Moreira-Muir 2017 'Volatility-Managed Portfolios'):
    # scale total exposure to keep realized portfolio vol near vol_target.
    # None = disabled. Exposure capped at 1.0 (no leverage), floored at 0.2.
    vol_target: float | None = None        # e.g. 0.15 = 15% annualized
    vol_lookback: int = 63                 # days for realized-vol estimate
    # Adaptive rebalancing: monthly by default, but adds WEEKLY rebalances
    # whenever market realized vol (21d, annualized) exceeds vol_trigger.
    # Responds faster in stressed regimes without paying weekly costs in calm ones.
    adaptive_rebalance: bool = False
    vol_trigger: float = 0.25              # 25% annualized market vol
    # Daniel-Moskowitz (2016) 'Momentum Crashes': momentum crasha in modo
    # prevedibile nei panic state (mercato sotto SMA200 + alta volatilità,
    # tipicamente durante i rimbalzi). panic_filter scala l'esposizione
    # equity di panic_scale in quegli stati. Usa solo dati fino a t-1.
    panic_filter: bool = False
    panic_scale: float = 0.5
    # Novy-Marx-Velikov (2016) buy/hold band: soglia severa per ENTRARE
    # (top_n), blanda per MANTENERE (top_n × hold_band_mult). Un titolo
    # comprato resta in portafoglio finché è nel band — taglia il turnover
    # del 30-50% quasi senza perdere segnale. 1.0 = disattivo.
    hold_band_mult: float = 2.0


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

        # ── Adaptive rebalancing ─────────────────────────────────────────────
        # Add weekly (Monday) rebalances during high-vol regimes. Market vol is
        # measured with data up to each candidate date only — no look-ahead.
        if self.config.adaptive_rebalance:
            market = (
                daily_returns["SPY"] if "SPY" in daily_returns.columns
                else daily_returns.mean(axis=1)
            )
            mkt_vol = market.rolling(21).std() * np.sqrt(252)
            extra = [
                d for d in prices.index
                if d.weekday() == 0                      # Mondays
                and not pd.isna(mkt_vol.loc[d])
                and mkt_vol.loc[d] > self.config.vol_trigger
            ]
            rebalance_dates = sorted(set(rebalance_dates) | set(extra))

        weights_history: dict[pd.Timestamp, pd.Series] = {}
        rank_history: dict[pd.Timestamp, pd.Series] = {}
        prev_equity_hold: list[str] = []

        for dt in rebalance_dates:
            # Apply PIT membership mask if provided
            prices_t = self._apply_pit_mask(prices, dt)
            try:
                w = self.strategy.weights(prices_t, dt)
                r = self.strategy.rank(prices_t, dt)
            except Exception as e:
                logger.warning(f"strategy failed at {dt}: {e}")
                continue

            # ── Buy/hold band (Novy-Marx-Velikov 2016) ───────────────────
            # Keep already-held names while they remain inside the wider
            # hold band; only the empty slots get filled with new top names.
            if (self.config.hold_band_mult > 1.0 and not w.empty
                    and not r.empty):
                gld_w = float(w.get("GLD", 0.0))
                eq_syms = [s for s in w.index if s != "GLD"]
                top_n = len(eq_syms)
                if top_n > 0:
                    ranked = r.sort_values(ascending=False)
                    band = set(ranked.head(
                        int(top_n * self.config.hold_band_mult)).index)
                    keep = [s for s in prev_equity_hold
                            if s in band and s in ranked.index]
                    fill = [s for s in ranked.index
                            if s not in keep][: max(0, top_n - len(keep))]
                    sel = keep + fill
                    if sel:
                        eq_w = (1.0 - gld_w) / len(sel)
                        w = pd.Series(eq_w, index=sel)
                        if gld_w > 0:
                            w.loc["GLD"] = gld_w
            prev_equity_hold = [s for s in w.index if s != "GLD"]

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

        # Reindex to daily FIRST, then shift by one DAY: weights decided at the
        # close of rebalance day t are held from t+1. (Shifting weights_df
        # directly would lag by a full rebalance PERIOD — audit finding C1.)
        held_weights = (
            weights_df.reindex(prices.index).ffill().shift(1).fillna(0.0)
        )

        # ── Circuit breaker ───────────────────────────────────────────────────
        # At each rebalance, check if portfolio is in deep drawdown.
        # If so, scale weights down until recovery.
        # Uses running equity computed incrementally — no look-ahead.
        if self.config.circuit_breaker_dd < 1.0:
            held_weights = self._apply_circuit_breaker(
                held_weights, daily_returns,
                self.config.circuit_breaker_dd,
                self.config.circuit_breaker_scale,
            )

        # ── Panic filter (Daniel-Moskowitz 2016) ─────────────────────────────
        # Panic state al giorno t-1 → esposizione scalata al giorno t.
        if self.config.panic_filter and "SPY" in prices.columns:
            spy = prices["SPY"]
            sma200 = spy.rolling(200).mean()
            vol21 = spy.pct_change().rolling(21).std() * np.sqrt(252)
            panic = ((spy < sma200) & (vol21 > self.config.vol_trigger)).shift(1).fillna(False)
            held_weights = held_weights.mul(
                pd.Series(np.where(panic, self.config.panic_scale, 1.0),
                          index=prices.index), axis=0)

        # Daily portfolio return
        portfolio_returns = (held_weights * daily_returns).sum(axis=1)

        # ── Volatility targeting ─────────────────────────────────────────────
        # scale_t = clip(vol_target / realized_vol_{t-1}, 0.2, 1.0)
        # Uses only past returns (shifted by 1) — no look-ahead. No leverage.
        if self.config.vol_target is not None:
            realized = (
                portfolio_returns.rolling(self.config.vol_lookback).std()
                * np.sqrt(252)
            ).shift(1)
            scale = (self.config.vol_target / realized).clip(0.2, 1.0).fillna(1.0)
            portfolio_returns = portfolio_returns * scale
            held_weights = held_weights.mul(scale, axis=0)
            # Charge the daily turnover generated by exposure scaling itself
            # (audit M5): |Δscale| × gross exposure × one-way cost.
            gross = held_weights.abs().sum(axis=1)
            scale_turnover = scale.diff().abs().fillna(0.0) * gross
            portfolio_returns = portfolio_returns - scale_turnover * (
                self.config.cost_bps / 10_000.0
            )

        # Transaction cost at each rebalance
        cost_per_bp = self.config.cost_bps / 10_000.0
        turnovers = (
            weights_df.fillna(0.0).diff().abs().sum(axis=1).shift(1).reindex(prices.index).fillna(0.0)
        )
        cost = turnovers * cost_per_bp
        portfolio_returns = portfolio_returns - cost

        # ── Commissione fissa per ordine (IBKR & co.) ────────────────────────
        # Conta gli ORDINI per rebalance (nomi il cui peso cambia) e ne addebita
        # il costo fisso come frazione del conto. Il conto cresce nel tempo →
        # la commissione fissa pesa sempre meno (realistico). Niente circolarità:
        # il valore conto usa l'equity pre-costo-fisso shiftata di un giorno.
        if self.config.capital_base and self.config.fixed_cost_per_trade > 0:
            wdiff = weights_df.fillna(0.0).diff()
            if len(weights_df) > 0:
                wdiff.iloc[0] = weights_df.iloc[0].fillna(0.0)  # apertura = tutti ordini
            n_trades = (wdiff.abs() > 1e-6).sum(axis=1)
            n_trades_daily = (
                n_trades.shift(1).reindex(prices.index).fillna(0.0)
            )
            pre_mult = (1 + portfolio_returns).cumprod().shift(1).fillna(1.0)
            account_value = (self.config.capital_base * pre_mult).clip(lower=1.0)
            fixed_drag = (
                n_trades_daily * self.config.fixed_cost_per_trade
            ) / account_value
            portfolio_returns = portfolio_returns - fixed_drag

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

    def _apply_pit_mask(self, prices: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
        if self.config.membership is None:
            return prices
        membership = self.config.membership
        valid_dates = membership.index[membership.index <= asof]
        if len(valid_dates) == 0:
            return prices
        last_row = membership.loc[valid_dates[-1]]
        live_symbols = last_row[last_row.astype(bool)].index
        # Keep service symbols even if not index members: regime symbol,
        # SPY (regime/stress) and GLD (gold-sleeve strategy parameter).
        regime = getattr(getattr(self.strategy, "cfg", None), "regime", None)
        regime_sym = getattr(regime, "symbol", None)
        cols = list(live_symbols)
        for service in (regime_sym, "SPY", "GLD", "HYG"):
            if service and service in prices.columns and service not in cols:
                cols.append(service)
        return prices[[c for c in cols if c in prices.columns]]

    @staticmethod
    def _apply_circuit_breaker(
        held_weights: pd.DataFrame,
        daily_returns: pd.DataFrame,
        dd_threshold: float,
        scale: float,
    ) -> pd.DataFrame:
        """Scale down weights when rolling drawdown exceeds threshold.

        Computed day-by-day using only past equity — no look-ahead.
        Circuit breaker activates when drawdown > dd_threshold.
        Deactivates when equity recovers past the previous peak.
        """
        scaled = held_weights.copy()
        equity = 1.0
        peak = 1.0
        breaker_on = False

        for i, dt in enumerate(held_weights.index):
            # Check drawdown BEFORE this day's return (use previous state)
            dd = (peak - equity) / peak if peak > 0 else 0.0

            if dd >= dd_threshold:
                breaker_on = True
            if breaker_on and equity >= peak:
                breaker_on = False  # recovered

            if breaker_on:
                scaled.iloc[i] = held_weights.iloc[i] * scale

            # Update equity with today's return
            w = scaled.iloc[i]
            r = daily_returns.iloc[i] if i < len(daily_returns) else pd.Series(0.0)
            day_ret = float((w * r.reindex(w.index, fill_value=0.0)).sum())
            equity *= (1 + day_ret)
            if equity > peak:
                peak = equity

        return scaled

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
