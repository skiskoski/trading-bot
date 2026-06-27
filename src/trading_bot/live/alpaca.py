"""Alpaca paper-trading client.

Handles order submission, position reconciliation, and account info.
Uses the REST API v2 — no third-party SDK dependency.

Configuration (add to .env):
    TRADEBOT_ALPACA_API_KEY=PKxxx
    TRADEBOT_ALPACA_SECRET_KEY=xxx
    TRADEBOT_ALPACA_BASE_URL=https://paper-api.alpaca.markets  # paper by default

Paper trading endpoint: https://paper-api.alpaca.markets
Live trading endpoint:  https://api.alpaca.markets  ← DO NOT USE until ready
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import requests

from trading_bot.utils.logging import get_logger

logger = get_logger(__name__)

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL  = "https://api.alpaca.markets"   # guarded — never default


@dataclass
class AlpacaPosition:
    symbol: str
    qty: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float   # percent
    current_price: float
    cost_basis: float


@dataclass
class AlpacaAccount:
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    currency: str = "USD"


@dataclass
class AlpacaOrder:
    id: str
    symbol: str
    side: str        # "buy" | "sell"
    qty: float
    status: str
    filled_avg_price: float | None
    submitted_at: str


class AlpacaClient:
    """Thin wrapper around Alpaca REST API v2 for paper trading."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        base_url: str = PAPER_URL,
    ):
        # Normalize BEFORE comparing — trailing slash / case must not bypass
        # the guard (audit finding M4).
        if base_url.rstrip("/").lower() == LIVE_URL:
            raise ValueError(
                "Live trading endpoint detected. "
                "This bot is paper-trading only. Set base_url to PAPER_URL."
            )
        self.base_url = base_url.rstrip("/")
        self._headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
            "Content-Type": "application/json",
        }

    def _get(self, path: str, params: dict | None = None) -> Any:
        r = requests.get(f"{self.base_url}{path}", headers=self._headers, params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict) -> Any:
        r = requests.post(f"{self.base_url}{path}", headers=self._headers,
                          data=json.dumps(body), timeout=15)
        r.raise_for_status()
        return r.json()

    def _delete(self, path: str) -> None:
        r = requests.delete(f"{self.base_url}{path}", headers=self._headers, timeout=15)
        r.raise_for_status()

    # ── Account ──────────────────────────────────────────────────────────────

    def get_account(self) -> AlpacaAccount:
        d = self._get("/v2/account")
        return AlpacaAccount(
            equity=float(d["equity"]),
            cash=float(d["cash"]),
            buying_power=float(d["buying_power"]),
            portfolio_value=float(d["portfolio_value"]),
        )

    def is_market_open(self) -> bool:
        d = self._get("/v2/clock")
        return bool(d.get("is_open", False))

    # ── Positions ─────────────────────────────────────────────────────────────

    def get_positions(self) -> list[AlpacaPosition]:
        positions = self._get("/v2/positions")
        return [
            AlpacaPosition(
                symbol=p["symbol"],
                qty=float(p["qty"]),
                market_value=float(p["market_value"]),
                unrealized_pl=float(p["unrealized_pl"]),
                unrealized_plpc=float(p["unrealized_plpc"]),
                current_price=float(p["current_price"]),
                cost_basis=float(p["cost_basis"]),
            )
            for p in positions
        ]

    def get_position(self, symbol: str) -> AlpacaPosition | None:
        try:
            p = self._get(f"/v2/positions/{symbol}")
            return AlpacaPosition(
                symbol=p["symbol"],
                qty=float(p["qty"]),
                market_value=float(p["market_value"]),
                unrealized_pl=float(p["unrealized_pl"]),
                unrealized_plpc=float(p["unrealized_plpc"]),
                current_price=float(p["current_price"]),
                cost_basis=float(p["cost_basis"]),
            )
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise

    def close_all_positions(self) -> None:
        """Close all open positions (use with care)."""
        self._delete("/v2/positions")
        logger.warning("All positions closed.")

    # ── Orders ────────────────────────────────────────────────────────────────

    def submit_order(
        self,
        symbol: str,
        qty: float | None,
        side: str,            # "buy" | "sell"
        order_type: str = "market",
        time_in_force: str = "day",
        notional: float | None = None,
    ) -> AlpacaOrder:
        """Submit an order by share qty OR dollar notional (exactly one).

        Notional orders remove any need to estimate prices client-side —
        Alpaca sizes the order at execution price (audit finding C3).
        """
        if (qty is None) == (notional is None):
            raise ValueError("Provide exactly one of qty or notional.")
        body = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        if notional is not None:
            body["notional"] = str(round(notional, 2))
        else:
            body["qty"] = str(round(qty, 4))
        o = self._post("/v2/orders", body)
        logger.info(f"Order submitted: {side} {qty:.4f} {symbol}  id={o['id']}")
        return AlpacaOrder(
            id=o["id"],
            symbol=o["symbol"],
            side=o["side"],
            qty=float(o["qty"]) if o.get("qty") else 0.0,  # notional orders: qty filled at execution
            status=o["status"],
            filled_avg_price=float(o["filled_avg_price"]) if o.get("filled_avg_price") else None,
            submitted_at=o.get("submitted_at", ""),
        )

    def get_orders(self, status: str = "open") -> list[AlpacaOrder]:
        orders = self._get("/v2/orders", params={"status": status, "limit": 100})
        return [
            AlpacaOrder(
                id=o["id"],
                symbol=o["symbol"],
                side=o["side"],
                qty=float(o["qty"]),
                status=o["status"],
                filled_avg_price=float(o["filled_avg_price"]) if o.get("filled_avg_price") else None,
                submitted_at=o.get("submitted_at", ""),
            )
            for o in orders
        ]

    def cancel_all_orders(self) -> None:
        self._delete("/v2/orders")

    # ── Rebalancing ───────────────────────────────────────────────────────────

    def rebalance(
        self,
        target_weights: dict[str, float],
        min_trade_pct: float = 0.01,
        dry_run: bool = False,
    ) -> list[AlpacaOrder]:
        """Execute a full portfolio rebalance.

        Args:
            target_weights: {symbol: weight} — weights should sum to ≤ 1.0.
            min_trade_pct: skip trades smaller than this fraction of portfolio.
            dry_run: if True, log orders but don't submit.

        Returns:
            List of submitted AlpacaOrder objects.
        """
        account = self.get_account()
        portfolio_value = account.portfolio_value
        logger.info(
            f"Rebalancing portfolio. Value=${portfolio_value:,.0f}  "
            f"targets={list(target_weights.keys())}"
        )

        # Current positions
        current_positions = {p.symbol: p for p in self.get_positions()}
        current_weights: dict[str, float] = {
            sym: pos.market_value / portfolio_value
            for sym, pos in current_positions.items()
        }

        # All symbols involved
        all_symbols = set(target_weights) | set(current_weights)

        orders_submitted: list[AlpacaOrder] = []

        # Sells first (free up cash), then buys
        sells = []
        buys = []

        for sym in all_symbols:
            target_w = target_weights.get(sym, 0.0)
            current_w = current_weights.get(sym, 0.0)
            delta_w = target_w - current_w

            if abs(delta_w) < min_trade_pct:
                continue

            delta_value = delta_w * portfolio_value
            if delta_value < 0:
                sells.append((sym, abs(delta_value)))
            else:
                buys.append((sym, delta_value))

        # Notional (dollar) orders: Alpaca sizes shares at execution price.
        # No client-side price estimation — the old fallback estimate produced
        # wildly wrong sizes for NEW positions (audit finding C3).
        for sym, value in sells:
            self._execute_leg(sym, value, "sell", current_positions,
                              orders_submitted, dry_run)
        for sym, value in buys:
            self._execute_leg(sym, value, "buy", current_positions,
                              orders_submitted, dry_run)

        logger.info(f"Rebalance complete. {len(orders_submitted)} orders submitted.")
        return orders_submitted

    def _execute_leg(
        self,
        sym: str,
        value: float,
        side: str,
        current_positions: dict,
        orders_submitted: list,
        dry_run: bool,
    ) -> None:
        """Submit one rebalance leg as a notional order (or full-position sell)."""
        if value < 1.0:    # Alpaca minimum notional is $1
            return
        # Full exit: selling (almost) the whole position → use qty to avoid
        # notional rounding leaving dust shares.
        pos = current_positions.get(sym)
        full_exit = (
            side == "sell" and pos is not None
            and value >= pos.market_value * 0.98
        )
        label = f"{side.upper()} {sym} ${value:,.0f}" + (" (full exit)" if full_exit else "")
        logger.info(f"  {'[DRY RUN] ' if dry_run else ''}{label}")
        if dry_run:
            return
        try:
            if full_exit:
                order = self.submit_order(sym, qty=pos.qty, side="sell")
            else:
                order = self.submit_order(sym, qty=None, side=side, notional=value)
            orders_submitted.append(order)
            time.sleep(0.1)  # respect rate limits
        except Exception as e:
            logger.error(f"Order failed for {sym}: {e}")

    # ── Portfolio history ─────────────────────────────────────────────────────

    def get_portfolio_history(
        self,
        period: str = "1M",
        timeframe: str = "1D",
    ) -> dict:
        """Get historical portfolio equity curve from Alpaca.

        period: "1D", "1W", "1M", "3M", "6M", "1A", "all"
        timeframe: "1Min", "5Min", "15Min", "1H", "1D"
        """
        return self._get(
            "/v2/account/portfolio/history",
            params={"period": period, "timeframe": timeframe, "extended_hours": False},
        )


def get_alpaca_client() -> AlpacaClient | None:
    """Build AlpacaClient from env settings. Returns None if keys not configured."""
    from trading_bot.config import settings

    api_key = getattr(settings, "alpaca_api_key", None)
    secret_key = getattr(settings, "alpaca_secret_key", None)
    base_url = getattr(settings, "alpaca_base_url", PAPER_URL)

    if not api_key or not secret_key or api_key == "YOUR_ALPACA_API_KEY":
        logger.warning(
            "Alpaca API keys not configured. "
            "Add TRADEBOT_ALPACA_API_KEY and TRADEBOT_ALPACA_SECRET_KEY to .env"
        )
        return None

    return AlpacaClient(api_key=api_key, secret_key=secret_key, base_url=base_url)
