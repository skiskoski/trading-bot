"""Telegram notification system.

Sends messages to a Telegram chat when:
- A strategy is promoted (★)
- Daily signals are generated
- A rebalancing is executed
- Drawdown exceeds a threshold
- Critical errors occur

Configuration (add to .env):
    TRADEBOT_TELEGRAM_BOT_TOKEN=123456:ABC-xxx
    TRADEBOT_TELEGRAM_CHAT_ID=987654321

How to get these:
    1. Open Telegram, search @BotFather
    2. Send /newbot → follow instructions → copy the TOKEN
    3. Send any message to your new bot
    4. Open: https://api.telegram.org/bot<TOKEN>/getUpdates
       Look for "chat": {"id": <YOUR_CHAT_ID>}
    5. Add both to .env
"""

from __future__ import annotations

import traceback
from datetime import date
from enum import Enum

import requests

from trading_bot.utils.logging import get_logger

logger = get_logger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class NotifyLevel(Enum):
    INFO    = "ℹ️"
    SUCCESS = "✅"
    WARNING = "⚠️"
    ERROR   = "🔴"
    TRADE   = "📊"
    PROMOTE = "⭐"


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.token = bot_token
        self.chat_id = chat_id
        self._url = TELEGRAM_API.format(token=bot_token)

    def send(self, text: str, level: NotifyLevel = NotifyLevel.INFO) -> bool:
        """Send a message. Returns True on success."""
        msg = f"{level.value} *TradingBot*\n{text}"
        try:
            r = requests.post(
                self._url,
                json={"chat_id": self.chat_id, "text": msg, "parse_mode": "Markdown"},
                timeout=10,
            )
            r.raise_for_status()
            return True
        except Exception as e:
            logger.warning(f"Telegram send failed: {e}")
            return False

    # ── Typed message helpers ─────────────────────────────────────────────────

    def notify_signals(
        self,
        strategy_name: str,
        asof: date,
        positions: dict[str, float],
        regime_active: bool | None = None,
    ) -> None:
        lines = [
            f"📅 *Signals — {asof}*",
            f"Strategy: `{strategy_name}`",
        ]
        if regime_active is not None:
            lines.append(f"Regime filter: {'🟢 ACTIVE' if regime_active else '🔴 OFF'}")
        lines.append("")
        if positions:
            lines.append("*Target positions:*")
            for sym, w in sorted(positions.items(), key=lambda x: -x[1]):
                bar = "█" * int(w * 20)
                lines.append(f"  `{sym:<6}` {w*100:5.1f}%  {bar}")
        else:
            lines.append("_No positions — regime filter may be blocking trades_")
        self.send("\n".join(lines), NotifyLevel.TRADE)

    def notify_rebalance(
        self,
        orders: list,
        portfolio_value: float,
        strategy_name: str,
    ) -> None:
        n_orders = len(orders)
        buys  = [o for o in orders if o.side == "buy"]
        sells = [o for o in orders if o.side == "sell"]
        lines = [
            f"🔄 *Rebalance executed — {strategy_name}*",
            f"Portfolio: ${portfolio_value:,.0f}",
            f"Orders: {n_orders} ({len(buys)} buys, {len(sells)} sells)",
        ]
        if buys:
            lines.append("*Buys:* " + ", ".join(f"`{o.symbol}`" for o in buys))
        if sells:
            lines.append("*Sells:* " + ", ".join(f"`{o.symbol}`" for o in sells))
        self.send("\n".join(lines), NotifyLevel.TRADE)

    def notify_promotion(
        self,
        strategy_name: str,
        oos_sharpe: float,
        pbo: float,
        dsr: float,
    ) -> None:
        lines = [
            f"⭐ *Strategy PROMOTED: `{strategy_name}`*",
            f"OOS Sharpe: `{oos_sharpe:.3f}`",
            f"PBO: `{pbo:.3f}` (< 0.5 ✅)",
            f"DSR: `{dsr:.3f}` (≥ 0.95 ✅)",
            "",
            "_This strategy passed all statistical gates and is ready for paper trading._",
        ]
        self.send("\n".join(lines), NotifyLevel.PROMOTE)

    def notify_drawdown_alert(
        self,
        current_dd: float,
        threshold: float,
        portfolio_value: float,
    ) -> None:
        lines = [
            f"⚠️ *Drawdown alert*",
            f"Current drawdown: `{current_dd*100:.1f}%`",
            f"Threshold: `{threshold*100:.0f}%`",
            f"Portfolio value: `${portfolio_value:,.0f}`",
            "",
            "_Consider reducing position sizes or activating regime filter._",
        ]
        self.send("\n".join(lines), NotifyLevel.WARNING)

    def notify_error(self, context: str, exc: Exception) -> None:
        tb = traceback.format_exc()[-500:]
        lines = [
            f"🔴 *Error in: {context}*",
            f"`{type(exc).__name__}: {exc}`",
            "",
            f"```\n{tb}\n```",
        ]
        self.send("\n".join(lines), NotifyLevel.ERROR)

    def notify_research_round(
        self,
        round_id: int,
        tested: int,
        promoted: int,
        best_sharpe: float,
    ) -> None:
        lines = [
            f"🔬 *Research round {round_id} complete*",
            f"Tested: {tested} strategies",
            f"Promoted: {promoted}",
            f"Best OOS Sharpe this round: `{best_sharpe:.3f}`",
        ]
        self.send("\n".join(lines), NotifyLevel.INFO)


def get_notifier() -> TelegramNotifier | None:
    """Build TelegramNotifier from env. Returns None if not configured."""
    from trading_bot.config import settings

    token = getattr(settings, "telegram_bot_token", None)
    chat_id = getattr(settings, "telegram_chat_id", None)

    if not token or not chat_id or token == "YOUR_TELEGRAM_BOT_TOKEN":
        logger.info("Telegram not configured — notifications disabled.")
        return None

    return TelegramNotifier(bot_token=token, chat_id=str(chat_id))


def notify(text: str, level: NotifyLevel = NotifyLevel.INFO) -> None:
    """Fire-and-forget notification. Safe to call even without config."""
    n = get_notifier()
    if n:
        n.send(text, level)
