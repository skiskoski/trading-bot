import logging

from rich.logging import RichHandler

from trading_bot.config import settings

_configured = False


def get_logger(name: str) -> logging.Logger:
    global _configured
    if not _configured:
        logging.basicConfig(
            level=settings.log_level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        )
        logging.getLogger("yfinance").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("peewee").setLevel(logging.WARNING)
        _configured = True
    return logging.getLogger(name)
