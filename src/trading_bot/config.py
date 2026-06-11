from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRADEBOT_", env_file=".env", extra="ignore")

    data_dir: Path = PROJECT_ROOT / "data"
    db_path: Path = PROJECT_ROOT / "data" / "trading_bot.db"
    start_date: str = "2010-01-01"
    risk_free_rate: float = 0.04
    log_level: str = "INFO"

    # Alpaca paper trading
    alpaca_api_key: str = "YOUR_ALPACA_API_KEY"
    alpaca_secret_key: str = "YOUR_ALPACA_SECRET_KEY"
    alpaca_base_url: str = "https://paper-api.alpaca.markets"

    # Telegram notifications
    telegram_bot_token: str = "YOUR_TELEGRAM_BOT_TOKEN"
    telegram_chat_id: str = "YOUR_TELEGRAM_CHAT_ID"

    # Holdout period — NEVER used during training
    holdout_start: str = "2024-01-01"

    # Portfolio overlays (applied at live/portfolio level, not in signal research)
    gold_sleeve_weight: float = 0.20      # TSMOM gold sleeve (0 = disabled)
    # 0.25 = Sharpe-optimal in the 2015-2026 A/B sweep (1.600 vs 1.527 off);
    # keeps 84% of raw CAGR while cutting MaxDD. Growth-oriented setting.
    vol_target: float = 0.25              # annualized vol target (0 = disabled)
    adaptive_rebalance: bool = True       # weekly rebalance when market vol > trigger
    vol_trigger: float = 0.25             # annualized market vol trigger
    panic_scale: float = 0.5              # equity scale in panic states (DM 2016)

    @property
    def db_url(self) -> str:
        return f"sqlite:///{self.db_path}"


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
