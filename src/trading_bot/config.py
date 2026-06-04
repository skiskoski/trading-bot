from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRADEBOT_", env_file=".env", extra="ignore")

    data_dir: Path = PROJECT_ROOT / "data"
    db_path: Path = PROJECT_ROOT / "data" / "trading_bot.db"
    universe_size: int = 100
    start_date: str = "2010-01-01"
    risk_free_rate: float = 0.04
    log_level: str = "INFO"

    @property
    def db_url(self) -> str:
        return f"sqlite:///{self.db_path}"


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
