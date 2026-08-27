"""Central configuration. Values may be overridden via .env or environment."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="ST_", extra="ignore")

    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    parquet_dir: Path = PROJECT_ROOT / "data" / "parquet"

    # Capital and risk. See plan: the flat DP charge forces a minimum position size.
    starting_capital: float = 50_000.0
    risk_per_trade: float = 0.01
    min_position_value: float = 10_000.0
    max_concurrent_positions: int = 5

    # The explain-chatbot must never make a paid API call. Guarded in backend/chat.
    allow_paid_llm: bool = False

    def ensure_dirs(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
