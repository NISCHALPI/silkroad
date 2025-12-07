import json
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class Settings(BaseModel):
    ALPACA_API_KEY: str
    ALPACA_API_SECRET: str
    DB_PATH: Path = Field(default_factory=lambda: Path().home() / ".market_data")
    ALPACA_API_PATH: Path = Field(
        default_factory=lambda: Path().home() / ".alpaca.json"
    )

    @classmethod
    def load(cls) -> "Settings":
        alpaca_api_path = Path().home() / ".alpaca.json"

        if not alpaca_api_path.exists():
            raise FileNotFoundError(
                f"Alpaca API key file not found at {alpaca_api_path}"
            )

        with open(alpaca_api_path, "r") as f:
            alpaca_keys = json.load(f)

        return cls(
            ALPACA_API_KEY=alpaca_keys["key"],
            ALPACA_API_SECRET=alpaca_keys["secret"],
            ALPACA_API_PATH=alpaca_api_path,
        )


settings = Settings.load()
