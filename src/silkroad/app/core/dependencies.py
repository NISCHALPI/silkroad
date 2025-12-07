from typing import Generator
from silkroad.app.db.market_data import MarketDataDB
from silkroad.core.enums import Horizon
from silkroad.app.core.config import settings
from silkroad.logging.logger import logger

# Global instance to keep connection open
_db_instance: MarketDataDB | None = None


def get_db() -> Generator[MarketDataDB, None, None]:
    """
    Dependency that yields the MarketDataDB instance.
    Ensures a single instance is initialized and reused.
    """
    global _db_instance

    if _db_instance is None:
        logger.info("Initializing MarketDataDB...")
        logger.info(f"Using database path: {settings.DB_PATH}")
        _db_instance = MarketDataDB(db_path=settings.DB_PATH, horizon=Horizon.DAILY)

    yield _db_instance
