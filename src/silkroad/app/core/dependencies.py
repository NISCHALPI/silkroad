from typing import Generator
from silkroad.db import AlpacaBackendProvider, ArcticDatabase
from silkroad.core.enums import Horizon
from silkroad.app.core.config import settings
from silkroad.logging.logger import logger

# Global instance to keep connection open
_db_instance: ArcticDatabase | None = None


def get_db() -> Generator[ArcticDatabase, None, None]:
    """
    Dependency that yields the MarketDataDB instance.
    Ensures a single instance is initialized and reused.
    """
    global _db_instance

    if _db_instance is None:
        logger.info("Initializing MarketDataDB...")
        logger.info(f"Using database path: {settings.DB_PATH}")
        uri = "lmdb://" + str(settings.DB_PATH.resolve())
        _db_instance = ArcticDatabase(
            uri=uri,
            backend=AlpacaBackendProvider(
                api_key=settings.ALPACA_API_KEY,
                api_secret=settings.ALPACA_API_SECRET,
            ),
        )

    yield _db_instance
