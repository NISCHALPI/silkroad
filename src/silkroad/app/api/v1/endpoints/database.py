import typing as tp
import datetime as dt
from fastapi import APIRouter, Depends, HTTPException, Body
from silkroad.app.db.market_data import MarketDataDB
from silkroad.app.core.dependencies import get_db
from silkroad.app.schemas.db_query import (
    TickerDataQuery,
    FetchTickerDataQuery,
    RefreshQuery,
)
from silkroad.core.data_models import Asset
from silkroad.core.enums import DataBackend, Exchange
from silkroad.app.core.config import settings
from silkroad.logging.logger import logger

router = APIRouter()

# --- Symbols Endpoints ---


@router.get("/symbols", response_model=tp.List[str])
def get_symbols(db: MarketDataDB = Depends(get_db)) -> tp.List[str]:
    """Endpoint to retrieve available market symbols."""
    return db.get_available_tickers()


@router.post("/symbols")
def add_symbol(
    query: FetchTickerDataQuery,
    update_mode: tp.Literal["full_refresh", "smart_merge"] = "smart_merge",
    db: MarketDataDB = Depends(get_db),
) -> dict[str, str]:
    """Endpoint to add a new market symbol along with its data to the database."""
    if query.backend != DataBackend.ALPACA:
        raise HTTPException(
            status_code=501, detail=f"Backend {query.backend} is not supported yet."
        )

    from_dt = db.ALPACA_START_DATE  # type: ignore
    to_dt = query.asset.exchange.previous_market_close()

    if query.start_date is not None:
        from_dt = query.start_date.replace(tzinfo=dt.timezone.utc)
    if query.end_date is not None:
        to_dt = query.end_date.replace(tzinfo=dt.timezone.utc)

    logger.info(
        f"Adding symbol {query.asset.ticker} from {from_dt} to {to_dt} using backend {query.backend}."
    )

    try:
        db.update_from_alpaca(
            api_key=settings.ALPACA_API_KEY,
            api_secret=settings.ALPACA_API_SECRET,
            assets=[query.asset],
            start=from_dt,
            end=to_dt,
            update_mode=update_mode,
        )
    except Exception as e:
        logger.error(f"Error adding symbol {query.asset.ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": f"Symbol {query.asset.ticker} added successfully."}


@router.delete("/symbols/{ticker}")
def delete_ticker(ticker: str, db: MarketDataDB = Depends(get_db)) -> dict[str, str]:
    """Delete all data for a specific ticker."""
    try:
        db.delete_ticker(ticker)
        return {"message": f"Ticker {ticker} deleted successfully."}
    except Exception as e:
        logger.error(f"Error deleting ticker {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/symbols/{ticker}/latest")
def get_latest_bar(
    ticker: str, db: MarketDataDB = Depends(get_db)
) -> tp.Dict[str, tp.Any]:
    """Retrieve the most recent bar for a ticker."""
    try:
        bar = db.get_latest_bar(ticker)
        if bar is None:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
        return bar  # type: ignore
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting latest bar for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Market Data Endpoints ---


@router.get("/data")
def get_market_data(
    query: TickerDataQuery = Depends(),
    db: MarketDataDB = Depends(get_db),
) -> tp.Dict[dt.datetime, tp.Dict[str, float]]:
    """Endpoint to retrieve market data for a given symbol."""
    try:
        # Get available data range
        asd, aed = db.get_available_date_range(query.ticker)
        if query.start_date is not None:
            asd = query.start_date.replace(tzinfo=dt.timezone.utc)

        if query.end_date is not None:
            aed = query.end_date.replace(tzinfo=dt.timezone.utc)

        df = db.get_ticker_data(ticker=query.ticker, start=asd, end=aed).df
        return df.to_dict(orient="index")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ohlcv")
def get_ohlcv(
    query: TickerDataQuery = Depends(),
    db: MarketDataDB = Depends(get_db),
) -> tp.Dict[dt.datetime, tp.Dict[str, float]]:
    """Endpoint to retrieve OHLCV data for a given symbol."""
    try:
        # Get available data range
        asd, aed = db.get_available_date_range(query.ticker)
        if query.start_date is not None:
            asd = query.start_date.replace(tzinfo=dt.timezone.utc)

        if query.end_date is not None:
            aed = query.end_date.replace(tzinfo=dt.timezone.utc)

        df = db.get_ticker_data(ticker=query.ticker, start=asd, end=aed).df

        # Columns: open, high, low, close, volume
        df = df[["open", "high", "low", "close", "volume"]]
        return df.to_dict(orient="index")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/close")
def get_close(
    query: TickerDataQuery = Depends(),
    db: MarketDataDB = Depends(get_db),
) -> tp.Dict[dt.datetime, float]:
    """Endpoint to retrieve closing prices for a given symbol."""
    try:
        # Get available data range
        asd, aed = db.get_available_date_range(query.ticker)
        if query.start_date is not None:
            asd = query.start_date.replace(tzinfo=dt.timezone.utc)

        if query.end_date is not None:
            aed = query.end_date.replace(tzinfo=dt.timezone.utc)

        df = db.get_ticker_data(ticker=query.ticker, start=asd, end=aed).df
        # Columns: close
        df = df["close"]
        return df.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Database Operations Endpoints ---


@router.post("/refresh")
def refresh_database(
    query: RefreshQuery, db: MarketDataDB = Depends(get_db)
) -> dict[str, str]:
    """Refresh database by updating assets to the latest available date."""
    try:
        db.refresh(
            api_key=settings.ALPACA_API_KEY,
            api_secret=settings.ALPACA_API_SECRET,
            assets=query.assets,
            end=query.end_date,
        )
        return {"message": "Database refresh completed successfully."}
    except Exception as e:
        logger.error(f"Error refreshing database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/reset")
def reset_database(db: MarketDataDB = Depends(get_db)) -> dict[str, str]:
    """Reset the database by deleting all stored data. Irreversible."""
    try:
        db.reset()
        return {"message": "Database reset successfully."}
    except Exception as e:
        logger.error(f"Error resetting database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
def get_summary(db: MarketDataDB = Depends(get_db)) -> tp.List[tp.Dict[str, tp.Any]]:
    """Generate a summary of all tickers in the database."""
    try:
        df = db.summary()
        # Reset index to include ticker in the dict
        return df.reset_index().to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))
