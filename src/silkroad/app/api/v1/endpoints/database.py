"""Database Endpoints"""

import typing as tp
import datetime as dt
from fastapi import APIRouter, Depends, HTTPException, Body
from silkroad.app.core.dependencies import get_db
from silkroad.db import ArcticDatabase
from silkroad.core.enums import Horizon, Sector, Exchange
from silkroad.core.data_models import Asset
from silkroad.app.schemas.database import (
    DBReadQuery,
    DBBatchReadQuery,
    DBSyncQuery,
    DBSymbolFilterQuery,
    DBPruneHistoryQuery,
    DBDeleteSymbolQuery,
)
import pandas as pd
from silkroad.logging.logger import logger

router = APIRouter()


@router.post("/sync")
def sync_market_data(
    query: DBSyncQuery, db: ArcticDatabase = Depends(get_db)
) -> tp.Dict[str, str]:
    """Sync market data for a list of assets."""
    try:
        db.sync(
            assets=query.assets,
            horizon=query.horizon,
            start_date=query.start_date,
            end_date=query.end_date,
            lookback_buffer=query.lookback_buffer,
        )
        return {
            "status": "success",
            "message": f"Synced {len(query.assets)} assets with horizon {query.horizon}",
        }
    except Exception as e:
        logger.error(f"Sync failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/read")
def read_data(
    query: DBReadQuery, db: ArcticDatabase = Depends(get_db)
) -> tp.List[tp.Dict[str, tp.Any]]:
    """Read data for a symbol as records."""
    df = db.read(
        symbol=query.symbol,
        horizon=query.horizon,
        start=query.start,
        end=query.end,
        columns=query.columns,
        as_of=query.as_of,
    )
    if df.empty:
        return []

    # Reset index to include timestamp in the result
    return df.reset_index().to_dict(orient="records")  # type: ignore


@router.post("/read/barset")
def read_barset(
    query: DBReadQuery, db: ArcticDatabase = Depends(get_db)
) -> tp.Dict[str, tp.Any]:
    """Read data as a UniformBarSet (serialized)."""
    try:
        bs = db.read_into_uniform_barset(
            symbol=query.symbol,
            horizon=query.horizon,
            start=query.start,
            end=query.end,
            columns=query.columns,
            as_of=query.as_of,
        )
        return bs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Read barset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/read/collection")
def read_collection(
    query: DBBatchReadQuery, db: ArcticDatabase = Depends(get_db)
) -> tp.Dict[str, tp.Any]:
    """Read data for multiple symbols as a UniformBarCollection (serialized)."""
    try:
        collection = db.read_into_uniform_barcollection(
            symbols=query.symbols,
            horizon=query.horizon,
            start=query.start,
            end=query.end,
            columns=query.columns,
            as_of=query.as_of,
        )
        return collection.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Read collection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/filter")
def filter_symbols(
    query: DBSymbolFilterQuery, db: ArcticDatabase = Depends(get_db)
) -> tp.List[str]:
    """Filter symbols by metadata."""
    criteria = {}
    if query.sector:
        criteria["sector"] = query.sector
    if query.exchange:
        criteria["exchange"] = query.exchange

    return db.filter_symbols(query.horizon, **criteria)


@router.get("/symbols")
def list_symbols(
    horizon: Horizon, db: ArcticDatabase = Depends(get_db)
) -> tp.List[str]:
    """List all symbols in a horizon."""
    return db.list_symbols(horizon)


@router.delete("/symbol")
def delete_symbol(
    query: DBDeleteSymbolQuery, db: ArcticDatabase = Depends(get_db)
) -> tp.Dict[str, str]:
    """Delete a symbol from the database."""
    try:
        db.delete(query.symbol, query.horizon)
        return {"status": "success", "message": f"Deleted {query.symbol}"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/prune")
def prune_history(
    query: DBPruneHistoryQuery, db: ArcticDatabase = Depends(get_db)
) -> tp.Dict[str, str]:
    """Prune version history for a symbol."""
    try:
        db.prune_history(query.symbol, query.horizon, query.max_versions)
        return {"status": "success", "message": f"Pruned history for {query.symbol}"}
    except Exception as e:
        logger.error(f"Prune failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
def get_summary(
    horizon: Horizon, db: ArcticDatabase = Depends(get_db)
) -> tp.List[tp.Dict[str, tp.Any]]:
    """Get a summary of the database library."""
    df = db.summary(horizon)
    if df.empty:
        return []
    return df.reset_index().to_dict(orient="records")  # type: ignore
