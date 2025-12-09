from pydantic import BaseModel, Field
import typing as tp
import datetime as dt
from silkroad.core.data_models import Asset
from silkroad.core.enums import Horizon, Sector, Exchange


class DBReadQuery(BaseModel):
    symbol: str = Field(..., description="The ticker symbol to retrieve data for.")
    horizon: Horizon = Field(
        ..., description="The time horizon/library containing the data."
    )
    start: tp.Optional[dt.datetime] = Field(None, description="Start date (inclusive).")
    end: tp.Optional[dt.datetime] = Field(None, description="End date (inclusive).")
    columns: tp.Optional[tp.List[str]] = Field(
        None, description="Specific columns to retrieve."
    )
    as_of: tp.Optional[tp.Union[int, str, dt.datetime]] = Field(
        None, description="Version specifier for point-in-time queries."
    )


class DBBatchReadQuery(BaseModel):
    symbols: tp.List[str] = Field(..., description="List of ticker symbols to load.")
    horizon: Horizon = Field(..., description="The time horizon/library.")
    start: tp.Optional[dt.datetime] = Field(None, description="Start date (inclusive).")
    end: tp.Optional[dt.datetime] = Field(None, description="End date (inclusive).")
    columns: tp.Optional[tp.List[str]] = Field(
        None, description="Specific columns to retrieve."
    )
    as_of: tp.Optional[tp.Union[int, str, dt.datetime]] = Field(
        None, description="Version identifier."
    )


class DBSyncQuery(BaseModel):
    assets: tp.List[Asset] = Field(..., description="List of assets to sync.")
    horizon: Horizon = Field(..., description="The time horizon to sync data for.")
    start_date: tp.Optional[dt.datetime] = Field(
        None, description="Start date for sync."
    )
    end_date: tp.Optional[dt.datetime] = Field(None, description="End date for sync.")
    lookback_buffer: int = Field(
        5, description="Lookback buffer for corporate action detection."
    )


class DBSymbolFilterQuery(BaseModel):
    horizon: Horizon = Field(..., description="The time horizon to search.")
    sector: tp.Optional[Sector] = Field(None, description="Filter by sector.")
    exchange: tp.Optional[Exchange] = Field(None, description="Filter by exchange.")


class DBPruneHistoryQuery(BaseModel):
    symbol: str = Field(..., description="The ticker symbol to prune.")
    horizon: Horizon = Field(..., description="The time horizon/library.")
    max_versions: int = Field(10, description="Version count threshold.")


class DBDeleteSymbolQuery(BaseModel):
    symbol: str = Field(..., description="The ticker symbol to delete.")
    horizon: Horizon = Field(..., description="The time horizon/library.")
