from pydantic import BaseModel, Field
import typing as tp
from silkroad.core.data_models import Asset
from silkroad.core.enums import DataBackend
import datetime as dt


__all__ = ["TickerDataQuery", "FetchTickerDataQuery", "TickerCollectionDataQuery"]


class TickerDataQuery(BaseModel):
    ticker: str = Field(..., description="Ticker of the asset")
    start_date: tp.Optional[dt.datetime] = Field(
        None, description="Start date of the data in UTC"
    )
    end_date: tp.Optional[dt.datetime] = Field(
        None, description="End date of the data in UTC"
    )


class TickerCollectionDataQuery(BaseModel):
    tickers: tp.List[str] = Field(..., description="List of tickers")
    start_date: tp.Optional[dt.datetime] = Field(
        None, description="Start date of the data in UTC"
    )
    end_date: tp.Optional[dt.datetime] = Field(
        None, description="End date of the data in UTC"
    )


class FetchTickerDataQuery(BaseModel):
    asset: Asset = Field(..., description="Asset to fetch data for")
    start_date: tp.Optional[dt.datetime] = Field(
        None, description="Start date of the data in UTC"
    )
    end_date: tp.Optional[dt.datetime] = Field(
        None, description="End date of the data in UTC"
    )
    backend: DataBackend = Field(DataBackend.ALPACA, description="Data backend to use")


class RefreshQuery(BaseModel):
    assets: tp.List[Asset] = Field(..., description="List of assets to refresh")
    end_date: tp.Optional[dt.datetime] = Field(
        None, description="End date for refresh in UTC"
    )
