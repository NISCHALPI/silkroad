from pydantic import BaseModel, Field
import typing as tp
import datetime as dt
import riskfolio as rp
from enum import Enum
from silkroad.core.enums import Horizon

__all__ = ["CovEstimationQuery", "MeanEstimationQuery"]


class MeanMethod(Enum):
    HIST = "hist"
    SEMI = "semi"
    EWMA1 = "ewma1"
    EWMA2 = "ewma2"
    JS = "JS"
    BS = "BS"
    BOP = "BOP"


class CovMethod(Enum):
    HIST = "hist"
    SEMI = "semi"
    EWMA1 = "ewma1"
    EWMA2 = "ewma2"
    LEDOIT = "ledoit"
    OAS = "oas"
    SHRUNK = "shrunk"
    GL = "gl"
    JLOGO = "jlogo"
    FIXED = "fixed"
    SPECTRAL = "spectral"
    SHRINK = "shrink"
    GERBER1 = "gerber1"
    GERBER2 = "gerber2"


class MeanEstimationQuery(BaseModel):
    tickers: tp.List[str] = Field(..., description="List of tickers")
    method: MeanMethod = Field(
        MeanMethod.HIST,
        description="Expected Airthmetic Return estimation method for collection of assets",
    )
    start: tp.Optional[dt.datetime] = Field(
        None, description="Start date of the data in UTC"
    )
    end: tp.Optional[dt.datetime] = Field(
        None, description="End date of the data in UTC"
    )
    horizon: Horizon = Field(Horizon.DAILY, description="Horizon/Frequency of the data")


class CovEstimationQuery(BaseModel):
    tickers: tp.List[str] = Field(
        ..., min_length=2, description="List of tickers with at least two tickers"
    )  # type: ignore
    method: CovMethod = Field(
        CovMethod.HIST,
        description="Covariance estimation method for collection of assets",
    )
    start: tp.Optional[dt.datetime] = Field(
        None, description="Start date of the data in UTC"
    )
    end: tp.Optional[dt.datetime] = Field(
        None, description="End date of the data in UTC"
    )
    horizon: Horizon = Field(Horizon.DAILY, description="Horizon/Frequency of the data")
