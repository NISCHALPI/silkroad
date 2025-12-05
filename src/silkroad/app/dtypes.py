"""Data types for Silkroad application."""

from pydantic import BaseModel, Field, model_validator
import typing as tp
from datetime import datetime
import numpy as np
from ..core.data_models import Asset


__all__ = ["TickerQueryParams", "ForcastParams", "RebalancingParams"]


class ForcastParams(BaseModel):
    timestamp: datetime = Field(..., description="Timestamp of the forcast")
    tickers: list[str] = Field(..., description="Tickers of the assets")
    expected_means: dict[str, float] = Field(
        ..., description="Expected means for rebalancing the portfolio"
    )
    expected_covariance: tp.List[tp.List[float]] = Field(
        ..., description="Expected covariance for rebalancing the portfolio"
    )

    @model_validator(mode="after")
    def check_tickers_and_means(self) -> "ForcastParams":
        if len(self.tickers) != len(self.expected_means):
            raise ValueError("Tickers and means must have the same length")
        if set(self.tickers) != set(self.expected_means.keys()):
            raise ValueError("Tickers and means must have the same keys")
        arr = np.array(self.expected_covariance)
        if arr.shape != (len(self.tickers), len(self.tickers)):
            raise ValueError("Covariance matrix must be square")
        return self


class RebalancingParams(BaseModel):
    timestamp: datetime = Field(..., description="Timestamp of the rebalancing")
    tickers: list[str] = Field(..., description="Tickers of the assets")
    weights: dict[str, float] = Field(..., description="Weights of the assets")

    @model_validator(mode="after")
    def check_weights_and_tkrs(self) -> "RebalancingParams":
        if len(self.tickers) != len(self.weights):
            raise ValueError("Tickers and weights must have the same length")
        if set(self.tickers) != set(self.weights.keys()):
            raise ValueError("Tickers and weights must have the same keys")
        if sum(self.weights.values()) != 1:
            raise ValueError("Weights must sum to 1")
        return self
