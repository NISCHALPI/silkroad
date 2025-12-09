"""Covariance Estimation Endpoints"""

import typing as tp
from fastapi import APIRouter, Depends, HTTPException
from silkroad.app.core.dependencies import get_db
from silkroad.app.schemas.rfolio_estimation_query import (
    CovEstimationQuery,
    MeanEstimationQuery,
)
from silkroad.db import ArcticDatabase
import pandas as pd
import riskfolio as rp
import datetime as dt
import numpy as np
from silkroad.logging.logger import logger

router = APIRouter()


def to_symmetric_positive_definite_matrix(
    matrix: np.ndarray,
    threshold: float = 1e-7,
) -> np.ndarray:
    """Coerce a matrix to be symmetric and positive definite.

    Diagonalize the matrix and clip negative eigenvalues for numerical stability.

    Args:
        matrix (np.ndarray): Input matrix
        threshold (float, optional): Threshold for eigenvalues. Defaults to 1e-7.

    Returns:
        np.ndarray: Coerced matrix
    """
    matrix = (matrix + matrix.T) / 2

    # Diagonalize
    D, V = np.linalg.eigh(matrix)
    D = np.diag(D)

    # Clip negative eigenvalues
    D = np.clip(D, threshold, None)

    # Reconstruct the matrix
    matrix = V @ D @ V.T

    return matrix


@router.post("/estimate_cov")
def estimate_cov(
    query: CovEstimationQuery, db: ArcticDatabase = Depends(get_db)
) -> tp.Dict[str, tp.Dict[str, float]]:
    """Estimate covariance matrix for a given collection of tickers

    Args:
        query (CovEstimationQuery): Query for covariance estimation
        db (MarketDataDB, optional): Database connection. Defaults to Depends(get_db).

    Returns:
        tp.Dict[str, tp.Dict[str, float]]: Covariance matrix
    """
    # Check availability of tickers
    available_tickers = db.list_symbols(horizon=query.horizon)
    if not set(query.tickers).issubset(set(available_tickers)):
        raise HTTPException(status_code=400, detail="Some tickers are not available")

    # Get the data range for the tickers
    summary = db.summary(horizon=query.horizon)
    ticker_summary = summary.loc[query.tickers]
    sd = ticker_summary["start_date"].max().to_pydatetime()
    ed = ticker_summary["end_date"].min().to_pydatetime()

    # Convert to UTC
    sd = sd.replace(tzinfo=dt.timezone.utc)
    ed = ed.replace(tzinfo=dt.timezone.utc)
    # Get the data for the tickers
    if query.start is None:
        query.start = sd
    else:
        query.start = query.start.replace(tzinfo=dt.timezone.utc)
    if query.end is None:
        query.end = ed
    else:
        query.end = query.end.replace(tzinfo=dt.timezone.utc)

    try:
        data = db.read_into_uniform_barcollection(
            symbols=query.tickers,
            horizon=query.horizon,
            start=query.start,
            end=query.end,
        )

        # Calculate the arithmetic returns
        returns = data.arithmetic_returns

        # Calculate the covariance matrix using riskfolio
        cov = rp.covar_matrix(returns, method=query.method.value)
        # Coerce the covariance matrix to be symmetric and positive definite
        cov = to_symmetric_positive_definite_matrix(cov.values)
        # Convert to dataframe
        cov = pd.DataFrame(cov, index=query.tickers, columns=query.tickers)  # type: ignore

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return cov.to_dict(orient="index")  # type: ignore


@router.post("/estimate_mean")
def estimate_mean(
    query: MeanEstimationQuery, db: ArcticDatabase = Depends(get_db)
) -> tp.Dict[str, float]:
    """Estimate mean for a given collection of tickers

    Args:
        query (MeanEstimationQuery): Query for mean estimation
        db (MarketDataDB, optional): Database connection. Defaults to Depends(get_db).

    Returns:
        tp.Dict[str, float]: Mean for each ticker
    """
    # Check availability of tickers
    available_tickers = db.list_symbols(horizon=query.horizon)
    if not set(query.tickers).issubset(set(available_tickers)):
        raise HTTPException(status_code=400, detail="Some tickers are not available")

    # Get the data range for the tickers
    summary = db.summary(horizon=query.horizon)
    ticker_summary = summary.loc[query.tickers]
    sd = ticker_summary["start_date"].max().to_pydatetime()
    ed = ticker_summary["end_date"].min().to_pydatetime()

    # Convert to UTC
    sd = sd.replace(tzinfo=dt.timezone.utc)
    ed = ed.replace(tzinfo=dt.timezone.utc)
    # Get the data for the tickers
    if query.start is None:
        query.start = sd
    else:
        query.start = query.start.replace(tzinfo=dt.timezone.utc)
    if query.end is None:
        query.end = ed
    else:
        query.end = query.end.replace(tzinfo=dt.timezone.utc)

    try:
        data = db.read_into_uniform_barcollection(
            symbols=query.tickers,
            horizon=query.horizon,
            start=query.start,
            end=query.end,
        )

        # Calculate the arithmetic returns
        returns = data.arithmetic_returns

        # Calculate the mean using riskfolio
        mean = rp.mean_vector(returns, method=query.method.value)
        # Convert the DataFrame to Dict
        if isinstance(mean, pd.DataFrame):
            mean = mean.iloc[0].to_dict()
        elif isinstance(mean, pd.Series):
            mean = mean.to_dict()
        elif isinstance(mean, np.ndarray):
            mean = pd.Series(mean).to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return mean
