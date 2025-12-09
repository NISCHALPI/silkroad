import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import datetime as dt
from silkroad.app.api.v1.endpoints.estimate_cov import router
from silkroad.db import ArcticDatabase
from silkroad.core.enums import Horizon
from silkroad.app.core.dependencies import get_db

from fastapi import FastAPI

app = FastAPI()
app.include_router(router)

client = TestClient(app)


@pytest.fixture
def mock_db():
    return MagicMock(spec=ArcticDatabase)


@pytest.fixture
def override_get_db(mock_db):
    app.dependency_overrides[get_db] = lambda: mock_db
    yield
    app.dependency_overrides = {}


@pytest.fixture
def sample_summary():
    tickers = ["AAPL", "MSFT"]
    data = {
        "start_date": [dt.datetime(2020, 1, 1), dt.datetime(2020, 1, 1)],
        "end_date": [dt.datetime(2023, 1, 1), dt.datetime(2023, 1, 1)],
    }
    df = pd.DataFrame(data, index=tickers)
    return df


@pytest.fixture
def mock_collection():
    mock_coll = MagicMock()
    # Mock arithmetic_returns to return a DataFrame
    dates = pd.date_range(start="2020-01-01", periods=10, freq="D")
    returns = pd.DataFrame(
        np.random.randn(10, 2), index=dates, columns=["AAPL", "MSFT"]
    )
    mock_coll.arithmetic_returns = returns
    return mock_coll


def test_estimate_cov_success(
    mock_db, override_get_db, sample_summary, mock_collection
):
    # Setup mocks
    mock_db.list_symbols.return_value = ["AAPL", "MSFT"]
    mock_db.summary.return_value = sample_summary
    mock_db.read_into_uniform_barcollection.return_value = mock_collection

    # Mock riskfolio
    with patch("silkroad.app.api.v1.endpoints.estimate_cov.rp") as mock_rp:
        # returns 2x2 covarianc matrix
        cov_matrix = pd.DataFrame(
            [[0.1, 0.05], [0.05, 0.1]], index=["AAPL", "MSFT"], columns=["AAPL", "MSFT"]
        )
        mock_rp.covar_matrix.return_value = cov_matrix

        response = client.post(
            "/estimate_cov",
            json={"tickers": ["AAPL", "MSFT"], "method": "hist", "horizon": "D"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "AAPL" in data
        assert "MSFT" in data
        assert data["AAPL"]["AAPL"] == pytest.approx(0.1, abs=1e-6)


def test_estimate_cov_ticker_not_available(mock_db, override_get_db):
    mock_db.list_symbols.return_value = ["AAPL"]

    response = client.post(
        "/estimate_cov",
        json={
            "tickers": ["AAPL", "MSFT"],  # MSFT not available
            "method": "hist",
            "horizon": "D",
        },
    )

    assert response.status_code == 400
    assert "not available" in response.json()["detail"]


def test_estimate_mean_success(
    mock_db, override_get_db, sample_summary, mock_collection
):
    # Setup mocks
    mock_db.list_symbols.return_value = ["AAPL", "MSFT"]
    mock_db.summary.return_value = sample_summary
    mock_db.read_into_uniform_barcollection.return_value = mock_collection

    # Mock riskfolio
    with patch("silkroad.app.api.v1.endpoints.estimate_cov.rp") as mock_rp:
        # returns mean series
        mean_vector = pd.Series([0.05, 0.06], index=["AAPL", "MSFT"])
        mock_rp.mean_vector.return_value = mean_vector

        response = client.post(
            "/estimate_mean",
            json={"tickers": ["AAPL", "MSFT"], "method": "hist", "horizon": "D"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["AAPL"] == 0.05
        assert data["MSFT"] == 0.06


def test_estimate_mean_ticker_not_available(mock_db, override_get_db):
    mock_db.list_symbols.return_value = ["AAPL"]

    response = client.post(
        "/estimate_mean",
        json={
            "tickers": ["AAPL", "MSFT"],  # MSFT not available
            "method": "hist",
            "horizon": "D",
        },
    )

    assert response.status_code == 400
    assert "not available" in response.json()["detail"]


def test_db_read_failure(mock_db, override_get_db, sample_summary):
    # Setup mocks
    mock_db.list_symbols.return_value = ["AAPL", "MSFT"]
    mock_db.summary.return_value = sample_summary
    mock_db.read_into_uniform_barcollection.side_effect = Exception("DB Error")

    response = client.post(
        "/estimate_cov",
        json={"tickers": ["AAPL", "MSFT"], "method": "hist", "horizon": "D"},
    )

    assert response.status_code == 500
    assert "DB Error" in response.json()["detail"]
