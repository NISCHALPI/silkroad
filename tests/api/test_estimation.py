import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import pandas as pd
import datetime as dt
import numpy as np
from silkroad.app.main import app
from silkroad.app.core.dependencies import get_db
from silkroad.core.data_models import (
    UniformBarCollection,
    UniformBarSet,
    Horizon,
    Asset,
    Exchange,
)
from silkroad.app.schemas.rfolio_estimation_query import CovMethod, MeanMethod

client = TestClient(app)


@pytest.fixture
def mock_db():
    db = MagicMock()

    # Mock get_available_tickers
    db.get_available_tickers.return_value = ["AAPL", "GOOG"]

    # Mock summary
    summary_df = pd.DataFrame(
        {
            "start_date": [dt.datetime(2023, 1, 1), dt.datetime(2023, 1, 1)],
            "end_date": [dt.datetime(2023, 12, 31), dt.datetime(2023, 12, 31)],
        },
        index=["AAPL", "GOOG"],
    )
    db.summary.return_value = summary_df

    # Mock get_uniform_bar_collection
    # Create dummy data
    dates = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")
    prices_aapl = np.linspace(100, 110, len(dates))
    prices_goog = np.linspace(200, 210, len(dates))

    df_aapl = pd.DataFrame({"close": prices_aapl}, index=dates)
    df_goog = pd.DataFrame({"close": prices_goog}, index=dates)

    # Mock UniformBarCollection and its arithmetic_returns property
    mock_collection = MagicMock(spec=UniformBarCollection)

    # Create returns DataFrame
    returns_df = pd.DataFrame(
        {
            "AAPL": df_aapl["close"].pct_change().dropna(),
            "GOOG": df_goog["close"].pct_change().dropna(),
        }
    )
    type(mock_collection).arithmetic_returns = PropertyMock(return_value=returns_df)

    db.get_uniform_bar_collection.return_value = mock_collection

    return db


from unittest.mock import PropertyMock


def test_estimate_cov(mock_db):
    app.dependency_overrides[get_db] = lambda: mock_db

    payload = {"tickers": ["AAPL", "GOOG"], "method": "hist"}

    response = client.post("/api/v1/estimate_cov", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "AAPL" in data
    assert "GOOG" in data
    assert "AAPL" in data["AAPL"]  # Check structure


def test_estimate_mean(mock_db):
    app.dependency_overrides[get_db] = lambda: mock_db

    payload = {"tickers": ["AAPL", "GOOG"], "method": "hist"}

    response = client.post("/api/v1/estimate_mean", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "AAPL" in data
    assert "GOOG" in data
    assert isinstance(data["AAPL"], (float, type(None)))


def test_invalid_ticker(mock_db):
    app.dependency_overrides[get_db] = lambda: mock_db

    payload = {"tickers": ["INVALID", "INVALID2"], "method": "hist"}

    response = client.post("/api/v1/estimate_cov", json=payload)
    assert response.status_code == 400
    assert "not available" in response.json()["detail"]


def test_estimate_cov_spd_coercion(mock_db):
    app.dependency_overrides[get_db] = lambda: mock_db

    # Non-SPD matrix (symmetric but negative eigenvalue)
    # [[1, 2], [2, 1]] -> eigenvalues 3, -1
    non_spd_cov = pd.DataFrame(
        [[1.0, 2.0], [2.0, 1.0]], index=["AAPL", "GOOG"], columns=["AAPL", "GOOG"]
    )

    with patch(
        "silkroad.app.api.v1.endpoints.estimate_cov.rp.covar_matrix",
        return_value=non_spd_cov,
    ):
        payload = {"tickers": ["AAPL", "GOOG"], "method": "hist"}

        response = client.post("/api/v1/estimate_cov", json=payload)
        assert response.status_code == 200
        data = response.json()

        # Convert back to numpy array
        cov_matrix = pd.DataFrame(data).values

        # Check symmetry
        assert np.allclose(cov_matrix, cov_matrix.T)

        # Check positive definiteness (all eigenvalues > 0)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        assert np.all(eigenvalues > 0)
