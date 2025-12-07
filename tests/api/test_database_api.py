import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import pandas as pd
import datetime as dt
from silkroad.app.main import app
from silkroad.app.core.dependencies import get_db
from silkroad.core.data_models import Asset
from silkroad.core.enums import AssetClass, Exchange

client = TestClient(app)


@pytest.fixture
def mock_db():
    db = MagicMock()
    return db


def test_delete_ticker(mock_db):
    app.dependency_overrides[get_db] = lambda: mock_db

    response = client.delete("/api/v1/symbols/AAPL")
    assert response.status_code == 200
    assert "deleted successfully" in response.json()["message"]
    mock_db.delete_ticker.assert_called_once_with("AAPL")


def test_get_latest_bar(mock_db):
    app.dependency_overrides[get_db] = lambda: mock_db

    mock_bar = {
        "open": 100.0,
        "high": 105.0,
        "low": 99.0,
        "close": 102.0,
        "volume": 1000,
    }
    mock_db.get_latest_bar.return_value = mock_bar

    response = client.get("/api/v1/symbols/AAPL/latest")
    assert response.status_code == 200
    assert response.json() == mock_bar
    mock_db.get_latest_bar.assert_called_once_with("AAPL")


def test_get_latest_bar_not_found(mock_db):
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.get_latest_bar.return_value = None

    response = client.get("/api/v1/symbols/INVALID/latest")
    assert response.status_code == 404
    assert "No data found" in response.json()["detail"]


def test_refresh_database(mock_db):
    app.dependency_overrides[get_db] = lambda: mock_db

    payload = {
        "assets": [
            {
                "ticker": "AAPL",
                "asset_class": "stock",
                "exchange": "nasdaq",
                "name": "Apple Inc.",
            }
        ],
    }

    response = client.post("/api/v1/refresh", json=payload)
    assert response.status_code == 200
    assert "completed successfully" in response.json()["message"]
    mock_db.refresh.assert_called_once()


def test_reset_database(mock_db):
    app.dependency_overrides[get_db] = lambda: mock_db

    response = client.delete("/api/v1/reset")
    assert response.status_code == 200
    assert "reset successfully" in response.json()["message"]
    mock_db.reset.assert_called_once()


def test_get_summary(mock_db):
    app.dependency_overrides[get_db] = lambda: mock_db

    summary_df = pd.DataFrame(
        {
            "start_date": [dt.datetime(2023, 1, 1)],
            "end_date": [dt.datetime(2023, 12, 31)],
            "num_bars": [252],
        },
        index=["AAPL"],
    )
    mock_db.summary.return_value = summary_df

    response = client.get("/api/v1/summary")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["index"] == "AAPL"
    assert data[0]["num_bars"] == 252
