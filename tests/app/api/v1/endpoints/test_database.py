import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import pandas as pd
import datetime as dt
from silkroad.app.api.v1.endpoints.database import router
from silkroad.db import ArcticDatabase
from silkroad.core.enums import Horizon, Sector, Exchange
from silkroad.core.data_models import Asset, UniformBarSet
from silkroad.app.core.dependencies import get_db

# Create a FastAPI app for testing just this router
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)

client = TestClient(app)


@pytest.fixture
def mock_db():
    db = MagicMock(spec=ArcticDatabase)
    return db


@pytest.fixture
def override_get_db(mock_db):
    app.dependency_overrides[get_db] = lambda: mock_db
    yield
    app.dependency_overrides = {}


@pytest.fixture
def sample_asset():
    return Asset(
        ticker="AAPL",
        name="Apple Inc.",
        exchange=Exchange.NASDAQ,
        sector=Sector.TECHNOLOGY,
    )


@pytest.fixture
def sample_df():
    dates = pd.date_range(start="2023-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [100.0] * 5,
            "high": [110.0] * 5,
            "low": [90.0] * 5,
            "close": [105.0] * 5,
            "volume": [1000] * 5,
            "trade_count": [100] * 5,
            "vwap": [102.0] * 5,
        },
        index=dates,
    )
    df.index.name = "timestamp"
    return df


def test_sync_market_data(mock_db, override_get_db, sample_asset):
    response = client.post(
        "/sync",
        json={
            "assets": [sample_asset.model_dump(mode="json")],
            "horizon": "D",
            "lookback_buffer": 5,
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    mock_db.sync.assert_called_once()


def test_read_data(mock_db, override_get_db, sample_df):
    mock_db.read.return_value = sample_df

    response = client.post("/read", json={"symbol": "AAPL", "horizon": "D"})
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 5
    assert "timestamp" in data[0]


def test_read_barset(mock_db, override_get_db, sample_df):
    # Mock return value to be a UniformBarSet
    bs = UniformBarSet.from_df("AAPL", Horizon.DAILY, sample_df)
    mock_db.read_into_uniform_barset.return_value = bs

    response = client.post("/read/barset", json={"symbol": "AAPL", "horizon": "D"})
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "AAPL"
    assert (
        len(data["bars"]) == 5
    )  # Note: logic might put them in buffer or df, test serialized output


def test_filter_symbols(mock_db, override_get_db):
    mock_db.filter_symbols.return_value = ["AAPL"]

    response = client.post("/filter", json={"horizon": "D", "sector": "technology"})
    assert response.status_code == 200
    assert response.json() == ["AAPL"]
    mock_db.filter_symbols.assert_called_with(Horizon.DAILY, sector=Sector.TECHNOLOGY)


def test_list_symbols(mock_db, override_get_db):
    mock_db.list_symbols.return_value = ["AAPL", "MSFT"]

    response = client.get("/symbols?horizon=D")
    assert response.status_code == 200
    assert response.json() == ["AAPL", "MSFT"]


def test_delete_symbol(mock_db, override_get_db):
    response = client.request(
        "DELETE", "/symbol", json={"symbol": "AAPL", "horizon": "D"}
    )
    assert response.status_code == 200
    mock_db.delete.assert_called_with("AAPL", Horizon.DAILY)


def test_prune_history(mock_db, override_get_db):
    response = client.post(
        "/prune", json={"symbol": "AAPL", "horizon": "D", "max_versions": 5}
    )
    assert response.status_code == 200
    mock_db.prune_history.assert_called_with("AAPL", Horizon.DAILY, 5)


def test_get_summary(mock_db, override_get_db):
    summary_df = pd.DataFrame([{"ticker": "AAPL", "rows": 100}])
    mock_db.summary.return_value = summary_df.set_index("ticker")

    response = client.get("/summary?horizon=D")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["ticker"] == "AAPL"
