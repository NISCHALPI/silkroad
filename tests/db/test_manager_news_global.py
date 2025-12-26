import pytest
import pandas as pd
import datetime as dt
from silkroad.core.enums import Horizon
from silkroad.core.news_models import MARKET_TICKER, NewsArticle
from silkroad.db.duckdb_store import DuckDBStore
from silkroad.db.manager import DataManager
from silkroad.db.backends import DataBackendProvider


class MockBackend(DataBackendProvider):
    def fetch_data(self, assets, start, end, horizon):
        return pd.DataFrame()

    def get_latest_bar(self, asset):
        return None


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test_news_market.duckdb"
    return DuckDBStore(str(db_path))


@pytest.fixture
def manager(store):
    backend = MockBackend()
    return DataManager(db=store, backend=backend)


def test_news_model_default_ticker():
    """Test that NewsArticle defaults to MARKET_TICKER if tickers empty."""
    now = dt.datetime.now(dt.timezone.utc)
    article = NewsArticle(
        timestamp=now, headline="Market News", content="Ct", source="Src", tickers=[]
    )
    assert article.tickers == [MARKET_TICKER]


def test_get_news_market_ticker(manager, store):
    """Test retrieving news using explicit MARKET ticker logic."""
    now = dt.datetime.now(dt.timezone.utc)

    # Add AAPL news
    df_ticker = pd.DataFrame(
        {
            "timestamp": [now],
            "headline": ["AAPL News"],
            "content": ["Body"],
            "source": ["Source"],
            "tickers": ["AAPL"],
        }
    )
    manager.add_news_from_dataframe(df_ticker)

    # Add Global news (empty ticker -> becomes MARKET)
    df_global = pd.DataFrame(
        {
            "timestamp": [now],
            "headline": ["Market News"],
            "content": ["Body"],
            "source": ["Source"],
            "tickers": [[]],  # Empty list
        }
    )
    manager.add_news_from_dataframe(df_global)

    # Verify in DB directly that it was saved as MARKET
    direct_res = store.get_news(tickers=[MARKET_TICKER], start=now, end=now)
    assert len(direct_res) == 1
    assert direct_res.iloc[0]["headline"] == "Market News"

    # 1. Fetch matching AAPL with include_global=True (Default)
    # Should automatically include MARKET ticker
    res_default = manager.get_news_history(tickers=["AAPL"], include_global=True)
    headlines = res_default["headline"].values
    assert "AAPL News" in headlines
    assert "Market News" in headlines
    assert len(res_default) == 2

    # 2. Fetch matching AAPL with include_global=False
    # Should not include MARKET ticker
    res_strict = manager.get_news_history(tickers=["AAPL"], include_global=False)
    headlines_strict = res_strict["headline"].values
    assert "AAPL News" in headlines_strict
    assert "Market News" not in headlines_strict
    assert len(res_strict) == 1

    # 3. Fetch explicit MARKET ticker
    res_market = manager.get_news_history(tickers=[MARKET_TICKER])
    assert len(res_market) == 1
    assert res_market.iloc[0]["headline"] == "Market News"


def test_get_news_all(manager):
    """Test fetching ALL news (no tickers provided)."""
    now = dt.datetime.now(dt.timezone.utc)
    df = pd.DataFrame(
        {
            "timestamp": [now],
            "headline": ["Some News"],
            "content": ["Body"],
            "source": ["Source"],
            "tickers": ["AAPL"],
        }
    )
    manager.add_news_from_dataframe(df)

    res = manager.get_news_history(tickers=None)
    assert len(res) == 1
    assert res.iloc[0]["headline"] == "Some News"
