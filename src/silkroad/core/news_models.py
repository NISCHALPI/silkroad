"""Data models for news and sentiment data."""

from datetime import datetime, timezone
import uuid
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


MARKET_TICKER = "MARKET"


class NewsArticle(BaseModel):
    """Model representing a single news article."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the article.",
    )
    timestamp: datetime = Field(..., description="Publication timestamp in UTC.")
    source: str = Field(
        ..., description="Source of the news (e.g., 'Reuters', 'Bloomberg')."
    )
    headline: str = Field(..., description="Headline or title of the article.")
    content: str = Field(..., description="Full content or summary of the article.")
    url: Optional[str] = Field(None, description="URL to the original article.")
    sentiment: float = Field(0.0, description="Sentiment score (-1.0 to 1.0).")
    tickers: List[str] = Field(
        default_factory=list, description="List of associated tickers."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        # Ensure timestamp is UTC
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)

        # Ensure at least one ticker exists (Market News)
        if not self.tickers:
            self.tickers = [MARKET_TICKER]
