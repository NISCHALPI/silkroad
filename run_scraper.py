import logging
from silkroad.utils.news_scraper import NewsScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    # Initialize with common tickers
    common_tickers = [
        "AAPL",
        "MSFT",
        "GOOG",
        "AMZN",
        "TSLA",
        "NVDA",
        "META",
        "AMD",
        "INTC",
    ]

    scraper = NewsScraper(tickers=common_tickers)

    # Run continuous loop
    # Limit to 50 articles per source, run every 15 minutes
    scraper.run(interval_minutes=15, limit=50, output_file="silkroad_news.csv")
