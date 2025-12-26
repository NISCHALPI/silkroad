"""Write a web scraper to fetch news articles from the web and store them in a database."""

import csv
import logging
import re
import asyncio
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Set

import aiohttp
import requests
import newspaper
import feedparser
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

from silkroad.core.news_models import NewsArticle

logger = logging.getLogger(__name__)


class NewsScraper:
    """Scraper to fetch news directly from financial news websites using Async I/O."""

    BENZINGA_RSS_URL = "http://feeds.benzinga.com/benzinga"
    AP_NEWS_URL = "https://apnews.com/hub/business"
    YAHOO_NEWS_URL = "https://finance.yahoo.com/topic/stock-market-news/"

    def __init__(self, tickers: Optional[List[str]] = None):
        """
        Args:
            tickers: Optional list of tickers to watch for.
        """
        self.tickers = tickers or []
        self.ticker_regex = None
        if self.tickers:
            pattern = r"\b(" + "|".join(map(re.escape, self.tickers)) + r")\b"
            self.ticker_regex = re.compile(pattern, re.IGNORECASE)

        # Initialize UserAgent rotator
        try:
            self.ua = UserAgent()
        except Exception:
            self.ua = None
            logger.warning(
                "Could not initialize fake_useragent, falling back to static."
            )

    def _get_headers(self) -> Dict[str, str]:
        """Generate random headers for requests."""
        if self.ua:
            try:
                user_agent = self.ua.random
            except Exception:
                user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        else:
            user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

        return {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

    async def fetch_latest(self, limit: int = 50) -> List[NewsArticle]:
        """Fetch latest news from all configured sources asynchronously.

        Args:
            limit: Number of articles to fetch per source. Defaults to 50.

        Returns:
            List[NewsArticle]: Combined list of fetched articles.
        """
        all_articles: List[NewsArticle] = []

        async with aiohttp.ClientSession() as session:
            # Fetch from all sources in parallel
            tasks = [
                self._fetch_yahoo(session, limit),
                self._fetch_benzinga(session, limit),
                self._fetch_ap_news(session, limit),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for res in results:
                if isinstance(res, list):
                    all_articles.extend(res)
                elif isinstance(res, Exception):
                    logger.error(f"Error checking source: {res}")

        # Deduplication by URL
        unique_articles = []
        seen_urls = set()
        for art in all_articles:
            if art.url not in seen_urls:
                unique_articles.append(art)
                seen_urls.add(art.url)

        return unique_articles

    async def _fetch_yahoo(
        self, session: aiohttp.ClientSession, limit: int
    ) -> List[NewsArticle]:
        """Fetch from Yahoo Finance using requests (for header stability)."""
        articles = []
        try:
            logger.info(f"Fetching {self.YAHOO_NEWS_URL}...")
            # Use requests in executor because aiohttp chokes on Yahoo's large headers/cookies
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                None,
                lambda: requests.get(
                    self.YAHOO_NEWS_URL, headers=self._get_headers(), timeout=15
                ).text,
            )

            soup = BeautifulSoup(text, "html.parser")

            links = set()
            for a_tag in soup.find_all("a", href=True):
                href = a_tag.get("href")
                if not isinstance(href, str):
                    continue

                if "/news/" in href or "/m/" in href:
                    if "finance.yahoo.com" in href:
                        links.add(href)
                    elif href.startswith("/"):
                        links.add(f"https://finance.yahoo.com{href}")

            sorted_links = list(links)
            sorted_links.sort(key=len, reverse=True)
            target_links = sorted_links[:limit]

            if target_links:
                logger.info(f"Yahoo: Processing {len(target_links)} links...")
                tasks = [
                    self._process_url(session, url, source="Yahoo Finance")
                    for url in target_links
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for res in results:
                    if isinstance(res, NewsArticle):
                        articles.append(res)
                    elif isinstance(res, Exception):
                        logger.warning(f"Failed to process Yahoo URL: {res}")

        except Exception as e:
            logger.error(f"Failed to fetch Yahoo Finance: {e}")
        return articles

    async def _fetch_benzinga(
        self, session: aiohttp.ClientSession, limit: int
    ) -> List[NewsArticle]:
        """Fetch from Benzinga RSS."""
        articles = []
        try:
            # RSS fetch can be async too
            logger.info(f"Fetching {self.BENZINGA_RSS_URL}...")
            async with session.get(
                self.BENZINGA_RSS_URL, headers=self._get_headers(), timeout=15
            ) as response:
                response.raise_for_status()
                rss_text = await response.text()

            # Feedparser is sync, but fast on memory string
            feed = feedparser.parse(rss_text)

            entries = feed.entries[:limit]
            if entries:
                logger.info(f"Benzinga: Processing {len(entries)} entries...")

                # Benzinga RSS items have data, but we might want full content
                # Let's parallelize full content fetch
                tasks = []
                for entry in entries:
                    url = entry.link
                    headline = entry.title
                    pub_date = datetime.now(timezone.utc)
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        import time

                        pub_date = datetime.fromtimestamp(
                            time.mktime(entry.published_parsed), timezone.utc
                        )

                    tasks.append(
                        self._process_url(
                            session,
                            url,
                            source="Benzinga",
                            known_headline=headline,
                            known_date=pub_date,
                        )
                    )

                results = await asyncio.gather(*tasks, return_exceptions=True)

                for res in results:
                    if isinstance(res, NewsArticle):
                        articles.append(res)
                    elif isinstance(res, Exception):
                        logger.warning(f"Failed to process Benzinga entry: {res}")

        except Exception as e:
            logger.error(f"Failed to fetch Benzinga: {e}")
        return articles

    async def _fetch_ap_news(
        self, session: aiohttp.ClientSession, limit: int
    ) -> List[NewsArticle]:
        """Fetch from AP News Business Hub."""
        articles = []
        try:
            logger.info(f"Fetching {self.AP_NEWS_URL}...")
            async with session.get(
                self.AP_NEWS_URL, headers=self._get_headers(), timeout=15
            ) as response:
                response.raise_for_status()
                text = await response.text()

            soup = BeautifulSoup(text, "html.parser")
            links = set()
            for a in soup.find_all("a", href=True):
                href = a.get("href")
                if isinstance(href, str) and "/article/" in href:
                    if href.startswith("http"):
                        links.add(href)
                    elif href.startswith("/"):
                        links.add(f"https://apnews.com{href}")

            target_links = list(links)[:limit]

            if target_links:
                logger.info(f"AP News: Processing {len(target_links)} links...")
                tasks = [
                    self._process_url(session, url, source="AP News")
                    for url in target_links
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for res in results:
                    if isinstance(res, NewsArticle):
                        articles.append(res)
                    elif isinstance(res, Exception):
                        logger.warning(f"Failed to process AP News URL: {res}")

        except Exception as e:
            logger.error(f"Failed to fetch AP News: {e}")
        return articles

    async def _process_url(
        self,
        session: aiohttp.ClientSession,
        url: str,
        source: str = "Unknown",
        known_headline: Optional[str] = None,
        known_date: Optional[datetime] = None,
    ) -> Optional[NewsArticle]:
        """Process a single article URL asynchronously."""
        try:
            html = ""
            # YAHOO SPECIAL HANDLING: Use requests because aiohttp cannot handle the large headers/cookies
            if "yahoo.com" in url:
                loop = asyncio.get_event_loop()
                try:
                    html = await loop.run_in_executor(
                        None,
                        lambda: requests.get(
                            url, headers=self._get_headers(), timeout=10
                        ).text,
                    )
                except Exception as e:
                    logger.debug(f"Yahoo requests fetch failed for {url}: {e}")
                    return None
            else:
                # Standard Async fetch
                async with session.get(
                    url, headers=self._get_headers(), timeout=10
                ) as response:
                    # Some sites return 403 on scraper, ensure we catch this
                    if response.status != 200:
                        logger.warning(f"Got {response.status} for {url}")
                        return None
                    html = await response.text()

            # 2. Parse with newspaper3k (CPU bound, technically should be in executor but fast enough)
            article = newspaper.Article(url)
            article.set_html(html)  # Inject async downloaded HTML
            article.parse()
            # article.nlp() # Optional, adds keywords but takes time

            headline = article.title or known_headline or "Unknown Title"
            content = article.text or ""

            pub_date = article.publish_date or known_date
            if not pub_date:
                pub_date = datetime.now(timezone.utc)
            elif pub_date.tzinfo is None:
                pub_date = pub_date.replace(tzinfo=timezone.utc)

            # 3. Ticker Association
            associated_tickers = self._find_tickers(headline, content)

            return NewsArticle(
                timestamp=pub_date,
                source=source,
                headline=headline,
                content=content,
                url=url,
                tickers=associated_tickers,
                metadata={},
            )

        except Exception as e:
            # logger.fine or debug usually, to avoid spam
            logger.debug(f"Failed to process URL {url}: {e}")
            return None

    def fetch_historical(
        self, topic: str, start_date: datetime, end_date: datetime
    ) -> List[NewsArticle]:
        """Historical fetching on static pages is not easily supported without an API."""
        logger.warning(
            "Historical fetching via scraping is not supported on these sources."
        )
        return []

    def _find_tickers(self, headline: str, content: str) -> List[str]:
        """Find tickers in text using hybrid regex approach."""
        if not headline and not content:
            return ["MARKET"]

        found_tickers = set()
        text = (headline + " " + content).upper()

        # 1. Cashtags
        cashtags = re.findall(r"\$([A-Z]{1,5})\b", text)
        found_tickers.update(cashtags)

        # 2. Explicit Tickers
        if self.ticker_regex:
            matches = self.ticker_regex.findall(text)
            found_tickers.update(matches)

        # 3. Exchange Patterns
        exchange_matches = re.findall(r"(?:NASDAQ|NYSE|AMEX)[:\s]+([A-Z]{1,5})\b", text)
        found_tickers.update(exchange_matches)

        STOPLIST = {
            "THE",
            "AND",
            "FOR",
            "ARE",
            "BUT",
            "NOT",
            "USA",
            "CEO",
            "IPO",
            "ETF",
            "GDP",
            "CPI",
            "USD",
            "EUR",
            "YEN",
            "GBP",
            "ALL",
            "NEW",
            "BIG",
            "LOW",
            "BUY",
            "SELL",
            "TOP",
            "NOW",
            "OUT",
            "DAY",
            "ONE",
            "TWO",
            "SIX",
            "TEN",
            "ART",
            "CAN",
            "DID",
            "GET",
            "GOT",
            "HAD",
            "HAS",
            "HIM",
            "HIS",
            "HOW",
            "ITS",
            "LET",
            "MAN",
            "MAY",
            "MET",
            "NOR",
            "OFF",
            "OLD",
            "OUR",
            "OWN",
            "PUT",
            "RUN",
            "SAW",
            "SAY",
            "SEE",
            "SET",
            "SHE",
            "SIT",
            "SON",
            "TOO",
            "USE",
            "WAS",
            "WAY",
            "WHO",
            "WHY",
            "WON",
            "YES",
            "YET",
            "YOU",
            "A",
            "I",
            "S",
        }

        final_tickers = {t for t in found_tickers if t not in STOPLIST and len(t) >= 2}

        return list(final_tickers) if final_tickers else ["MARKET"]

    def save_to_csv(self, articles: List[NewsArticle], filepath: str):
        """Save articles to a CSV file with deduplication."""
        if not articles:
            logger.info("No articles to process.")
            return

        import os

        existing_urls = set()
        file_exists = os.path.isfile(filepath)

        # 1. Read existing URLs if file exists
        if file_exists:
            try:
                with open(filepath, mode="r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if "url" in row:
                            existing_urls.add(row["url"])
            except Exception as e:
                logger.error(f"Error reading existing CSV: {e}")

        # 2. Filter new articles
        new_articles = [a for a in articles if a.url not in existing_urls]

        if not new_articles:
            logger.info("No new unique articles to save.")
            return

        # 3. Append new articles
        fieldnames = [
            "id",
            "timestamp",
            "source",
            "headline",
            "content",
            "url",
            "sentiment",
            "tickers",
            "metadata",
        ]

        try:
            mode = "a" if file_exists else "w"
            with open(filepath, mode=mode, newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()

                for article in new_articles:
                    row = {
                        "id": article.id,
                        "timestamp": article.timestamp.isoformat(),
                        "source": article.source,
                        "headline": article.headline,
                        "content": article.content,
                        "url": article.url,
                        "sentiment": article.sentiment,
                        "tickers": ",".join(article.tickers),
                        "metadata": str(article.metadata),
                    }
                    writer.writerow(row)
            logger.info(f"Saved {len(new_articles)} new articles to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")

    def run(
        self,
        interval_minutes: int = 15,
        limit: int = 50,
        output_file: str = "silkroad_news.csv",
    ):
        """
        Run the news scraper continuously at a specified interval using Async loop.
        """
        logger.info("Initializing Continuous News Scraper Loop (Async)...")
        logger.info(
            f"Settings: Interval={interval_minutes}m, Limit={limit}, Output={output_file}"
        )

        while True:
            try:
                start_time = datetime.now()
                logger.info(f"--- Starting Async Fetch Cycle at {start_time} ---")

                # Run the async fetch within the synchoroncus loop using asyncio.run
                articles = asyncio.run(self.fetch_latest(limit=limit))

                if articles:
                    logger.info(f"Fetched {len(articles)} articles. Saving...")
                    self.save_to_csv(articles, output_file)
                else:
                    logger.info("No articles found this cycle.")

                elapsed = (datetime.now() - start_time).total_seconds()
                sleep_seconds = (interval_minutes * 60) - elapsed

                if sleep_seconds > 0:
                    logger.info(
                        f"Cycle complete. Scraped {len(articles)} items. Sleeping for {sleep_seconds:.1f} seconds..."
                    )
                    import time

                    time.sleep(sleep_seconds)
                else:
                    logger.warning(
                        "Fetch took longer than interval! Starting next cycle immediately."
                    )

            except KeyboardInterrupt:
                logger.info("Stopping scraper (KeyboardInterrupt).")
                break
            except Exception as e:
                logger.error(f"Unexpected error in scrape loop: {e}")
                import time

                logger.info("Retrying in 60 seconds...")
                time.sleep(60)
