"""Enumerations used throughout the Silkroad package."""

from enum import Enum
import pandas as pd


class Horizon(Enum):
    """Enumeration for different trading horizons using their respective number of trading days."""

    SECONDS = -2
    MINUTE = -1
    HOURLY = 0
    DAILY = 1
    WEEKLY = 5
    MONTHLY = 21
    QUARTERLY = 63
    SEMI_ANNUAL = 126
    ANNUAL = 252

    def to_days(self) -> int:
        """Convert the horizon to the equivalent number of trading days."""
        return self.value  # type: ignore

    def to_pandas_freq(self) -> str:
        """Convert the horizon to a pandas frequency string."""
        mapping = {
            Horizon.SECONDS: "S",
            Horizon.MINUTE: "T",
            Horizon.HOURLY: "H",
            Horizon.DAILY: "D",
            Horizon.WEEKLY: "W",
            Horizon.MONTHLY: "M",
            Horizon.QUARTERLY: "Q",
            Horizon.SEMI_ANNUAL: "2Q",
            Horizon.ANNUAL: "YE",
        }
        return mapping[self]

    def to_pandas_timedelta(self):
        """Convert the horizon to a pandas Timedelta object."""
        mapping = {
            Horizon.SECONDS: pd.Timedelta(seconds=1),
            Horizon.MINUTE: pd.Timedelta(minutes=1),
            Horizon.HOURLY: pd.Timedelta(hours=1),
            Horizon.DAILY: pd.Timedelta(days=1),
            Horizon.WEEKLY: pd.Timedelta(weeks=1),
            Horizon.MONTHLY: pd.Timedelta(days=21),
            Horizon.QUARTERLY: pd.Timedelta(days=63),
            Horizon.SEMI_ANNUAL: pd.Timedelta(days=126),
            Horizon.ANNUAL: pd.Timedelta(days=252),
        }
        return mapping[self]

    def check_valid(self, delta: pd.Timedelta) -> bool:
        """Check if a given pandas Timedelta matches the horizon within a tolerance."""
        expected = self.to_pandas_timedelta()
        tolerance = pd.Timedelta(milliseconds=1)
        return abs(delta - expected) <= tolerance


class DataBackend(Enum):
    """Enumeration for different data backends."""

    YFINANCE = "yfinance"
    ALPACA = "alpaca"


class AssetClass(Enum):
    """Enumeration for different asset classes."""

    STOCK = 0
    BOND = 1
    COMMODITY = 2
    FOREX = 3
    CRYPTOCURRENCY = 4
    REAL_ESTATE = 5
    OTHER = -1


class Sector(Enum):
    """Enumeration for different industry sectors."""

    TECHNOLOGY = 0
    HEALTHCARE = 1
    FINANCIALS = 2
    CONSUMER_DISCRETIONARY = 3
    CONSUMER_STAPLES = 4
    ENERGY = 5
    UTILITIES = 6
    INDUSTRIALS = 7
    MATERIALS = 8
    REAL_ESTATE = 9
    COMMUNICATION_SERVICES = 10
    OTHER = -1


class Exchange(Enum):
    """Enumeration for different stock exchanges."""

    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    AMEX = "AMEX"
    OTC = "OTC"
    OTHER = "OTHER"
