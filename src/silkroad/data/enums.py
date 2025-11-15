"""Enumerations used throughout the Silkroad package."""

from enum import Enum


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
        return max(self.value, 1)

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
