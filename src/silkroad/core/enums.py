"""Enumerations for trading horizons, asset classes, sectors, and exchanges."""

from enum import Enum
import pandas as pd


class Horizon(Enum):
    """Trading horizons represented by their equivalent number of trading days."""

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
        """Convert the horizon to the equivalent number of trading days.

        Returns:
            Number of trading days.
        """
        return self.value  # type: ignore

    def periods_annually(self) -> float:
        """Number of periods in a year for the given horizon.

        Returns:
            Number of periods annually as a float.
        """
        mapping = {
            Horizon.SECONDS: 252 * 6.5 * 60 * 60,
            Horizon.MINUTE: 252 * 6.5 * 60,
            Horizon.HOURLY: 252 * 6.5,
            Horizon.DAILY: 252,
            Horizon.WEEKLY: 52,
            Horizon.MONTHLY: 12,
            Horizon.QUARTERLY: 4,
            Horizon.SEMI_ANNUAL: 2,
            Horizon.ANNUAL: 1,
        }
        return mapping[self]

    def to_pandas_freq(self) -> str:
        """Convert the horizon to a pandas frequency string.

        Returns:
            Pandas frequency string (e.g., 'D', 'W', 'M').
        """
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
        """Convert the horizon to a pandas Timedelta object.

        Returns:
            Pandas Timedelta object corresponding to the horizon.
        """
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
        """Check if a given pandas Timedelta matches the horizon within tolerance.

        Args:
            delta: Pandas Timedelta to check.

        Returns:
            True if delta matches horizon within 1ms tolerance, False otherwise.
        """
        expected = self.to_pandas_timedelta()
        tolerance = pd.Timedelta(milliseconds=1)
        return abs(delta - expected) <= tolerance


class DataBackend(Enum):
    """Data backend providers."""

    YFINANCE = "yfinance"
    ALPACA = "alpaca"


class AssetClass(Enum):
    """Asset class types."""

    STOCK = 0
    BOND = 1
    COMMODITY = 2
    FOREX = 3
    CRYPTOCURRENCY = 4
    REAL_ESTATE = 5
    OTHER = -1


class Sector(Enum):
    """Industry sectors."""

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
    """Stock exchanges."""

    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    AMEX = "AMEX"
    OTC = "OTC"
    OTHER = "OTHER"
