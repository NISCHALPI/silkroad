"""Enumerations for trading horizons, asset classes, sectors, and exchanges."""

from enum import Enum
import pandas as pd
import datetime as dt
import typing as tp
import pandas_market_calendars as mcal
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


class Horizon(Enum):
    """Trading horizons represented by their equivalent number of trading days."""

    SECONDS = "s"
    MINUTE = "min"
    HOURLY = "h"
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M"
    QUARTERLY = "Q"
    SEMI_ANNUAL = "2Q"
    ANNUAL = "YE"

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
        return self.value  # type: ignore

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

    def to_alpaca_timeframe_unit(self) -> TimeFrame:
        """Convert the horizon to an Alpaca TimeFrame unit.

        Args:
            None

        Returns:
            TimeFrame: Corresponding Alpaca TimeFrame unit.
        """
        if self == Horizon.SECONDS:
            raise ValueError("Seconds horizon is not supported by Alpaca TimeFrame.")

        mapping = {
            Horizon.MINUTE: TimeFrame.Minute,
            Horizon.HOURLY: TimeFrame.Hour,
            Horizon.DAILY: TimeFrame.Day,
            Horizon.WEEKLY: TimeFrame.Week,
            Horizon.MONTHLY: TimeFrame.Month,
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


class AssetClass(Enum):
    """Asset classes."""

    STOCK = "stock"
    BOND = "bond"
    COMMODITY = "commodity"
    CRYPTO = "crypto"
    REAL_ESTATE = "real_estate"
    CASH = "cash"
    OTHER = "other"


class Sector(Enum):
    """GICS Sectors for Equities."""

    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    FINANCIALS = "financials"
    CONSUMER_DISC = "consumer_discretionary"  # Discretionary
    CONSUMER_STAP = "consumer_staples"  # Staples
    ENERGY = "energy"
    UTILITIES = "utilities"
    INDUSTRIALS = "industrials"
    MATERIALS = "materials"
    REAL_ESTATE = "real_estate"
    COMM_SERVICES = "communication_services"
    DIVERSIFIED = "diversified"
    GOVERNMENT = "government"
    PRECIOUS_METALS = "precious_metals"
    OTHERS = (
        "other"  # Use this for Assets that don't have a sector (like Cash/Crypto/Bonds)
    )


class Exchange(Enum):
    """Stock exchanges."""

    NYSE = "nyse"
    NASDAQ = "nasdaq"
    LSE = "lse"
    HKEX = "hkex"
    SSE = "sse"
    AMEX = "amex"
    OTC = "otc"
    OTHER = "other"

    @property
    def calendar(self):
        """Returns the pandas_market_calendars object for the exchange.

        This property lazily loads the appropriate market calendar from the
        `pandas_market_calendars` library. It maps the internal `Exchange` enum
        members to the corresponding calendar names used by the library.

        Returns:
            pandas_market_calendars.MarketCalendar: The market calendar object
            configured for this exchange.
        """

        mapping = {
            Exchange.NYSE: "NYSE",
            Exchange.NASDAQ: "NASDAQ",
            Exchange.AMEX: "NYSE",  # AMEX uses NYSE calendar
            Exchange.OTC: "OTC",  # Might need custom or fallback
            Exchange.LSE: "LSE",
            Exchange.HKEX: "HKEX",
            Exchange.SSE: "SSE",
        }

        cal_name = mapping.get(self, "NYSE")  # Default to NYSE
        return mcal.get_calendar(cal_name)

    @property
    def timezone(self) -> str:
        """Returns the IANA timezone string for the exchange.

        Retrieves the timezone information associated with the exchange's
        market calendar. Handles differences between `pytz` and `zoneinfo`
        implementations of timezone objects.

        Returns:
            str: The IANA timezone string (e.g., "America/New_York", "Asia/Shanghai").
        """
        # pandas_market_calendars uses pytz or ZoneInfo.
        try:
            return self.calendar.tz.zone  # pytz
        except AttributeError:
            return self.calendar.tz.key  # ZoneInfo

    @property
    def current_market_open(self) -> dt.datetime:
        """Returns the current market open time in UTC.

        Retrieves the open time for the current trading session. If the market
        is currently closed, this method raises a ValueError.

        Returns:
            dt.datetime: The current market open time in UTC.

        Raises:
            ValueError: If the market is closed for the current timestamp.
        """
        now = dt.datetime.now(dt.timezone.utc)
        if not self.is_open(now):
            raise ValueError("Market is currently closed.")
        return self.market_open_at(now)

    @property
    def current_market_close(self) -> dt.datetime:
        """Returns the current market close time in UTC.

        Retrieves the close time for the current trading session. If the market
        is currently closed, this method raises a ValueError.

        Returns:
            dt.datetime: The current market close time in UTC.

        Raises:
            ValueError: If the market is closed for the current timestamp.
        """
        now = dt.datetime.now(dt.timezone.utc)
        if not self.is_open(now):
            raise ValueError("Market is currently closed.")
        return self.market_close_at(now)

    @property
    def current_market_hours(self) -> tuple[dt.datetime, dt.datetime]:
        """Returns the current market hours in UTC.

        Retrieves the (open, close) tuple for the current trading session.
        If the market is currently closed, this method raises a ValueError.

        Returns:
            tuple[dt.datetime, dt.datetime]: The (open, close) times in UTC.

        Raises:
            ValueError: If the market is closed for the current timestamp.
        """
        now = dt.datetime.now(dt.timezone.utc)
        if not self.is_open(now):
            raise ValueError("Market is currently closed.")
        return self.market_hours_at(now)

    def is_open(self, timestamp: tp.Optional[dt.datetime] = None) -> bool:
        """Returns True if the market is currently open, False otherwise.

        Args:
            timestamp (dt.datetime, optional): The timestamp to check. Defaults to now.
        """
        if timestamp is None:
            timestamp = dt.datetime.now(dt.timezone.utc)
        return self.is_open_at(timestamp)

    def holiday_at(self, dt_obj: dt.datetime) -> bool:
        """Returns True if the given datetime is a holiday, False otherwise.

        Checks if the date of the provided timestamp is a holiday for the exchange.

        Args:
            dt_obj (dt.datetime): The timestamp to check.

        Returns:
            bool: True if the date is a holiday, False otherwise.
        """
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=dt.timezone.utc)

        # Check if it's a trading day
        schedule = self.calendar.schedule(start_date=dt_obj, end_date=dt_obj)
        return schedule.empty

    def is_open_at(self, dt_obj: dt.datetime) -> bool:
        """Returns True if the market is open at the given datetime, False otherwise.

        Args:
            dt_obj (dt.datetime): The timestamp to check.

        Returns:
            bool: True if open, False otherwise.
        """
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=dt.timezone.utc)

        # Add buffer to ensure coverage
        schedule = self.calendar.schedule(
            start_date=dt_obj - dt.timedelta(days=1),
            end_date=dt_obj + dt.timedelta(days=1),
        )
        if schedule.empty:
            return False

        try:
            return self.calendar.open_at_time(schedule=schedule, timestamp=dt_obj)
        except ValueError:
            return False

    def is_closed_at(self, dt_obj: dt.datetime) -> bool:
        """Returns True if the market is closed at the given datetime, False otherwise.

        Args:
            dt_obj (dt.datetime): The timestamp to check.

        Returns:
            bool: True if closed, False otherwise.
        """
        return not self.is_open_at(dt_obj)

    def market_open_at(self, dt_obj: dt.datetime) -> dt.datetime:
        """Returns the market open time at the given datetime in UTC.

        If the market is open at `dt_obj`, returns the open time of that session.
        If the market is closed, raises ValueError.

        Args:
            dt_obj (dt.datetime): The timestamp to check.

        Returns:
            dt.datetime: The market open time in UTC.

        Raises:
            ValueError: If the market is closed for the given datetime.
        """
        if not self.is_open_at(dt_obj):
            raise ValueError(f"Market is closed at {dt_obj}")

        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=dt.timezone.utc)

        schedule = self.calendar.schedule(
            start_date=dt_obj - dt.timedelta(days=1),
            end_date=dt_obj + dt.timedelta(days=1),
        )
        # Find the session that covers dt_obj
        # Since is_open_at is true, there must be a session where open <= dt <= close
        for _, row in schedule.iterrows():
            if row["market_open"] <= dt_obj <= row["market_close"]:
                return row["market_open"]

        raise ValueError(f"Could not find session for {dt_obj}")

    def market_close_at(self, dt_obj: dt.datetime) -> dt.datetime:
        """Returns the market close time at the given datetime in UTC.

        If the market is open at `dt_obj`, returns the close time of that session.
        If the market is closed, raises ValueError.

        Args:
            dt_obj (dt.datetime): The timestamp to check.

        Returns:
            dt.datetime: The market close time in UTC.

        Raises:
            ValueError: If the market is closed for the given datetime.
        """
        if not self.is_open_at(dt_obj):
            raise ValueError(f"Market is closed at {dt_obj}")

        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=dt.timezone.utc)

        schedule = self.calendar.schedule(
            start_date=dt_obj - dt.timedelta(days=1),
            end_date=dt_obj + dt.timedelta(days=1),
        )
        for _, row in schedule.iterrows():
            if row["market_open"] <= dt_obj <= row["market_close"]:
                return row["market_close"]

        raise ValueError(f"Could not find session for {dt_obj}")

    def market_hours_at(self, dt_obj: dt.datetime) -> tuple[dt.datetime, dt.datetime]:
        """Returns the market hours at the given datetime in UTC.

        Retrieves the open and close times for the trading session active at the
        specified datetime. If the market is closed at that time, raises a ValueError.

        Args:
            dt_obj (dt.datetime): The timestamp to check.

        Returns:
            tuple[dt.datetime, dt.datetime]: The (open, close) times in UTC.

        Raises:
            ValueError: If the market is closed for the given datetime.
        """
        return (self.market_open_at(dt_obj), self.market_close_at(dt_obj))

    def next_market_open(self) -> dt.datetime:
        """Returns the next market open time in UTC.

        Searches for the next scheduled market open event starting from the current
        UTC time. This method looks ahead up to 14 days to find the next open.

        Returns:
            dt.datetime: The next market open time in UTC.
        """
        now = dt.datetime.now(dt.timezone.utc)
        schedule = self.calendar.schedule(
            start_date=now, end_date=now + dt.timedelta(days=14)
        )
        for _, row in schedule.iterrows():
            if row["market_open"] > now:
                return row["market_open"]
        return now  # Should not happen

    def next_market_close(self) -> dt.datetime:
        """Returns the next market close time in UTC.

        Searches for the next scheduled market close event starting from the current
        UTC time. This method looks ahead up to 14 days.

        Returns:
            dt.datetime: The next market close time in UTC.
        """
        now = dt.datetime.now(dt.timezone.utc)
        schedule = self.calendar.schedule(
            start_date=now, end_date=now + dt.timedelta(days=14)
        )
        for _, row in schedule.iterrows():
            if row["market_close"] > now:
                return row["market_close"]
        return now

    def previous_market_open(self) -> dt.datetime:
        """Returns the previous market open time in UTC.

        Searches for the most recent past market open event relative to the current
        UTC time. This method looks back up to 14 days.

        Returns:
            dt.datetime: The previous market open time in UTC.
        """
        now = dt.datetime.now(dt.timezone.utc)
        schedule = self.calendar.schedule(
            start_date=now - dt.timedelta(days=14), end_date=now
        )
        # Iterate backwards
        for i in range(len(schedule) - 1, -1, -1):
            row = schedule.iloc[i]
            if row["market_open"] < now:
                return row["market_open"]
        return now

    def previous_market_close(self) -> dt.datetime:
        """Returns the previous market close time in UTC.

        Searches for the most recent past market close event relative to the current
        UTC time. This method looks back up to 14 days.

        Returns:
            dt.datetime: The previous market close time in UTC.
        """
        now = dt.datetime.now(dt.timezone.utc)
        schedule = self.calendar.schedule(
            start_date=now - dt.timedelta(days=14), end_date=now
        )
        # Iterate backwards
        for i in range(len(schedule) - 1, -1, -1):
            row = schedule.iloc[i]
            if row["market_close"] < now:
                return row["market_close"]
        return now

    def upcoming_holiday(self) -> dt.datetime:
        """Returns the next upcoming holiday in UTC.

        Identifies the next weekday that is not a trading day for the exchange.
        This method scans ahead up to 1 year.

        Returns:
            dt.datetime: The date of the next upcoming holiday (midnight UTC).
        """
        now = dt.datetime.now(dt.timezone.utc)
        current = now
        while True:
            current += dt.timedelta(days=1)
            if current.weekday() >= 5:  # Skip weekends
                continue

            sch = self.calendar.schedule(start_date=current, end_date=current)
            if sch.empty:
                return current.replace(hour=0, minute=0, second=0, microsecond=0)

            if (current - now).days > 365:
                return current  # Fallback

    def previous_holiday(self) -> dt.datetime:
        """Returns the previous holiday in UTC.

        Identifies the most recent past weekday that was not a trading day for
        the exchange. This method scans back up to 1 year.

        Returns:
            dt.datetime: The date of the previous holiday (midnight UTC).
        """
        now = dt.datetime.now(dt.timezone.utc)
        current = now
        while True:
            current -= dt.timedelta(days=1)
            if current.weekday() >= 5:  # Skip weekends
                continue

            sch = self.calendar.schedule(start_date=current, end_date=current)
            if sch.empty:
                return current.replace(hour=0, minute=0, second=0, microsecond=0)

            if (now - current).days > 365:
                return current
