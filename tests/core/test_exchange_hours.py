import pytest
import datetime as dt
import typing as tp
from silkroad.core.enums import Exchange

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo


@pytest.fixture
def utc_now():
    return dt.datetime(2023, 10, 25, 14, 0, 0, tzinfo=dt.timezone.utc)  # Wednesday


def test_timezone_mapping():
    assert Exchange.NYSE.timezone == "America/New_York"
    assert Exchange.LSE.timezone == "Europe/London"
    assert Exchange.HKEX.timezone in ["Asia/Hong_Kong", "Asia/Shanghai"]


def test_is_open_at():
    # NYSE Open: 9:30 - 16:00 ET
    # 2023-10-25 (Wed)
    # 14:00 UTC = 10:00 ET (Open)
    dt_open = dt.datetime(2023, 10, 25, 14, 0, 0, tzinfo=dt.timezone.utc)
    assert Exchange.NYSE.is_open_at(dt_open)
    assert not Exchange.NYSE.is_closed_at(dt_open)

    # 13:00 UTC = 09:00 ET (Closed)
    dt_closed = dt.datetime(2023, 10, 25, 13, 0, 0, tzinfo=dt.timezone.utc)
    assert not Exchange.NYSE.is_open_at(dt_closed)
    assert Exchange.NYSE.is_closed_at(dt_closed)


def test_market_hours_at():
    # 2023-10-25 (Wed)
    dt_open = dt.datetime(2023, 10, 25, 14, 0, 0, tzinfo=dt.timezone.utc)
    open_time, close_time = Exchange.NYSE.market_hours_at(dt_open)

    # Expected: 9:30 ET -> 13:30 UTC, 16:00 ET -> 20:00 UTC
    expected_open = dt.datetime(2023, 10, 25, 13, 30, 0, tzinfo=dt.timezone.utc)
    expected_close = dt.datetime(2023, 10, 25, 20, 0, 0, tzinfo=dt.timezone.utc)

    assert open_time == expected_open
    assert close_time == expected_close

    # Test closed time raises ValueError
    dt_closed = dt.datetime(2023, 10, 25, 13, 0, 0, tzinfo=dt.timezone.utc)
    with pytest.raises(ValueError):
        Exchange.NYSE.market_hours_at(dt_closed)


def test_holiday_at():
    # Christmas 2023 (Monday)
    dt_xmas = dt.datetime(2023, 12, 25, 12, 0, 0, tzinfo=dt.timezone.utc)
    assert Exchange.NYSE.holiday_at(dt_xmas)

    # Regular Wednesday
    dt_wed = dt.datetime(2023, 10, 25, 12, 0, 0, tzinfo=dt.timezone.utc)
    assert not Exchange.NYSE.holiday_at(dt_wed)


def test_next_market_open_close():
    # Mocking datetime.now is hard without a library like freezegun.
    # Instead, we rely on the logic being correct relative to "now".
    # But we can test the specific methods if we could inject "now".
    # Since we can't easily, we will test the logic by using a known past date
    # if the methods allowed passing "now", but they don't.
    # So we will just smoke test them to ensure they return future dates.

    now = dt.datetime.now(dt.timezone.utc)
    next_open = Exchange.NYSE.next_market_open()
    next_close = Exchange.NYSE.next_market_close()

    assert next_open >= now
    assert next_close >= now


def test_previous_market_open_close():
    now = dt.datetime.now(dt.timezone.utc)
    prev_open = Exchange.NYSE.previous_market_open()
    prev_close = Exchange.NYSE.previous_market_close()

    assert prev_open <= now
    assert prev_close <= now


def test_upcoming_holiday():
    # This should return a future date
    holiday = Exchange.NYSE.upcoming_holiday()
    now = dt.datetime.now(dt.timezone.utc)
    assert holiday > now
    # Check if it's really a holiday (empty schedule)
    assert Exchange.NYSE.holiday_at(holiday)


def test_previous_holiday():
    holiday = Exchange.NYSE.previous_holiday()
    now = dt.datetime.now(dt.timezone.utc)
    assert holiday < now
    assert Exchange.NYSE.holiday_at(holiday)


def test_current_methods_smoke():
    # Smoke test current methods.
    # Depending on when this test runs, it might raise ValueError or return values.
    try:
        Exchange.NYSE.current_market_open
        Exchange.NYSE.current_market_close
        Exchange.NYSE.current_market_hours
        assert Exchange.NYSE.is_open()
    except ValueError:
        assert not Exchange.NYSE.is_open()
