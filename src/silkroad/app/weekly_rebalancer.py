"""Weekly rebalancer strategy implementation."""

from ..logger.logger import logger
from ..portfolio import Portfolio
from ..core import Horizon, UniformBarSet
import pandas as pd
from .feature_extrator import ohlcv_to_features
from alpaca.data.models import Bar
from typing import List


class WeeklyRebalancer:
    DFREQ = Horizon.DAILY  # We ingest daily bars
    LOOKBACK = Horizon.ANNUAL
    REBALANCE_ON = "FRIDAY"

    def __init__(
        self,
        portfolio: Portfolio,
    ) -> None:

        if portfolio.universe.horizon != Horizon.DAILY:
            raise ValueError(
                "WeeklyRebalancer only currently supports daily horizon universe."
            )
        self.portfolio = portfolio

        # Check that we have sufficient lookback data for all tickers
        if not self._check_lookback_data():
            raise ValueError(
                """Insufficient lookback data in the portfolio universe.
                Add more Bars to the universe to cover the lookback period.
                """
            )

    def _check_lookback_data(
        self,
    ) -> bool:
        """Check if there is sufficient lookback data in the portfolio universe.

        Returns:
            bool: True if sufficient data is available, False otherwise.


        """
        if not self.portfolio.universe.n_bars >= self.LOOKBACK.periods_annually():
            logger.debug(
                f"Insufficient lookback data in the portfolio universe. Required: {self.LOOKBACK.periods_annually()}, Available: {self.portfolio.universe.n_bars}"
            )
            return False

        return True

    def get_state(
        self,
        ticker: str,
        spy_ohlcv: pd.DataFrame,
        macro_data: pd.DataFrame,
    ) -> pd.Series:
        """Get the current state for a given ticker.

        Args:
            ticker (str): The ticker symbol.
        Returns:
            pd.Series: The current state features.
        """
        ohlcv = self.portfolio.universe.bar_map[ticker].df
        return ohlcv_to_features(
            ohlcv=ohlcv,
            market_ohlcv=spy_ohlcv,
            macro_data=macro_data,
        )

    def push_new_bars(
        self,
        new_bars: dict[str, List[Bar]],
    ) -> None:
        """Push new bars to the portfolio universe.

        Args:
            new_bars (dict[str, pd.DataFrame]): A dictionary mapping ticker symbols to their new OHLCV data.
        """
        bar_map = {
            ticker: UniformBarSet(
                symbol=ticker,
                horizon=self.DFREQ,
                bars=new_bars[ticker],
            )
            for ticker in new_bars
        }

        self.portfolio.universe.push(new_bars=bar_map)

    def should_rebalance(
        self,
        current_date: pd.Timestamp,
    ) -> bool:
        """Check if the portfolio should be rebalanced on the current date.

        Args:
            current_date (pd.Timestamp): The current date.
        Returns:
            bool: True if rebalancing should occur, False otherwise.
        """
        return current_date.weekday() == pd.Timestamp(self.REBALANCE_ON).weekday()
