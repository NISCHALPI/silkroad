"""Portfolio management module for tracking and optimizing multi-asset investments.

This module provides the Portfolio class which enables tracking of asset positions,
rebalancing using mean-variance optimization, and strategy backtesting with
comprehensive performance metrics.

Typical usage example:

    from silkroad.portfolio import Portfolio
    from silkroad.core.base_models import UniformBarCollection
    import numpy as np

    # Create portfolio with equal weights
    weights = np.array([0.5, 0.5])
    portfolio = Portfolio(
        w=weights,
        cash=100000.0,
        universe=universe_data
    )

    # Run a strategy
    mask = np.ones(len(portfolio.log_returns), dtype=bool)
    results = portfolio.run_strategy(mask, risk_free_rate=0.02)
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
"""

from typing import Dict
from pydantic import BaseModel, Field, model_validator, ValidationInfo, ConfigDict
from ..core.base_models import UniformBarCollection
import numpy as np
import pandas as pd
from typing import Union
from ..functional.metrics import sortin_ratio, sharpe_ratio, max_drawdown
from ..functional.mean_variance import mean_variance_optimization

__all__ = [
    "Portfolio",
]


class Portfolio(BaseModel):
    """Manages a collection of investments with performance tracking and optimization.

    The Portfolio class represents a multi-asset investment portfolio that tracks
    positions over time. It provides functionality for calculating returns, rebalancing
    using mean-variance optimization, and running trading strategies with comprehensive
    performance analytics.

    Attributes:
        w: Array of asset weights that must sum to 1.0. Shape: (n_assets,)
        cash: Cash balance held in the portfolio (not currently used in calculations).
        universe: Historical price data for all investable assets.

    Examples:
        Create a portfolio with equal weights across two assets:

        >>> import numpy as np
        >>> weights = np.array([0.5, 0.5])
        >>> portfolio = Portfolio(
        ...     w=weights,
        ...     cash=100000.0,
        ...     universe=universe_data
        ... )

        Calculate log returns:

        >>> log_returns = portfolio.log_returns

        Rebalance to maximize returns:

        >>> portfolio.rebalance_portfolio(
        ...     mu=expected_returns,
        ...     cov=covariance_matrix,
        ...     target_return=0.10,
        ...     short_selling=False
        ... )
    """

    w: np.ndarray = Field(..., description="Weights of each asset in the portfolio.")
    cash: float = Field(..., description="Amount of cash in the portfolio.")
    universe: UniformBarCollection = Field(
        ...,
        description="Collection of historical bars for assets which the portfolio can invest in.",
    )
    model_config = ConfigDict(
        validate_assignment=True, validate_return=True, arbitrary_types_allowed=True
    )

    @model_validator(mode="after")
    def check_weights_length(self, info: ValidationInfo) -> "Portfolio":
        """Validates portfolio weights match universe and sum to 1.0.

        Args:
            info: Validation context information from Pydantic.

        Returns:
            The validated Portfolio instance.

        Raises:
            ValueError: If weights length doesn't match number of assets.
            ValueError: If weights don't sum to 1.0 (within floating point tolerance).
        """
        if len(self.w) != len(self.universe.symbols):
            raise ValueError(
                f"Length of weights {len(self.w)} does not match number of assets in universe {len(self.universe.symbols)}."
            )
        if not np.isclose(np.sum(self.w), 1.0):
            raise ValueError(f"Weights must sum to 1.0, but sum to {np.sum(self.w)}.")
        return self

    @property
    def closing_prices(self) -> pd.DataFrame:
        """Returns closing prices for all assets in wide format.

        Pivots the universe data to create a DataFrame with timestamps as index
        and asset symbols as columns.

        Returns:
            DataFrame with closing prices. Index is timestamp, columns are symbols.
        """
        return self.universe.df.reset_index().pivot(
            index="timestamp", columns="symbol", values="close"
        )

    @property
    def log_returns(self) -> pd.DataFrame:
        """Calculates log returns for all portfolio assets.

        Computes logarithmic returns as log(P_t / P_{t-1}) for each asset.
        The first row is dropped as it contains NaN values.

        Returns:
            DataFrame of log returns with same structure as closing_prices,
            but with one fewer row (first period dropped).
        """
        closing_prices = self.closing_prices
        return (closing_prices / closing_prices.shift(1)).apply(np.log).dropna()

    @property
    def current_timestamp(self) -> pd.Timestamp:
        """Returns the most recent timestamp in the universe data.

        Returns:
            The last timestamp as a pandas Timestamp object.
        """
        return self.universe.timestamps[-1]

    @property
    def first_timestamp(self) -> pd.Timestamp:
        """Returns the earliest timestamp in the universe data.

        Returns:
            The first timestamp as a pandas Timestamp object.
        """
        return self.universe.timestamps[0]

    @property
    def n_assets(self) -> int:
        """Returns the number of assets in the portfolio universe.

        Returns:
            Count of unique asset symbols in the universe.
        """
        return len(self.universe.symbols)

    def rebalance_portfolio(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        target_return: float,
        short_selling: bool = False,
    ) -> None:
        """Rebalances portfolio weights using mean-variance optimization.

        Updates the portfolio weights by solving the mean-variance optimization
        problem: minimize portfolio variance subject to achieving a target expected
        return and weights summing to 1.0. Optionally allows short selling.

        Args:
            mu: Expected returns for each asset. Shape: (n_assets,)
            cov: Covariance matrix of asset returns. Shape: (n_assets, n_assets)
            target_return: Desired annualized portfolio return (e.g., 0.10 for 10%).
            short_selling: If True, allows negative weights (short positions).
                If False, constrains all weights to be non-negative. Default is False.

        Raises:
            ValueError: If mu length doesn't match number of assets.
            ValueError: If cov dimensions don't match (n_assets, n_assets).

        Note:
            This method modifies the portfolio weights in-place. The optimization
            may fail if the target return is not achievable given the constraints.

        Examples:
            Rebalance with long-only constraint:

            >>> portfolio.rebalance_portfolio(
            ...     mu=np.array([0.10, 0.15]),
            ...     cov=np.array([[0.04, 0.01], [0.01, 0.09]]),
            ...     target_return=0.12,
            ...     short_selling=False
            ... )
        """
        # Check dimensions
        if mu.shape[0] != self.n_assets:
            raise ValueError(
                f"Expected returns vector length {mu.shape[0]} does not match number of assets {self.n_assets}."
            )
        if cov.shape[0] != self.n_assets or cov.shape[1] != self.n_assets:
            raise ValueError(
                f"Covariance matrix dimensions {cov.shape} do not match number of assets {self.n_assets}."
            )

        # Perform mean-variance optimization
        result = mean_variance_optimization(
            mu=mu,  # type: ignore
            cov=cov,  # type: ignore
            target_return=target_return,
            short_selling=short_selling,
        )

        # Rebalance weights
        self.w = result["weights"]  # type: ignore

    def run_strategy(
        self,
        mask: Union[np.ndarray, pd.Series],
        risk_free_rate: float = 0.0,
    ) -> Dict[str, float | pd.Series]:
        """Execute a trading strategy and compute comprehensive performance metrics.

        This method simulates a dynamic trading strategy where portfolio positions are
        controlled by a boolean mask signal. It applies the mask to control exposure,
        computes portfolio returns, and calculates key risk-adjusted performance metrics
        including Sharpe ratio, Sortino ratio, and maximum drawdown.

        The calculation process:
        1. Applies the boolean mask to log returns (True = invested, False = cash)
        2. Converts masked log returns to simple returns for accurate compounding
        3. Computes weighted portfolio returns using asset weights
        4. Converts back to log returns for the time series
        5. Calculates risk-adjusted performance metrics

        Note:
            The mask is applied to all returns except the last period (mask[:-1]) to
            prevent look-ahead bias - trading decisions at time t should only affect
            returns from t to t+1.

        Args:
            mask: Boolean array or Series indicating when to invest in risky assets.
                Must have length equal to number of timestamps. True means fully invested
                according to weights, False means hold cash (zero returns).
            risk_free_rate: Annualized risk-free rate used for Sharpe and Sortino ratio
                calculations. Default is 0.0 (assumes zero risk-free rate).

        Returns:
            Dictionary containing strategy performance metrics with keys:
                - 'log_returns': Time series of strategy log returns (pd.Series)
                - 'sharpe_ratio': Annualized Sharpe ratio (float)
                - 'sortin_ratio': Annualized Sortino ratio (float)
                - 'max_drawdown': Maximum drawdown as positive float (float)

        Examples:
            >>> # Simple buy-and-hold strategy
            >>> mask = np.ones(len(portfolio.log_returns), dtype=bool)
            >>> results = portfolio.run_strategy(mask, risk_free_rate=0.02)
            >>> print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")

            >>> # Trend-following strategy with moving average
            >>> prices = portfolio.closing_prices.mean(axis=1)
            >>> mask = prices > prices.rolling(20).mean()
            >>> results = portfolio.run_strategy(mask, risk_free_rate=0.03)
            >>> cumulative_returns = np.exp(results['log_returns'].cumsum())
        """
        masked_log_rets = (self.log_returns.values.T * mask[:-1]).T
        portfolio_rets = (np.exp(masked_log_rets) - 1) @ self.w  # type: ignore
        log_returns = pd.Series(np.log1p(portfolio_rets), index=self.log_returns.index)

        # Calculate performance metrics
        sharpe = sharpe_ratio(
            returns=log_returns.to_numpy(),
            risk_free_rate=risk_free_rate,
            periods=self.universe.horizon.periods_annually(),  # Get periods based on universe horizon
        )
        sortino = sortin_ratio(
            returns=log_returns.to_numpy(),
            minimum_acceptable_return=risk_free_rate,
            periods=self.universe.horizon.periods_annually(),  # Get periods based on universe horizon
        )
        max_dd = max_drawdown(returns=log_returns.to_numpy())

        return {
            "log_returns": log_returns,
            "sharpe_ratio": sharpe,
            "sortin_ratio": sortino,
            "max_drawdown": max_dd,
        }
