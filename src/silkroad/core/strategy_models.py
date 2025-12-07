"""Trading Strategy Models.

This module provides a compact and efficient interface for defining trading strategies.
The core abstraction is the `Strategy` class, which requires implementing a `compute_weights`
method. This method is designed to handle the entire dataset at once (batch processing),
enabling vectorized execution where possible.
"""

import abc
from typing import Any, Callable, Dict, Optional, Union
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from rich import progress
import riskfolio as rp

from silkroad.core.data_models import UniformBarCollection
from silkroad.core.enums import Horizon
from silkroad.functional.backtest import backtest_with_metrics

__all__ = ["Strategy", "BuyAndHoldStrategy", "RiskfolioStrategy"]


class Strategy(abc.ABC):
    """Abstract base class for trading strategies.

    Strategies must implement `compute_weights` to map market data to asset weights.
    """

    @abc.abstractmethod
    def compute_weights(self, data: Any, **kwargs) -> Any:
        """Compute weights for the entire dataset.

        Args:
            data: Input market data (e.g., UniformBarCollection, DataFrame, jax.Array).
            **kwargs: Strategy-specific arguments.

        Returns:
            Asset weights (e.g., DataFrame, jax.Array).
        """
        pass

    def __call__(self, data: Any, **kwargs) -> Any:
        return self.compute_weights(data, **kwargs)

    def backtest(
        self,
        universe: UniformBarCollection,
        capital: float,
        risk_free_rate: float = 0.04,
        transaction_cost_bp: float = 0.0,
        alpha: float = 0.05,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run a backtest using the strategy.

        Args:
            universe: Historical market data.
            capital: Initial capital.
            risk_free_rate: Risk-free rate for metrics.
            transaction_cost_bp: Transaction cost in basis points.
            alpha: Significance level for VaR/CVaR.
            **kwargs: Arguments passed to `compute_weights`.

        Returns:
            Dictionary containing performance metrics.
        """
        # 1. Compute weights for the entire history
        # We assume compute_weights returns a DataFrame or we need to standardize it
        weights_output = self.compute_weights(universe, **kwargs)

        # Standardize weights to DataFrame if not already
        if isinstance(weights_output, pd.DataFrame):
            weights_df = weights_output
            # Check for rebalance flag, default to True if missing
            if "should_rebalance" not in weights_df.columns:
                rebalance_mask = jnp.ones(len(weights_df), dtype=bool)
                target_weights = jnp.array(weights_df.values)
            else:
                rebalance_mask = jnp.array(weights_df["should_rebalance"].values)
                target_weights = jnp.array(
                    weights_df.drop(columns=["should_rebalance"]).values
                )
        elif isinstance(weights_output, (jnp.ndarray, np.ndarray)):
            # Assume it's just weights, rebalance every step
            target_weights = jnp.array(weights_output)
            rebalance_mask = jnp.ones(len(target_weights), dtype=bool)
        else:
            raise TypeError(f"Unsupported weights output type: {type(weights_output)}")

        # 2. Prepare data for functional backtest
        log_returns = jnp.array(universe.log_returns.values)
        periods_annually = universe.horizon.periods_annually()

        # Align lengths if necessary (weights usually start from t=1 or t=0 depending on logic)
        # functional.backtest expects log_returns[t] and target_weights[t] (target for end of t)
        # Usually we generate weights based on data up to t, to hold for t+1.
        # Let's assume weights aligned with returns for now.
        min_len = min(len(log_returns), len(target_weights))
        log_returns = log_returns[:min_len]
        target_weights = target_weights[:min_len]
        rebalance_mask = rebalance_mask[:min_len]

        # 3. Run backtest
        results = backtest_with_metrics(
            init_value=capital,
            init_weights=target_weights[0],  # Start with first target weights
            log_returns=log_returns,
            target_weights=target_weights,
            transaction_cost_bp=transaction_cost_bp,
            rebalance_mask=rebalance_mask,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_annually,
            alpha=alpha,
        )

        return {
            "metrics": pd.Series(
                {k: float(v) for k, v in results.items() if jnp.ndim(v) == 0}
            ),
            "log_returns": pd.Series(results["returns_history"]),  # type: ignore
        }


class BuyAndHoldStrategy(Strategy):
    """Efficient Buy and Hold Strategy."""

    def compute_weights(
        self, data: Any, hold_assets: Optional[list[str]] = None, **kwargs
    ) -> pd.DataFrame:
        """Compute weights using pandas broadcasting.

        Args:
            data: UniformBarCollection or DataFrame.
            hold_assets: List of assets to hold.

        Returns:
            DataFrame of weights.
        """
        if isinstance(data, UniformBarCollection):
            df = data.arthimetic_returns
        else:
            df = data

        if hold_assets is None:
            hold_assets = df.columns.tolist()

        weight = 1.0 / len(hold_assets)  # type: ignore

        # Create weights DataFrame with constant values
        weights_df = pd.DataFrame(0.0, index=df.index, columns=df.columns)
        weights_df[hold_assets] = weight  # type: ignore

        # Add rebalance flag (False)
        weights_df["should_rebalance"] = False

        return weights_df


class RiskfolioStrategy(Strategy):
    """Riskfolio-Lib based strategy (Iterative)."""

    def compute_weights(
        self,
        data: UniformBarCollection,
        risk_free_rate: float = 0.04,
        min_data_points: int = 50,
        model: str = "Classic",
        rm: str = "MV",
        obj: str = "Sharpe",
        method_mu: str = "hist",
        method_cov: str = "hist",
        **kwargs,
    ) -> pd.DataFrame:
        """Compute weights iteratively using Riskfolio.

        Args:
            data: UniformBarCollection.
            risk_free_rate: Annualized risk-free rate.
            min_data_points: Minimum history required.
            model: Model type for Riskfolio.
            rm: Risk measure.
            obj: Objective function.
            method_mu: Method for expected returns.
            method_cov: Method for covariance.
            **kwargs: Additional Riskfolio Portfolio parameters.

        Returns:
            DataFrame of weights.
        """
        universe = data
        returns = universe.arithmetic_returns
        weights_list = []
        rebalance_flags = []

        warnings.filterwarnings("ignore")

        for i in progress.track(
            range(0, len(returns)), description="Computing Riskfolio Weights"
        ):
            ## We lookupto current time index i and forcast weights for next period
            ## Hence we use returns up to i (inclusive). Normally, in live trading,
            ## we wait till almost the end of period i to compute weights for i+1.
            ## For eg, if data is daily and we are at day i, we use returns up to day i
            ## to compute weights for day i+1 and rebalance at either very end of day i
            ## or at the beginning of day i+1.
            df = returns.iloc[: i + 1]

            # Optimization logic
            if len(df) < min_data_points:
                w = pd.Series(1.0 / len(df.columns), index=df.columns)
                should_rebalance = False
            else:
                try:
                    # Simplified Riskfolio call for brevity, can be expanded
                    port = rp.Portfolio(returns=df, **kwargs)
                    port.assets_stats(method_mu=method_mu, method_cov=method_cov)
                    w = port.optimization(
                        model=model,
                        rm=rm,
                        obj=obj,
                        rf=risk_free_rate / universe.horizon.periods_annually(),  # type: ignore  # Need to convert to per-period
                    )
                    if w is None:
                        w = pd.Series(1.0 / len(df.columns), index=df.columns)
                        should_rebalance = False
                    else:
                        w = w.iloc[:, 0]  # Extract series
                        should_rebalance = True
                except Exception:
                    w = pd.Series(1.0 / len(df.columns), index=df.columns)
                    should_rebalance = False

            weights_list.append(w)
            rebalance_flags.append(should_rebalance)

        warnings.resetwarnings()

        weights_df = pd.DataFrame(weights_list, index=universe.timestamps[1:])
        rebalance_df = pd.DataFrame(
            {"should_rebalance": rebalance_flags}, index=universe.timestamps[1:]
        )
        return pd.concat([weights_df, rebalance_df], axis=1)
