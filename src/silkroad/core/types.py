"""Reusable type definitions for the Silkroad data module.

This module provides type aliases and constrained types that can be used across
different parts of the package for type safety and validation.

Type Aliases:
    BarList: A list of Bar objects with at least one element.
    SymbolDict: A dictionary mapping symbols (str) to any value type.
    TimeSeriesDict: A dictionary mapping symbols to their respective time series data.

These types can be reused across different modules for consistent type checking
and validation.
"""

from typing import Annotated, List
from alpaca.data.models import Bar
import annotated_types as at
from pydantic.functional_validators import BeforeValidator

__all__ = [
    "BarList",
]

# A list of Bar objects with at least one element
BarList = Annotated[List[Bar], at.MinLen(1)]


# Check if Bar are of same symbol and Bars are increasing in time
def validate_uniform_bars(bars: List[Bar]) -> List[Bar]:
    """Validator to ensure all bars in the list have the same symbol.

    Args:
        bars (List[Bar]): The list of Bar objects to validate.
    Returns:
        List[Bar]: The original list if validation passes.
    Raises:
        ValueError: If bars have differing symbols.
    """
    if not bars:
        return bars  # Empty list is considered valid

    first_symbol = bars[0].symbol
    for bar in bars:
        if bar.symbol != first_symbol:
            raise ValueError(
                f"All bars must have the same symbol. Found '{bar.symbol}' and '{first_symbol}'."
            )

    # Chekc if bars are sorted by time
    for i in range(1, len(bars)):
        if bars[i].timestamp <= bars[i - 1].timestamp:
            raise ValueError("Bars must be in strictly increasing order of timestamp.")
    return bars


# A list of Uniform Bar objects with at least one element and validated for same symbol
UniformBarList = Annotated[
    List[Bar], at.MinLen(1), BeforeValidator(validate_uniform_bars)
]
