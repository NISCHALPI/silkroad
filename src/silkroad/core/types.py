"""Type definitions and validators for time series data."""

from typing import Annotated, List
from alpaca.data.models.bars import Bar
import annotated_types as at
from pydantic.functional_validators import BeforeValidator

__all__ = [
    "BarList",
]

# A list of Bar objects with at least one elements
BarList = Annotated[List[Bar], at.MinLen(1)]


# Check if Bar are of same symbol and Bars are increasing in time
def validate_uniform_bars(bars: List[Bar]) -> List[Bar]:
    """Validator to ensure all bars have the same symbol and are time-ordered.

    Args:
        bars: List of Bar objects to validate.

    Returns:
        Original list if validation passes.s

    Raises:
        ValueError: If bars have differing symbols or are not in strictly
            increasing order of timestamp.
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
