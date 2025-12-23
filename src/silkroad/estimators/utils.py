import typing as tp

import jax
import jax.numpy as jnp

__all__ = [
    "centered_moving_average",
    "detrend",
    "sliding_window",
]


def centered_moving_average(x: jax.Array, kernel_size: int) -> jax.Array:
    """Computes the centered moving average of a 1D array.


    Args:
        x: Input 1D array.
        kernel_size: Window size for the moving average. Must be an odd integer.

    Returns:
        A 1D array containing the centered moving average.
    """
    # 1. Create Kernel (Shape depends on k, so k must be static)
    kernel = jnp.ones(kernel_size) / kernel_size

    # 2. Calculate Pad Size (Value depends on k)
    pad_size = (kernel_size - 1) // 2

    # 3. Pad (Amount depends on k)
    # mode='edge' works fine in JIT
    x_padded = jnp.pad(x, pad_size, mode="edge")

    # 4. Convolve
    return jnp.convolve(x_padded, kernel, mode="valid")


def detrend(
    x: jax.Array,
    kernel_size: int,
) -> tp.Tuple[jax.Array, jax.Array]:
    """Detrends a 1D array using centered moving average.

    Args:
        x: Input 1D array.
        kernel_size: Window size for the moving average. Must be an odd integer.

    Returns:
        A tuple containing:
            - The trend component (centered moving average).
            - The detrended component (original array minus trend).
    """
    t = centered_moving_average(x, kernel_size)
    return t, x - t


def sliding_window(
    n: int,
    lookback: int,
    lookforward: int,
    step: int = 1,
) -> tp.Iterator[tp.Tuple[slice, slice]]:
    """Generates sliding window indices for time series data.

    Args:
        n: Length of the time series.
        lookback: Number of past time steps to include in each input window.
        lookforward: Number of future time steps to include in each output window.
        step: Step size between windows.

    Yields:
        Tuples of slices representing input and output windows.

    Note:
        - The window starts at starts at least from lookback and ends at most at n - lookforward.
        - The function yields slices for indexing into the time series data.
    """
    if n < lookback + lookforward:
        raise ValueError(
            "Time series length n must be at least lookback + lookforward."
        )

    start = lookback
    while start + lookforward <= n:
        input_slice = slice(start - lookback, start)
        output_slice = slice(start, start + lookforward)
        yield input_slice, output_slice
        start += step
