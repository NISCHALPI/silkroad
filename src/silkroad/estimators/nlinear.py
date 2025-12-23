"""NLinear model for time series forecasting and statistical decomposition.

This module provides the NLinear model, a neural time series forecasting architecture
that decomposes input signals into trend and seasonal components before applying
linear transformations independently to each component.

The NLinear model is particularly effective for time series with strong seasonal
patterns and can scale efficiently across multiple features using shared weight
matrices. It leverages JAX for efficient computation and automatic differentiation,
with support for distributed training via Flax NNX.

Key Features:
    - Decomposition: Separates trend and seasonal components using centered
      moving average filtering.
    - Feature Sharing: By default, weights are shared across all input features,
      reducing parameter count while maintaining expressivity.
    - JAX-based: Fully differentiable implementation using JAX arrays and primitives.
    - Progress Tracking: Training includes rich progress bar visualization with
      real-time loss monitoring.
    - Flexible Training: Supports configurable learning rates, optimizers, and
      batch sizes via sliding window sampling.

Example:
    >>> import jax
    >>> from silkroad.estimators.nlinear import NLinear
    >>> import optax
    >>>
    >>> # Create dummy time series data (100 timesteps, 3 features)
    >>> x = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
    >>>
    >>> # Initialize model with lookback=10, lookforward=5
    >>> rngs = jax.numpy as jnp.random.default_rng(seed=0)
    >>> model = NLinear(lookback=10, lookforward=5, kernel_size=5, rngs=rngs)
    >>>
    >>> # Create optimizer
    >>> optimizer = optax.adam(learning_rate=0.01)
    >>> # Fit the model with progress bar
    >>> fitted_model = NLinear.fit(
    ...     x=x,
    ...     model=model,
    ...     gt=optimizer,
    ...     max_iterations=100,
    ...     sliding_step=1,
    ...     verbose=True
    ... )

Attributes:
    lookback (int): Number of historical timesteps used for prediction.
    lookforward (int): Number of future timesteps to predict.
    kernel_size (int): Size of the kernel for centered moving average detrending.
        Must be odd.
    Ws (flax.nnx.Linear): Linear layer applied to seasonal component.
    Wt (flax.nnx.Linear): Linear layer applied to trend component.
"""

import typing as tp

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
from rich.progress import Progress

from silkroad.estimators.utils import detrend, sliding_window
from silkroad.logging.logger import logger


class NLinear(nnx.Module):
    """Decomposition-based linear model for univariate and multivariate time series forecasting.

    NLinear is a neural time series forecasting model that achieves competitive
    performance by decomposing the input signal into trend and seasonal components
    using centered moving average filtering, then applying independent linear
    transformations to each component.

    The key insight is that many time series can be effectively forecasted using
    simple linear models when the input is properly decomposed. This approach is
    particularly effective for seasonal data and can scale to high-dimensional
    feature spaces due to parameter sharing.

    Architecture:
        1. **Decomposition**: The input is normalized by subtracting its last value,
           then decomposed into trend (via centered moving average) and seasonal
           (residual) components.
        2. **Padding**: The lookback window is extended by centered padding based
           on kernel_size to capture additional historical context for detrending.
        3. **Linear Projection**: Independent linear layers (Wt and Ws) project
           trend and seasonal components from lookback to lookforward timesteps.
        4. **Reconstruction**: The forecast is obtained by summing the trend forecast,
           seasonal forecast, and the normalization constant (last value).

    Attributes:
        lookback (int): Number of past timesteps used to condition the model.
            Must be positive.
        lookforward (int): Number of future timesteps to predict. Must be positive.
        kernel_size (int): Size of the moving average kernel for trend extraction.
            Must be a positive odd integer.
        Ws (flax.nnx.Linear): Learnable linear transformation for seasonal component.
            Shape: (lookback, lookforward).
        Wt (flax.nnx.Linear): Learnable linear transformation for trend component.
            Shape: (lookback, lookforward).

    Raises:
        ValueError: If kernel_size is not an odd integer.

    Example:
        >>> import jax
        >>> from silkroad.estimators.nlinear import NLinear
        >>>  model = NLinear(lookback=24, lookforward=6, kernel_size=3, rngs=rngs)
        >>> x_forecast = model(x)  # (6, num_features)
        >>> fitted_model = NLinear.fit(x, model, optimizer, max_iterations=100)

    References:
        The decomposition-based approach is inspired by time series analysis
        literature and adaptive filtering techniques. For details on NLinear and
        related methods, see recent time series forecasting benchmarks.
    """

    def __init__(
        self, lookback: int, lookforward: int, kernel_size: int, rngs: nnx.Rngs
    ) -> None:
        """Initialize the NLinear forecasting model.

        Creates the NLinear model by instantiating two independent linear layers
        for trend and seasonal component transformations. The linear layers are
        initialized using the provided Flax RNG state and are trainable parameters.

        Args:
            lookback (int): Number of historical timesteps to use as input features.
                Typical values: 24 (hourly), 7 (daily), or 12-60 (high-frequency).
                Must be positive.
            lookforward (int): Number of future timesteps to predict. Also called
                the forecasting horizon or prediction length. Must be positive.
                Example: lookforward=1 for one-step-ahead, 24 for a day ahead.
            kernel_size (int): Window size for the centered moving average filter
                used to extract trend. Must be a positive odd integer.
                - Smaller kernels (3, 5): Capture local trends.
                - Larger kernels (7, 13): Smooth over longer periods.
            rngs (flax.nnx.Rngs): Flax RNG state for weight initialization.
                Ensures reproducibility of weight initialization.

        Raises:
            ValueError: If kernel_size is not an odd positive integer.

        Example:
            >>> import jax
            >>> import flax.nnx as nnx
            >>> rngs = nnx.Rngs(0)
            >>> model = NLinear(lookback=12, lookforward=3, kernel_size=5, rngs=rngs)
        """
        self.lookback = lookback
        self.lookforward = lookforward

        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")
        self.kernel_size = kernel_size

        self.Wt = nnx.Linear(
            in_features=self.lookback, out_features=lookforward, rngs=rngs
        )
        self.Ws = nnx.Linear(
            in_features=self.lookback, out_features=lookforward, rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Perform a forward pass to generate forecasts.

        This method implements the core NLinear forecasting computation:
            1. Validates and reshapes input to 2D (seq_length, num_features).
            2. Extracts the last true_lookback timesteps with centered padding.
            3. Normalizes the data by subtracting its final value.
            4. Decomposes the normalized signal into trend and seasonal components
               using centered moving average filtering.
            5. Applies independent linear transformations to each component.
            6. Reconstructs the forecast by summing components and adding back
               the normalization constant.

        Args:
            x (jax.Array): Input time series data. Can be:
                - 1D array of shape (seq_length,): Automatically expanded to
                  (seq_length, 1).
                - 2D array of shape (seq_length, num_features): Multiple features
                  or independent time series.
                Must have at least lookback + (kernel_size - 1) // 2 timesteps.

        Returns:
            jax.Array: Forecasted values of shape (lookforward, num_features).
                The forecast contains lookforward future timesteps for each feature.

        Raises:
            ValueError: If input has more than 2 dimensions.

        Example:
            >>> x = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
            >>> model = NLinear(lookback=10, lookforward=5, kernel_size=3, rngs=rngs)
            >>> forecast = model(x)  # Shape: (5, 3)
        """
        # If n-dim is greater than 2, raise error
        if jnp.ndim(x) > 2:
            raise ValueError("Input array must be 1D or 2D.")
        # If one dimensional, expand dimensions
        if jnp.ndim(x) == 1:
            x = jnp.expand_dims(x, axis=-1)

        # Calculate true lookback based on kernel size
        true_lookback = self.lookback + (self.kernel_size - 1) // 2  # Centered Padding
        # Check if input length is sufficient
        if x.shape[0] < true_lookback:
            raise ValueError(
                f"Input length {x.shape[0]} is less than required "
                f"true_lookback {true_lookback}."
            )
        # Extract the last true_lookback time steps
        x = x[-true_lookback:, :]
        last = x[[-1], :]

        x_hat = x - last  # Normalization step

        # Detrend the input
        trend, seasonal = jax.vmap(detrend, in_axes=(1, None), out_axes=1)(
            x_hat, self.kernel_size
        )
        # Cutoff to lookback
        seasonal = seasonal[-self.lookback :, :]
        trend = trend[-self.lookback :, :]

        # Apply linear transformations
        seasonal_forecast = self.Ws(seasonal.T).T
        trend_forecast = self.Wt(trend.T).T
        forecast = seasonal_forecast + trend_forecast + last
        return forecast

    @staticmethod
    def fit(
        x: jax.Array,
        model: "NLinear",
        gt: optax.GradientTransformation,
        max_iterations: int,
        sliding_step: int = 1,
        verbose: bool = False,
    ) -> "NLinear":
        """Train the NLinear model using stochastic gradient descent.

        This static method trains the provided NLinear model using a sliding window
        approach over the input time series. The training loop:
            1. Splits the model into graph definition and trainable state.
            2. Initializes the optimizer state.
            3. For each iteration, slides a window over the data (with stride `sliding_step`)
               to create multiple training examples per epoch.
            4. Computes the mean squared error loss and backpropagates gradients.
            5. Updates model parameters using the provided optimizer.
            6. Displays progress via a rich progress bar showing average loss per epoch.
            7. Optionally logs detailed information for each iteration if verbose=True.

        The sliding window generates training pairs (x_in, y_true) where:
            - x_in: The last lookback timesteps, shape (lookback, num_features).
            - y_true: The next lookforward timesteps, shape (lookforward, num_features).

        Args:
            x (jax.Array): Training time series data of shape (seq_length, num_features).
                Must have at least lookback + lookforward timesteps to generate
                at least one training example.
            model (NLinear): An initialized NLinear instance to train.
            gt (optax.GradientTransformation): Optimizer (gradient transformation) from optax.
                Example: optax.adam(learning_rate=1e-3).
            max_iterations (int): Number of training epochs. Each epoch slides a window
                over the entire dataset once.
            sliding_step (int, optional): Stride for the sliding window. Controls how many
                timesteps to advance per training example within an epoch.
                - sliding_step=1 (default): Create examples at every timestep.
                - sliding_step=lookforward: Non-overlapping training examples.
                Default: 1.
            verbose (bool, optional): If True, logs iteration-level loss information.
                If False, only displays progress bar without console logs.
                Default: False.

        Returns:
            NLinear: The trained model with updated parameters. The model graph
                definition (graph structure) remains unchanged; only the state
                (weights and biases) is updated.

        Raises:
            ValueError: If the time series is shorter than lookback + lookforward,
                preventing generation of at least one training example.

        Example:
            >>> import jax
            >>> import optax
            >>> from silkroad.estimators.nlinear import NLinear
            >>>
            >>> # Generate synthetic data
            >>> x = jax.random.normal(jax.random.PRNGKey(0), (500, 2))
            >>>
            >>> # Initialize model
            >>> rngs = nnx.Rngs(0)
            >>> model = NLinear(lookback=24, lookforward=6, kernel_size=5, rngs=rngs)
            >>>
            >>> # Train with Adam optimizer
            >>> optimizer = optax.adam(learning_rate=0.01)
            >>> fitted_model = NLinear.fit(
            ...     x=x,
            ...     model=model,
            ...     gt=optimizer,
            ...     max_iterations=50,
            ...     sliding_step=1,
            ...     verbose=True
            ... )

        Notes:
            - Loss is computed as mean squared error (MSE) across all features.
            - Average loss per epoch is displayed in the progress bar.
            - For high-dimensional data, consider increasing sliding_step to reduce
              the number of training examples per epoch.
            - The training loop is JAX JIT-compiled for efficiency.
        """
        # Expand the dimensions if input is 1D
        if jnp.ndim(x) == 1:
            if verbose:
                logger.info("Expanding input dimensions for 1D input.")
            x = jnp.expand_dims(x, axis=-1)

        # Split model into gdef and state
        gdef, state = nnx.split(
            model
        )  # No need to filter here since no stateful params
        opt_state = gt.init(state)

        @jax.jit
        def loss_fn(state, x, y):
            model = nnx.merge(gdef, state)
            y_pred = model(x)
            loss = jnp.mean((y - y_pred) ** 2)
            return loss

        @jax.jit
        def update(state, opt_state, x, y):
            loss, grads = jax.value_and_grad(loss_fn)(state, x, y)
            updates, opt_state = gt.update(grads, opt_state, state)
            state = optax.apply_updates(state, updates)
            return state, opt_state, loss

        # Run the iteration
        if verbose:
            logger.info("Starting NLinear fitting process.")
        with Progress() as progress:
            task = progress.add_task(
                "[cyan]Fitting NLinear model...", total=max_iterations
            )
            for iterations in range(max_iterations):

                slice_iterator = sliding_window(
                    n=x.shape[0],
                    lookback=model.lookback,
                    lookforward=model.lookforward,
                    step=sliding_step,
                )
                epoch_loss = 0.0
                batch_count = 0
                for start_window, end_window in slice_iterator:
                    x_in = x[start_window, :]
                    y_true = x[end_window, :]
                    state, opt_state, batch_loss = update(
                        state, opt_state, x_in, y_true
                    )
                    epoch_loss += float(batch_loss)
                    batch_count += 1

                avg_loss = epoch_loss / batch_count if batch_count > 0 else float("nan")
                progress.update(
                    task,
                    advance=1,
                    description=f"[cyan]Avg Loss: {avg_loss:.6f}",
                )
                if verbose:
                    logger.info(
                        f"Iteration {iterations + 1}/{max_iterations}, Avg Loss: {avg_loss:.6f}"
                    )

        # Merge the final state back into the model
        fitted_model = nnx.merge(gdef, state)
        return fitted_model

    def forcast(
        self,
        x: jax.Array,
        n_steps: int,
    ) -> jax.Array:
        """Generate multi-step ahead forecasts using autoregressive generation.

        This method performs iterative forecasting by:
            1. Taking an initial history window of length lookback or longer.
            2. Predicting lookforward timesteps into the future.
            3. Appending predictions to the history.
            4. Repeating steps 2-3 until n_steps future values are generated.

        The autoregressive approach allows forecasting beyond the model's native
        lookforward horizon by reusing its own predictions as input.

        Args:
            x (jax.Array): Initial historical time series data. Can be:
                - 1D array of shape (seq_length,): Automatically expanded to
                  (seq_length, 1).
                - 2D array of shape (seq_length, num_features): Multiple features.
                Must have at least lookback timesteps.
            n_steps (int): Number of future timesteps to forecast.
                Example: n_steps=24 for a 24-step ahead forecast.
                Must be positive.

        Returns:
            jax.Array: All forecasted values (from step 1 to n_steps) concatenated
                along the time axis, shape (n_steps, num_features).
                Note: Returns only the new predictions, not including the input history.

        Raises:
            ValueError: If input has more than 2 dimensions.

        Example:
            >>> # Historical data: 100 timesteps, 1 feature
            >>> x_history = jax.random.normal(jax.random.PRNGKey(0), (100, 1))
            >>> model = NLinear(lookback=24, lookforward=6, kernel_size=3, rngs=rngs)
            >>>
            >>> # Forecast 48 steps (8 * 6) into the future
            >>> forecast = model.forcast(x_history, n_steps=48)  # Shape: (48, 1)

        Notes:
            - For long-term forecasts, prediction errors can accumulate due to
              the autoregressive nature. Consider using ensemble methods or
              retraining for improved accuracy on very long forecasts.
            - Each iteration uses lookforward steps from the model; the total
              number of forward passes is ceil(n_steps / lookforward).
        """
        # If n-dim is greater than 2, raise error
        if jnp.ndim(x) > 2:
            raise ValueError("Input array must be 1D or 2D.")
        # If one dimensional, expand dimensions
        if jnp.ndim(x) == 1:
            x = jnp.expand_dims(x, axis=-1)

        forecasts = []
        carry = x
        for _ in range(n_steps):
            y_pred = self(carry)
            forecasts.append(y_pred)
            carry = jnp.concatenate([carry, y_pred], axis=0)

        return jnp.concatenate(forecasts, axis=0)
