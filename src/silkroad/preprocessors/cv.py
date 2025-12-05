"""Module providing purged walk-forward cross-validation for MultiIndex DataFrames."""

import pandas as pd
import numpy as np
import typing as tp


def purged_walk_forward_cv(
    data: pd.DataFrame,
    train_size_days: int = 252,
    test_size_days: int = 21,
    embargo_days: int = 5,
) -> tp.Generator[tp.Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generate purged walk-forward cross-validation splits for dataframe with MultiIndex.

    This function handles the time-series nature of the data by ensuring:
    1. Splits are done based on unique dates (not just row counts).
    2. An embargo period separates Train and Test to prevent look-ahead bias/leakage.
    3. Data doesn't need to be sorted by date (though it helps), the logic sorts unique dates internally.

    Args:
        data (pd.DataFrame): DataFrame with a MultiIndex containing a Datetime level.
        train_size_days (int): Number of unique dates to include in training.
        test_size_days (int): Number of unique dates to include in testing.
        embargo_days (int): Gap between Train end and Test start (must be >= target horizon).

    Yields:
        (train_indices, test_indices): Tuples of numpy arrays containing integer location
                                       indices (iloc) for the train and test sets.
    """

    # 1. Identify the Date Index Level
    # We iterate through levels to find the one containing datetime objects
    date_level_name = None
    for name, level in zip(data.index.names, data.index.levels):  # type: ignore
        if pd.api.types.is_datetime64_any_dtype(level):
            date_level_name = name
            break

    if date_level_name is None:
        raise ValueError("Could not find a Datetime level in the MultiIndex.")

    # 2. Extract and Sort Unique Dates
    # We work with the underlying array for speed
    all_dates = data.index.get_level_values(date_level_name)
    unique_dates = np.sort(np.unique(all_dates))
    n_dates = len(unique_dates)

    # 3. Validation
    min_needed = train_size_days + embargo_days + test_size_days
    if n_dates < min_needed:
        raise ValueError(
            f"Not enough data. Have {n_dates} unique dates, need at least {min_needed} "
            f"(Train: {train_size_days} + Embargo: {embargo_days} + Test: {test_size_days})"
        )

    # 4. Generate Indices Mapping
    # Creating a map of {Date -> [Array of Integer Indices]} is significantly faster
    # than querying the dataframe repeatedly inside the loop.
    # We use np.searchsorted to map the full date column to integer IDs of unique_dates
    date_to_idx_map = np.searchsorted(unique_dates, all_dates)

    # 5. Walk-Forward Loop
    # We iterate through the unique_dates array
    # Step size is test_size_days to ensure non-overlapping test sets

    curr_train_end_idx = train_size_days

    while curr_train_end_idx + embargo_days + test_size_days <= n_dates:

        # Define indices in the unique_dates array
        # Rolling Window Logic:
        train_start_date_idx = curr_train_end_idx - train_size_days
        # For Expanding Window (Start from beginning), change above line to:
        # train_start_date_idx = 0

        train_end_date_idx = curr_train_end_idx

        test_start_date_idx = curr_train_end_idx + embargo_days
        test_end_date_idx = test_start_date_idx + test_size_days

        # Get the integer IDs for the dates in this fold
        # These are indices relative to `unique_dates`
        fold_train_date_ids = np.arange(train_start_date_idx, train_end_date_idx)
        fold_test_date_ids = np.arange(test_start_date_idx, test_end_date_idx)

        # Map these Date IDs back to the original DataFrame row indices
        # We use np.isin on the pre-calculated map
        train_indices = np.flatnonzero(np.isin(date_to_idx_map, fold_train_date_ids))
        test_indices = np.flatnonzero(np.isin(date_to_idx_map, fold_test_date_ids))

        yield train_indices, test_indices

        # Move forward
        curr_train_end_idx += test_size_days
