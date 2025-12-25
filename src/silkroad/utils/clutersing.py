"""Clustering the assets based on their correlation matrix of returns or log-returns."""

import scipy.cluster.hierarchy as sch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing as tp
from scipy.spatial.distance import squareform


def get_linkage_matrix(
    log_returns: pd.DataFrame, method: str = "ward", **kwargs
) -> np.ndarray:
    """Compute the linkage matrix from the correlation matrix.

    Args:
        log_returns (pd.DataFrame): DataFrame of log returns of asset prices.
        method (str, optional): Linkage method to use. Defaults to 'ward'.
        **kwargs: Additional arguments for the linkage function.

    Returns:
        np.ndarray: Linkage matrix.
    """
    # Convert correlation to distance
    corr = log_returns.corr().values
    distance = np.sqrt(0.5 * (1 - corr))
    distance = squareform(distance)  # Convert to condensed form
    # Compute the linkage matrix
    linkage_matrix = sch.linkage(distance, method=method, **kwargs)
    return linkage_matrix


def plot_dendrogram(
    log_returns: pd.DataFrame,
    method: str = "ward",
    color_threshold: float = None,
    ax: plt.Axes = None,
    linkage_kwargs: tp.Dict["str", tp.Any] = {},
    dendogram_kwargs: tp.Dict["str", tp.Any] = {},
) -> None:
    """Plot the dendrogram of the hierarchical clustering.

    Args:
        log_returns (pd.DataFrame): DataFrame of log returns of asset prices.
        method (str, optional): Linkage method to use. Defaults to 'ward'.
        color_threshold (float, optional): Threshold to color clusters. Defaults to None.
        ax (plt.Axes, optional): Matplotlib Axes to plot on. Defaults to None.
        linkage_kwargs (tp.Dict[str, tp.Any], optional): Additional arguments for the linkage function. Defaults to {}.
        dendogram_kwargs (tp.Dict[str, tp.Any], optional): Additional arguments for the dendrogram function. Defaults to {}.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    linkage_matrix = get_linkage_matrix(log_returns, method=method, **linkage_kwargs)
    sch.dendrogram(
        linkage_matrix,
        labels=log_returns.columns,
        color_threshold=color_threshold,
        ax=ax,
        **dendogram_kwargs,
    )
    ax.set_title("Hierarchical Clustering Dendrogram")
    ax.set_xlabel("Assets")
    ax.set_ylabel("Distance")
    plt.show()


def get_clusters(
    log_returns: pd.DataFrame,
    method: str = "ward",
    t: float = 0.75,
    criterion: str = "distance",
    linkage_kwargs: tp.Dict["str", tp.Any] = {},
    cluster_kwargs: tp.Dict["str", tp.Any] = {},
) -> pd.Series:
    """Get cluster assignments for each asset.

    Args:
        log_returns (pd.DataFrame): DataFrame of log returns of asset prices.
        method (str, optional): Linkage method to use. Defaults to 'ward'.
        t (float, optional): Threshold to form flat clusters. Defaults to 0.75.
        criterion (str, optional): Criterion to use in forming flat clusters. Defaults to 'distance'.
        linkage_kwargs (tp.Dict[str, tp.Any], optional): Additional arguments for the linkage function. Defaults to {}.
        cluster_kwargs (tp.Dict[str, tp.Any], optional): Additional arguments for the clustering function. Defaults to {}.

    Returns:
        pd.Series: Series with asset names as index and cluster labels as values.
    """
    linkage_matrix = get_linkage_matrix(log_returns, method=method, **linkage_kwargs)
    cluster_labels = sch.fcluster(
        linkage_matrix, t=t, criterion=criterion, **cluster_kwargs
    )
    return pd.Series(cluster_labels, index=log_returns.columns).sort_values()
