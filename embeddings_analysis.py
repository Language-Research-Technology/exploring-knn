import pynndescent  # https://pynndescent.readthedocs.io/en/latest/how_to_use_pynndescent.html#Nearest-neighbors-of-the-training-set

import numpy as np
from sklearn.cluster import KMeans

from typing import Tuple


def get_knn(embedding: np.ndarray, n_neighbours: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieves the nearest neighbors and distances for each embedding using NNDescent.

    Params:
        - embedding: A 2D NumPy array of shape (n_samples, n_features), where each row 
          corresponds to an embedding of an image.
        - n_neighbours: The number of nearest neighbors to return for each embedding.

    Returns:
        - nearest_neighbour_array: A 2D NumPy array of shape (n_samples, n_neighbours), 
          where the ith row contains the indices of the nearest neighbors for the ith sample.
        - neighbours_distance_array: A 2D NumPy array of shape (n_samples, n_neighbours), 
          where the ith row contains the distances to the nearest neighbors for the ith sample.
    """
    index = pynndescent.NNDescent(
        embedding, n_neighbors=n_neighbours, random_state=42)
    return index.neighbor_graph  # nearest_neighbours_array, neighbours_distance_array


def get_count_neighbour_occurances(nearest_neighbours_array: np.ndarray) -> np.ndarray:
    """
    Returns an array where each element corresponds to the number of times each image 
    appears as a neighbor (including itself) across all other embeddings.

    Params:
        - nearest_neighbours_array: A 2D NumPy array of shape (n_samples, n_neighbours), 
          where each row contains the indices of the nearest neighbors for each embedding.

    Returns:
        - ranking_array: A 1D NumPy array, where each element represents the count of 
          occurrences of each sample as a neighbor across all other samples.
    """
    _, counts = np.unique(
        nearest_neighbours_array.flatten(), return_counts=True)
    return counts


def rank_neighbour_occurances(count_neighbour_occurances_array: np.ndarray, ascending: bool = True) -> np.ndarray:
    """
    Ranks the samples based on how often they appear as neighbors. Optionally sorts the result 
    in ascending or descending order.

    Params:
        - count_neighbour_occurances_array: A 1D NumPy array containing the count of 
          occurrences of each sample as a neighbor.
        - ascending: A boolean indicating whether the sorting should be in ascending 
          (True) or descending (False) order.

    Returns:
        - ranked_array: A 1D NumPy array of indices, where the order corresponds to the 
          ranked samples based on their neighbor occurrences.
    """
    return np.argsort(count_neighbour_occurances_array)[::-1 + 2 * ascending]


def get_mean_distance(neighbours_distance_array: np.ndarray, exclude_self: bool = True) -> np.ndarray:
    """
    Computes the mean distance of each sample to its nearest neighbors.

    Params:
        - neighbours_distance_array: A 2D NumPy array of shape (n_samples, n_neighbours), 
          where each row contains the distances to the nearest neighbors for each embedding.
        - exclude_self: A boolean indicating whether to exclude the distance to the sample 
          itself (usually zero) from the mean calculation.

    Returns:
        - mean_distances: A 1D NumPy array containing the mean distance of each sample 
          to its neighbors.
    """
    if exclude_self:
        mean_distances = np.mean(neighbours_distance_array[:, 1:], axis=1)
    else:
        mean_distances = np.mean(neighbours_distance_array, axis=1)
    return mean_distances


def get_cluster_labels(embedding: np.ndarray, n_clusters: int = 10, random_state: int = 42) -> np.ndarray:
    """
    Clusters the embeddings into the specified number of clusters using KMeans.

    Params:
        - embedding: A 2D NumPy array of shape (n_samples, n_features), where each row 
          corresponds to an embedding of an image.
        - n_clusters: The number of clusters to form.
        - random_state: The seed used by the random number generator for reproducibility.

    Returns:
        - labels: A 1D NumPy array of shape (n_samples,) containing the cluster labels 
          for each sample.
    """
    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=random_state).fit(embedding)
    return kmeans.labels_
