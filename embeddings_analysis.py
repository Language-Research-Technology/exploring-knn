import pynndescent  # https://pynndescent.readthedocs.io/en/latest/how_to_use_pynndescent.html#Nearest-neighbors-of-the-training-set

import numpy as np
from sklearn.cluster import KMeans

from typing import Tuple


def get_knn(embedding: np.ndarray, n_neighbours: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieves the nearest neighbors and distances for each embedding using NNDescent.

    This function finds the n_neighbours closest embeddings for each sample 
    using a nearest neighbor search algorithm.

    Args:
        embedding (np.ndarray): A 2D NumPy array of shape (n_samples, n_features), 
            where each row corresponds to an embedding of an image.
        n_neighbours (int): The number of nearest neighbors to retrieve for each embedding.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - np.ndarray: A 2D array of shape (n_samples, n_neighbours), 
              where each row contains the indices of the nearest neighbors.
            - np.ndarray: A 2D array of shape (n_samples, n_neighbours), 
              where each row contains the corresponding distances to the nearest neighbors.
    """
    index = pynndescent.NNDescent(
        embedding, n_neighbors=n_neighbours, random_state=42)
    return index.neighbor_graph  # nearest_neighbours_array, neighbours_distance_array


def get_count_neighbour_occurances(nearest_neighbours_array: np.ndarray) -> np.ndarray:
    """
    Computes the number of times each image appears as a neighbor across all embeddings.

    This function returns an array where each element represents the count of times 
    a particular image appears as a nearest neighbor (including itself) in the dataset.

    Args:
        nearest_neighbours_array (np.ndarray): A 2D NumPy array of shape (n_samples, n_neighbours), 
            where each row contains the indices of the nearest neighbors for each embedding.

    Returns:
        np.ndarray: A 1D NumPy array of shape (n_samples,), where each element represents 
        the number of times the corresponding sample appears as a neighbor across all samples.
    """
    _, counts = np.unique(
        nearest_neighbours_array.flatten(), return_counts=True)
    return counts


def rank_neighbour_occurances(count_neighbour_occurances_array: np.ndarray, ascending: bool = True) -> np.ndarray:
    """
    Ranks the samples based on how often they appear as neighbors.

    This function sorts the samples based on their neighbor occurrence counts, 
    with an option to sort in ascending or descending order.

    Args:
        count_neighbour_occurances_array (np.ndarray): A 1D NumPy array containing 
            the count of occurrences of each sample as a neighbor.
        ascending (bool, optional): A boolean indicating whether to sort in ascending 
            (True) or descending (False) order. Defaults to False (descending).

    Returns:
        np.ndarray: A 1D NumPy array of indices, where the order corresponds to the 
        ranked samples based on their neighbor occurrences.
    """
    return np.argsort(count_neighbour_occurances_array)[::-1 + 2 * ascending]


def get_mean_distance(neighbours_distance_array: np.ndarray, exclude_self: bool = True) -> np.ndarray:
    """
    Computes the mean distance of each sample to its nearest neighbors.

    This function calculates the mean distance of each sample to its nearest neighbors, 
    with an option to exclude the distance to the sample itself (usually zero).

    Args:
        neighbours_distance_array (np.ndarray): A 2D NumPy array of shape (n_samples, n_neighbours), 
            where each row contains the distances to the nearest neighbors for each embedding.
        exclude_self (bool, optional): A boolean indicating whether to exclude the distance to the sample 
            itself from the mean calculation. Defaults to True.

    Returns:
        np.ndarray: A 1D NumPy array containing the mean distance of each sample to its neighbors.
    """
    if exclude_self:
        mean_distances = np.mean(neighbours_distance_array[:, 1:], axis=1)
    else:
        mean_distances = np.mean(neighbours_distance_array, axis=1)
    return mean_distances


def get_cluster_labels(embedding: np.ndarray, n_clusters: int = 10, random_state: int = 42) -> np.ndarray:
    """
    Clusters the embeddings into the specified number of clusters using KMeans.

    This function applies KMeans clustering to the provided embeddings and assigns 
    a cluster label to each sample.

    Args:
        embedding (np.ndarray): A 2D NumPy array of shape (n_samples, n_features), 
            where each row corresponds to an embedding of an image.
        n_clusters (int): The number of clusters to form.
        random_state (int, optional): The seed used by the random number generator 
            for reproducibility. Defaults to 42.

    Returns:
        np.ndarray: A 1D NumPy array of shape (n_samples,) containing the cluster labels 
        assigned to each sample.
    """
    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=random_state).fit(embedding)
    return kmeans.labels_
