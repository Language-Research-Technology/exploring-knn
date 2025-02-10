import pynndescent  # https://pynndescent.readthedocs.io/en/latest/how_to_use_pynndescent.html#Nearest-neighbors-of-the-training-set

import numpy as np

from sklearn.cluster import KMeans


def get_knn(embedding, n_neighbours=10) -> ("nearest_neighbours_array", "neighbours_distance_array"):
    """
    Params:
        - embedding: numpy array of embeddings, embedding at index i corresponds to filename at index i.
        - n_neighbours: number of nearest neighbours returned.

    Returns:
        - nearest_neighbour_array: an array where the ith row corresponds to the n
        nearest neighbours for the ith point.
        - neighbours_distance_array: an array where the ith row corresponds to 
        the distances of the n nearest neighbours from the ith point.
    """
    index = pynndescent.NNDescent(
        embedding, n_neighbors=n_neighbours, random_state=42)
    return index.neighbor_graph  # nearest_neighbours_array, neighbours_distance_array


def get_count_neighbour_occurances(nearest_neighbours_array) -> "ranking_array":
    """return array where the ith index is how many times the ith number occured as a neighbour 
    (including as a neighbour of itself)"""
    _, counts = np.unique(
        nearest_neighbours_array.flatten(), return_counts=True)
    return counts


def rank_neighbour_occurances(count_neighbour_occurances_array, ascending=True):
    """takes get_count_neighbour_occurances() or get_mean_distance() output as input and returns a ranked array 
    of indexes in desired order (the index corresponds to a filename in the image_filenames variable)"""
    # first part returns ascending list of sorted indices
    # second part conditionally flips the order
    # ascending = True -> 1,  ascending = False -> -1
    return np.argsort(count_neighbour_occurances_array)[::-1+2*ascending]


def get_mean_distance(neighbours_distance_array, exclude_self=True):
    """Takes neighbours_distance_array and returns an array where the ith row
    is the mean distance across the n neighbours from the ith embedded image.

    This mean can conditionally include its distance from itself (0) in the mean calculation"""

    if exclude_self:
        # exclude first column if exclude_self
        mean_distances = np.mean(neighbours_distance_array[:, 1:], axis=1)

    else:
        mean_distances = np.mean(neighbours_distance_array, axis=1)
    return mean_distances


def get_cluster_labels(embedding, n_clusters=10, random_state=42) -> ['labels']:
    # add docstring
    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=random_state).fit(embedding)
    return kmeans.labels_
