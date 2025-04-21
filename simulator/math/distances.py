import numpy as np
from numba import njit


@njit(cache=True)
def relative_distances(reference: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """
    Efficiently calculates the relative distances from a reference position to
    multiple positions in an M-dimensional space.

    Parameters
    ----------
    reference : np.ndarray
        A reference position as a (M,) array, where M is the dimensionality of
        the space.
    positions : np.ndarray
        An array of positions with shape (N, M), where M is the dimensionality
        of the space and N is the number of positions.

    Returns
    -------
    np.ndarray
        An array of relative distances with shape (N,).
    """
    deltas = positions - reference
    distances = np.sqrt(np.sum(deltas**2, axis=-1))
    return distances


@njit(cache=True)
def pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """
    Computes the pairwise Euclidean distances between all positions.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, M), where N is the number of positions and M is the dimensionality.

    Returns
    -------
    np.ndarray
        Array of shape (N, N) with pairwise distances.
    """
    deltas = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.sqrt(np.sum(deltas**2, axis=-1))
    return distances


if __name__ == "__main__":
    # Test for relative_distances
    ref = np.array([0.0, 0.0])
    points = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]]
    )
    rel_dists = relative_distances(ref, points)
    print("Relative distances from", ref, "to each point:")
    print(rel_dists)

    # Test for pairwise_distances
    pair_dists = pairwise_distances(points)
    print("\nPairwise distances between points:")
    print(pair_dists)

    # Manual check (distance between [1,0] and [0,1] should be sqrt(2) ≈ 1.414)
    expected = np.linalg.norm(points[0] - points[1])
    print(
        f"\nDistance between {points[0]} and {points[1]}: {pair_dists[0,1]} (expected: {expected})"
    )
