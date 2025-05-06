import numpy as np
from numba import njit


def distances_from_point(reference: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """
    Computes Euclidean distances from a single reference point to multiple
    positions.

    Parameters
    ----------
    reference : np.ndarray
        A single point of shape (D,) where D is the number of dimensions.
    positions : np.ndarray
        An array of shape (N, D) containing N points in D-dimensional space.

    Returns
    -------
    np.ndarray
        An array of shape (N,) with distances from the reference to each point.
    """
    distances = pairwise_cross_distances(reference[None, :], positions)
    return np.squeeze(distances)


def pairwise_self_distances(positions: np.ndarray) -> np.ndarray:
    """
    Computes pairwise Euclidean distances between all points in a single set.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, D) where N is the number of points and D is the
        number of dimensions of the space.

    Returns
    -------
    np.ndarray
        An array of shape (N, N) with distances between all pairs of points.
    """
    distances = pairwise_cross_distances(positions, positions)
    return distances


@njit(cache=True)
def pairwise_cross_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Computes pairwise Euclidean distances between points in two different sets.

    Parameters
    ----------
    a : np.ndarray
        Array of shape (N, D), N points in D-dimensional space.
    b : np.ndarray
        Array of shape (M, D), M points in D-dimensional space.

    Returns
    -------
    np.ndarray
        Array of shape (N, M), where each (i,j) element is the distance between
        a[i] and b[j].
    """
    deltas = a[:, None, :] - b[None, :, :]
    distances = np.sqrt(np.sum(deltas**2, axis=-1))
    return distances


if __name__ == "__main__":
    ref = np.array([0.0, 0.0])
    points = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 2.0],
        ]
    )

    print("Reference point:", ref)
    print("Points:\n", points)

    # Test distances_from_point
    rel_dists = distances_from_point(ref, points)
    print("\nDistances from reference to each point:")
    for i, p in enumerate(points):
        print(f"Distance to {p}: {rel_dists[i]:.4f}")

    # Test pairwise_self_distances
    pair_dists = pairwise_self_distances(points)
    print("\nPairwise distances between all points:")
    print(pair_dists)

    # Manual check: distance between [1.0, 0.0] and [0.0, 1.0]
    expected = np.linalg.norm(points[0] - points[1])
    print(f"\nCheck: distance between {points[0]} and {points[1]}:")
    print(f"Computed: {pair_dists[0, 1]:.4f}, Expected: {expected:.4f}")

    # Test pairwise_cross_distances with different sets
    a = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    b = np.array([[1.0, 0.0], [0.0, 1.0]])
    cross_dists = pairwise_cross_distances(a, b)
    print("\nPairwise distances between two sets:")
    print("Set A:\n", a)
    print("Set B:\n", b)
    print("Distances:\n", cross_dists)
