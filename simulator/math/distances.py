import numpy as np
from numba import njit


@njit(cache=True)
def calculate_relative_distances(
    reference: np.ndarray, positions: np.ndarray
) -> np.ndarray:
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
    distances = np.sqrt(np.sum(deltas**2, axis=1))
    return distances
