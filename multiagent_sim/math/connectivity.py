"""
Connectivity module
"""

import numpy as np
from scipy.sparse.csgraph import connected_components

from .path_loss_model import signal_strength


def wireless_connectivity_matrix(
    positions: np.ndarray,
    tx_power: float = 20.0,
    min_rssi: float = -80.0,
    f: float = 2412,
    n: float = 2.4,
) -> np.ndarray:
    """
    Computes a boolean connectivity matrix based on pairwise RSSI values between nodes.

    Each entry [i, j] in the matrix is True if node i can directly reach node j
    with RSSI greater than `min_rssi`, based on the path loss model.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 3) representing the 3D positions of N nodes.
    tx_power : float, optional
        Transmit power in dBm. Default is 20.0.
    min_rssi : float, optional
        Minimum RSSI threshold (in dBm) to consider a connection valid. Default is -80.0.
    f : float, optional
        Frequency in MHz used for signal strength calculation. Default is 2412 (2.4 GHz Wi-Fi).
    n : float, optional
        Path loss exponent. Default is 2.4 (typical for indoor environments).

    Returns
    -------
    np.ndarray
        A (N, N) boolean matrix where entry [i, j] is True if node i can reach node j.
    """
    positions = np.atleast_2d(positions)
    N = positions.shape[0]
    matrix = np.zeros((N, N), dtype=bool)

    for i in range(N):
        tx = positions[i, :]
        rx = positions[:, :]
        rssi = signal_strength(tx, rx, f=f, n=n, tx_power=tx_power, mode="max")
        matrix[i, np.where(rssi > min_rssi)[0]] = True

    np.fill_diagonal(matrix, False)  # No self-connections
    return matrix

def connected_clusters(conn: np.ndarray) -> list[np.ndarray]:
    """
    Identifies clusters of connected nodes from a connectivity matrix.

    Parameters
    ----------
    conn : np.ndarray
        Boolean NxN connectivity matrix.

    Returns
    -------
    list of np.ndarray
        List of arrays, where each array contains the indices of one connected cluster.
    """
    n_components, labels = connected_components(
        conn, directed=False, return_labels=True
    )
    clusters = [np.where(labels == i)[0] for i in range(n_components)]
    return clusters

def global_connected(
    positions: np.ndarray,
    tx_power: float = 20.0,
    min_rssi: float = -80.0,
    f: float = 2412,
    n: float = 2.4,
):
    """
    Returns the indices of the positions connected to the largest cluster.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 3) representing the 3D positions of N nodes.
    tx_power : float, optional
        Transmit power in dBm. Default is 20.0.
    min_rssi : float, optional
        Minimum RSSI threshold (in dBm) to consider a connection valid. Default is -80.0.
    f : float, optional
        Frequency in MHz used for signal strength calculation. Default is 2412 (2.4 GHz Wi-Fi).
    n : float, optional
        Path loss exponent. Default is 2.4 (typical for indoor environments).

    Returns
    -------
    np.ndarray
        Indices of nodes in the largest connected cluster.
    """
    conn = wireless_connectivity_matrix(positions, tx_power, min_rssi, f, n)
    clusters = connected_clusters(conn)
    largest_cluster = np.argmax(len(cluster) for cluster in clusters)
    return clusters[largest_cluster]