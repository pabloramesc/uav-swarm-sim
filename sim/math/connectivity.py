"""
Wireless connectivity module
"""

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.csgraph import connected_components

from .path_loss_model import signal_strength


def pairwise_connectivity_matrix(
    positions: np.ndarray,
    tx_power: float = 20.0,
    min_rssi: float = -80.0,
    freq_mhz: float = 2412,
    path_loss_exp: float = 2.4,
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
    freq_mhz : float, optional
        Frequency in MHz used for signal strength calculation. Default is 2412 (2.4 GHz Wi-Fi).
    path_loss_exp : float, optional
        Path loss exponent. Default is 2.4 (typical for indoor environments).

    Returns
    -------
    np.ndarray
        A (N, N) boolean matrix where entry [i, j] is True if node i can reach node j.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.size == 0:
        return np.zeros((0, 0), dtype=bool)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Positions must have shape (N, 3).")

    num_positions = positions.shape[0]
    matrix = np.zeros((num_positions, num_positions), dtype=bool)

    for i in range(num_positions):
        tx = positions[i, :]
        rx = positions[:, :]
        rssi = signal_strength(
            tx, rx, f=freq_mhz, n=path_loss_exp, tx_power=tx_power, mode="max"
        )
        matrix[i, np.where(rssi > min_rssi)[0]] = True

    np.fill_diagonal(matrix, False)  # No self-connections
    return matrix


def directly_connected(
    positions: np.ndarray,
    tx_power: float = 20.0,
    min_rssi: float = -80.0,
    freq_mhz: float = 2412.0,
    path_loss_exp: float = 2.4,
) -> np.ndarray:
    """
    Returns the indices of nodes that have at least one direct link
    with another node above the given RSSI threshold.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 3) representing the 3D positions of N nodes.
    tx_power : float, optional
        Transmit power in dBm. Default is 20.0.
    min_rssi : float, optional
        Minimum RSSI threshold (in dBm) to consider a direct connection. Default is -80.0.
    freq_mhz : float, optional
        Frequency in MHz. Default is 2412 (2.4 GHz Wi-Fi).
    path_loss_exp : float, optional
        Path loss exponent. Default is 2.4.

    Returns
    -------
    np.ndarray
        Array of node indices that are directly connected to at least one other node.
    """
    matrix = pairwise_connectivity_matrix(
        positions=positions,
        tx_power=tx_power,
        min_rssi=min_rssi,
        freq_mhz=freq_mhz,
        path_loss_exp=path_loss_exp,
    )
    return np.flatnonzero(matrix.any(axis=1))


def connected_clusters(conn: np.ndarray) -> list[NDArray[np.intp]]:
    """
    Identifies clusters of connected nodes from a connectivity matrix.

    Parameters
    ----------
    conn : np.ndarray
        Boolean (N, N) connectivity matrix.

    Returns
    -------
    list of np.ndarray
        List of arrays, where each array contains the indices of one connected cluster.
    """
    conn = np.asarray(conn, dtype=bool)
    if conn.ndim != 2 or conn.shape[0] != conn.shape[1]:
        raise ValueError("Connectivity matrix must be square.")
    if conn.shape[0] == 0:
        return []

    n_components, labels = connected_components(
        conn, directed=False, return_labels=True
    )
    clusters = [np.where(labels == i)[0] for i in range(n_components)]
    return clusters


def globally_connected(
    positions: np.ndarray,
    tx_power: float = 20.0,
    min_rssi: float = -80.0,
    freq_mhz: float = 2412,
    path_loss_exp: float = 2.4,
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
    freq_mhz : float, optional
        Frequency in MHz used for signal strength calculation. Default is 2412 (2.4 GHz Wi-Fi).
    path_loss_exp : float, optional
        Path loss exponent. Default is 2.4 (typical for indoor environments).

    Returns
    -------
    np.ndarray
        Indices of nodes in the largest connected cluster.
    """
    conn = pairwise_connectivity_matrix(
        positions, tx_power, min_rssi, freq_mhz, path_loss_exp
    )
    clusters = connected_clusters(conn)
    if not clusters:
        return np.array([], dtype=np.intp)

    # ``max`` keeps the first cluster when sizes tie, making the result stable.
    return max(clusters, key=len)
