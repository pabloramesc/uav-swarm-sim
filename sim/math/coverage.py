"""
Wireless coverage module.

Provides functions to compute which receivers are covered by transmitters
based on signal strength and a path-loss model.
"""

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from .distances import pairwise_cross_distances


@lru_cache(maxsize=128)
def max_coverage_distance(
    tx_power: float,
    min_rssi: float,
    f_mhz: float = 2412.0,
    path_loss_exp: float = 2.4,
    d0_km: float = 1e-3,
) -> float:
    """
    Compute maximum coverage distance (in meters) for given RSSI threshold.

    Parameters
    ----------
    tx_power : float
        Transmit power in dBm.
    min_rssi : float
        Minimum RSSI threshold in dBm.
    f_mhz : float
        Carrier frequency in MHz. Default 2412 MHz.
    path_loss_exp : float
        Path loss exponent. Default 2.4.
    d0_km : float
        Reference distance in km. Default 1 m.

    Returns
    -------
    float
        Maximum distance (meters) where min_rssi is still received.
    """
    # Free-space path loss at reference distance d0
    pl0 = 20 * np.log10(d0_km) + 20 * np.log10(f_mhz) + 32.44
    d_max = (d0_km * 1e3) * 10 ** ((tx_power - min_rssi - pl0) / (10 * path_loss_exp))
    return d_max


def covered_positions(
    tx_positions: NDArray[np.floating],
    rx_positions: NDArray[np.floating],
    tx_power: float = 20.0,
    min_rssi: float = -80.0,
    freq_mhz: float = 2412.0,
    path_loss_exp: float = 2.4,
) -> NDArray[np.bool_]:
    """
    Compute a boolean mask of receivers covered by a set of transmitters.

    Parameters
    ----------
    tx_positions : NDArray[np.floating], shape (N, 3)
        3D positions of N transmitters.
    rx_positions : NDArray[np.floating], shape (M, 3)
        3D positions of M receivers.
    tx_power : float, optional
        Transmit power in dBm. Default is 20.0.
    min_rssi : float, optional
        Minimum RSSI threshold in dBm to consider a receiver covered.
        Default is -80.0.
    freq_mhz : float, optional
        Carrier frequency in MHz. Default is 2412 (2.4 GHz Wi-Fi).
    path_loss_exp : float, optional
        Path loss exponent. Default is 2.4.

    Returns
    -------
    NDArray[np.bool_], shape (M,)
        Boolean array where True indicates a receiver is covered.
    """
    if tx_positions.size == 0 or rx_positions.size == 0:
        return np.zeros(rx_positions.shape[0], dtype=bool)

    # Compute max distance from RSSI threshold
    d_max = max_coverage_distance(tx_power, min_rssi, freq_mhz, path_loss_exp)

    # Compute all pairwise distances (N x M)
    distances = pairwise_cross_distances(tx_positions, rx_positions)

    # Boolean mask: True if any TX covers RX
    covered_mask = np.any(distances <= d_max, axis=0)

    return covered_mask


def coverage_matrix(
    tx_positions: NDArray[np.floating],
    rx_positions: NDArray[np.floating],
    tx_power: float = 20.0,
    min_rssi: float = -80.0,
    freq_mhz: float = 2412.0,
    path_loss_exp: float = 2.4,
) -> NDArray[np.bool_]:
    """
    Compute a boolean coverage matrix: rows=transmitters, columns=receivers.

    Each element (i, j) is True if transmitter-i covers receiver-j.

    Parameters
    ----------
    tx_positions : NDArray[np.floating], shape (N, 3)
        3D positions of N transmitters.
    rx_positions : NDArray[np.floating], shape (M, 3)
        3D positions of M receivers.
    tx_power : float, optional
        Transmit power in dBm. Default is 20.0.
    min_rssi : float, optional
        Minimum RSSI threshold (in dBm) to consider a receiver covered.
        Default is -80.0.
    freq_mhz : float, optional
        Carrier frequency in MHz. Default is 2412.
    path_loss_exp : float, optional
        Path loss exponent. Default is 2.4.

    Returns
    -------
    NDArray[np.bool_], shape (N, M)
        Boolean matrix indicating coverage. True if transmitter covers receiver.
    """
    num_tx = tx_positions.shape[0]
    num_rx = rx_positions.shape[0]
    coverage = np.zeros((num_tx, num_rx), dtype=bool)

    for i in range(num_tx):
        covered_mask = covered_positions(
            tx_positions[i : i + 1],
            rx_positions,
            tx_power=tx_power,
            min_rssi=min_rssi,
            freq_mhz=freq_mhz,
            path_loss_exp=path_loss_exp,
        )
        coverage[i, :] = covered_mask

    return coverage
