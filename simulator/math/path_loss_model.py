import numpy as np
from numba import njit


def calculate_signal_strength(
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    f: float = 10.0,
    n: float = 3.0,
    tx_power: float = 20.0,
    mode: str = "total",
) -> np.ndarray:
    """
    Calculate the transmitted power at receiver positions from transmitter positions.

    Parameters
    ----------
    tx_positions : np.ndarray
        Array of shape (num_tx, 3) with x, y, z coordinates of transmitters.
    rx_positions : np.ndarray
        Array of shape (num_rx, 3) with x, y, z coordinates of receivers.
    f : float, optional
        Frequency in MHz (default is 10.0).
    n : float, optional
        Path loss exponent (default is 3.0).
    tx_power : float, optional
        Transmit power in dBm (default is 20.0).
    mode : str, optional
        Mode of calculation. Can be "total" to sum contributions from all transmitters
        or "max" to return the maximum received power from a single transmitter (default is "total").

    Returns
    -------
    np.ndarray
        Array of shape (num_rx,) with the received power at each receiver position.
        The result depends on the selected mode:
        - "total": Total received power from all transmitters.
        - "max": Maximum received power from a single transmitter.
    """
    rx_power = _calculate_rx_powers(tx_positions, rx_positions, f, n, tx_power)

    if mode == "total":
        # Compute received power in linear scale (mW)
        rx_power_linear = 10 ** (rx_power / 10)  # Convert dBm to mW

        # Sum contributions from all transmitters in linear scale
        total_rx_power_linear = np.sum(rx_power_linear, axis=0)  # Sum in mW

        # Convert total received power back to dBm
        total_rx_power_dbm = 10 * np.log10(total_rx_power_linear)
        return total_rx_power_dbm

    elif mode == "max":
        # Return the maximum received power for each receiver position (num_rx,)
        max_rx_power = np.max(rx_power, axis=0)
        return max_rx_power

    else:
        raise ValueError("Invalid mode. Choose 'total' or 'max'.")


def signal_strength_map(
    tx_positions: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    f: float = 10.0,
    n: float = 3.0,
    tx_power: float = 20.0,
    mode: str = "total",
) -> np.ndarray:
    """
    Generate a heatmap of received power over a grid of points.

    Parameters
    ----------
    tx_positions : np.ndarray
        Array of shape (num_tx, 3) with x, y, z coordinates of transmitters.
    xs : np.ndarray
        1D array of x-coordinates for the grid.
    ys : np.ndarray
        1D array of y-coordinates for the grid.
    f : float, optional
        Frequency in MHz (default is 10.0).
    n : float, optional
        Path loss exponent (default is 3.0).
    tx_power : float, optional
        Transmit power in dBm (default is 20.0).
    mode : str, optional
        Mode of calculation. Can be "total" to sum contributions from all transmitters
        or "max" to return the maximum received power from a single transmitter (default is "total").

    Returns
    -------
    np.ndarray
        2D array of received power over the grid, with shape (len(ys), len(xs)).
    """
    # Create grid points
    x_grid, y_grid = np.meshgrid(xs, ys)
    z_grid = np.zeros_like(x_grid)
    grid_points = np.stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()], axis=-1)

    # Calculate received power at grid points
    rx_power = calculate_signal_strength(
        tx_positions, grid_points, f, n, tx_power, mode
    )

    # Reshape to grid shape
    rx_power_map = rx_power.reshape(ys.size, xs.size)

    return rx_power_map


@njit(cache=True)
def _calculate_rx_powers(
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    f: float = 10.0,
    n: float = 3.0,
    tx_power: float = 20.0,
) -> np.ndarray:
    d0 = 1.0  # Path loss reference is calculated with d0 in km and f in MHz
    pl0 = 20 * np.log10(d0 * 1e-3) + 20 * np.log10(f) + 32.44

    # Compute delta vectors (num_tx x num_rx x 3)
    delta = tx_positions[:, None, :] - rx_positions[None, :, :]

    # Compute distances (num_tx x num_rx)
    distances = np.sqrt(np.sum(delta**2, axis=-1))  # Euclidean distance

    # Avoid division by zero
    distances = np.maximum(distances, d0)

    # Compute path loss (num_tx x num_rx)
    path_loss = pl0 + 10 * n * np.log10(distances / d0)

    # Compute received power (num_tx x num_rx)
    rx_power = tx_power - path_loss
    return rx_power


if __name__ == "__main__":
    num_tx = 10
    space = 100.0

    # Generate random transmitter positions
    tx_positions = np.zeros((num_tx, 3))
    tx_positions[:, 0:2] = np.random.uniform(-space, +space, (num_tx, 2))
    tx_positions[:, 2] = np.random.uniform(0.0, 0.0, (num_tx,))
    xs = np.linspace(-space, +space, 100)
    ys = np.linspace(-space, +space, 100)

    # Generate the heatmap
    heatmap = signal_strength_map(tx_positions, xs, ys, f=10.0, n=3.0, mode="max")

    # Plot the heatmap
    from matplotlib import pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot()
    im = ax.imshow(
        heatmap, extent=[-space, space, -space, space], origin="lower", cmap="plasma"
    )
    plt.colorbar(im, ax=ax, label="Received Power (dBm)")

    # Overlay transmitter positions
    ax.scatter(
        tx_positions[:, 0],
        tx_positions[:, 1],
        color="red",
        label="Transmitters",
        marker="x",
    )

    # Add labels and legend
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend()

    plt.show()
