import numpy as np

def random_positions(
    num_points: int,
    origin: np.ndarray = np.zeros(2),
    space: float = 1.0,
    altitude: float = 0.0,
) -> np.ndarray:
    """
    Generate random positions around a given origin.

    Parameters
    ----------
    num_points : int
        Number of positions to generate.
    origin : np.ndarray, optional
        Center of the random distribution [x, y] (default is [0, 0]).
    space : float, optional
        Standard deviation of the random distribution (default is 1.0).
    altitude : float, optional
        Initial altitude for all points (default is 0.0).

    Returns
    -------
    np.ndarray
        Array of shape (num_points, 3) containing the generated positions.
    """
    positions = np.zeros((num_points, 3))
    positions[:, 0:2] = np.random.normal(origin, space, (num_points, 2))
    positions[:, 2] = altitude
    return positions

def grid_positions(
    num_points: int,
    origin: np.ndarray = np.zeros(2),
    space: float = 1.0,
    altitude: float = 0.0,
) -> np.ndarray:
    """
    Generate positions in a grid formation.

    Parameters
    ----------
    num_points : int
        Number of positions to generate.
    origin : np.ndarray, optional
        Bottom-left corner of the grid [x, y] (default is [0, 0]).
    space : float, optional
        Spacing between positions in the grid (default is 1.0).
    altitude : float, optional
        Initial altitude for all positions (default is 0.0).

    Returns
    -------
    np.ndarray
        Array of shape (num_points, 3) containing the generated positions.
    """
    positions = np.zeros((num_points, 3))
    positions[:, 2] = altitude
    positions[:, 3:6] = 0.0
    grid_size = int(np.ceil(np.sqrt(num_points)))
    drone_id = 0
    for row in range(grid_size):
        for col in range(grid_size):
            positions[drone_id, 0] = origin[0] + space * row
            positions[drone_id, 1] = origin[1] + space * col
            drone_id += 1
            if drone_id >= num_points:
                return positions