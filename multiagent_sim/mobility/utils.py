import numpy as np

from ..environment import Environment


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
    index = 0
    for row in range(grid_size):
        for col in range(grid_size):
            positions[index, 0] = origin[0] + space * row
            positions[index, 1] = origin[1] + space * col
            index += 1
            if index >= num_points:
                return positions
    return positions


def environment_random_positions(num_positions: int, env: Environment) -> np.ndarray:
    """
    Generate random positions within the environment.

    Parameters
    ----------
    num_positions : int
        Number of random positions to generate.
    env : Environment
        The environment object containing the boundaries.

    Returns
    -------
    np.ndarray
        Array of shape (num_positions, 3) containing random positions in the
        format [x, y, z].
    """
    if num_positions <= 0:
        return np.zeros((0, 3))

    if env.boundary is None:
        raise RuntimeError("Environment boundary not initialized.")

    positions = []
    max_iter = num_positions * 100
    for _ in range(max_iter):
        x = np.random.uniform(env.boundary.bounds.xmin, env.boundary.bounds.xmax)
        y = np.random.uniform(env.boundary.bounds.ymin, env.boundary.bounds.ymax)
        z = env.get_elevation(pos=np.array([x, y])).item()
        if env.is_collision(
            pos=np.array([x, y, z]), check_altitude=False, check_boundary=True
        ).item():
            continue
        positions.append([x, y, z])
        if len(positions) == num_positions:
            break

    if len(positions) != num_positions:
        raise RuntimeError("Cannot generate random positions inside environment")

    return np.array(positions)
