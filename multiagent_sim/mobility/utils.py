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
    drone_id = 0
    for row in range(grid_size):
        for col in range(grid_size):
            positions[drone_id, 0] = origin[0] + space * row
            positions[drone_id, 1] = origin[1] + space * col
            drone_id += 1
            if drone_id >= num_points:
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
    positions = []
    max_iter = num_positions * 100
    for _ in range(max_iter):
        x = np.random.uniform(env.boundary.left, env.boundary.right)
        y = np.random.uniform(env.boundary.bottom, env.boundary.top)
        z = env.get_elevation([x, y])
        if (not env.is_inside([x, y])) or env.is_collision(
            [x, y, z], check_altitude=False
        ):
            continue
        positions.append([x, y, z])
        if len(positions) == num_positions:
            break

    if len(positions) != num_positions:
        raise RuntimeError("Cannot generate random positions inside environment")

    return np.array(positions)
