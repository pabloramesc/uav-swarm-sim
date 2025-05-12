import numpy as np

from .environment import Environment


def random_positions(num_positions: int, env: Environment) -> np.ndarray:
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
    while len(positions) < num_positions:
        x = np.random.uniform(env.boundary.xmin, env.boundary.xmax)
        y = np.random.uniform(env.boundary.ymin, env.boundary.ymax)
        z = env.get_elevation([x, y])
        if not env.is_inside([x, y]) or env.is_collision(
            [x, y, z], check_altitude=False
        ):
            continue
        positions.append([x, y, z])
    return np.array(positions)
