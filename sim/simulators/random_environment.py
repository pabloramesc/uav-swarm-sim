import numpy as np

from ..environment import Environment


from typing import Union, Tuple


def reset_random_environment(
    env: Environment,
    boundary_size: Union[float, Tuple[float, float]],
    num_obstacles: Union[int, Tuple[int, int]] = 0,
) -> None:
    """
    Reset the environment with random boundary and obstacles.

    Args:
        env: Environment instance to reset.
        boundary_size: Either a fixed float or a tuple (min, max) to sample uniformly.
        num_obstacles: Either a fixed int or a tuple (min, max) to sample uniformly.
    """
    # Sample boundary size if a range is provided
    if isinstance(boundary_size, tuple):
        boundary_size = np.random.uniform(boundary_size[0], boundary_size[1])

    env.set_rectangular_boundary(
        bottom_left=[0, 0], top_right=[boundary_size, boundary_size]
    )

    # Sample number of obstacles if a range is provided
    if isinstance(num_obstacles, tuple):
        num_obstacles = np.random.randint(num_obstacles[0], num_obstacles[1] + 1)

    env.clear_obstacles()
    for _ in range(num_obstacles):
        if np.random.rand() > 0.5:
            add_circular_obstacle(env)
        else:
            add_rectangular_obstacle(env)


def add_circular_obstacle(env: Environment) -> None:
    center = np.random.uniform(
        env.boundary.bounds.xy_min,
        env.boundary.bounds.xy_max,
        size=(2,),
    )
    min_size = min(env.boundary.bounds.size)
    radius = np.random.uniform(0.01 * min_size, 0.1 * min_size)
    env.add_circular_obstacle(center, radius)


def add_rectangular_obstacle(env: Environment) -> None:
    bottom_left = np.random.uniform(
        env.boundary.bounds.xy_min,
        env.boundary.bounds.xy_max,
        size=(2,),
    )
    min_size = min(env.boundary.bounds.size)
    width_height = np.random.uniform(0.02 * min_size, 0.2 * min_size, size=(2,))
    top_right = bottom_left + width_height
    env.add_rectangular_obstacle(bottom_left, top_right)
