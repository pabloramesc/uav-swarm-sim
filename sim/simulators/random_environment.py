import numpy as np

from ..environment import Environment


def reset_random_environment(
    env: Environment,
    boundary_size: float,
    num_obstacles: int = 0,
) -> None:
    env.set_rectangular_boundary(
        bottom_left=[-boundary_size, -boundary_size],
        top_right=[+boundary_size, +boundary_size],
    )

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
    radius = np.random.uniform(0.02 * min_size, 0.2 * min_size)
    env.add_circular_obstacle(center, radius)


def add_rectangular_obstacle(env: Environment) -> None:
    bottom_left = np.random.uniform(
        env.boundary.bounds.xy_min,
        env.boundary.bounds.xy_max,
        size=(2,),
    )
    min_size = min(env.boundary.bounds.size)
    width_height = np.random.uniform(0.01 * min_size, 0.1 * min_size, size=(2,))
    top_right = bottom_left + width_height
    env.add_rectangular_obstacle(bottom_left, top_right)
