"""Helpers for randomized training environments."""

from __future__ import annotations

import numpy as np

from .environment import Environment

type FloatRange = float | tuple[float, float]
type IntRange = int | tuple[int, int]


def reset_random_environment(
    environment: Environment,
    boundary_size: FloatRange,
    num_obstacles: IntRange = 0,
    *,
    rng: np.random.Generator | None = None,
) -> None:
    """Replace an environment's square boundary and random obstacles."""

    generator = rng or np.random.default_rng()
    size = _sample_float(boundary_size, generator)
    count = _sample_int(num_obstacles, generator)
    if size <= 0:
        raise ValueError("boundary_size must be positive.")
    if count < 0:
        raise ValueError("num_obstacles cannot be negative.")

    environment.set_rectangular_boundary((0.0, 0.0), (size, size))
    environment.clear_obstacles()

    for _ in range(count):
        if generator.random() < 0.5:
            _add_circle(environment, size, generator)
        else:
            _add_rectangle(environment, size, generator)


def _sample_float(value: FloatRange, rng: np.random.Generator) -> float:
    if isinstance(value, tuple):
        low, high = map(float, value)
        if low > high:
            raise ValueError("Range minimum cannot exceed its maximum.")
        return float(rng.uniform(low, high))
    return float(value)


def _sample_int(value: IntRange, rng: np.random.Generator) -> int:
    if isinstance(value, tuple):
        low, high = map(int, value)
        if low > high:
            raise ValueError("Range minimum cannot exceed its maximum.")
        return int(rng.integers(low, high + 1))
    return int(value)


def _add_circle(
    environment: Environment, size: float, rng: np.random.Generator
) -> None:
    radius = float(rng.uniform(0.01 * size, 0.1 * size))
    center = rng.uniform(radius, size - radius, size=2)
    environment.add_circular_obstacle(center, radius)


def _add_rectangle(
    environment: Environment, size: float, rng: np.random.Generator
) -> None:
    dimensions = rng.uniform(0.02 * size, 0.2 * size, size=2)
    bottom_left = rng.uniform((0.0, 0.0), size - dimensions)
    environment.add_rectangular_obstacle(bottom_left, bottom_left + dimensions)
