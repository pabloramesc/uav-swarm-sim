"""Initial-position generators for simulation scenarios."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .environment import Environment


def grid_positions(
    count: int,
    *,
    origin: ArrayLike = (0.0, 0.0),
    spacing: float = 1.0,
    altitude: float = 0.0,
) -> NDArray[np.float64]:
    """Place points row-by-row on the smallest enclosing square grid."""

    if count < 0:
        raise ValueError("count cannot be negative.")
    if spacing < 0:
        raise ValueError("spacing cannot be negative.")

    origin_array = np.asarray(origin, dtype=np.float64)
    if origin_array.shape != (2,):
        raise ValueError("origin must have shape (2,).")

    positions = np.zeros((count, 3), dtype=np.float64)
    positions[:, 2] = float(altitude)
    if count == 0:
        return positions

    side = int(np.ceil(np.sqrt(count)))
    indices = np.arange(count)
    positions[:, 0] = origin_array[0] + (indices // side) * spacing
    positions[:, 1] = origin_array[1] + (indices % side) * spacing
    return positions


def random_positions(
    count: int,
    *,
    origin: ArrayLike = (0.0, 0.0),
    spread: float = 1.0,
    altitude: float = 0.0,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Sample normally distributed positions around an origin."""

    if count < 0:
        raise ValueError("count cannot be negative.")
    if spread < 0:
        raise ValueError("spread cannot be negative.")

    origin_array = np.asarray(origin, dtype=np.float64)
    if origin_array.shape != (2,):
        raise ValueError("origin must have shape (2,).")

    generator = rng or np.random.default_rng()
    positions = np.zeros((count, 3), dtype=np.float64)
    positions[:, :2] = generator.normal(origin_array, spread, (count, 2))
    positions[:, 2] = float(altitude)
    return positions


def sample_positions(
    count: int,
    environment: Environment,
    *,
    altitude: float = 0.0,
    rng: np.random.Generator | None = None,
    max_attempts_per_position: int = 100,
) -> NDArray[np.float64]:
    """Sample collision-free positions inside an environment.

    ``altitude`` is interpreted as height above local ground.
    """

    if count < 0:
        raise ValueError("count cannot be negative.")
    if max_attempts_per_position <= 0:
        raise ValueError("max_attempts_per_position must be positive.")
    if count == 0:
        return np.zeros((0, 3), dtype=np.float64)

    boundary = environment.require_boundary()
    bounds = boundary.bounds
    generator = rng or np.random.default_rng()
    positions: list[list[float]] = []

    for _ in range(count * max_attempts_per_position):
        xy = np.array(
            [
                generator.uniform(bounds.xmin, bounds.xmax),
                generator.uniform(bounds.ymin, bounds.ymax),
            ]
        )
        ground = environment.get_elevation(xy).item()
        position = np.array([xy[0], xy[1], ground + altitude])
        if environment.is_collision(
            position, check_altitude=False, check_boundary=True
        ).item():
            continue
        positions.append(position.tolist())
        if len(positions) == count:
            return np.asarray(positions, dtype=np.float64)

    raise RuntimeError(
        f"Could not sample {count} collision-free positions inside the environment."
    )
