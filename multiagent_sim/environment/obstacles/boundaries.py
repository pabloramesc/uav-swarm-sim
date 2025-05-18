"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np
from numpy.typing import ArrayLike

from shapely import Polygon

from .obstacles import (
    Obstacle,
    CircularObstacle,
    RectangularObstacle,
    PolygonalObstacle,
)
from .numba_helpers import (
    center_distances_numba,
    center_distances_and_directions_numba,
    is_inside_rectangle_numba,
    rectangle_closest_point_numba,
    rectangle_distances_numba,
)


class Boundary(Obstacle):

    def __init__(self, shape: Polygon) -> None:
        super().__init__(shape)

    def distance(self, pos: ArrayLike) -> float:
        pos = np.atleast_2d(pos)
        is_inside = self.is_inside(pos)
        distances = super()._get_distances(pos)
        is_inside = np.atleast_1d(is_inside)
        distances[~is_inside] = 0.0
        return distances if distances.shape[0] > 1 else distances.item()

    def direction(self, pos: ArrayLike) -> np.ndarray:
        direction = super().direction(pos)
        return -direction


class CircularBoundary(CircularObstacle, Boundary):

    def __init__(self, center: ArrayLike, radius: float, quad_segs: int = 4) -> None:
        super().__init__(center, radius, quad_segs)

    def distance(self, pos: ArrayLike) -> float:
        pos = np.atleast_2d(pos)
        distances = center_distances_numba(pos, self.center)
        distances = np.maximum(self.radius - distances, 0.0)
        return distances if distances.shape[0] > 1 else distances.item()

    def direction(self, pos: ArrayLike) -> np.ndarray:
        pos = np.atleast_2d(pos)
        _, directions = center_distances_and_directions_numba(pos, self.center)
        return np.squeeze(-directions)


class RectangularBoundary(RectangularObstacle, Boundary):

    def __init__(self, bottom_left: ArrayLike, top_right: ArrayLike) -> None:
        super().__init__(bottom_left, top_right)

    def distance(self, pos: ArrayLike) -> float:
        pos = np.atleast_2d(pos)
        is_inside = is_inside_rectangle_numba(
            pos, self.left, self.right, self.bottom, self.top
        )
        distances = rectangle_distances_numba(
            pos, self.left, self.right, self.bottom, self.top
        )
        distances[~is_inside] = 0.0
        return distances

    def direction(self, pos: ArrayLike) -> np.ndarray:
        directions = super().direction(pos)
        return -directions


class PolygonalBoundary(PolygonalObstacle, Boundary):

    def __init__(self, vertices: ArrayLike):
        super().__init__(vertices)
