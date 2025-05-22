"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from abc import ABC

import numpy as np
from numpy.typing import ArrayLike
from shapely import Point, Polygon, shortest_line, box

from .numba_helpers import (
    center_distances_numba,
    center_distances_and_directions_numba,
    circle_closest_point_numba,
    is_inside_rectangle_numba,
    rectangle_closest_point_numba,
    rectangle_distances_and_directions_numba,
    rectangle_external_distances_numba,
)


class Obstacle(ABC):

    def __init__(self, shape: Polygon) -> None:
        self.shape = shape
        self.centroid = np.array(self.shape.centroid.coords[0])

    def is_inside(self, pos: ArrayLike) -> np.ndarray:
        pos = np.asarray(pos)
        points = [Point(p) for p in pos]
        is_inside = np.array([self.shape.contains(p) for p in points])
        return is_inside

    def distance(self, pos: ArrayLike) -> np.ndarray:
        pos = np.atleast_2d(pos)
        distances = self._get_distances(pos)
        is_inside = self.is_inside(pos)
        distances[is_inside] = 0.0
        return distances

    def direction(self, pos: ArrayLike) -> np.ndarray:
        pos = np.atleast_2d(pos)
        closest, distances = self._get_closest_and_distances(pos)
        deltas = closest - pos
        directions = np.zeros_like(deltas)
        non_zero = distances > 0.0
        directions[non_zero] = deltas[non_zero] / distances[non_zero, None]
        is_inside = self.is_inside(pos)
        directions[is_inside] *= -1
        return directions

    def closest_point(self, pos: ArrayLike) -> np.ndarray:
        pos = np.atleast_2d(pos)
        closest, _ = self._get_closest_and_distances(pos)
        return closest

    def _get_distances(self, pos: np.ndarray) -> np.ndarray:
        pos = np.atleast_2d(pos)
        points = [Point(p) for p in pos]
        distances = np.array([self.shape.boundary.distance(p) for p in points])
        return distances

    def _get_closest_and_distances(
        self, pos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        pos = np.atleast_2d(pos)
        points = [Point(p) for p in pos]
        lines = [shortest_line(self.shape.boundary, p) for p in points]
        closest = np.array([line.coords[0] for line in lines])
        distances = np.array([line.length for line in lines])
        return closest, distances

    @property
    def bounds(self) -> np.ndarray:
        """Returns [left, bottom, right, top] as a NumPy array."""
        return np.array(self.shape.bounds, dtype=float)

    @property
    def left(self) -> float:
        return self.bounds[0]

    @property
    def bottom(self) -> float:
        return self.bounds[1]

    @property
    def right(self) -> float:
        return self.bounds[2]

    @property
    def top(self) -> float:
        return self.bounds[3]


class CircularObstacle(Obstacle):

    def __init__(self, center: ArrayLike, radius: float, quad_segs: int = 4) -> None:
        self.center = np.array(center)
        self.radius = float(radius)
        super().__init__(Point(self.center).buffer(self.radius, quad_segs))

    def is_inside(self, pos: ArrayLike) -> np.ndarray:
        pos = np.atleast_2d(pos)
        distances = center_distances_numba(pos, self.center)
        is_inside = distances <= self.radius
        return is_inside

    def distance(self, pos: ArrayLike) -> np.ndarray:
        pos = np.atleast_2d(pos)
        distances = center_distances_numba(pos, self.center)
        distances = np.maximum(distances - self.radius, 0.0)
        return distances

    def direction(self, pos: ArrayLike) -> np.ndarray:
        pos = np.atleast_2d(pos)
        _, directions = center_distances_and_directions_numba(pos, self.center)
        return directions

    def closest_point(self, pos: ArrayLike) -> np.ndarray:
        pos = np.atleast_2d(pos)
        closest = circle_closest_point_numba(pos, self.center, self.radius)
        return closest


class RectangularObstacle(Obstacle):

    def __init__(self, bottom_left: ArrayLike, top_right: ArrayLike) -> None:
        self.bottom_left = np.array(bottom_left)
        self.top_right = np.array(top_right)
        super().__init__(box(*self.bottom_left, *self.top_right))

    def is_inside(self, pos: ArrayLike) -> np.ndarray:
        pos = np.atleast_2d(pos)
        is_inside = is_inside_rectangle_numba(
            pos, left=self.left, right=self.right, bottom=self.bottom, top=self.top
        )
        return is_inside

    def distance(self, pos: ArrayLike) -> np.ndarray:
        pos = np.atleast_2d(pos)
        distances = rectangle_external_distances_numba(
            pos, left=self.left, right=self.right, bottom=self.bottom, top=self.top
        )
        return distances

    def direction(self, pos: ArrayLike) -> np.ndarray:
        pos = np.atleast_2d(pos)
        _, directions = rectangle_distances_and_directions_numba(
            pos, left=self.left, right=self.right, bottom=self.bottom, top=self.top
        )
        is_inside = is_inside_rectangle_numba(
            pos, left=self.left, right=self.right, bottom=self.bottom, top=self.top
        )
        directions[is_inside] *= -1
        return directions

    def closest_point(self, pos: ArrayLike) -> np.ndarray:
        pos = np.asarray(pos)
        closest = rectangle_closest_point_numba(
            pos, left=self.left, right=self.right, bottom=self.bottom, top=self.top
        )
        return closest


class PolygonalObstacle(Obstacle):

    def __init__(self, vertices: ArrayLike):
        self.vertices = np.array(vertices)
        super().__init__(Polygon(self.vertices))
