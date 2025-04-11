"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from abc import ABC

import numpy as np
from numpy.typing import ArrayLike
from shapely import Point, Polygon, shortest_line


class AvoidRegion(ABC):
    """
    Abstract base class for regions to avoid in the simulation environment.

    Attributes
    ----------
    shape : Polygon
        The geometric shape of the region.
    centroid : np.ndarray
        The centroid of the region as a 2D array [x, y].
    """

    def __init__(self, shape: Polygon) -> None:
        """
        Initializes the avoid region with a given shape.

        Parameters
        ----------
        shape : Polygon
            The geometric shape of the region.
        """
        self.shape = shape
        self.centroid = np.array(self.shape.centroid.coords[0])

    def is_inside(self, pos: ArrayLike) -> bool:
        """
        Checks if a position is inside the region.

        Parameters
        ----------
        pos : ArrayLike
            The position [x, y] to check.

        Returns
        -------
        bool
            True if the position is inside the region, False otherwise.
        """
        return self.shape.contains(Point(pos))

    def distance(self, pos: ArrayLike) -> float:
        """
        Calculates the distance from a position to the boundary of the region.

        Parameters
        ----------
        pos : ArrayLike
            The position [x, y] to calculate the distance from.

        Returns
        -------
        float
            The distance to the boundary of the region.
        """
        return self.shape.boundary.distance(Point(pos))

    def direction(self, pos: ArrayLike) -> np.ndarray:
        """
        Calculates the normalized direction vector from a position to the closest point on the boundary.

        Parameters
        ----------
        pos : ArrayLike
            The position [x, y] to calculate the direction from.

        Returns
        -------
        np.ndarray
            A normalized direction vector [dx, dy].
        """
        closest, distance = self._get_closest_and_distance(pos)
        direction = (closest - pos) / distance if distance > 0.0 else np.zeros(2)
        return direction

    def closest_point(self, pos: ArrayLike) -> np.ndarray:
        """
        Finds the closest point on the boundary of the region to a given position.

        Parameters
        ----------
        pos : ArrayLike
            The position [x, y] to find the closest point to.

        Returns
        -------
        np.ndarray
            The closest point [x, y] on the boundary.
        """
        closest, _ = self._get_closest_and_distance(pos)
        return np.array(closest)

    def _get_closest_and_distance(self, pos: ArrayLike) -> tuple[np.ndarray, float]:
        """
        Helper method to calculate the closest point and distance to the boundary.

        Parameters
        ----------
        pos : ArrayLike
            The position [x, y] to calculate the closest point and distance from.

        Returns
        -------
        tuple[np.ndarray, float]
            The closest point [x, y] and the distance to the boundary.
        """
        line = shortest_line(self.shape.boundary, Point(pos))
        closest = np.array(line.coords[0])
        distance = line.length
        return closest, distance
