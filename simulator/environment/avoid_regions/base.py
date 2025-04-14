"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from abc import ABC

import numpy as np
from numpy.typing import ArrayLike
from shapely import Point, Polygon, shortest_line, box


class Region(ABC):
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
        Initializes the region with a given shape.

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
        """
        line = shortest_line(self.shape.boundary, Point(pos))
        closest = np.array(line.coords[0])
        distance = line.length
        return closest, distance


class CircularRegion(Region):
    """
    Represents a circular region in the simulation environment.

    Attributes
    ----------
    center : np.ndarray
        The center of the circular region [x, y].
    radius : float
        The radius of the circular region.
    """

    def __init__(self, center: ArrayLike, radius: float, quad_segs: int = 2) -> None:
        """
        Initializes the circular region with a center and radius.

        Parameters
        ----------
        center : ArrayLike
            The center of the circular region [x, y].
        radius : float
            The radius of the circular region.
        quad_segs : int, optional
            Number of segments to approximate the circle (default is 2).
        """
        self.center = np.array(center)
        self.radius = float(radius)
        super().__init__(Point(self.center).buffer(self.radius, quad_segs))

    def is_inside(self, pos: ArrayLike) -> bool:
        """
        Checks if a position is inside the circular region.

        Parameters
        ----------
        pos : ArrayLike
            The position [x, y] to check.

        Returns
        -------
        bool
            True if the position is inside the region, False otherwise.
        """
        delta = self.center - pos
        return np.linalg.norm(delta) <= self.radius

    def distance(self, pos: ArrayLike) -> float:
        """
        Calculates the distance from a position to the circular region.

        Parameters
        ----------
        pos : ArrayLike
            The position [x, y] to calculate the distance from.

        Returns
        -------
        float
            The distance to the region. Returns 0.0 if the position is inside the region.
        """
        delta = self.center - pos
        return abs(np.linalg.norm(delta) - self.radius)

    def direction(self, pos: ArrayLike) -> np.ndarray:
        """
        Calculates the normalized direction vector from a position to the circular region.

        Parameters
        ----------
        pos : ArrayLike
            The position [x, y] to calculate the direction from.

        Returns
        -------
        np.ndarray
            A normalized direction vector [dx, dy].
        """
        delta = self.center - pos
        norm = np.linalg.norm(delta)
        direction = delta / norm if norm > 0.0 else np.zeros(2)
        if self.is_inside(pos):
            return -direction
        return direction

    def closest_point(self, pos: ArrayLike) -> np.ndarray:
        """
        Finds the closest point on the boundary of the circular region to a given position.

        Parameters
        ----------
        pos : ArrayLike
            The position [x, y] to find the closest point to.

        Returns
        -------
        np.ndarray
            The closest point [x, y] on the boundary.
        """
        delta = pos - self.center
        norm = np.linalg.norm(delta)
        if norm == 0.0:
            # If the position is exactly at the center, return a point on the boundary
            return self.center + np.array([self.radius, 0])
        return self.center + (delta / norm) * self.radius


class RectangularRegion(Region):
    """
    Represents a rectangular region in the simulation environment.

    Attributes
    ----------
    bottom_left : np.ndarray
        The bottom-left corner of the rectangle [x, y].
    top_right : np.ndarray
        The top-right corner of the rectangle [x, y].
    """

    def __init__(self, bottom_left: ArrayLike, top_right: ArrayLike) -> None:
        """
        Initializes the rectangular region with bottom-left and top-right corners.

        Parameters
        ----------
        bottom_left : ArrayLike
            The bottom-left corner of the rectangle [x, y].
        top_right : ArrayLike
            The top-right corner of the rectangle [x, y].
        """
        self.bottom_left = np.array(bottom_left)
        self.top_right = np.array(top_right)
        super().__init__(box(*self.bottom_left, *self.top_right))

    @property
    def bottom(self) -> float:
        return self.bottom_left[1]

    @property
    def left(self) -> float:
        return self.bottom_left[0]

    @property
    def top(self) -> float:
        return self.top_right[1]

    @property
    def right(self) -> float:
        return self.top_right[0]

    def is_inside(self, pos: ArrayLike) -> bool:
        """
        Checks if a position is inside the rectangular region.

        Parameters
        ----------
        pos : ArrayLike
            The position [x, y] to check.

        Returns
        -------
        bool
            True if the position is inside the region, False otherwise.
        """
        return self.left <= pos[0] <= self.right and self.bottom <= pos[1] <= self.top

    def distance(self, pos: ArrayLike) -> float:
        """
        Calculates the distance from a position to the rectangular region.

        Parameters
        ----------
        pos : ArrayLike
            The position [x, y] to calculate the distance from.

        Returns
        -------
        float
            The distance to the region.
        """
        closest_point = self.closest_point(pos)
        return np.linalg.norm(pos - closest_point)

    def direction(self, pos: ArrayLike) -> np.ndarray:
        """
        Calculates the normalized direction vector from a position to the rectangular region.

        Parameters
        ----------
        pos : ArrayLike
            The position [x, y] to calculate the direction from.

        Returns
        -------
        np.ndarray
            A normalized direction vector [dx, dy].
        """
        closest = self.closest_point(pos)
        delta = closest - pos
        norm = np.linalg.norm(delta)
        direction = delta / norm if norm > 0.0 else np.zeros(2)
        return direction

    def closest_point(self, pos: ArrayLike) -> np.ndarray:
        """
        Finds the closest point on the boundary of the rectangular region to a given position.

        Parameters
        ----------
        pos : ArrayLike
            The position [x, y] to find the closest point to.

        Returns
        -------
        np.ndarray
            The closest point [x, y] on the boundary.
        """
        # Determine if the position is inside the rectangle
        if self.is_inside(pos):
            # If inside, find the closest edge
            distances = [
                abs(pos[0] - self.left),  # Distance to left edge
                abs(pos[0] - self.right),  # Distance to right edge
                abs(pos[1] - self.bottom),  # Distance to bottom edge
                abs(pos[1] - self.top),  # Distance to top edge
            ]
            min_index = np.argmin(distances)
            if min_index == 0:  # Closest to left edge
                return np.array([self.left, pos[1]])
            elif min_index == 1:  # Closest to right edge
                return np.array([self.right, pos[1]])
            elif min_index == 2:  # Closest to bottom edge
                return np.array([pos[0], self.bottom])
            else:  # Closest to top edge
                return np.array([pos[0], self.top])
        else:
            # If outside, return the clamped position
            closest_x = np.clip(pos[0], self.left, self.right)
            closest_y = np.clip(pos[1], self.bottom, self.top)
            return np.array([closest_x, closest_y])


class PolygonalRegion(Region):
    """
    Represents a polygonal region in the simulation environment.

    Attributes
    ----------
    vertices : np.ndarray
        The vertices of the polygon as an array of [x, y] coordinates.
    """

    def __init__(self, vertices: ArrayLike):
        """
        Initializes the polygonal region with a set of vertices.

        Parameters
        ----------
        vertices : ArrayLike
            The vertices of the polygon as an array of [x, y] coordinates.
        """
        self.vertices = np.array(vertices)
        super().__init__(Polygon(self.vertices))
