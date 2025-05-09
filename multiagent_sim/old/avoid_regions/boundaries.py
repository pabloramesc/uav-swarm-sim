"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np
from numpy.typing import ArrayLike

from shapely import Polygon

from .base import (
    Region,
    CircularRegion,
    RectangularRegion,
    PolygonalRegion,
)


class Boundary(Region):
    """
    Represents a boundary in the simulation environment.
    """

    def __init__(self, shape: Polygon) -> None:
        """
        Initializes the boundary with a given shape.

        Parameters
        ----------
        shape : Polygon
            The geometric shape of the boundary.
        """
        super().__init__(shape)

    def distance(self, pos: ArrayLike) -> float:
        """
        Calculates the distance from a position to the boundary.

        Parameters
        ----------
        pos : ArrayLike
            The position [x, y] to calculate the distance from.

        Returns
        -------
        float
            The distance to the boundary. Returns 0.0 if the position is outside the boundary.
        """
        if not self.is_inside(pos):
            return 0.0
        return super().distance(pos)

    def direction(self, pos: ArrayLike) -> np.ndarray:
        """
        Calculates the normalized direction vector from a position to the boundary.

        Parameters
        ----------
        pos : ArrayLike
            The position [x, y] to calculate the direction from.

        Returns
        -------
        np.ndarray
            A normalized direction vector [dx, dy].
        """
        direction = super().direction(pos)
        if not self.is_inside(pos):
            return -direction
        return direction


class CircularBoundary(CircularRegion, Boundary):
    """
    Represents a circular boundary in the simulation environment.

    Attributes
    ----------
    center : np.ndarray
        The center of the circular boundary [x, y].
    radius : float
        The radius of the circular boundary.
    """

    def __init__(self, center: ArrayLike, radius: float, quad_segs: int = 2) -> None:
        """
        Initializes the circular boundary with a center and radius.

        Parameters
        ----------
        center : ArrayLike
            The center of the circular region [x, y].
        radius : float
            The radius of the circular region.
        quad_segs : int, optional
            Number of segments to approximate the circle (default is 2).
        """
        super().__init__(center, radius, quad_segs)

    def distance(self, pos: ArrayLike) -> float:
        """
        Calculates the distance from a position to the circular boundary.

        Parameters
        ----------
        pos : ArrayLike
            The position [x, y] to calculate the distance from.

        Returns
        -------
        float
            The distance to the boundary. Returns 0.0 if the position is outside the boundary.
        """
        delta = self.center - pos
        return max(0.0, self.radius - np.linalg.norm(delta))
    
    def direction(self, pos: ArrayLike) -> np.ndarray:
        """
        Calculates the normalized direction vector from a position to the circular boundary.

        Parameters
        ----------
        pos : ArrayLike
            The position [x, y] to calculate the direction from.

        Returns
        -------
        np.ndarray
            A normalized direction vector [dx, dy].
        """
        delta = pos - self.center
        norm = np.linalg.norm(delta)
        direction = delta / norm if norm > 0.0 else np.zeros(2)
        return direction


class RectangularBoundary(RectangularRegion, Boundary):
    """
    Represents a rectangular boundary in the simulation environment.

    Attributes
    ----------
    bottom_left : np.ndarray
        The bottom-left corner of the rectangle [x, y].
    top_right : np.ndarray
        The top-right corner of the rectangle [x, y].
    """

    def __init__(self, bottom_left: ArrayLike, top_right: ArrayLike) -> None:
        """
        Initializes the rectangular boundary with bottom-left and top-right corners.

        Parameters
        ----------
        bottom_left : ArrayLike
            The bottom-left corner of the rectangle [x, y].
        top_right : ArrayLike
            The top-right corner of the rectangle [x, y].
        """
        super().__init__(bottom_left, top_right)
        
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
            The distance to the region. Returns 0.0 if the position is outside the region.
        """
        if not self.is_inside(pos):
            return 0.0
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
        if not self.is_inside(pos):
            return -direction
        return direction


class PolygonalBoundary(PolygonalRegion, Boundary):
    """
    Represents a polygonal boundary in the simulation environment.

    Attributes
    ----------
    vertices : np.ndarray
        The vertices of the polygon as an array of [x, y] coordinates.
    """

    def __init__(self, vertices: ArrayLike):
        """
        Initializes the polygonal boundary with a set of vertices.

        Parameters
        ----------
        vertices : ArrayLike
            The vertices of the polygon as an array of [x, y] coordinates.
        """
        super().__init__(vertices)
