"""
 Copyright (c) 2025 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np
from numpy.typing import ArrayLike
from shapely import Point, Polygon, box

from .base import AvoidRegion


class Obstacle(AvoidRegion):
    """
    Represents an obstacle in the simulation environment.
    """

    def __init__(self, shape: Polygon) -> None:
        """
        Initializes the obstacle with a given shape.

        Parameters
        ----------
        shape : Polygon
            The geometric shape of the obstacle.
        """
        super().__init__(shape)

    def distance(self, pos: ArrayLike) -> float:
        """
        Calculates the distance from a position to the obstacle.

        Parameters
        ----------
        pos : ArrayLike
            The position [x, y] to calculate the distance from.

        Returns
        -------
        float
            The distance to the obstacle. Returns 0.0 if the position is inside the obstacle.
        """
        if self.is_inside(pos):
            return 0.0
        return super().distance(pos)

    def direction(self, pos: ArrayLike) -> np.ndarray:
        """
        Calculates the normalized direction vector from a position to the obstacle.

        Parameters
        ----------
        pos : ArrayLike
            The position [x, y] to calculate the direction from.

        Returns
        -------
        np.ndarray
            A normalized direction vector [dx, dy]. If the position is inside the obstacle,
            the direction points outward.
        """
        direction = super().direction(pos)
        if self.is_inside(pos):
            return -direction
        return direction


class CircularObstacle(Obstacle):
    """
    Represents a circular obstacle in the simulation environment.

    Attributes
    ----------
    center : np.ndarray
        The center of the circular obstacle [x, y].
    radius : float
        The radius of the circular obstacle.
    """

    def __init__(self, center: ArrayLike, radius: float) -> None:
        """
        Initializes the circular obstacle with a center and radius.

        Parameters
        ----------
        center : ArrayLike
            The center of the circular obstacle [x, y].
        radius : float
            The radius of the circular obstacle.
        """
        self.center = np.array(center)
        self.radius = float(radius)
        super().__init__(Point(self.center).buffer(self.radius))

    def is_inside(self, pos: ArrayLike) -> bool:
        """
        Checks if a position is inside the circular obstacle.

        Parameters
        ----------
        pos : ArrayLike
            The position [x, y] to check.

        Returns
        -------
        bool
            True if the position is inside the obstacle, False otherwise.
        """
        delta = self.center - pos
        return np.linalg.norm(delta) < self.radius

    def distance(self, pos: ArrayLike) -> float:
        """
        Calculates the distance from a position to the circular obstacle.

        Parameters
        ----------
        pos : ArrayLike
            The position [x, y] to calculate the distance from.

        Returns
        -------
        float
            The distance to the obstacle. Returns 0.0 if the position is inside the obstacle.
        """
        delta = self.center - pos
        return max(0.0, np.linalg.norm(delta) - self.radius)

    def direction(self, pos: ArrayLike) -> np.ndarray:
        """
        Calculates the normalized direction vector from a position to the circular obstacle.

        Parameters
        ----------
        pos : ArrayLike
            The position [x, y] to calculate the direction from.

        Returns
        -------
        np.ndarray
            A normalized direction vector [dx, dy]. If the position is inside the obstacle,
            the direction points outward.
        """
        delta = self.center - pos
        norm = np.linalg.norm(delta)
        return delta / norm if norm > 0.0 else np.zeros(2)


class RectangularObstacle(Obstacle):
    """
    Represents a rectangular obstacle in the simulation environment.

    Attributes
    ----------
    bottom_left : np.ndarray
        The bottom-left corner of the rectangle [x, y].
    top_right : np.ndarray
        The top-right corner of the rectangle [x, y].
    """

    def __init__(self, bottom_left: ArrayLike, top_right: ArrayLike) -> None:
        """
        Initializes the rectangular obstacle with bottom-left and top-right corners.

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


class PolygonalObstacle(Obstacle):
    """
    Represents a polygonal obstacle in the simulation environment.

    Attributes
    ----------
    vertices : np.ndarray
        The vertices of the polygon as an array of [x, y] coordinates.
    """

    def __init__(self, vertices: ArrayLike):
        """
        Initializes the polygonal obstacle with a set of vertices.

        Parameters
        ----------
        vertices : ArrayLike
            The vertices of the polygon as an array of [x, y] coordinates.
        """
        self.vertices = np.array(vertices)
        super().__init__(Polygon(self.vertices))
