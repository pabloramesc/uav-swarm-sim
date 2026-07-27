"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..math.geo import enu2geo, geo2enu
from .obstacles.boundaries import Boundary, PolygonalBoundary, RectangularBoundary
from .obstacles.obstacles import CircularObstacle, Obstacle, RectangularObstacle


class Environment:
    """Manages the environment, including elevation data, boundaries,
    and obstacles."""

    def __init__(
        self,
        dem_path: str | None = None,
        boundary: Boundary | None = None,
        obstacles: list[Obstacle] | None = None,
        *,
        fetch_satellite: bool = False,
    ) -> None:
        """Initializes the environment with elevation data, calculates the home
        reference point, and initializes empty boundary/obstacle lists.

        Args:
            dem_path: Path to the DEM (Digital Elevation Model) file.
            boundary: A boundary object defining the limits of the environment.
            obstacles: Obstacle objects to add to the environment.
            fetch_satellite: Download terrain background tiles when true.
        """
        if dem_path is not None:
            # Terrain support is optional and relatively heavy.  Keep it out of
            # flat-world imports and load it only when a DEM is requested.
            from .elevation_map import ElevationMap

            self.elevation_map: ElevationMap | None = ElevationMap(
                dem_path,
                fetch_satellite=fetch_satellite,
            )
        else:
            self.elevation_map = None
        self._boundary = boundary
        self._obstacles = list(obstacles) if obstacles is not None else []

        # Calculate the home reference point (bottom-left corner of the elevation map)
        self.home = (
            np.array(
                [
                    self.elevation_map.bounds.bottom,
                    self.elevation_map.bounds.left,
                    0.0,
                ]
            )
            if self.elevation_map is not None
            else np.zeros(3)
        )

    @property
    def boundary(self) -> Boundary | None:
        return self._boundary

    def require_boundary(self) -> Boundary:
        """Return the configured boundary or raise a useful lifecycle error."""

        if self._boundary is None:
            raise RuntimeError("Environment boundary is not configured.")
        return self._boundary

    @property
    def obstacles(self) -> tuple[Obstacle, ...]:
        """Read-only structural view of configured obstacles."""

        return tuple(self._obstacles)

    @property
    def boundary_and_obstacles(self) -> tuple[Obstacle, ...]:
        """All avoidance regions, with the boundary first."""

        return (self.require_boundary(), *self.obstacles)

    def set_boundary(self, boundary: Boundary) -> None:
        """Sets the boundary of the environment.

        Args:
            boundary: The boundary object defining the limits of the environment.
        """
        self._boundary = boundary

    def add_obstacle(self, obstacle: Obstacle) -> None:
        """Adds an obstacle to the environment.

        Args:
            obstacle: The obstacle object to add to the environment.
        """
        self._obstacles.append(obstacle)

    def clear_obstacles(self) -> None:
        """Delete all obstacles."""
        self._obstacles.clear()

    def set_rectangular_boundary(
        self, bottom_left: ArrayLike, top_right: ArrayLike
    ) -> None:
        """Sets a rectangular boundary for the environment.

        Args:
            bottom_left: Coordinates of the bottom-left corner of the boundary [x, y].
            top_right: Coordinates of the top-right corner of the boundary [x, y].
        """
        rect = RectangularBoundary(bottom_left, top_right)
        self.set_boundary(rect)

    def set_polygonal_boundary(self, vertices: ArrayLike) -> None:
        """Sets a polygonal boundary for the environment.

        Args:
            vertices: List of vertices defining the polygonal boundary.
        """
        poly = PolygonalBoundary(vertices)
        self.set_boundary(poly)

    def add_circular_obstacle(self, center: ArrayLike, radius: float) -> None:
        """Adds a circular obstacle to the environment.

        Args:
            center: Coordinates of the center of the obstacle [x, y].
            radius: Radius of the circular obstacle.
        """
        circ = CircularObstacle(center, radius)
        self.add_obstacle(circ)

    def add_rectangular_obstacle(
        self, bottom_left: ArrayLike, top_right: ArrayLike
    ) -> None:
        """Adds a rectangular obstacle to the environment.

        Args:
            bottom_left: Coordinates of the bottom-left corner of the obstacle [x, y].
            top_right: Coordinates of the top-right corner of the obstacle [x, y].
        """
        rect = RectangularObstacle(bottom_left, top_right)
        self.add_obstacle(rect)

    def is_inside(self, pos: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Checks if one or more positions are inside the environment boundary.

        Args:
            pos: Position(s) [x, y, z] in meters. Can be a (3,) array for a
                single position or a (N, 3) array for multiple positions.

        Returns:
            A boolean array of shape (N,) indicating whether each position is
                inside the boundary.
        """
        pos = np.atleast_2d(pos)  # Ensure pos is (N, 3)
        return self.require_boundary().is_inside(pos[:, 0:2])

    def is_collision(
        self,
        pos: NDArray[np.float64],
        check_altitude: bool = False,
        check_boundary: bool = False,
    ) -> NDArray[np.bool_]:
        """Checks if one or more positions collide with any obstacle or the ground.

        Parameters
        ----------
        pos : ArrayLike
            Position(s) [x, y, z] in meters. Can be a (3,) array for a single
            position or an (N, 3) array for multiple positions.

        Returns:
            A boolean array of shape (N,) indicating whether each position
                collides with an obstacle or the ground.
        """
        pos = np.atleast_2d(pos)  # Ensure pos is (N, 3)

        boundary_collisions = np.zeros(pos.shape[0], dtype=bool)
        if check_boundary and self.boundary is not None:
            boundary_collisions = ~self.boundary.is_inside(pos[:, 0:2])

        obstacle_collisions = np.array(
            [obstacle.is_inside(pos[:, 0:2]) for obstacle in self._obstacles]
        )
        obstacle_collisions = np.any(obstacle_collisions, axis=0)

        collisions = boundary_collisions | obstacle_collisions

        if check_altitude:
            ground_elevations = self.get_elevation(
                pos[:, 0:2]
            )  # Get ground elevation for all positions
            below_ground = pos[:, 2] < ground_elevations
            collisions = collisions | below_ground

        return collisions

    def get_elevation(self, pos: NDArray[np.float64]) -> NDArray[np.float64]:
        """Gets the elevation at a specific position.

        Args:
            pos: Horizontal position(s) [x, y] in meters. Can be a (2,) array
                for a single position or an (N, 2) array for multiple positions.

        Returns:
            A (N,) array with elevation values in meters.
        """
        pos = np.atleast_2d(pos)

        if self.elevation_map is None:
            return np.zeros(pos.shape[0])

        # Convert local Cartesian coordinates to geographic coordinates
        enu = np.zeros((pos.shape[0], 3))
        enu[:, 0:2] = pos[:, 0:2]
        geo = enu2geo(enu, self.home)
        geo = np.atleast_2d(geo)
        lat, lon = geo[:, 0], geo[:, 1]
        return self.elevation_map.get_elevation(lat, lon)

    def enu2geo(self, pos: ArrayLike) -> NDArray[np.float64]:
        """Converts local ENU (East-North-Up) coordinates to geographic coordinates.

        Args:
            pos: Local ENU coordinates [e, n, u] in meters.

        Returns:
            Geographic coordinates [latitude, longitude, altitude] in
            (degrees, degrees, meters).
        """
        return enu2geo(pos, self.home)

    def geo2enu(self, geo: ArrayLike) -> NDArray[np.float64]:
        """Converts geographic coordinates to local ENU (East-North-Up) coordinates.

        Args:
            geo: Geographic coordinates [latitude, longitude, altitude] in
                (degrees, degrees, meters).

        Returns:
            Local ENU coordinates [e, n, u] in meters.
        """
        return geo2enu(geo, self.home)
