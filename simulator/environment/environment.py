"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np

from .elevation_map import ElevationMap
from .avoid_regions import Boundary, Obstacle, AvoidRegion
from .geo import geo2enu, enu2geo


class Environment:
    """
    Manages the simulation environment, including elevation data, boundaries,
    and obstacles.
    """

    def __init__(
        self,
        dem_path: str = None,
        boundary: Boundary = None,
        obstacles: list[Obstacle] = [],
    ) -> None:
        """
        Initializes the environment with elevation data, calculates the home
        reference point, and initializes empty boundary/obstacle lists.

        Parameters
        ----------
        dem_path : str, optional
            Path to the DEM (Digital Elevation Model) file. Default is None.
        boundary : Boundary, optional
            The boundary object defining the limits of the environment.
            Default is None.
        obstacle : list[Obstacle], optional
            A list with obstacle objects to add to the environment.
            Default is [].
        """
        self.elevation_map = ElevationMap(dem_path) if dem_path is not None else None
        self.boundary = boundary
        self.obstacles = obstacles

        # Calculate the home reference point (bottom-left corner of the elevation map)
        self.home = (
            np.array(
                [self.elevation_map.bounds.bottom, self.elevation_map.bounds.left, 0.0]
            )
            if self.elevation_map is not None
            else np.zeros(3)
        )

    @property
    def avoid_regions(self) -> list[AvoidRegion]:
        """
        A list of all avoid regions, including the boundary and obstacles.
        """
        return [self.boundary] + self.obstacles
    
    @property
    def boundary_xlim(self) -> tuple[float, float]:
        if self.boundary is None:
            return None
        x = self.boundary.shape.exterior.xy[0]
        return min(x), max(x)
    
    @property
    def boundary_ylim(self) -> tuple[float, float]:
        if self.boundary is None:
            return None
        y = self.boundary.shape.exterior.xy[1]
        return min(y), max(y)

    def set_boundary(self, boundary: Boundary) -> None:
        """
        Sets the boundary of the environment.

        Parameters
        ----------
        boundary : Boundary
            The boundary object defining the limits of the environment.
        """
        self.boundary = boundary

    def add_obstacle(self, obstacle: Obstacle) -> None:
        """
        Adds an obstacle to the environment.

        Parameters
        ----------
        obstacle : Obstacle
            The obstacle object to add to the environment.
        """
        self.obstacles.append(obstacle)

    def clear_obstacles(self) -> None:
        """
        Delete all obstacles.
        """
        self.obstacles = []

    def is_inside(self, pos: np.ndarray) -> np.ndarray:
        """
        Checks if one or more positions are inside the environment boundary.

        Parameters
        ----------
        pos : np.ndarray
            Position(s) [x, y, z] in meters. Can be a (3,) array for a single
            position or an (N, 3) array for multiple positions.

        Returns
        -------
        np.ndarray
            A boolean array of shape (N,) indicating whether each position is
            inside the boundary. If a single position is provided, a single
            boolean value is returned.
        """
        pos = np.atleast_2d(pos)  # Ensure pos is (N, 3)
        if self.boundary is None:
            raise ValueError("Boundary is not defined.")
        inside = np.array([self.boundary.is_inside(p[0:2]) for p in pos])
        return inside if len(inside) > 1 else inside[0]

    def is_collision(self, pos: np.ndarray) -> np.ndarray:
        """
        Checks if one or more positions collide with any obstacle or the ground.

        Parameters
        ----------
        pos : np.ndarray
            Position(s) [x, y, z] in meters. Can be a (3,) array for a single
            position or an (N, 3) array for multiple positions.

        Returns
        -------
        np.ndarray
            A boolean array of shape (N,) indicating whether each position
            collides with an obstacle or the ground. If a single position is
            provided, a single boolean value is returned.
        """
        pos = np.atleast_2d(pos)  # Ensure pos is (N, 3)

        # Check collision with the ground
        ground_elevations = self.get_elevation(pos[:, 0:2])  # Get ground elevation for all positions
        below_ground = pos[:, 2] < ground_elevations

        # Check collision with obstacles
        obstacle_collisions = np.array([
            any(obstacle.is_inside(p[0:2]) for obstacle in self.obstacles)
            for p in pos
        ])

        collisions = below_ground | obstacle_collisions
        return collisions if len(collisions) > 1 else collisions[0]

    def get_elevation(self, pos: np.ndarray) -> np.ndarray:
        """
        Gets the elevation at a specific position.

        Parameters
        ----------
        pos : np.ndarray
            Horizontal position(s) [x, y] in meters. Can be a (2,) array for a single
            position or an (N, 2) array for multiple positions.

        Returns
        -------
        np.ndarray
            An (N,2) array with elevation values in meters.
        """
        pos = np.atleast_2d(pos)
        if self.elevation_map is None:
            return np.zeros(pos.shape[0])
        # Convert local Cartesian coordinates to geographic coordinates
        enu = np.zeros((pos.shape[0], 3))
        enu[:, 0:2] = pos[:, 0:2]
        geo = enu2geo(enu, self.home)
        lat, lon = geo[:, 0], geo[:, 1]
        return self.elevation_map.get_elevation(lat, lon)

    def enu_to_geo(self, pos: np.ndarray) -> np.ndarray:
        """
        Converts local ENU (East-North-Up) coordinates to geographic coordinates.

        Parameters
        ----------
        pos : np.ndarray
            Local ENU coordinates [e, n, u] in meters.

        Returns
        -------
        np.ndarray
            Geographic coordinates [latitude, longitude, altitude] in
            (degrees, degrees, meters).
        """
        return enu2geo(pos, self.home)

    def geo_to_enu(self, geo: np.ndarray) -> np.ndarray:
        """
        Converts geographic coordinates to local ENU (East-North-Up) coordinates.

        Parameters
        ----------
        geo : np.ndarray
            Geographic coordinates [latitude, longitude, altitude] in
            (degrees, degrees, meters).

        Returns
        -------
        np.ndarray
            Local ENU coordinates [e, n, u] in meters.
        """
        return geo2enu(geo, self.home)

    def plot_environment(self) -> None:
        """
        Visualizes the environment, including the boundary, obstacles, and
        elevation map.
        """
        import matplotlib.pyplot as plt

        # Plot the elevation map
        self.elevation_map.plot()

        # Create a new figure for boundaries and obstacles
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot the boundary
        if self.boundary:
            x, y = self.boundary.shape.exterior.xy
            ax.plot(x, y, color="blue", label="Boundary")

        # Plot the obstacles
        for obstacle in self.obstacles:
            x, y = obstacle.shape.exterior.xy
            ax.plot(x, y, color="red", label="Obstacle")

        # Configure the plot
        ax.set_title("Simulation Environment")
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.legend()
        plt.show()
