"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np

from .elevation_map import ElevationMap
from .avoid_regions import Boundary, Obstacle, AvoidRegion
from .geo import geo2xyz, xyz2geo


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
        self.elevation = ElevationMap(dem_path) if dem_path is not None else None
        self.boundary = boundary
        self.obstacles = obstacles

        # Calculate the home reference point (bottom-left corner of the elevation map)
        self.home = (
            np.array([self.elevation.bounds.bottom, self.elevation.bounds.left, 0.0])
            if self.elevation is not None
            else np.zeros(3)
        )

    @property
    def avoid_regions(self) -> list[AvoidRegion]:
        """
        A list of all avoid regions, including the boundary and obstacles.
        """
        return [self.boundary] + self.obstacles

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

    def is_inside(self, pos: np.ndarray) -> bool:
        """
        Checks if a position is inside the environment boundary.

        Parameters
        ----------
        pos : np.ndarray
            Position [x, y, z] in meters.

        Returns
        -------
        bool
            True if the position is inside the boundary, False otherwise.
        """
        if self.boundary is None:
            raise ValueError("Boundary is not defined.")
        return self.boundary.is_inside(pos[0:2])

    def is_collision(self, pos: np.ndarray) -> bool:
        """
        Checks if a position collides with any obstacle or the ground.

        Parameters
        ----------
        pos : np.ndarray
            Position [x, y, z] in meters.

        Returns
        -------
        bool
            True if the position collides with an obstacle or the ground,
            False otherwise.
        """
        # Check collision with the ground
        ground_elevation = self.get_elevation(pos)
        if pos[2] <= ground_elevation:  # Check if altitude is below or at ground level
            return True

        # Check collision with obstacles
        for obstacle in self.obstacles:
            if obstacle.is_inside(pos[0:2]):  # Check only x, y for obstacles
                return True

        return False

    def get_elevation(self, pos: np.ndarray) -> float:
        """
        Gets the elevation at a specific position.

        Parameters
        ----------
        pos : np.ndarray
            A (2,) array with horizontal position [x, y] in meters.

        Returns
        -------
        float
            Elevation in meters.
        """
        if self.elevation is None:
            return 0.0
        # Convert local Cartesian coordinates to geographic coordinates
        xyz = np.zeros(3)
        xyz[0:2] = pos
        geo = xyz2geo(xyz, self.home)
        lat, lon = geo[0], geo[1]
        return self.elevation.get_elevation(lat, lon)

    def xyz_to_geo(self, pos: np.ndarray) -> np.ndarray:
        """
        Converts local Cartesian coordinates to geographic coordinates.

        Parameters
        ----------
        pos : np.ndarray
            Local Cartesian coordinates [x, y, z] in meters.

        Returns
        -------
        np.ndarray
            Geographic coordinates [latitude, longitude, altitude] in
            (degrees, degrees, meters).
        """
        return xyz2geo(pos, self.home)

    def geo_to_xyz(self, geo: np.ndarray) -> np.ndarray:
        """
        Converts geographic coordinates to local Cartesian coordinates.

        Parameters
        ----------
        geo : np.ndarray
            Geographic coordinates [latitude, longitude, altitude] in
            (degrees, degrees, meters).

        Returns
        -------
        np.ndarray
            Local Cartesian coordinates [x, y, z] in meters.
        """
        return geo2xyz(geo, self.home)

    def plot_environment(self) -> None:
        """
        Visualizes the environment, including the boundary, obstacles, and
        elevation map.
        """
        import matplotlib.pyplot as plt

        # Plot the elevation map
        self.elevation.plot()

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
