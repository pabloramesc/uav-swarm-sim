import numpy as np

from simulator.environment.elevation_map import ElevationMap
from simulator.environment.avoid_regions import Boundary, Obstacle, AvoidRegion
from simulator.environment.geo import geo2xyz, xyz2geo


class Environment:
    """
    Manages the simulation environment, including elevation data, boundaries, and obstacles.
    """

    def __init__(self, dem_path: str) -> None:
        """
        Initializes the environment with elevation data, calculates the home reference point,
        and initializes empty boundary/obstacle lists.

        Parameters
        ----------
        dem_path : str
            Path to the DEM (Digital Elevation Model) file.
        """
        self.elevation = ElevationMap(dem_path)
        self.boundary: Boundary = None
        self.obstacles: list[Obstacle] = []

        # Calculate the home reference point (bottom-left corner of the elevation map)
        self.home = np.array([self.elevation.bounds.bottom, self.elevation.bounds.left, 0.0])

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

    def get_avoid_regions(self) -> list[AvoidRegion]:
        """
        Returns a list of all avoid regions, including the boundary and obstacles.

        Returns
        -------
        list[AvoidRegion]
            List of avoid regions.
        """
        return [self.boundary] + self.obstacles

    def is_inside(self, pos: np.ndarray) -> bool:
        """
        Checks if a position is inside the environment boundary.

        Parameters
        ----------
        pos : np.ndarray
            Position [x, y] in meters.

        Returns
        -------
        bool
            True if the position is inside the boundary, False otherwise.
        """
        if self.boundary is None:
            raise ValueError("Boundary is not defined.")
        return self.boundary.is_inside(pos)

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
            True if the position collides with an obstacle or the ground, False otherwise.
        """
        # Check collision with the ground
        ground_elevation = self.get_elevation(pos[:2])
        if pos[2] <= ground_elevation:  # Check if altitude is below or at ground level
            return True

        # Check collision with obstacles
        for obstacle in self.obstacles:
            if obstacle.is_inside(pos[:2]):  # Check only x, y for obstacles
                return True
            
        return False

    def get_elevation(self, pos: np.ndarray) -> float:
        """
        Gets the elevation at a specific position.

        Parameters
        ----------
        pos : np.ndarray
            Position [x, y] in meters.

        Returns
        -------
        float
            Elevation in meters.
        """
        # Convert local Cartesian coordinates to geographic coordinates
        geo = xyz2geo(pos, self.home)
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
            Geographic coordinates [latitude, longitude, altitude] in degrees and meters.
        """
        return xyz2geo(pos, self.home)

    def geo_to_xyz(self, geo: np.ndarray) -> np.ndarray:
        """
        Converts geographic coordinates to local Cartesian coordinates.

        Parameters
        ----------
        geo : np.ndarray
            Geographic coordinates [latitude, longitude, altitude] in degrees and meters.

        Returns
        -------
        np.ndarray
            Local Cartesian coordinates [x, y, z] in meters.
        """
        return geo2xyz(geo, self.home)

    def plot_environment(self) -> None:
        """
        Visualizes the environment, including the boundary, obstacles, and elevation map.
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