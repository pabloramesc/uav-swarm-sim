"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.coords import BoundingBox
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D


class ElevationMap:
    def __init__(self, dem_path: str):
        self.dem_path = dem_path
        self.bounds: BoundingBox = None
        self.resolution: tuple[float, float] = None
        self.elevation_data: np.ndarray = None

        # Cargar el archivo DEM
        self.load_dem(self.dem_path)

    @property
    def min_elevation(self) -> float:
        return self.elevation_data.min()

    @property
    def max_elevation(self) -> float:
        return self.elevation_data.max()

    def load_dem(self, dem_path: str):
        with rasterio.open(dem_path) as dem:
            self.bounds = dem.bounds
            self.resolution = dem.res
            self.elevation_data = dem.read(1)

    def get_elevation(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """
        Gets the elevation for one or more geographic coordinates.

        Parameters
        ----------
        lat : np.ndarray
            Latitude(s) in degrees. Can be a scalar or a 1D array of length N.
        lon : np.ndarray
            Longitude(s) in degrees. Can be a scalar or a 1D array of length N.

        Returns
        -------
        np.ndarray
            Elevation(s) in meters. Returns a scalar if inputs are scalars,
            or a 1D array of length N if inputs are arrays.
        """
        # Ensure lat and lon are numpy arrays
        lat = np.atleast_1d(lat)
        lon = np.atleast_1d(lon)

        # Check if coordinates are within bounds
        in_bounds = (
            (self.bounds.left <= lon)
            & (lon <= self.bounds.right)
            & (self.bounds.bottom <= lat)
            & (lat <= self.bounds.top)
        )

        # Initialize elevation array with zeros
        elevations = np.zeros_like(lat, dtype=float)

        # Process only the coordinates within bounds
        valid_indices = np.where(in_bounds)[0]
        if valid_indices.size > 0:
            valid_lat = lat[valid_indices]
            valid_lon = lon[valid_indices]

            # Compute row and column indices for valid coordinates
            cols = ((valid_lon - self.bounds.left) / self.resolution[0]).astype(int)
            rows = ((self.bounds.top - valid_lat) / self.resolution[1]).astype(int)

            # Get elevation values for valid coordinates
            elevations[valid_indices] = self.elevation_data[rows, cols]

        # Return scalar if input was scalar, otherwise return array
        return elevations if elevations.size > 1 else elevations.item()

    def plot(self, ax: Axes = None, show: bool = False):
        """
        Visualizes the elevation map.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            External axes to plot on. If None, a new figure and axes are created.
        """
        if self.elevation_data is None:
            raise ValueError("Elevation data is not loaded.")

        # Create a new figure and axes if ax is None
        if ax is None:
            fig, ax = plt.subplots()
            show = True

        im = ax.imshow(
            self.elevation_data,
            cmap="terrain",
            extent=(
                self.bounds.left,
                self.bounds.right,
                self.bounds.bottom,
                self.bounds.top,
            ),
            origin="upper",
        )
        plt.colorbar(im, ax=ax, label="Elevation (m)")
        ax.set_title("Elevation Map")
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")

        # Show the plot only if ax is None (external axes won't call plt.show())
        if show:
            plt.show()

    def plot_3d(self, ax: Axes3D = None, show: bool = False):
        """
        Visualizes the elevation map in 3D.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.Axes3DSubplot, optional
            External 3D axes to plot on. If None, a new figure and axes are created.
        """
        if self.elevation_data is None:
            raise ValueError("Elevation data is not loaded.")

        # Create a new figure and axes if ax is None
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            show = True

        # Create a meshgrid for the elevation data
        lon = np.linspace(
            self.bounds.left, self.bounds.right, self.elevation_data.shape[1]
        )
        lat = np.linspace(
            self.bounds.bottom, self.bounds.top, self.elevation_data.shape[0]
        )
        lon_grid, lat_grid = np.meshgrid(lon, lat)

        # Plot the surface
        surf = ax.plot_surface(
            lon_grid,
            lat_grid,
            np.flipud(self.elevation_data),
            cmap="terrain",
            edgecolor="none",
            linewidth=0,
            antialiased=False,
            rstride=5,
            cstride=5,
        )

        # Convert latitude and longitude to meters
        lat_to_meters = 111320.0  # Approximate conversion factor for latitude
        lon_to_meters = 111320.0 * np.cos(np.deg2rad(lat_grid))  # Varies with latitude
        x = (lon_grid - self.bounds.left) * lon_to_meters
        y = (lat_grid - self.bounds.bottom) * lat_to_meters

        # Adjust axis limits to bounds
        ax.set_xlim(lon.min(), lon.max())
        ax.set_ylim(lat.min(), lat.max())
        ax.set_zlim(self.min_elevation, self.max_elevation)
        ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(self.elevation_data)))

        # Add a colorbar if ax is None
        ax.figure.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Elevation (m)")

        # Set labels and title
        ax.set_title("Elevation Map (3D)")
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.set_zlabel("Elevation (m)")

        # Show the plot only if ax is None
        if show:
            plt.show()


# Ejemplo de uso
if __name__ == "__main__":
    # Ruta al archivo DEM
    dir_path = "data"
    file_name = "barcelona_dem.tif"
    file_path = os.path.join(dir_path, file_name)

    # Crear el objeto ElevationMap
    elevation_map = ElevationMap(file_path)

    # Consultar la altitud en una coordenada específica
    latitude = 41.3851  # Latitud de Barcelona
    longitude = 2.1734  # Longitud de Barcelona
    altitude = elevation_map.get_elevation(latitude, longitude)
    print(f"Altitud en latitud {latitude}, longitud {longitude}: {altitude} m")

    # Visualizar el mapa de elevación
    elevation_map.plot()

    # Visualizar el mapa de elevación en 3D
    elevation_map.plot_3d()
