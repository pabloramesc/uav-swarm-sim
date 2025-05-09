"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import os

import contextily as ctx
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from rasterio.coords import BoundingBox
import geopandas as gpd
from shapely.geometry import box  # Add this import


class ElevationMap:
    def __init__(self, dem_path: str):
        self.dem_path = dem_path
        self.bounds: BoundingBox = None
        self.resolution: tuple[float, float] = None
        self.elevation_data: np.ndarray = None

        # Load the DEM file
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
        lat = np.atleast_1d(lat)
        lon = np.atleast_1d(lon)

        in_bounds = (
            (self.bounds.left <= lon)
            & (lon <= self.bounds.right)
            & (self.bounds.bottom <= lat)
            & (lat <= self.bounds.top)
        )

        elevations = np.zeros_like(lat, dtype=float)
        valid_indices = np.where(in_bounds)[0]
        if valid_indices.size > 0:
            valid_lat = lat[valid_indices]
            valid_lon = lon[valid_indices]

            cols = ((valid_lon - self.bounds.left) / self.resolution[0]).astype(int)
            rows = ((self.bounds.top - valid_lat) / self.resolution[1]).astype(int)

            cols = np.clip(cols, 0, self.elevation_data.shape[1] - 1)
            rows = np.clip(rows, 0, self.elevation_data.shape[0] - 1)

            elevations[valid_indices] = self.elevation_data[rows, cols]

        return elevations if elevations.size > 1 else elevations.item()

    def _generate_grid(
        self, output_resolution: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        width, height = output_resolution
        lon = np.linspace(self.bounds.left, self.bounds.right, width)
        lat = np.linspace(self.bounds.top, self.bounds.bottom, height)
        return np.meshgrid(lon, lat)

    def _normalize_elevation(self, elevation_grid: np.ndarray) -> np.ndarray:
        return (elevation_grid - self.min_elevation) / (
            self.max_elevation - self.min_elevation
        )

    def _fetch_satellite_image(self) -> np.ndarray:
        """
        Fetch the satellite image for the elevation map bounds and crop it to match the elevation data.
        """
        # Convert bounds from EPSG:4326 (degrees) to EPSG:3857 (meters)
        bounds_box = box(
            self.bounds.left, self.bounds.bottom, self.bounds.right, self.bounds.top
        )
        gdf = gpd.GeoDataFrame(geometry=[bounds_box], crs="EPSG:4326").to_crs(epsg=3857)
        bounds_3857 = gdf.total_bounds  # [x_min, x_max, y_min, y_max] in EPSG:3857

        # Fetch the satellite image using the converted bounds
        img, ext = ctx.bounds2img(
            *bounds_3857, zoom=14, source=ctx.providers.Esri.WorldImagery
        )

        # Crop the satellite image to match the elevation bounds
        x_min, x_max, y_min, y_max = ext  # ext is in EPSG:3857
        lon_extent = np.linspace(
            x_min, x_max, img.shape[1]
        )  # X-axis (longitude in meters)
        lat_extent = np.linspace(
            y_max, y_min, img.shape[0]
        )  # Y-axis (latitude in meters, inverted)

        # Find indices for cropping
        lon_idx = np.where(
            (lon_extent >= bounds_3857[0]) & (lon_extent <= bounds_3857[1])
        )[0]
        lat_idx = np.where(
            (lat_extent >= bounds_3857[2]) & (lat_extent <= bounds_3857[3])
        )[0]

        # Crop the image
        cropped_img = img[
            np.min(lat_idx) : np.max(lat_idx) + 1, np.min(lon_idx) : np.max(lon_idx) + 1
        ]

        return cropped_img

    def save_satellite_image(self, image_path: str = None) -> None:
        if image_path is None:
            directory, _ = os.path.split(self.dem_path)
            image_path = os.path.join(directory, "satellite_image.png")

        img = self._fetch_satellite_image()
        img_pil = Image.fromarray(img)
        img_pil.save(image_path)
        print("Satellite image saved to", image_path)

    def save_elevation_image(
        self, image_path: str = None, output_resolution: tuple[int, int] = (1000, 1000)
    ) -> None:
        if image_path is None:
            directory, _ = os.path.split(self.dem_path)
            image_path = os.path.join(directory, "elevation_image.png")

        lon_grid, lat_grid = self._generate_grid(output_resolution)
        elevations = self.get_elevation(lat_grid.ravel(), lon_grid.ravel())
        elevation_grid = elevations.reshape(output_resolution[1], output_resolution[0])
        normalized_data = self._normalize_elevation(elevation_grid)

        plt.figure(figsize=(10, 10 * (output_resolution[1] / output_resolution[0])))
        plt.imshow(normalized_data, cmap="terrain")
        plt.axis("off")
        plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
        plt.close()
        print("Elevation image saved to", image_path)

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
    print(elevation_map.bounds)

    # Consultar la altitud en una coordenada específica
    latitude = 41.3851  # Latitud de Barcelona
    longitude = 2.1734  # Longitud de Barcelona
    altitude = elevation_map.get_elevation(latitude, longitude)
    print(f"Altitud en latitud {latitude}, longitud {longitude}: {altitude} m")

    # Visualizar el mapa de elevación
    elevation_map.plot()

    # Visualizar el mapa de elevación en 3D
    elevation_map.plot_3d()
