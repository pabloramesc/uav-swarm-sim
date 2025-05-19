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
from pyproj import Transformer


class ElevationMap:
    def __init__(self, dem_path: str):
        self.dem_path = dem_path
        self.bounds: BoundingBox = None
        self.resolution: tuple[float, float] = None
        self.elevation_data: np.ndarray = None

        self.load_dem(self.dem_path)

        self.elevation_img = self.generate_elevation_image()
        self.satellite_img = self.generate_satellite_image()
        self.fused_img = self.generate_fused_image()

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
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        left, bottom = transformer.transform(self.bounds.left, self.bounds.bottom)
        right, top = transformer.transform(self.bounds.right, self.bounds.top)
        bounds_3857 = (left, bottom, right, top)

        # print("Bounds in WGS84:", self.bounds)
        # print("Bounds in EPSG:3857:", bounds_3857)

        try:
            img, ext = ctx.bounds2img(
                *bounds_3857, zoom=14, source=ctx.providers.Esri.WorldImagery
            )
            # print("Satellite image downloaded successfully")
        except Exception as e:
            print("Error downloading tiles:", e)

        # print("Returned extent (EPSG:3857):", ext)

        x_min, x_max, y_min, y_max = ext
        height, width, _ = img.shape

        pixel_size_x = (x_max - x_min) / width
        pixel_size_y = (y_max - y_min) / height

        left_px = int((bounds_3857[0] - x_min) / pixel_size_x)
        right_px = int((bounds_3857[2] - x_min) / pixel_size_x)
        top_px = int((y_max - bounds_3857[3]) / pixel_size_y)
        bottom_px = int((y_max - bounds_3857[1]) / pixel_size_y)

        if left_px < 0 or right_px > width or top_px < 0 or bottom_px > height:
            raise ValueError(
                "Calculated crop bounds fall outside the image dimensions."
            )

        cropped_img = img[top_px:bottom_px, left_px:right_px]
        return cropped_img

    def generate_elevation_image(
        self, output_resolution: tuple[int, int] = (1000, 1000)
    ) -> None:
        lon_grid, lat_grid = self._generate_grid(output_resolution)
        elevations = self.get_elevation(lat_grid.ravel(), lon_grid.ravel())
        elevation_grid = elevations.reshape(output_resolution[1], output_resolution[0])
        normalized_data = self._normalize_elevation(elevation_grid)
        return normalized_data

    def generate_satellite_image(self) -> None:
        img = self._fetch_satellite_image()
        return img

    def generate_fused_image(
        self,
        output_resolution: tuple[int, int] = (1000, 1000),
        alpha: float = 0.5,
    ) -> None:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("Alpha must be between 0 and 1.")

        sat_img = (
            self.satellite_img
            if self.satellite_img is not None
            else self.generate_satellite_image()
        )
        sat_resized = np.array(Image.fromarray(sat_img).resize(output_resolution[::-1]))
        elev_norm = self.generate_elevation_image(output_resolution)
        elev_rgb = plt.cm.terrain(elev_norm)[:, :, :3]

        elev_uint8 = (elev_rgb * 255).astype(np.uint8)
        sat_uint8 = sat_resized[:, :, :3]

        fused = ((1 - alpha) * sat_uint8 + alpha * elev_uint8).astype(np.uint8)
        self.fused_img = fused
        return fused


if __name__ == "__main__":
    file_name = "barcelona_dem.tif"
    file_path = os.path.join("data", "elevation", file_name)

    elevation_map = ElevationMap(file_path)
    print(elevation_map.bounds)

    # Generate and plot
    elev_img = elevation_map.generate_elevation_image()
    sat_img = elevation_map.generate_satellite_image()
    fused_img = elevation_map.generate_fused_image()

    # Example plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(elev_img, cmap="terrain")
    axes[0].set_title("Elevation (normalized)")
    axes[1].imshow(sat_img)
    axes[1].set_title("Satellite")
    axes[2].imshow(fused_img)
    axes[2].set_title("Fused")
    plt.tight_layout()
    plt.show()
