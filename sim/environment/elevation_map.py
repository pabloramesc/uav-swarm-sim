"""Digital elevation map loading and optional satellite imagery."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple, cast

import numpy as np
import rasterio
from rasterio.coords import BoundingBox

logger = logging.getLogger(__name__)


class Bounds(NamedTuple):
    left: float
    bottom: float
    right: float
    top: float


class ElevationMap:
    """Load a GeoTIFF and prepare images used by terrain viewers.

    Satellite tiles are opt-in because fetching them performs network I/O.
    Without tiles, ``fused_img`` is a terrain-colored elevation image.
    """

    def __init__(self, dem_path: str | Path, *, fetch_satellite: bool = False):
        self.dem_path = Path(dem_path)
        self.load_dem()

        self.elevation_img = self.generate_elevation_image()
        self.satellite_img: np.ndarray | None = None
        if fetch_satellite:
            self.satellite_img = self.generate_satellite_image()
        self.fused_img = self.generate_fused_image()

    @property
    def min_elevation(self) -> float:
        return float(self.elevation_data.min())

    @property
    def max_elevation(self) -> float:
        return float(self.elevation_data.max())

    def load_dem(self) -> None:
        """Load elevation values and geographic metadata from the GeoTIFF."""

        with rasterio.open(self.dem_path) as dem:
            if dem.crs is None or dem.crs.to_epsg() != 4326:
                raise ValueError(
                    "Elevation maps must use the EPSG:4326 coordinate system."
                )
            bounds = cast(BoundingBox, dem.bounds)
            self.bounds = Bounds(*bounds)
            self.resolution = cast(tuple[float, float], dem.res)
            self.elevation_data = np.asarray(dem.read(1))

    def get_elevation(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Return elevations for latitude/longitude pairs.

        Coordinates outside the DEM return zero, matching the historical API.
        """

        latitudes = np.atleast_1d(np.asarray(lat, dtype=float))
        longitudes = np.atleast_1d(np.asarray(lon, dtype=float))
        if latitudes.shape != longitudes.shape:
            raise ValueError("Latitude and longitude arrays must have the same shape.")

        in_bounds = (
            (self.bounds.left <= longitudes)
            & (longitudes <= self.bounds.right)
            & (self.bounds.bottom <= latitudes)
            & (latitudes <= self.bounds.top)
        )

        elevations = np.zeros_like(latitudes, dtype=float)
        valid_indices = np.flatnonzero(in_bounds)
        if valid_indices.size == 0:
            return elevations

        valid_latitudes = latitudes[valid_indices]
        valid_longitudes = longitudes[valid_indices]
        columns = ((valid_longitudes - self.bounds.left) / self.resolution[0]).astype(
            int
        )
        rows = ((self.bounds.top - valid_latitudes) / self.resolution[1]).astype(int)

        columns = np.clip(columns, 0, self.elevation_data.shape[1] - 1)
        rows = np.clip(rows, 0, self.elevation_data.shape[0] - 1)
        elevations[valid_indices] = self.elevation_data[rows, columns]
        return elevations

    def generate_elevation_image(
        self, output_resolution: tuple[int, int] = (1_000, 1_000)
    ) -> np.ndarray:
        """Interpolate the DEM onto a normalized image grid."""

        longitude_grid, latitude_grid = self._generate_grid(output_resolution)
        elevations = self.get_elevation(latitude_grid.ravel(), longitude_grid.ravel())
        elevation_grid = elevations.reshape(output_resolution[1], output_resolution[0])
        return self._normalize_elevation(elevation_grid)

    def generate_satellite_image(self) -> np.ndarray:
        """Fetch and crop satellite tiles covering the DEM bounds."""

        import contextily as ctx
        from pyproj import Transformer

        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        left, bottom = transformer.transform(self.bounds.left, self.bounds.bottom)
        right, top = transformer.transform(self.bounds.right, self.bounds.top)
        projected_bounds = (left, bottom, right, top)
        logger.debug("DEM bounds (WGS84): %s", self.bounds)
        logger.debug("DEM bounds (EPSG:3857): %s", projected_bounds)

        try:
            image, extent = ctx.bounds2img(
                *projected_bounds,
                zoom=14,
                source=ctx.providers.Esri.WorldImagery,
            )
        except Exception as exc:
            raise RuntimeError("Could not download satellite tiles.") from exc

        x_min, x_max, y_min, y_max = extent
        height, width, _ = image.shape
        pixel_size_x = (x_max - x_min) / width
        pixel_size_y = (y_max - y_min) / height

        left_px = int((projected_bounds[0] - x_min) / pixel_size_x)
        right_px = int((projected_bounds[2] - x_min) / pixel_size_x)
        top_px = int((y_max - projected_bounds[3]) / pixel_size_y)
        bottom_px = int((y_max - projected_bounds[1]) / pixel_size_y)
        if left_px < 0 or right_px > width or top_px < 0 or bottom_px > height:
            raise ValueError("Satellite crop falls outside the downloaded image.")

        return image[top_px:bottom_px, left_px:right_px]

    def generate_fused_image(
        self,
        output_resolution: tuple[int, int] = (1_000, 1_000),
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Blend elevation colors with loaded satellite tiles."""

        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1.")

        from matplotlib import colormaps

        elevation = self.generate_elevation_image(output_resolution)
        elevation_rgb = colormaps["terrain"](elevation)[..., :3]
        elevation_uint8 = (elevation_rgb * 255).astype(np.uint8)
        if self.satellite_img is None:
            return elevation_uint8

        from PIL import Image

        satellite = np.array(
            Image.fromarray(self.satellite_img).resize(output_resolution)
        )[..., :3]
        return ((1.0 - alpha) * satellite + alpha * elevation_uint8).astype(np.uint8)

    def _generate_grid(
        self, output_resolution: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        width, height = output_resolution
        if width <= 0 or height <= 0:
            raise ValueError("output_resolution dimensions must be positive.")
        longitude = np.linspace(self.bounds.left, self.bounds.right, width)
        latitude = np.linspace(self.bounds.top, self.bounds.bottom, height)
        return np.meshgrid(longitude, latitude)

    def _normalize_elevation(self, elevation: np.ndarray) -> np.ndarray:
        span = self.max_elevation - self.min_elevation
        if span == 0.0:
            return np.zeros_like(elevation, dtype=float)
        return (elevation - self.min_elevation) / span
