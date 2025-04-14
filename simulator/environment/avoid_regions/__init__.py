"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from .base import CircularRegion, PolygonalRegion, RectangularRegion, Region
from .boundaries import (
    Boundary,
    CircularBoundary,
    PolygonalBoundary,
    RectangularBoundary,
)
from .obstacles import (
    CircularObstacle,
    Obstacle,
    PolygonalObstacle,
    RectangularObstacle,
)
from .visualization import plot_limited_region
