from .environment import Environment
from .obstacles import (
    Boundary,
    CircularBoundary,
    CircularObstacle,
    Obstacle,
    PolygonalBoundary,
    PolygonalObstacle,
    RectangularBoundary,
    RectangularObstacle,
)
from .placement import grid_positions, random_positions, sample_positions

__all__ = [
    "Boundary",
    "CircularBoundary",
    "CircularObstacle",
    "ElevationMap",
    "Environment",
    "Obstacle",
    "PolygonalBoundary",
    "PolygonalObstacle",
    "RectangularBoundary",
    "RectangularObstacle",
    "grid_positions",
    "random_positions",
    "sample_positions",
]


def __getattr__(name: str):
    """Load optional terrain dependencies only when explicitly requested."""

    if name == "ElevationMap":
        from .elevation_map import ElevationMap

        return ElevationMap
    raise AttributeError(name)
