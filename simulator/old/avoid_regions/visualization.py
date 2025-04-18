"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np
from matplotlib import pyplot as plt

from .base import (
    Region,
    CircularRegion,
    PolygonalRegion,
    RectangularRegion,
)
from .boundaries import CircularBoundary, PolygonalBoundary, RectangularBoundary
from .obstacles import CircularObstacle, PolygonalObstacle, RectangularObstacle


def plot_limited_region(
    region: Region, x_range=(-20, 20), y_range=(-20, 20), resolution=100
):
    """
    Visualizes the distance and direction to a given region in the simulation environment.

    This function generates a contour plot showing the distance to the region and overlays
    a quiver plot to visualize the direction vectors pointing toward or away from the region.

    Parameters
    ----------
    region : AvoidRegion
        The region to visualize (e.g., an obstacle or boundary).
    x_range : tuple[float, float], optional
        The range of x-coordinates to visualize (default is (-20, 20)).
    y_range : tuple[float, float], optional
        The range of y-coordinates to visualize (default is (-20, 20)).
    resolution : int, optional
        The resolution of the dense grid for the contour plot (default is 100).

    Returns
    -------
    None
        Displays the plot directly using matplotlib.
    """
    # Generate dense grid for contour plot
    x_dense = np.linspace(*x_range, resolution)
    y_dense = np.linspace(*y_range, resolution)
    X_dense, Y_dense = np.meshgrid(x_dense, y_dense)
    pos_dense = np.column_stack([X_dense.ravel(), Y_dense.ravel()])

    # Generate sparse grid for quiver plot
    x_sparse = np.linspace(*x_range, 25)
    y_sparse = np.linspace(*y_range, 25)
    X_sparse, Y_sparse = np.meshgrid(x_sparse, y_sparse)
    pos_sparse = np.column_stack([X_sparse.ravel(), Y_sparse.ravel()])

    # Calculate distances and directions
    distances = np.array([region.distance(p) for p in pos_dense]).reshape(X_dense.shape)
    directions = np.array([region.direction(p) for p in pos_sparse]).reshape(
        X_sparse.shape + (2,)
    )

    # Plot distance contour
    plt.figure(figsize=(8, 6))
    plt.contourf(X_dense, Y_dense, distances, levels=50, cmap="viridis", alpha=0.75)
    plt.colorbar(label="Distance to limited region")

    # Plot direction vectors
    plt.quiver(
        X_sparse, Y_sparse, directions[..., 0], directions[..., 1], color="white"
    )

    # Plot the region's boundary
    plt.plot(*region.shape.exterior.xy, color="black")

    # Configure plot labels and title
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title(f"Distance and direction to {region.__class__.__name__}")
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":

    """TEST AVOID REGIONS"""
    # Create a circular avoid region
    circ_obs = CircularRegion([0, 0], 10.0, quad_segs=16)
    plot_limited_region(circ_obs)

    # Create a rectangular avoid region
    rect_obs = RectangularRegion([-10, -10], [+10, +10])
    plot_limited_region(rect_obs)

    # Create a polygonal avoid region
    poly_obs = PolygonalRegion([[-10, 0], [-10, -5], [10, -5], [5, 10]])
    plot_limited_region(poly_obs)

    """TEST OBSTACLES"""
    # Create a circular obstacle
    circ_obs = CircularObstacle([0, 0], 10.0, quad_segs=16)
    plot_limited_region(circ_obs)

    # Create a rectangular obstacle
    rect_obs = RectangularObstacle([-10, -10], [+10, +10])
    plot_limited_region(rect_obs)

    # Create a polygonal obstacle
    poly_obs = PolygonalObstacle([[-10, 0], [-10, -5], [10, -5], [5, 10]])
    plot_limited_region(poly_obs)

    """TEST BOUNDARIES"""
    # Create a circular boundary
    circ_bound = CircularBoundary([0, 0], 10.0, quad_segs=16)
    plot_limited_region(circ_bound)

    # Create a rectangular boundary
    rect_bound = RectangularBoundary([-10, -10], [+10, +10])
    plot_limited_region(rect_bound)

    # Create a polygonal boundary
    poly_bound = PolygonalBoundary([[-10, 0], [-10, -5], [10, -5], [5, 10]])
    plot_limited_region(poly_bound)
