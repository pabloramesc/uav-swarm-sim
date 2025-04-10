from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike
from shapely import Point, shortest_line, box, Polygon
from matplotlib import pyplot as plt


class AvoidRegion(ABC):
    def __init__(self, shape: Polygon) -> None:
        self.shape = shape
        self.centroid = np.array(self.shape.centroid.coords[0])

    def is_inside(self, pos: ArrayLike) -> bool:
        return self.shape.contains(Point(pos))

    def distance(self, pos: ArrayLike) -> float:
        return self.shape.boundary.distance(Point(pos))

    def direction(self, pos: ArrayLike) -> np.ndarray:
        closest, distance = self._get_closest_and_distance(pos)
        direction = (closest - pos) / distance if distance > 0.0 else np.zeros(2)
        return direction

    def closest_point(self, pos: ArrayLike) -> np.ndarray:
        closest, _ = self._get_closest_and_distance(pos)
        return np.array(closest)

    def _get_closest_and_distance(self, pos: ArrayLike) -> tuple[np.ndarray, float]:
        line = shortest_line(self.shape.boundary, Point(pos))
        closest = np.array(line.coords[0])
        distance = line.length
        return closest, distance


class Obstacle(AvoidRegion):
    def __init__(self, shape) -> None:
        super().__init__(shape)

    def distance(self, pos: ArrayLike) -> float:
        if self.is_inside(pos):
            return 0.0
        return super().distance(pos)

    def direction(self, pos: ArrayLike) -> np.ndarray:
        direction = super().direction(pos)
        if self.is_inside(pos):
            return -direction
        return direction


class CircularObstacle(Obstacle):
    def __init__(self, center: ArrayLike, radius: float) -> None:
        self.center = np.array(center)
        self.radius = float(radius)
        super().__init__(Point(self.center).buffer(self.radius))

    def is_inside(self, pos: ArrayLike) -> bool:
        delta = self.center - pos
        return np.linalg.norm(delta) < self.radius

    def distance(self, pos: ArrayLike) -> float:
        delta = self.center - pos
        return max(0.0, np.linalg.norm(delta) - self.radius)

    def direction(self, pos: ArrayLike) -> np.ndarray:
        delta = self.center - pos
        norm = np.linalg.norm(delta)
        return delta / norm if norm > 0.0 else np.zeros(2)


class RectangularObstacle(Obstacle):
    def __init__(self, bottom_left: ArrayLike, top_right: ArrayLike) -> None:
        self.bottom_left = np.array(bottom_left)
        self.top_right = np.array(top_right)
        super().__init__(box(*self.bottom_left, *self.top_right))


class PolygonalObstacle(Obstacle):
    def __init__(self, vertices: ArrayLike):
        self.vertices = np.array(vertices)
        super().__init__(Polygon(self.vertices))


class Boundary(AvoidRegion):
    def __init__(self, shape):
        super().__init__(shape)

    def distance(self, pos):
        if not self.is_inside(pos):
            return 0.0
        return super().distance(pos)

    def direction(self, pos):
        direction = super().direction(pos)
        if not self.is_inside(pos):
            return -direction
        return direction


class CircularBoundary(Boundary):
    def __init__(self, center: ArrayLike, radius: float) -> None:
        self.center = np.array(center)
        self.radius = float(radius)
        super().__init__(Point(self.center).buffer(self.radius))

    def is_inside(self, pos: ArrayLike) -> bool:
        delta = self.center - pos
        return np.linalg.norm(delta) < self.radius

    def distance(self, pos: ArrayLike) -> float:
        delta = self.center - pos
        return max(0.0, self.radius - np.linalg.norm(delta))

    def direction(self, pos: ArrayLike) -> np.ndarray:
        delta = pos - self.center
        norm = np.linalg.norm(delta)
        return delta / norm if norm > 0.0 else np.zeros(2)


class RectangularBoundary(Boundary):
    def __init__(self, bottom_left: ArrayLike, top_right: ArrayLike) -> None:
        self.bottom_left = np.array(bottom_left)
        self.top_right = np.array(top_right)
        super().__init__(box(*self.bottom_left, *self.top_right))


class PolygonalBoundary(Boundary):
    def __init__(self, vertices: ArrayLike):
        self.vertices = np.array(vertices)
        super().__init__(Polygon(self.vertices))


def plot_limited_region(
    region: AvoidRegion, x_range=(-20, 20), y_range=(-20, 20), resolution=100
):
    x_dense = np.linspace(*x_range, resolution)
    y_dense = np.linspace(*y_range, resolution)
    X_dense, Y_dense = np.meshgrid(x_dense, y_dense)
    pos_dense = np.column_stack([X_dense.ravel(), Y_dense.ravel()])

    x_sparse = np.linspace(*x_range, 25)
    y_sparse = np.linspace(*y_range, 25)
    X_sparse, Y_sparse = np.meshgrid(x_sparse, y_sparse)
    pos_sparse = np.column_stack([X_sparse.ravel(), Y_sparse.ravel()])

    distances = np.array([region.distance(p) for p in pos_dense]).reshape(X_dense.shape)
    directions = np.array([region.direction(p) for p in pos_sparse]).reshape(
        X_sparse.shape + (2,)
    )

    plt.figure(figsize=(8, 6))
    plt.contourf(X_dense, Y_dense, distances, levels=50, cmap="viridis", alpha=0.75)
    plt.colorbar(label="Distance to limited region")
    plt.quiver(
        X_sparse, Y_sparse, directions[..., 0], directions[..., 1], color="white"
    )
    plt.plot(*region.shape.exterior.xy, color="black")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title(f"Distance and direction to {region.__class__.__name__}")
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    circ_obs = CircularObstacle([0, 0], 10.0)
    plot_limited_region(circ_obs)

    rect_obs = RectangularObstacle([-10, -10], [+10, +10])
    plot_limited_region(rect_obs)

    poly_obs = PolygonalObstacle([[-10, 0], [-10, -5], [10, -5], [5, 10]])
    plot_limited_region(poly_obs)

    circ_bound = CircularBoundary([0, 0], 10.0)
    plot_limited_region(circ_bound)

    rect_bound = RectangularBoundary([-10, -10], [+10, +10])
    plot_limited_region(rect_bound)

    poly_bound = PolygonalBoundary([[-10, 0], [-10, -5], [10, -5], [5, 10]])
    plot_limited_region(poly_bound)
