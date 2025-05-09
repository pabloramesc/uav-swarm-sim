from abc import ABC, abstractmethod

import numpy as np
from shapely import Point, shortest_line, box, Polygon
from matplotlib import pyplot as plt


class Obstacle(ABC):
    def __init__(self, shape: Polygon) -> None:
        self.shape = shape
        self.centroid = np.array(self.shape.centroid.coords[0])

    def is_inside(self, pos: np.ndarray) -> bool:
        return self.shape.contains(Point(pos))

    def distance(self, pos: np.ndarray) -> float:
        return self.shape.distance(Point(pos))

    def direction(self, pos: np.ndarray) -> np.ndarray:
        closest, distance = self.get_closest_distance(pos)
        direction = (closest - pos) / distance if distance > 0.0 else np.zeros(2)
        if self.is_inside(pos):
            return -direction
        return direction

    def closest_point(self, pos: np.ndarray) -> np.ndarray:
        closest = shortest_line(self.shape.boundary, Point(pos)).coords[0]
        return np.array(closest)

    def get_closest_distance(self, pos: np.ndarray) -> tuple[np.ndarray, float]:
        line = shortest_line(self.shape.boundary, Point(pos))
        closest = np.array(line.coords[0])
        distance = line.length
        return closest, distance


class CircularObstacle(Obstacle):
    def __init__(self, center: np.ndarray, radius: float) -> None:
        self.center = np.array(center)
        self.radius = float(radius)
        super().__init__(Point(self.center).buffer(self.radius))

    def is_inside(self, pos: np.ndarray) -> bool:
        delta = self.center - pos
        return np.linalg.norm(delta) < self.radius

    def distance(self, pos: np.ndarray) -> float:
        delta = self.center - pos
        return max(0.0, np.linalg.norm(delta) - self.radius)

    def direction(self, pos: np.ndarray) -> np.ndarray:
        delta = self.center - pos
        norm = np.linalg.norm(delta)
        return delta / norm if norm > 0.0 else np.zeros(2)


class RectangularObstacle(Obstacle):
    def __init__(self, bottom_left: np.ndarray, top_right: np.ndarray) -> None:
        self.bottom_left = np.array(bottom_left)
        self.top_right = np.array(top_right)
        super().__init__(box(*self.bottom_left, *self.top_right))


class PolygonalObstacle(Obstacle):
    def __init__(self, vertices: np.ndarray):
        self.vertices = np.array(vertices)
        super().__init__(Polygon(self.vertices))


def plot_obstacle(
    obstacle: Obstacle, x_range=(-20, 20), y_range=(-20, 20), resolution=100
):
    x_dense = np.linspace(*x_range, resolution)
    y_dense = np.linspace(*y_range, resolution)
    X_dense, Y_dense = np.meshgrid(x_dense, y_dense)
    pos_dense = np.column_stack([X_dense.ravel(), Y_dense.ravel()])

    x_sparse = np.linspace(*x_range, 25)
    y_sparse = np.linspace(*y_range, 25)
    X_sparse, Y_sparse = np.meshgrid(x_sparse, y_sparse)
    pos_sparse = np.column_stack([X_sparse.ravel(), Y_sparse.ravel()])

    distances = np.array([obstacle.distance(p) for p in pos_dense]).reshape(
        X_dense.shape
    )
    directions = np.array([obstacle.direction(p) for p in pos_sparse]).reshape(
        X_sparse.shape + (2,)
    )

    plt.figure(figsize=(8, 6))
    plt.contourf(X_dense, Y_dense, distances, levels=50, cmap="viridis", alpha=0.75)
    plt.colorbar(label="Distance to Obstacle")
    plt.quiver(
        X_sparse, Y_sparse, directions[..., 0], directions[..., 1], color="white"
    )

    if isinstance(obstacle, CircularObstacle):
        patch = plt.Circle(obstacle.center, obstacle.radius, color="black", fill=False)
    elif isinstance(obstacle, RectangularObstacle):
        patch = plt.Rectangle(
            obstacle.bottom_left,
            *(obstacle.top_right - obstacle.bottom_left),
            color="black",
            fill=False,
        )
    elif isinstance(obstacle, PolygonalObstacle):
        patch = plt.Polygon(obstacle.vertices, color="black", fill=False)
    else:
        patch = None

    if patch is not None:
        plt.gca().add_patch(patch)

    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title(f"Distance and Direction to {obstacle.__class__.__name__}")
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    circ_obs = CircularObstacle([0, 0], 10.0)
    plot_obstacle(circ_obs)

    rect_obs = RectangularObstacle([-10, -10], [+10, +10])
    plot_obstacle(rect_obs)

    poly_obs = PolygonalObstacle([[-10, 0], [-10, -5], [10, -5], [5, 10]])
    plot_obstacle(poly_obs)
