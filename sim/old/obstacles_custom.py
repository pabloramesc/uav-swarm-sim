from abc import ABC, abstractmethod

import numpy as np


class Obstacle(ABC):
    @abstractmethod
    def is_inside(self, pos: np.ndarray) -> bool:
        pass

    @abstractmethod
    def distance_to(self, pos: np.ndarray) -> float:
        pass

    @abstractmethod
    def direction_to(self, pos: np.ndarray) -> np.ndarray:
        pass


class CircularObstacle(Obstacle):
    def __init__(self, center: np.ndarray, radius: float):
        self.center = np.array(center)
        self.radius = radius

    def is_inside(self, pos: np.ndarray) -> bool:
        delta = self.center - pos
        return np.linalg.norm(delta) < self.radius

    def distance_to(self, pos: np.ndarray) -> float:
        delta = self.center - pos
        return max(0.0, np.linalg.norm(delta) - self.radius)

    def direction_to(self, pos: np.ndarray) -> np.ndarray:
        delta = self.center - pos
        norm = np.linalg.norm(delta)
        return delta / norm if norm > 0.0 else np.zeros(2)


class RectangularObstacle(Obstacle):
    def __init__(self, bottom_left: np.ndarray, top_right: np.ndarray):
        self.bottom_left = np.array(bottom_left)
        self.top_right = np.array(top_right)
        self.width, self.height = np.abs(self.bottom_left - self.top_right)
        self.center = (self.bottom_left + self.top_right) / 2

    def is_inside(self, pos: np.ndarray):
        return (self.bottom_left[0] <= pos[0] <= self.top_right[0]) and (
            self.bottom_left[1] <= pos[1] <= self.top_right[1]
        )

    def distance_to(self, pos: np.ndarray) -> float:
        closest = np.clip(pos, self.bottom_left, self.top_right)
        return np.linalg.norm(pos - closest)

    def direction_to(self, pos: np.ndarray) -> np.ndarray:
        closest = np.clip(pos, self.bottom_left, self.top_right)
        delta = closest - pos
        norm = np.linalg.norm(delta)
        if norm > 0.0:
            return delta / norm
        delta = self.center - pos
        norm = np.linalg.norm(delta)
        return delta / norm if norm > 0.0 else np.zeros(2)


class PolygonalObstacle(Obstacle):
    def __init__(self, vertices: np.ndarray):
        self.vertices = np.array(vertices)
        self.num_vertices = self.vertices.shape[0]
        if self.num_vertices < 3:
            raise ValueError("Polygon need at least 3 vertices")
        self.center = np.mean(self.vertices, axis=0)

    def is_inside(self, pos: np.ndarray) -> bool:
        x, y = pos
        inside = False
        for i in range(self.num_vertices):
            x1, y1 = self.vertices[i]
            x2, y2 = self.vertices[(i + 1) % self.num_vertices]
            if y > min(y1, y2):
                if y <= max(y1, y2):
                    if x <= max(x1, x2):
                        if y1 != y2:
                            xinters = (y - y1) * (x2 - x1) / (y2 - y1) + x1
                        if x1 == x2 or x <= xinters:
                            inside = not inside
        return inside

    def distance_to(self, pos: np.ndarray) -> float:
        if self.is_inside(pos):
            return 0.0
        return self._get_closest_distance_segment(pos)[0]

    def direction_to(self, pos: np.ndarray) -> np.ndarray:
        if self.is_inside(pos):
            direction = self.center - pos
            dir_norm = np.linalg.norm(direction)
            return direction / dir_norm if dir_norm > 0.0 else np.zeros(2)
        closest_seg = self._get_closest_distance_segment(pos)[1]
        seg_vec = closest_seg[1] - closest_seg[0]
        seg_len = np.linalg.norm(seg_vec)
        seg_normal = np.array([seg_vec[1], -seg_vec[0]]) / seg_len
        return -seg_normal

    def _distance_to_segment(
        self, pos: np.ndarray, p1: np.ndarray, p2: np.ndarray
    ) -> float:
        seg_vec = p2 - p1
        seg_len = np.linalg.norm(seg_vec)
        pos_vec = pos - p1
        proj_dist = np.dot(pos_vec, seg_vec) / seg_len
        # if projection is out of segment, return distance to closest vertex
        if proj_dist < 0.0:
            return np.linalg.norm(pos - p1)
        if proj_dist > seg_len:
            return np.linalg.norm(pos - p2)
        # if projection is inside the segment, return distance to projected point
        proj_point = p1 + proj_dist * (seg_vec / seg_len)
        return np.linalg.norm(pos - proj_point)

    def _get_closest_distance_segment(
        self, pos: np.ndarray
    ) -> tuple[float, tuple[np.ndarray, np.ndarray]]:
        min_dist = float("inf")
        closest_seg = None
        for i in range(self.num_vertices):
            p1 = self.vertices[i]
            p2 = self.vertices[(i + 1) % self.num_vertices]
            dist = self._distance_to_segment(pos, p1, p2)
            if dist < min_dist:
                min_dist = dist
                closest_seg = (p1, p2)
        return min_dist, closest_seg


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    obstacles = [
        CircularObstacle([0, 0], 10.0),
        RectangularObstacle([-10, -10], [+10, +10]),
        PolygonalObstacle([[-10, 0], [-10, -5], [10, -5], [5, 10]]),
    ]

    x_dense = np.linspace(-20, 20, 100)
    y_dense = np.linspace(-20, 20, 100)
    X_dense, Y_dense = np.meshgrid(x_dense, y_dense)
    pos_dense = np.column_stack([X_dense.ravel(), Y_dense.ravel()])

    x_sparse = np.linspace(-20, 20, 20)
    y_sparse = np.linspace(-20, 20, 20)
    X_sparse, Y_sparse = np.meshgrid(x_sparse, y_sparse)
    pos_sparse = np.column_stack([X_sparse.ravel(), Y_sparse.ravel()])

    for i, obs in enumerate(obstacles):
        distances = np.array([obs.distance_to(p) for p in pos_dense]).reshape(
            X_dense.shape
        )
        directions = np.array([obs.direction_to(p) for p in pos_sparse]).reshape(
            X_sparse.shape + (2,)
        )

        plt.figure(figsize=(8, 6))
        plt.contourf(X_dense, Y_dense, distances, levels=50, cmap="viridis", alpha=0.75)
        plt.colorbar(label="Distance to Obstacle")
        plt.quiver(
            X_sparse, Y_sparse, directions[..., 0], directions[..., 1], color="white"
        )

        if isinstance(obs, CircularObstacle):
            plt.gca().add_patch(
                plt.Circle(obs.center, obs.radius, color="black", fill=False)
            )
        elif isinstance(obs, RectangularObstacle):
            rect = plt.Rectangle(
                obs.bottom_left, obs.width, obs.height, color="black", fill=False
            )
            plt.gca().add_patch(rect)
        elif isinstance(obs, PolygonalObstacle):
            plt.gca().add_patch(plt.Polygon(obs.vertices, color="black", fill=False))

        plt.xlabel("X position")
        plt.ylabel("Y position")
        plt.title(f"Distance and Direction to {obs.__class__.__name__}")
        plt.axis("equal")
        plt.show()
