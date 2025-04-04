from abc import ABC, abstractmethod
import numpy as np

from simulator.limited_regions import LimitedRegion
from simulator.math.angles import vector_angle, diff_angle
from shapely import Point, Polygon


class EVSM:
    def __init__(
        self, ln: float = 10.0, ks: float = 0.2, kd: float = 0.8, mass: float = 1.0
    ) -> None:
        self.ln = ln
        self.ks = ks
        self.kd = kd
        self.expl_multiplier = 1.0
        self.d_min_obstacle = 1.0

        self.mass = mass
        self.state = np.zeros(4)  # px, py, vx, vy

        self.neighbors = np.zeros((0, 2))  # (px, py)
        self.links = np.zeros((0,), dtype=np.bool)

        self.limited_regions: list[LimitedRegion] = []

        self.edge_robot = False
        self.swap_angles = (None, None)

    @property
    def position(self) -> np.ndarray:
        return self.state[0:2]

    @property
    def velocity(self) -> np.ndarray:
        return self.state[2:4]

    def update(
        self, state: np.ndarray, neighbors: np.ndarray, dt: float = 0.01
    ) -> np.ndarray:
        self.state = state
        self.neighbors = neighbors
        return self.update_from_internal(dt)  # return the target position

    def update_from_internal(self, dt: float = 0.01) -> np.ndarray:
        self.links = self.calculate_links(self.position, self.neighbors)
        self.edge_robot, angle1, angle2 = self.is_edge_robot(
            self.position, self.neighbors[self.links]
        )
        self.swap_angles = (angle1, angle2) if self.edge_robot else (None, None)
        force = self.calculate_control_force(
            self.position, self.neighbors[self.links]
        ) + self.calculate_damping_force(self.velocity)
        self.update_state(force, dt)
        return self.position  # return the target position from the updated state

    def update_state(self, force: np.ndarray, dt: float = 0.01) -> None:
        self.state[0:2] += self.velocity * dt
        self.state[2:4] += force / self.mass * dt

    def calculate_links(
        self, position: np.ndarray, neighbors: np.ndarray
    ) -> np.ndarray:
        num_neighbors = neighbors.shape[0]
        links = np.zeros((num_neighbors,), dtype=np.bool)
        for neighbor1_idx in range(num_neighbors):
            neighbor1_pos = neighbors[neighbor1_idx, :]
            center = (position + neighbor1_pos) / 2
            radius = np.linalg.norm(position - neighbor1_pos) / 2
            distance = 0.0
            for neighbor2_idx in range(num_neighbors):
                if neighbor2_idx == neighbor1_idx:
                    continue
                neighbor2_pos = neighbors[neighbor2_idx, :]
                distance = np.linalg.norm(neighbor2_pos - center)
                if distance <= radius:
                    break
            if distance > radius:
                links[neighbor1_idx] = True
        return links

    def calculate_control_force(
        self, position: np.ndarray, linked_neighbors: np.ndarray
    ) -> np.ndarray:
        deltas = linked_neighbors - position
        distances = np.linalg.norm(deltas, axis=1)
        directions = np.where(
            distances > 0.0, deltas / distances[:, np.newaxis], np.zeros(2)
        )
        spring_forces = self.ks * (distances - self.ln) * directions
        return spring_forces

    def calculate_damping_force(self, velocity: np.ndarray) -> np.ndarray:
        damping_force += -self.kd * velocity
        return damping_force

    def calculate_exploration_force(
        self,
        position: np.ndarray,
        limited_regions: list[LimitedRegion],
        swap_angles: tuple[float, float],
    ) -> np.ndarray:
        num_regions = len(limited_regions)
        distances = np.zeros(num_regions)
        directions = np.zeros((num_regions, 2))
        for i, region in enumerate(limited_regions):
            # Check if the obstacle is between the swap angles
            direction = region.direction(position)
            # TODO: Check if the direction is between the swap angles
            distances[i] = region.distance(position)
        weights = distances / np.linalg.norm(distances)
        expl_forces = self.expl_multiplier * weights * self.ks * distances * directions
        return expl_forces

    def is_edge_robot(
        self, position: np.ndarray, linked_neighbors: np.ndarray
    ) -> tuple[bool, float, float]:
        num_neighbors = linked_neighbors.shape[0]
        for i in range(num_neighbors):
            for j in range(i + 1, num_neighbors):
                v1 = linked_neighbors[i] - position
                v2 = linked_neighbors[j] - position
                angle1 = vector_angle(v1)
                angle2 = vector_angle(v2)

                # Check if swap angle is >= 90 degrees
                swap_angle = diff_angle(angle1, angle2)
                if np.abs(swap_angle) < np.pi / 2:
                    continue

                # Check if there are neighbors inside the triangle formed by the two vectors
                triangle = Polygon([position, linked_neighbors[i], linked_neighbors[j]])
                for k in range(num_neighbors):
                    if k == i or k == j:
                        continue
                    if triangle.contains(Point(linked_neighbors[k])):
                        continue

                return True, angle1, angle2
        return False, None, None

    def is_near_obstacle(self, position: np.ndarray) -> bool:
        for region in self.limited_regions:
            if region.distance(position) < self.d_min_obstacle:
                return True
        return False
