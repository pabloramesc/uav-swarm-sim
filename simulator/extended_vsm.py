from abc import ABC, abstractmethod
import numpy as np

from obstacles import Obstacle


class EVSM:
    def __init__(
        self, ln: float = 10.0, ks: float = 0.2, kd: float = 0.8, mass: float = 1.0
    ) -> None:
        self.ln = ln
        self.ks = ks
        self.kd = kd

        self.mass = mass
        self.state = np.zeros(4)  # px, py, vx, vy

        self.neighbors = np.zeros((0, 2))  # (px, py)
        self.links = np.zeros((0,), dtype=np.bool)

        self.obstacles: list[Obstacle] = []

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
        force = np.zeros(2)
        force += self.calculate_control_force(
            self.position, self.velocity, self.neighbors[self.links]
        )
        self.update_state(force, dt)
        return self.position  # return the target position

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
        self, position: np.ndarray, velocity: np.ndarray, linked_neighbors: np.ndarray
    ) -> np.ndarray:
        deltas = linked_neighbors - position
        distances = np.linalg.norm(deltas, axis=1)
        directions = np.where(
            distances > 0.0, deltas / distances[:, np.newaxis], np.zeros(2)
        )
        spring_forces = self.ks * (distances - self.ln) * directions
        damping_force += -self.kd * velocity
        return spring_forces + damping_force

    def update_state(self, force: np.ndarray, dt: float = 0.01) -> None:
        self.state[0:2] += self.velocity * dt
        self.state[2:4] += force / self.mass * dt
        
    def calculate_
