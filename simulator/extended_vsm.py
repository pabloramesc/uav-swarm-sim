from abc import ABC, abstractmethod
import numpy as np
from numba import njit, prange


class PositioningAlgorithm:
    def __init__(self):
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.neighbors_positions: np.ndarray = None
        self.obstacles = None

    def set_data(self, position: np.ndarray, neighbors_positions: np.ndarray) -> None:
        self.position = position
        self.neighbors_positions = neighbors_positions

    @abstractmethod
    def calculate_target_position(self) -> np.ndarray:
        pass


class EVSM(PositioningAlgorithm):
    def __init__(self):
        super().__init__()
        
        self.mass = 1.0
        self.osbtacle_min_distance = 10.0
        self.spring_natural_length = 100.0
        self.spring_constant = 0.2
        self.damping_coeff = 0.8

        self.links: list[bool] = []
        self.force = np.zeros(2)
        
    def update_state(self, dt: float):
        self.position += self.velocity * dt
        self.velocity += self.force / self.mass * dt

    def calculate_target_position(self) -> np.ndarray:
        self.links = calculate_links(self.position, self.neighbors_positions)
        self.force = calculate_control_force(
            self.position,
            self.velocity,
            self.neighbors_positions,
            self.links,
            self.spring_natural_length,
            self.spring_constant,
            self.damping_coeff,
        )
        self.update_state()
        return self.position


@njit(parallel=True)
def calculate_links(
    position: np.ndarray, neighbors_positions: np.ndarray
) -> list[bool]:
    num_neighbors = neighbors_positions.shape[0]
    links = [False] * num_neighbors
    for neighbor1_idx in prange(num_neighbors):
        neighbor1_pos = neighbors_positions[neighbor1_idx, :]
        center = (position + neighbor1_pos) / 2
        radius = np.linalg.norm(position - neighbor1_pos) / 2
        distance = 0.0
        for neighbor2_idx in range(num_neighbors):
            neighbor2_pos = neighbors_positions[neighbor2_idx, :]
            distance = np.linalg.norm(neighbor2_pos - center)
            if distance <= radius:
                break
        if distance > radius:
            links[neighbor1_idx] = True
    return links


@njit(parallel=True)
def calculate_control_force(
    position: np.ndarray,
    velocity: np.ndarray,
    neighbors_positions: np.ndarray,
    links: list[bool],
    ln: float = 10.0,
    ks: float = 0.2,
    kd: float = 0.8,
) -> np.ndarray:
    num_neighbors = neighbors_positions.shape[0]
    force = np.zeros(2)
    for neighbor_idx in prange(num_neighbors):
        if not links[neighbor_idx]:
            continue
        neighbor_pos = neighbors_positions[neighbor_idx, :]
        delta = neighbor_pos - position
        distance = np.linalg.norm(delta)
        direction = delta / distance if distance > 0 else np.zeros(2)
        spring_force = ks * (distance - ln) * direction
        force += spring_force
    damping_force = -kd * velocity
    force += damping_force
    return force
