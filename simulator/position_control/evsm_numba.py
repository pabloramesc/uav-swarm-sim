import numpy as np
from numba import njit, prange

from simulator.math.angles import vector_angle, diff_angle_2pi


@njit(parallel=True)
def calculate_links(position: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    num_neighbors = neighbors.shape[0]
    links = np.zeros((num_neighbors,), dtype=np.bool)
    for i in prange(num_neighbors):
        neighbor1 = neighbors[i, :]
        center = (position + neighbor1) / 2
        radius = np.linalg.norm(position - neighbor1) / 2
        distance = np.inf
        for j in range(num_neighbors):
            if j == i:
                continue
            neighbor2 = neighbors[j, :]
            distance = np.linalg.norm(neighbor2 - center)
            if distance <= radius:
                break
        if distance > radius:
            links[i] = True
    return links


@njit
def calculate_control_force(
    position: np.ndarray, linked_neighbors: np.ndarray, ln: float = 1.0, ks: float = 1.0
) -> np.ndarray:
    deltas = linked_neighbors - position
    # distances = np.linalg.norm(deltas, axis=1)[:, np.newaxis]
    distances = np.sqrt(np.sum(deltas**2, axis=1))[:, np.newaxis]
    directions = np.where(distances > 0.0, deltas / distances, np.zeros_like(deltas))
    spring_forces = (distances - ln) * directions
    control_force = ks * np.sum(spring_forces, axis=0)
    return control_force


@njit
def calculate_damping_force(velocity: np.ndarray, kd: float = 1.0) -> np.ndarray:
    damping_force = -kd * velocity * np.linalg.norm(velocity)
    return damping_force


@njit
def calculate_sweep_angle(
    position: np.ndarray, neighbors: np.ndarray
) -> tuple[float, float]:
    num_neighbors = neighbors.shape[0]
    if num_neighbors == 0:
        return (0.0, 0.0)
    vectors = neighbors - position
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    angles.sort()
    for i in range(num_neighbors):
        angle1 = angles[i - 1]
        angle2 = angles[i]
        sweep_angle = (angle2 - angle1) % (2 * np.pi)
        if sweep_angle >= np.pi / 2:
            return (angle1, angle2)
    return (np.nan, np.nan)
