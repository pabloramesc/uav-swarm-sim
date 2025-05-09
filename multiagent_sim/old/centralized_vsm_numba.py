"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np
from numba import njit, prange


@njit(parallel=True)
def links_matrix(positions: np.ndarray) -> np.ndarray:
    num_drones = positions.shape[0]
    links = np.full((num_drones, num_drones), False, dtype=bool)
    for drone1_id in prange(num_drones):
        drone1_pos = positions[drone1_id, :]
        for drone2_id in list(range(num_drones))[:drone1_id]:
            drone2_pos = positions[drone2_id, :]
            center = (drone1_pos + drone2_pos) / 2
            radius = np.linalg.norm(drone1_pos - drone2_pos) / 2
            distance = 0.0
            for drone3_id in range(num_drones):
                if drone3_id == drone1_id or drone3_id == drone2_id:
                    continue
                drone3_pos = positions[drone3_id]
                distance = np.linalg.norm(drone3_pos - center)
                if distance <= radius:
                    break
            if distance > radius:
                links[drone1_id, drone2_id] = True
                links[drone2_id, drone1_id] = True
    return links


@njit(parallel=True)
def control_force(
    states: np.ndarray,
    links: np.ndarray,
    ln: float = 10.0,
    ks: float = 0.2,
    kd: float = 0.8,
) -> np.ndarray:
    num_drones = states.shape[0]
    forces = np.zeros((num_drones, 2))
    for drone1_id in prange(num_drones):
        drone1_pos = states[drone1_id, 0:2]
        for drone2_id in range(num_drones):
            if drone1_id == drone2_id:
                continue
            if not links[drone1_id, drone2_id]:
                continue
            drone2_pos = states[drone2_id, 0:2]
            delta = drone2_pos - drone1_pos
            distance = np.linalg.norm(delta)
            direction = delta / distance if distance > 0 else np.zeros(2)
            spring_force = ks * (distance - ln) * direction
            forces[drone1_id, :] += spring_force
        drone1_vel = states[drone1_id, 2:4]
        damping_force = -kd * drone1_vel
        forces[drone1_id, :] += damping_force
    return forces


@njit(parallel=True)
def update_states(
    states: np.ndarray, forces: np.ndarray, dt: float = 0.01
) -> np.ndarray:
    num_drones = states.shape[0]
    new_states = np.zeros_like(states)
    for drone_id in prange(num_drones):
        new_states[drone_id, 0:2] = states[drone_id, 0:2] + states[drone_id, 2:4] * dt
        new_states[drone_id, 2:4] = states[drone_id, 2:4] + forces[drone_id, :] * dt
    return new_states
