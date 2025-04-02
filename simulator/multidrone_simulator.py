"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np

from simulator.agents import Drone

from simulator.virtual_spring_mesh import (
    calculate_links,
    calculate_forces,
    update_states,
)


class MultiDroneSimulator:

    def __init__(self, num_drones: int, dt: float = 0.01) -> None:
        self.num_drones = num_drones
        self.dt = dt

        self.time = 0.0
        self.step = 0

        self.drones: list[Drone] = []
        for _ in range(num_drones):
            drone = Drone()
            self.drones.append(drone)

        self.drone_states = np.zeros((num_drones, 4))  # px, py, vx, vy
        self.links_matrix = np.full((num_drones, num_drones), False, dtype=bool)

    def _get_drone_states(self) -> None:
        for i, drone in enumerate(self.drones):
            self.drone_states[i, 0:2] = drone.position
            self.drone_states[i, 2:4] = drone.velocity

    def _set_drone_states(self) -> None:
        for i, drone in enumerate(self.drones):
            drone.position = self.drone_states[i, 0:2]
            drone.velocity = self.drone_states[i, 2:4]

    def initialize_random_positions(
        self, origin: np.ndarray = np.zeros(2), space: float = 1.0
    ) -> None:
        self.drone_states[:, 0:2] = np.random.normal(
            origin, space, (self.num_drones, 2)
        )
        self._set_drone_states()

    def initialize_grid_positions(
        self, origin: np.ndarray = np.zeros(2), space: float = 1.0
    ):
        grid_size = int(np.ceil(np.sqrt(self.num_drones)))
        drone_id = 0
        for row in range(grid_size):
            for col in range(grid_size):
                self.drone_states[drone_id, 0] = origin[0] + space * row
                self.drone_states[drone_id, 1] = origin[1] + space * col
                drone_id += 1
                if drone_id >= self.num_drones:
                    self._set_drone_states()
                    return

    def update(self, dt: float = None) -> None:
        dt = dt or self.dt
        self.time += dt
        self._get_drone_states()
        self.links_matrix = calculate_links(self.drone_states[:, 0:2])
        forces = calculate_forces(self.drone_states, self.links_matrix, ln=100.0)
        self.drone_states = update_states(self.drone_states, forces, dt)
        self._set_drone_states()

    def update_neighbors(self, max_distance: float) -> None:
        self._get_drone_states()
        positions = self.drone_states[:, 0:2]
        for drone in self.drones:
            deltas = positions - drone.position
            distances = np.linalg.norm(deltas, axis=1)
            neighbors = np.where(distances <= max_distance)
            # TODO: complete update neighbors method
