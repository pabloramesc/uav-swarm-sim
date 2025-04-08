"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np
from numpy.typing import ArrayLike

from simulator.agents.drone import Drone
from simulator.environment.limited_regions import (
    LimitedRegion,
    Boundary,
    Obstacle,
    RectangularBoundary,
    PolygonalBoundary,
    CircularObstacle,
    RectangularObstacle,
)


class MultiDroneSimulator:

    def __init__(self, num_drones: int, dt: float = 0.01) -> None:
        self.num_drones = num_drones
        self.dt = dt

        self.time = 0.0
        self.step = 0

        self.drones: list[Drone] = []
        self.drone_ids = np.zeros((num_drones,), dtype=np.int32)
        for i in range(num_drones):
            drone = Drone()
            self.drones.append(drone)
            self.drone_ids[i] = drone.id

        self.drone_states = np.zeros((num_drones, 6))  # px, py, pz, vx, vy, vz
        self.links_matrix = np.full((num_drones, num_drones), False, dtype=bool)
        self.edge_drones_mask = np.full((num_drones,), False, dtype=bool)

        self.boundary: Boundary = None
        self.obstacles: list[Obstacle] = []

    @property
    def drone_positions(self) -> np.ndarray:
        return self.drone_states[:, 0:3]

    @property
    def drone_velocities(self) -> np.ndarray:
        return self.drone_states[:, 3:6]

    @property
    def limited_regions(self) -> list[LimitedRegion]:
        return [self.boundary] + self.obstacles

    def set_rectangular_boundary(
        self, bottom_left: ArrayLike, top_right: ArrayLike
    ) -> None:
        rect = RectangularBoundary(bottom_left, top_right)
        self.boundary = rect
        self._update_limited_regions_info()

    def set_polygonal_boundary(self, vertices: ArrayLike) -> None:
        poly = PolygonalBoundary(vertices)
        self.boundary = poly
        self._update_limited_regions_info()

    def clear_obstacles(self) -> None:
        self.obstacles = []
        self._update_limited_regions_info()

    def add_circular_obstacle(self, center: ArrayLike, radius: float) -> None:
        circ = CircularObstacle(center, radius)
        self.obstacles.append(circ)
        self._update_limited_regions_info()

    def add_rectangular_obstacle(
        self, bottom_left: ArrayLike, top_right: ArrayLike
    ) -> None:
        rect = RectangularObstacle(bottom_left, top_right)
        self.obstacles.append(rect)
        self._update_limited_regions_info()

    def _update_limited_regions_info(self) -> None:
        for drone in self.drones:
            drone.position_control.limited_regions = self.limited_regions

    def _get_drone_states(self) -> None:
        for i, drone in enumerate(self.drones):
            self.drone_states[i, 0:3] = drone.position
            self.drone_states[i, 3:6] = drone.velocity

    def _set_drone_states(self) -> None:
        for i, drone in enumerate(self.drones):
            drone.state[0:3] = self.drone_states[i, 0:3]
            drone.state[3:6] = self.drone_states[i, 3:6]

    def initialize_random_positions(
        self,
        origin: np.ndarray = np.zeros(2),
        space: float = 1.0,
        altitude: float = 0.0,
    ) -> None:
        self.drone_states[:, 0:2] = np.random.normal(
            origin, space, (self.num_drones, 2)
        )
        self.drone_states[:, 2] = altitude
        self.drone_states[:, 3:6] = 0.0
        self._set_drone_states()

    def initialize_grid_positions(
        self,
        origin: np.ndarray = np.zeros(2),
        space: float = 1.0,
        altitude: float = 0.0,
    ):
        self.drone_states[:, 2] = altitude
        self.drone_states[:, 3:6] = 0.0
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
        self.update_visible_neighbors(100.0)
        self.update_drones(dt)
        self.update_links_matrix()
        self.update_edge_drones_mask()

    def update_drones(self, dt: float = None) -> None:
        dt = dt or self.dt
        for drone in self.drones:
            drone.update(dt)
        self.time += dt
        self.step += 1

    def update_visible_neighbors(self, max_distance: float) -> None:
        self._get_drone_states()
        positions = self.drone_states[:, 0:3]
        for drone in self.drones:
            deltas = positions - drone.position
            distances = np.linalg.norm(deltas, axis=1)
            visible_neighbors = (distances < max_distance) & (distances > 0.0)
            drone.set_visible_neighbors(
                self.drone_ids[visible_neighbors], positions[visible_neighbors]
            )

    def update_links_matrix(self) -> None:
        self.links_matrix = np.full(
            (self.num_drones, self.num_drones), False, dtype=bool
        )
        for drone in self.drones:
            neighbor_indexes = drone.visible_neighbors_ids - 1
            links_mask = drone.position_control.links_mask
            drone_links = np.zeros((self.num_drones,), dtype=bool)
            drone_links[neighbor_indexes] = links_mask
            self.links_matrix[drone.id - 1, :] = drone_links

    def update_edge_drones_mask(self) -> None:
        for i, drone in enumerate(self.drones):
            self.edge_drones_mask[i] = drone.position_control.is_edge_robot()
