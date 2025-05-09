"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""
import time

import numpy as np
from numpy.typing import ArrayLike

from simulator.agents.drone import Drone
from simulator.environment.avoid_regions import (
    Region,
    Boundary,
    Obstacle,
    RectangularBoundary,
    PolygonalBoundary,
    CircularObstacle,
    RectangularObstacle,
)

from simulator.virtual_spring_mesh import (
    calculate_links,
    calculate_forces,
    update_states,
)

from multiprocessing import Process, Manager


def drone_process(
    drone_id: int,
    shared_dict: dict,
    dt: float = 0.01,
    initial_state: np.ndarray = np.zeros(6),
    limited_regions: list[Region] = [],
):
    local_time = 0.0
    drone = Drone()
    drone.id = drone_id
    drone.state = initial_state
    drone.position_control.avoid_regions = limited_regions

    num_drones = shared_dict["num_drones"]

    print(f"Drone {drone_id} of {num_drones} initiated")

    while True:
        sim_time = shared_dict["time"]
        if sim_time > local_time:
            local_time += dt
            print(f"Drone {drone_id} local time: {local_time}")

            # update visible neighbors
            neighbor_ids = []
            neighbor_positions = []
            for id in range(num_drones):
                if id == drone.id:
                    continue
                state = shared_dict["drones"][id]["state"]
                position = state[0:3]
                print(f"Drone {drone_id}: found drone {id} at position {position}")
                distance = np.linalg.norm(drone.position - position)
                if distance < 100.0:
                    neighbor_ids.append(id)
                    neighbor_positions.append(position)
            neighbor_ids = np.array(neighbor_ids)
            neighbor_positions = np.array(neighbor_positions)
            drone.set_visible_neighbors(neighbor_ids, neighbor_positions)

            # update drone
            drone.update(dt)

            # update shared dict
            shared_dict["drones"][id]["time"] = local_time
            shared_dict["drones"][id]["state"] = drone.state.tolist()
            links_mask = drone.position_control.links_mask
            shared_dict["drones"][id]["links"] = neighbor_ids[links_mask].tolist()
            is_edge = drone.position_control.is_edge_robot()
            shared_dict["drones"][id]["is_edge"] = is_edge
        time.sleep(0.01)


class MultiDroneSimulator:

    def __init__(self, num_drones: int, dt: float = 0.01) -> None:
        self.num_drones = num_drones
        self.dt = dt

        self.time = 0.0
        self.step = 0

        self.drone_states = np.zeros((num_drones, 6))  # px, py, pz, vx, vy, vz
        self.links_matrix = np.full((num_drones, num_drones), False, dtype=bool)
        self.edge_drones_mask = np.full((num_drones,), False, dtype=bool)

        self.boundary: Boundary = None
        self.obstacles: list[Obstacle] = []

        self.processes: list[Process] = []
        manager = Manager()
        self.shared_dict = manager.dict()
        self.shared_dict["time"] = 0.0
        self.shared_dict["num_drones"] = self.num_drones
        self.shared_dict["drones"] = manager.dict()

        for id in range(self.num_drones):
            self.shared_dict["drones"][id] = manager.dict()
            self.shared_dict["drones"][id]["time"] = 0.0
            self.shared_dict["drones"][id]["state"] = np.zeros(6).tolist()
            self.shared_dict["drones"][id]["links"] = []
            self.shared_dict["drones"][id]["is_edge"] = None

    @property
    def drone_positions(self) -> np.ndarray:
        return self.drone_states[:, 0:3]

    @property
    def drone_velocities(self) -> np.ndarray:
        return self.drone_states[:, 3:6]

    @property
    def limited_regions(self) -> list[Region]:
        return [self.boundary] + self.obstacles

    def set_rectangular_boundary(
        self, bottom_left: ArrayLike, top_right: ArrayLike
    ) -> None:
        rect = RectangularBoundary(bottom_left, top_right)
        self.boundary = rect

    def set_polygonal_boundary(self, vertices: ArrayLike) -> None:
        poly = PolygonalBoundary(vertices)
        self.boundary = poly

    def clear_obstacles(self) -> None:
        self.obstacles = []

    def add_circular_obstacle(self, center: ArrayLike, radius: float) -> None:
        circ = CircularObstacle(center, radius)
        self.obstacles.append(circ)

    def add_rectangular_obstacle(
        self, bottom_left: ArrayLike, top_right: ArrayLike
    ) -> None:
        rect = RectangularObstacle(bottom_left, top_right)
        self.obstacles.append(rect)

    def update(self) -> None:
        self.time += self.dt
        self.shared_dict["time"] = self.time
        self.update_drone_states()
        self.update_links_matrix()
        self.update_edge_drones_mask()

    def update_drone_states(self) -> None:
        for id in range(self.num_drones):
            state = self.shared_dict["drones"][id]["state"]
            self.drone_states[id, :] = state

    def update_links_matrix(self) -> None:
        self.links_matrix = np.full(
            (self.num_drones, self.num_drones), False, dtype=bool
        )
        for id in range(self.num_drones):
            indices = self.shared_dict["drones"][id]["links"]
            mask = np.zeros(self.num_drones, dtype=bool)
            mask[indices] = True
            self.links_matrix[id, :] = mask

    def update_edge_drones_mask(self) -> None:
        for id in range(self.num_drones):
            is_edge = self.shared_dict["drones"][id]["is_edge"]
            self.edge_drones_mask[id] = is_edge

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
                    return

    def launch_simulation(self):
        self.processes = []

        for id in range(self.num_drones):
            initial_state = self.drone_states[id, :].copy()
            p = Process(
                target=drone_process,
                args=(
                    id,
                    self.shared_dict,
                    self.dt,
                    initial_state,
                    self.limited_regions,
                ),
            )
            p.start()
            self.processes.append(p)

    def stop_simulation(self):
        for p in self.processes:
            p.terminate()
        for p in self.processes:
            p.join()
