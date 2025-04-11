"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np
from numpy.typing import ArrayLike

from simulator.agents import Drone
from simulator.environment import (
    Environment,
    RectangularBoundary,
    PolygonalBoundary,
    CircularObstacle,
    RectangularObstacle,
)
from simulator.swarming import EVSMConfig, EVSMPositionController


class MultiDroneSimulator:
    """
    Simulates a swarm of drones in a 3D environment.

    This class manages the initialization, state updates, and interactions of multiple drones
    in a simulation environment. It supports features such as obstacle avoidance, neighbor detection,
    and swarm behavior using position controllers.

    Attributes
    ----------
    num_drones : int
        Number of drones in the simulation.
    dt : float
        Time step for the simulation in seconds.
    time : float
        Current simulation time in seconds.
    step : int
        Current simulation step.
    environment : Environment
        The simulation environment containing boundaries and obstacles.
    drones : list[Drone]
        List of Drone objects in the simulation.
    drone_ids : np.ndarray
        Array of unique IDs for each drone.
    drone_states : np.ndarray
        Array of drone states, where each state is [px, py, pz, vx, vy, vz].
    links_matrix : np.ndarray
        Boolean adjacency matrix representing links between drones.
    edge_drones_mask : np.ndarray
        Boolean array indicating which drones are at the edge of the swarm.
    """

    def __init__(self, num_drones: int, dt: float = 0.01) -> None:
        """
        Initializes the MultiDroneSimulator.

        Parameters
        ----------
        num_drones : int
            Number of drones in the simulation.
        dt : float, optional
            Time step for the simulation in seconds (default is 0.01).
        """
        self.num_drones = num_drones
        self.dt = dt

        self.time = 0.0
        self.step = 0

        self.environment = Environment()

        self.drones: list[Drone] = []
        self.drone_ids = np.zeros((num_drones,), dtype=np.int32)
        for i in range(num_drones):
            config = EVSMConfig()
            controller = EVSMPositionController(config, env=self.environment)
            drone = Drone(
                id=i + 1, env=self.environment, position_controller=controller
            )
            self.drones.append(drone)
            self.drone_ids[i] = drone.id

        self.drone_states = np.zeros((num_drones, 6))  # px, py, pz, vx, vy, vz
        self.links_matrix = np.full((num_drones, num_drones), False, dtype=bool)
        self.edge_drones_mask = np.full((num_drones,), False, dtype=bool)

    @property
    def drone_positions(self) -> np.ndarray:
        """
        A (N, 3) shape array with drone positions [px, py, pz] in meters,
        where N is the number of drones.
        """
        return self.drone_states[:, 0:3]

    @property
    def drone_velocities(self) -> np.ndarray:
        """
        A (N, 3) shape array with drone velocities [vx, vy, vz] in m/s,
        where N is the number of drones.
        """
        return self.drone_states[:, 3:6]

    def set_rectangular_boundary(
        self, bottom_left: ArrayLike, top_right: ArrayLike
    ) -> None:
        """
        Sets a rectangular boundary for the simulation environment.

        Parameters
        ----------
        bottom_left : ArrayLike
            Coordinates of the bottom-left corner of the boundary [x, y].
        top_right : ArrayLike
            Coordinates of the top-right corner of the boundary [x, y].
        """
        rect = RectangularBoundary(bottom_left, top_right)
        self.environment.set_boundary(rect)

    def set_polygonal_boundary(self, vertices: ArrayLike) -> None:
        """
        Sets a polygonal boundary for the simulation environment.

        Parameters
        ----------
        vertices : ArrayLike
            List of vertices defining the polygonal boundary.
        """
        poly = PolygonalBoundary(vertices)
        self.environment.set_boundary(poly)

    def add_circular_obstacle(self, center: ArrayLike, radius: float) -> None:
        """
        Adds a circular obstacle to the simulation environment.

        Parameters
        ----------
        center : ArrayLike
            Coordinates of the center of the obstacle [x, y].
        radius : float
            Radius of the circular obstacle.
        """
        circ = CircularObstacle(center, radius)
        self.environment.add_obstacle(circ)

    def add_rectangular_obstacle(
        self, bottom_left: ArrayLike, top_right: ArrayLike
    ) -> None:
        """
        Adds a rectangular obstacle to the simulation environment.

        Parameters
        ----------
        bottom_left : ArrayLike
            Coordinates of the bottom-left corner of the obstacle [x, y].
        top_right : ArrayLike
            Coordinates of the top-right corner of the obstacle [x, y].
        """
        rect = RectangularObstacle(bottom_left, top_right)
        self.environment.add_obstacle(rect)

    def clear_obstacles(self) -> None:
        """
        Removes all obstacles from the simulation environment.
        """
        self.environment.clear_obstacles()

    def set_random_positions(
        self,
        origin: np.ndarray = np.zeros(2),
        space: float = 1.0,
        altitude: float = 0.0,
    ) -> None:
        """
        Initializes drones with random positions around a given origin.

        Parameters
        ----------
        origin : np.ndarray, optional
            Center of the random distribution [x, y] (default is [0, 0]).
        space : float, optional
            Standard deviation of the random distribution (default is 1.0).
        altitude : float, optional
            Initial altitude for all drones (default is 0.0).
        """
        self.drone_states[:, 0:2] = np.random.normal(
            origin, space, (self.num_drones, 2)
        )
        self.drone_states[:, 2] = altitude
        self.drone_states[:, 3:6] = 0.0
        self._set_drone_states()

    def set_grid_positions(
        self,
        origin: np.ndarray = np.zeros(2),
        space: float = 1.0,
        altitude: float = 0.0,
    ) -> None:
        """
        Initializes drones in a grid formation.

        Parameters
        ----------
        origin : np.ndarray, optional
            Bottom-left corner of the grid [x, y] (default is [0, 0]).
        space : float, optional
            Spacing between drones in the grid (default is 1.0).
        altitude : float, optional
            Initial altitude for all drones (default is 0.0).
        """
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

    def initialize(self, verbose: bool = True) -> None:
        if verbose:
            print("Initializing simulation ...")
            
        self.update(dt=0.0)
        
        if verbose:
            print("✅ Initialization completed.")

    def update(self, dt: float = None) -> None:
        """
        Updates the simulation state, including drones and their interactions.

        Parameters
        ----------
        dt : float, optional
            Time step for the update in seconds (default is `self.dt`).
        """
        dt = dt if dt is not None else self.dt
        self._get_drone_states()
        self._update_visible_neighbors(100.0)
        self._update_drones(dt)
        self._update_links_matrix()
        self._update_edge_drones_mask()

    def _get_drone_states(self) -> None:
        """
        Updates the `drone_states` array with the current states of all drones.
        """
        for i, drone in enumerate(self.drones):
            self.drone_states[i, 0:3] = drone.position
            self.drone_states[i, 3:6] = drone.velocity

    def _set_drone_states(self) -> None:
        """
        Updates the states of all drones with the values in the `drone_states` array.
        """
        for i, drone in enumerate(self.drones):
            drone.state[0:3] = self.drone_states[i, 0:3]
            drone.state[3:6] = self.drone_states[i, 3:6]

    def _update_drones(self, dt: float) -> None:
        """
        Advances the state of all drones in the simulation by one time step.
        """
        for drone in self.drones:
            drone.update(dt)
        self.time += dt
        self.step += 1

    def _update_visible_neighbors(self, max_distance: float) -> None:
        """
        Updates the list of visible neighbors for each drone based on a maximum
        visibility distance.
        """
        positions = self.drone_states[:, 0:3]
        for drone in self.drones:
            deltas = positions - drone.position
            distances = np.linalg.norm(deltas, axis=1)
            visible_neighbors = (distances < max_distance) & (distances > 0.0)
            drone.set_neighbors(
                self.drone_ids[visible_neighbors], positions[visible_neighbors]
            )

    def _update_links_matrix(self) -> None:
        """
        Updates the adjacency matrix representing links between drones.
        """
        self.links_matrix = np.full(
            (self.num_drones, self.num_drones), False, dtype=bool
        )
        for drone in self.drones:
            neighbor_indexes = drone.neighbor_ids - 1
            controller: EVSMPositionController = drone.position_controller
            if not isinstance(controller, EVSMPositionController):
                raise Exception(f"Drone {drone.id} position controller is not EVSM")
            links_mask = controller.evsm.links_mask
            drone_links = np.zeros((self.num_drones,), dtype=bool)
            drone_links[neighbor_indexes] = links_mask
            self.links_matrix[drone.id - 1, :] = drone_links

    def _update_edge_drones_mask(self) -> None:
        """
        Updates the mask indicating which drones are at the edge of the swarm.
        """
        for i, drone in enumerate(self.drones):
            controller: EVSMPositionController = drone.position_controller
            if not isinstance(controller, EVSMPositionController):
                raise Exception(f"Drone {drone.id} position controller is not EVSM")
            self.edge_drones_mask[i] = controller.evsm.is_edge_robot()
