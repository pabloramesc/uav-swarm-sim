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
from simulator.math.path_loss_model import (
    signal_strength,
    signal_strength_map,
)


class MultiDroneSimulatorEVSM:
    """
    Simulates a swarm of drones in a 3D environment.

    This class manages the initialization, state updates, and interactions of multiple drones
    in a simulation environment. It supports features such as obstacle avoidance, neighbor detection,
    and swarm behavior using position controllers.
    """

    def __init__(
        self,
        num_drones: int,
        dt: float = 0.01,
        dem_path: str = None,
        config: EVSMConfig = None,
        visible_distance: float = 100.0,
    ) -> None:
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
        self.environment = Environment(dem_path)
        self.config = config if config is not None else EVSMConfig()
        self.visible_distance = visible_distance

        self._init_states()
        self._init_drones()

    def _init_states(self) -> None:
        self.time = 0.0
        self.step = 0
        self.drone_states = np.zeros((self.num_drones, 6))  # px, py, pz, vx, vy, vz
        self.links_matrix = np.full(
            (self.num_drones, self.num_drones), False, dtype=bool
        )
        self.edge_drones_mask = np.full((self.num_drones,), False, dtype=bool)

    def _init_drones(self) -> None:
        self.drones: list[Drone] = []
        for id in range(self.num_drones):
            evsm = EVSMPositionController(self.config, self.environment)
            drone = Drone(id, self.environment, evsm)
            self.drones.append(drone)

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

    def initialize(self, positions: np.ndarray = None, verbose: bool = True) -> None:
        """
        Initializes the simulation by updating the initial state of all drones.

        Parameters
        ----------
        positions : np.ndarray, optional
            A (N, 3) array specifying the initial positions [px, py, pz] of the drones.
            If None, the positions remain unchanged (default is None).
        verbose : bool, optional
            If True, prints initialization progress (default is True).
        """
        if verbose:
            print("Initializing simulation ...")

        self._init_states()
        self._init_drones()

        if positions is not None:
            self.drone_states[:, 0:3] = positions  # Fix assignment to update all drones
            self._set_drone_states()

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
        self.time += dt
        self.step += 1
        self._get_drone_states()
        self._update_visible_neighbors()
        self._update_drones(dt)
        self._update_links_matrix()
        self._update_edge_drones_mask()

    def signal_strength(self, positions: np.ndarray) -> np.ndarray:
        """
        Calculates the signal strength at given positions based on the drone positions.

        Parameters
        ----------
        positions : np.ndarray
            A (N, 3) array of positions [x, y, z] where the signal strength is to be calculated.

        Returns
        -------
        np.ndarray
            A (N,) array of signal strength values in dBm at the given positions.
        """
        return signal_strength(self.drone_positions, positions, f=2.4e3, mode="max")

    def signal_strength_map(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """
        Generates a 2D heatmap of signal strength over a grid of points.

        Parameters
        ----------
        xs : np.ndarray
            A 1D array of x-coordinates for the grid.
        ys : np.ndarray
            A 1D array of y-coordinates for the grid.

        Returns
        -------
        np.ndarray
            A 2D array of signal strength values in dBm over the grid.
        """
        return signal_strength_map(self.drone_positions, xs, ys, f=2.4e3, mode="max")

    def coverage_map(
        self, xs: np.ndarray, ys: np.ndarray, rx_sens: float = -80.0
    ) -> np.ndarray:
        """
        Generates a binary coverage map indicating areas with sufficient signal strength.

        Parameters
        ----------
        xs : np.ndarray
            A 1D array of x-coordinates for the grid.
        ys : np.ndarray
            A 1D array of y-coordinates for the grid.
        rx_sens : float, optional
            Receiver sensitivity threshold in dBm (default is -80.0).

        Returns
        -------
        np.ndarray
            A 2D binary array where True indicates sufficient signal strength.
        """
        return self.signal_strength_map(xs, ys) > rx_sens

    def area_coverage_ratio(
        self, num_points: int = 1000, rx_sens: float = -80.0
    ) -> float:
        """
        Calculates the ratio of the area covered by sufficient signal strength.

        Parameters
        ----------
        num_points : int, optional
            Number of random points to sample within the environment (default is 1000).
        rx_sens : float, optional
            Receiver sensitivity threshold in dBm (default is -80.0).

        Returns
        -------
        float
            The ratio of the area covered by sufficient signal strength.
        """
        eval_points = np.zeros((num_points, 3))
        eval_points[:, 0] = np.random.uniform(
            *self.environment.boundary_xlim, num_points
        )
        eval_points[:, 1] = np.random.uniform(
            *self.environment.boundary_ylim, num_points
        )
        eval_points[:, 2] = self.environment.get_elevation(eval_points[:, 0:2])
        in_area = self.environment.is_inside(
            eval_points
        ) & ~self.environment.is_collision(eval_points)
        tx_power = self.signal_strength(eval_points[in_area])
        in_range = tx_power > rx_sens
        return np.sum(in_range) / np.sum(in_area)

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

    def _update_visible_neighbors(self) -> None:
        """
        Updates the list of visible neighbors for each drone based on a maximum
        visibility distance.
        """
        positions = self.drone_states[:, 0:3]
        for drone in self.drones:
            deltas = positions - drone.position
            distances = np.linalg.norm(deltas, axis=1)
            is_visible = (distances < self.visible_distance) & (distances > 0.0)
            indices = np.where(is_visible)[0]
            drone.set_neighbors(indices, positions[indices])

    def _update_links_matrix(self) -> None:
        """
        Updates the adjacency matrix representing links between drones.
        """
        self.links_matrix = np.full(
            (self.num_drones, self.num_drones), False, dtype=bool
        )
        for drone in self.drones:
            indices = drone.neighbor_ids
            controller: EVSMPositionController = drone.position_controller
            if not isinstance(controller, EVSMPositionController):
                raise Exception(f"Drone {drone.id} position controller is not EVSM")
            links_mask = controller.evsm.links_mask
            drone_links = np.zeros((self.num_drones,), dtype=bool)
            drone_links[indices] = links_mask
            self.links_matrix[drone.id, :] = drone_links

    def _update_edge_drones_mask(self) -> None:
        """
        Updates the mask indicating which drones are at the edge of the swarm.
        """
        for i, drone in enumerate(self.drones):
            controller: EVSMPositionController = drone.position_controller
            if not isinstance(controller, EVSMPositionController):
                raise Exception(f"Drone {drone.id} position controller is not EVSM")
            self.edge_drones_mask[i] = controller.evsm.is_edge_robot()
