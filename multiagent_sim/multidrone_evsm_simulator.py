"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np

from .agents import Drone, AgentsManager, AgentsConfig
from .environment import Environment
from .mobility.evsm_swarming import EVSMConfig, EVSMController
from .math.path_loss_model import signal_strength


class MultiDroneEVSMSimulator:
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
        evsm_config: EVSMConfig = None,
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
        self.visible_distance = visible_distance

        self.agents_manager = AgentsManager(
            agents_config=AgentsConfig(num_drones=num_drones),
            swarming_config=evsm_config,
            env=self.environment,
        )

        self.time = 0.0
        self.step = 0

        self.edge_drones_mask = np.zeros((self.num_drones,), dtype=bool)

    @property
    def drone_states(self) -> np.ndarray:
        """
        A (N, 6) shape array with drone states [px, py, pz, vx, vy, vz] in meters and m/s,
        where N is the number of drones.
        """
        return self.agents_manager.drones.get_states_array()

    @property
    def drone_positions(self) -> np.ndarray:
        """
        A (N, 3) shape array with drone positions [px, py, pz] in meters,
        where N is the number of drones.
        """
        return self.agents_manager.drones.get_states_array()[:, 0:3]

    @property
    def drone_velocities(self) -> np.ndarray:
        """
        A (N, 3) shape array with drone velocities [vx, vy, vz] in m/s,
        where N is the number of drones.
        """
        return self.agents_manager.drones.get_states_array()[:, 3:6]

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

        if positions is not None:
            states = np.zeros((self.num_drones, 6), dtype=np.float32)
            states[:, 0:3] = positions
            self.agents_manager.initialize_drones(states)

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
        self.agents_manager.update_agents(dt=dt)
        self._update_links_matrix()
        self._update_edge_drones_mask()

    def area_coverage_ratio(
        self,
        num_points: int = 1000,
        tx_power: float = 20.0,
        rx_sens: float = -80.0,
        freq: float = 2.4,
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
        eval_points = np.zeros((num_points, 3), dtype=np.float32)
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
        tx_power_map = signal_strength(
            self.drone_positions,
            eval_points[in_area],
            f=freq,
            tx_power=tx_power,
            mode="max",
        )
        in_range = tx_power_map > rx_sens
        return np.sum(in_range) / np.sum(in_area)

    def _update_links_matrix(self) -> None:
        """
        Updates the adjacency matrix representing links between drones.
        """
        self.links_matrix = np.full(
            (self.num_drones, self.num_drones), False, dtype=bool
        )
        for drone in self.agents_manager.drones.get_all():
            controller: EVSMController = drone.swarming

            if controller is None:
                raise Exception(f"Drone {drone.agent_id} has no position controller")

            if not isinstance(controller, EVSMController):
                raise Exception(
                    f"Drone {drone.agent_id} position controller is not EVSM"
                )

            indices = drone.neighbor_drone_ids
            links_mask = controller.evsm.links_mask
            drone_links = np.zeros((self.num_drones,), dtype=bool)
            drone_links[indices] = links_mask
            self.links_matrix[drone.agent_id, :] = drone_links

    def _update_edge_drones_mask(self) -> None:
        """
        Updates the mask indicating which drones are at the edge of the swarm.
        """
        for i, drone in enumerate(self.agents_manager.drones.get_all()):
            controller: EVSMController = drone.swarming

            if controller is None:
                raise Exception(f"Drone {drone.id} has no position controller")

            if not isinstance(controller, EVSMController):
                raise Exception(f"Drone {drone.id} position controller is not EVSM")

            self.edge_drones_mask[i] = controller.evsm.is_edge_robot()
