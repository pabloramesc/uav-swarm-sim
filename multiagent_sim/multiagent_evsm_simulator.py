"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import time

import numpy as np
from numpy.typing import ArrayLike

from .agents import Drone, AgentsManager, AgentsConfig, NeighborProvider
from .environment import Environment
from .mobility.evsm_position_controller import (
    EVSMPositionConfig,
    EVSMPositionController,
)
from .math.path_loss_model import signal_strength
from .network import NetworkSimulator
from .utils.logger import create_logger
from .mobility.utils import grid_positions, environment_random_positions


class MultiAgentEVSMSimulator:
    
    SYNC_TOLERANCE = 0.1

    def __init__(
        self,
        num_drones: int,
        num_users: int = 0,
        dt: float = 0.01,
        dem_path: str = None,
        evsm_config: EVSMPositionConfig = None,
        neihgbor_provider: NeighborProvider = "registry",
    ) -> None:
        self.num_drones = num_drones
        self.num_users = num_users
        self.dt = dt
        self.environment = Environment(dem_path)
        self.evsm_config = evsm_config

        if neihgbor_provider == "network":
            self.network_simulator = NetworkSimulator(
                num_gcs=1,
                num_drones=self.num_drones,
                num_users=self.num_users,
                verbose=True,
            )
            self.network_simulator.launch_simulator(max_attempts=2)
        else:
            self.network_simulator = None

        agents_config = AgentsConfig(
            num_gcs=1,
            num_drones=num_drones,
            num_users=num_users,
            drones_neighbor_provider=neihgbor_provider,
        )
        self.agents_manager = AgentsManager(
            agents_config=agents_config,
            swarming_config=evsm_config,
            environment=self.environment,
            network_simulator=self.network_simulator,
        )

        self.init_time: float = None
        self.sim_time = 0.0
        self.sim_step = 0

        self.edge_drones_mask = np.zeros((self.num_drones,), dtype=bool)

        self.logger = create_logger(name="MultiDroneEVSMSimulator", level="INFO")

    @property
    def real_time(self) -> float:
        """
        Returns the real time elapsed since the simulation started.
        """
        if self.init_time is None:
            return 0.0
        return time.time() - self.init_time

    def initialize(self, home: ArrayLike) -> None:
        self.logger.info("Initializing simulation ...")

        gcs_state = np.zeros(6)
        gcs_state[0:3] = np.asarray(home)
        self.agents_manager.initialize_agent(global_id=0, state=gcs_state)

        drone_states = np.zeros((self.num_drones, 6))
        drone_states[:, 0:3] = grid_positions(
            num_points=self.num_drones,
            origin=home,
            space=5.0,
            altitude=self.evsm_config.target_altitude,
        )
        self.agents_manager.initialize_all_agents(
            states=drone_states, agent_type="drone"
        )

        user_states = np.zeros((self.num_users, 6))
        user_states[:, 0:3] = environment_random_positions(
            num_positions=self.num_users, env=self.environment
        )
        self.agents_manager.initialize_all_agents(states=user_states, agent_type="user")

        self.init_time = time.time()
        self.sim_time = 0.0
        self.sim_step = 0

        self.update(dt=0.0)

        self.logger.info("✅ Initialization completed.")

    def update(self, dt: float = None) -> None:
        """
        Updates the simulation state, including drones and their interactions.

        Parameters
        ----------
        dt : float, optional
            Time step for the update in seconds (default is `self.dt`).
        """
        dt = dt if dt is not None else self.dt
        self.sim_time += dt
        self.sim_step += 1

        if self.network_simulator is not None:
            agent_positions = None
            if self.sim_step % 10 == 0:
                agent_states = np.array(
                    [agent.state for agent in self.agents_manager.agents]
                )
                agent_positions = agent_states[:, 0:3]
            check = self.sim_step % 100 == 0
            self.network_simulator.update(agent_positions, check)

        self.agents_manager.update_agents(dt=dt)

        self._update_links_matrix()
        self._update_edge_drones_mask()

    def sync_to_real_time(self) -> None:
        """
        Synchronizes the simulation time with real time, ensuring that the simulation
        does not run faster than real time.
        """
        if self.sim_time - self.real_time > self.dt:
            time.sleep(self.sim_time - self.real_time - self.dt)
        
        self._sync_to_real_time()

    def _sync_to_real_time(self) -> None:
        """
        Synchronizes the simulation time with real time, ensuring that the simulation
        does not run faster than real time.
        """
        real_delta = self.sim_time - self.real_time
        if real_delta > self.SYNC_TOLERANCE:
            time.sleep(real_delta)

        while True:
            ns3_delta = self.sim_time - self.network_simulator.ns3_time
            if ns3_delta < self.SYNC_TOLERANCE:
                break
            try:
                self.network_simulator.bridge.request_sim_time(timeout=ns3_delta)
            except TimeoutError:
                self.network_simulator.fetch_packets()

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
            drone: Drone = drone
            controller: EVSMPositionController = drone.position_controller

            if controller is None:
                raise Exception(f"Drone {drone.agent_id} has no position controller")

            if not isinstance(controller, EVSMPositionController):
                raise Exception(
                    f"Drone {drone.agent_id} position controller is not EVSM"
                )

            neighbor_ids = np.array(list(drone.drone_positions.keys()))
            neighbor_indices = self.agents_manager.drones.get_indices(neighbor_ids)
            links_mask = controller.evsm.links_mask

            drone_links = np.zeros((self.num_drones,), dtype=bool)
            if neighbor_indices.size > 0 and neighbor_indices.shape == links_mask.shape:
                drone_links[neighbor_indices] = links_mask

            if not np.any(drone_links):
                self.logger.info(f"Drone {drone.agent_id} has no links.")

            drone_index = self.agents_manager.drones.get_index(drone.agent_id)
            self.links_matrix[drone_index, :] = drone_links

    def _update_edge_drones_mask(self) -> None:
        """
        Updates the mask indicating which drones are at the edge of the swarm.
        """
        for i, drone in enumerate(self.agents_manager.drones.get_all()):
            drone: Drone = drone
            controller: EVSMPositionController = drone.position_controller

            if controller is None:
                raise Exception(f"Drone {drone.id} has no position controller")

            if not isinstance(controller, EVSMPositionController):
                raise Exception(f"Drone {drone.id} position controller is not EVSM")

            self.edge_drones_mask[i] = controller.evsm.is_edge_robot()
