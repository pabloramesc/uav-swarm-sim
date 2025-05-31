import numpy as np
from numpy.typing import ArrayLike

from ..agents import AgentsRegistry, Drone, Agent, ControlStation, User
from ..mobility.evsm_position_controller import EVSMPositionController, EVSMConfig
from ..mobility.utils import grid_positions, environment_random_positions
from .multiagent_simulator import MultiAgentSimulator


class EVSMMonitor:
    def __init__(self, drones: AgentsRegistry):
        self.registry = drones
        self.edge_mask = np.zeros(drones.num_agents, dtype=bool)
        self.springs_matrix = np.zeros(
            (drones.num_agents, drones.num_agents), dtype=bool
        )

    def update(self):
        for i, agent in enumerate(self.registry):
            drone: Drone = agent
            controller: EVSMPositionController = drone.position_controller

            if controller is None:
                raise Exception(f"Drone {drone.id} has no position controller")

            if not isinstance(controller, EVSMPositionController):
                raise Exception(f"Drone {drone.id} position controller is not EVSM")

            self.edge_mask[i] = controller.evsm.is_edge_robot()
            self.springs_matrix[i] = self._drone_springs_mask(controller)

    def _drone_springs_mask(self, controller: EVSMPositionController) -> np.ndarray:
        drone_springs = np.zeros(self.registry.num_agents, dtype=bool)

        neighbor_ids = np.array(list(controller.drone_positions.keys()))
        neighbor_indices = self.registry.get_indices(neighbor_ids)

        if neighbor_indices.size == 0:
            return drone_springs

        springs_mask = controller.evsm.springs_mask
        if neighbor_indices.shape != springs_mask.shape:
            raise ValueError("Springs mask shape do not match neighbor indices")

        drone_springs[neighbor_indices] = springs_mask
        return drone_springs


class EVSMSimulator(MultiAgentSimulator):

    def __init__(
        self,
        num_drones: int,
        num_users: int = 0,
        dt: float = 0.01,
        dem_path: str = None,
        use_network: bool = False,
        evsm_config: EVSMConfig = None,
    ) -> None:
        self.evsm_config = evsm_config or EVSMConfig()

        super().__init__(
            num_drones,
            num_users,
            dt,
            dem_path,
            use_network,
            evsm_config=self.evsm_config,
        )

        self.evsm_monitor = EVSMMonitor(drones=self.drones)

    def _create_drone(self, evsm_config: EVSMConfig = None) -> Drone:
        evsm = EVSMPositionController(
            config=evsm_config, environment=self.environment
        )
        drone = Drone(
            agent_id=len(self.agents),
            environment=self.environment,
            position_controller=evsm,
            network_sim=self.netsim,
            drones_registry=self.drones,
            users_registry=self.users,
            neighbor_provider="network" if self.network else "registry",
        )
        return drone

    def initialize(self, home: ArrayLike = [0.0, 0.0], spacing: float = 5.0) -> None:
        self.logger.info("Initializing simulation ...")

        gcs_state = np.zeros(6)
        gcs_state[0:2] = np.asarray(home[0:2])
        gcs_state[2] = self.environment.get_elevation(home[0:2])
        self.gcs.initialize(state=gcs_state)

        drone_states = np.zeros((self.num_drones, 6))
        drone_states[:, 0:3] = grid_positions(
            num_points=self.num_drones,
            origin=home,
            space=spacing,
            altitude=self.evsm_config.target_altitude,
        )
        self.drones.initialize(states=drone_states)

        user_states = np.zeros((self.num_users, 6))
        user_states[:, 0:3] = environment_random_positions(
            num_positions=self.num_users, env=self.environment
        )
        self.users.initialize(states=user_states)

        super().initialize()

        self.logger.info("✅ Initialization completed.")

    def update(self, dt=None):
        super().update(dt)
        self.evsm_monitor.update()
        self._sync_to_real_time()
