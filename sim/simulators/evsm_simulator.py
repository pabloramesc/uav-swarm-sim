import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..agents import (
    AgentsManager,
    ControlStation,
    Drone,
    RegistryNeighborProvider,
    SwarmLinkNeighborProvider,
    User,
)
from ..environment import Environment
from ..mobility.evsm_position_controller import EVSMConfig, EVSMPositionController
from ..mobility.utils import environment_random_positions, grid_positions
from ..network import SwarmLink
from .evsm_monitor import EVSMMonitor
from .simulator import MultiAgentSimulator


class EVSMSimulator:

    logger = logging.getLogger("EVSMSimulator")

    def __init__(
        self,
        environment: Environment,
        num_drones: int,
        num_users: int = 0,
        num_gcs: int = 1,
        use_network: bool = False,
        evsm_config: Optional[EVSMConfig] = None,
        dt: float = 0.01,
    ) -> None:
        # Configuration
        self.evsm_config = evsm_config or EVSMConfig()
        self.use_network = use_network

        # Core components
        self.agents = AgentsManager()
        self.environment = environment

        # Agents
        self._create_gcs(num_gcs)
        self._create_drones(num_drones)
        self._create_users(num_users)

        # Simulation
        self.sim = MultiAgentSimulator(
            agents=self.agents,
            environment=self.environment,
            use_network=use_network,
            dt=dt,
        )

        self.evsm_monitor = EVSMMonitor(drones=self.sim.drones)

    def _create_gcs(self, num_gcs: int) -> None:
        for _ in range(num_gcs):
            station_id = self.agents.size + 1

            if self.use_network and self.sim.network is not None:
                link = SwarmLink(
                    agent_id=station_id,
                    network_sim=self.sim.network.netsim,
                    position_timeout=5.0,
                )
            else:
                link = None

            gcs = ControlStation(
                agent_id=station_id, env=self.environment, swarm_link=link
            )
            self.agents.register_agent(gcs)
        return

    def _create_drones(self, num_drones: int) -> None:
        for _ in range(num_drones):
            drone_id = self.agents.size + 1

            if self.use_network and self.sim.network is not None:
                link = SwarmLink(
                    agent_id=drone_id,
                    network_sim=self.sim.network.netsim,
                    local_bcast_interval=0.1,
                    global_bcast_interval=1.0,
                    position_timeout=5.0,
                )
                provider = SwarmLinkNeighborProvider(swarm_link=link)

            else:
                link = None
                provider = RegistryNeighborProvider(
                    agent_id=drone_id,
                    drones_registry=self.agents.drones,
                    users_registry=self.agents.users,
                )

            evsm = EVSMPositionController(
                config=self.evsm_config, environment=self.environment
            )

            drone = Drone(
                agent_id=drone_id,
                env=self.environment,
                controller=evsm,
                provider=provider,
                swarm_link=link,
            )

            self.agents.register_agent(drone)
        return

    def _create_users(self, num_users: int) -> None:
        for _ in range(num_users):
            user = User(
                agent_id=self.agents.size + 1, env=self.environment, swarm_link=None
            )
            self.agents.register_agent(user)
        return

    def reset(
        self, home: NDArray = np.zeros(2), spacing: float = 5.0, altitude: float = 0.0
    ) -> None:
        self.logger.info("Initializing simulation ...")

        gcs_states = np.zeros((self.sim.gcs.size, 6))
        gcs_states[:, 0:2] = home[0:2]
        gcs_states[:, 2] = self.environment.get_elevation(home[0:2])

        drone_states = np.zeros((self.sim.drones.size, 6))
        drone_states[:, 0:3] = grid_positions(
            num_points=self.sim.drones.size,
            origin=home,
            space=spacing,
            altitude=altitude,
        )

        user_states = np.zeros((self.sim.users.size, 6))
        user_states[:, 0:3] = environment_random_positions(
            num_positions=self.sim.users.size, env=self.environment
        )

        states = np.vstack([gcs_states, drone_states, user_states])
        self.sim.reset(states=states)

        self.logger.info("✅ Initialization completed.")

    def step(self, dt=None):
        self.sim.step(dt)
        self.evsm_monitor.update()
