from dataclasses import dataclass

import numpy as np

from ..environment import Environment
from ..mobility.base_swarming import SwarmingController, SwarmingType, SwarmingConfig
from ..mobility.evsm_swarming import EVSMConfig, EVSMController
from ..mobility.sdqn_swarming import SDQNConfig, SDQNController
from ..network.network_simulator import NetworkSimulator
from ..network.swarm_interface import SwarmProtocolInterface
from .agent import Agent
from .agents_registry import AgentsRegistry
from .control_station import ControlStation
from .drone import Drone, NeighborProvider
from .user import User


@dataclass
class AgentsConfig:
    num_gcs: int = 0
    num_drones: int = 0
    num_users: int = 0
    drones_swarming_type: SwarmingType = "evsm"
    drones_neighbor_provider: NeighborProvider = "registry"


class AgentsManager:

    def __init__(
        self,
        agents_config: AgentsConfig,
        swarming_config: SwarmingConfig = None,
        env: Environment = None,
        net_sim: NetworkSimulator = None,
    ):
        self.agents_config = agents_config
        self.swarming_config = swarming_config
        self.environment = env if env is not None else Environment()
        self.network_simulator = net_sim

        self.agents: list[Agent] = []
        self.drones = AgentsRegistry()
        self.users = AgentsRegistry()
        self._create_agents()

    def get_agent(self, global_id: int) -> Agent:
        return self.agents[global_id]

    def initialize_agents(self, states: np.ndarray) -> None:
        for i, agent in enumerate(self.agents):
            agent.initialize(states[i])

    def initialize_drones(self, states: np.ndarray) -> None:
        for i, drone in enumerate(self.drones.get_all()):
            drone.initialize(states[i])

    def update_agents(self, dt: float = 0.01) -> None:
        for agent in self.agents:
            agent.update(dt)

    def _create_agents(self) -> None:
        global_id = 0
        for id in range(self.agents_config.num_gcs):
            gcs = self._create_control_station(global_id=global_id, type_id=id)
            self.agents.append(gcs)
            global_id += 1

        for id in range(self.agents_config.num_drones):
            drone = self._create_drone(agent_id=global_id, type_id=id)
            self.agents.append(drone)
            self.drones.register(drone)
            global_id += 1

        for id in range(self.agents_config.num_users):
            user = self._create_user(global_id=global_id, type_id=id)
            self.agents.append(user)
            self.users.register(user)
            global_id += 1

    def _create_control_station(self, global_id: int, type_id: int) -> ControlStation:
        interface = SwarmProtocolInterface(
            agent_id=global_id, network_sim=self.network_simulator
        )
        gcs = ControlStation(
            global_id=global_id, type_id=type_id, env=self.environment, net=interface
        )
        return gcs

    def _create_drone(self, agent_id: int, type_id: int) -> Drone:
        swarming: SwarmingController = None
        if self.agents_config.drones_swarming_type == "evsm":
            config = (
                self.swarming_config
                if self.swarming_config is not None
                else EVSMConfig()
            )
            swarming = EVSMController(config=config, env=self.environment)
        elif self.agents_config.drones_swarming_type == "sdqn":
            config = (
                self.swarming_config
                if self.swarming_config is not None
                else SDQNConfig()
            )
            swarming = SDQNController(config=config, env=self.environment)
        else:
            raise ValueError(
                f"Invalid swarming controller type: {self.agents_config.drones_swarming_type}"
            )

        interface = (
            SwarmProtocolInterface(
                agent_id=agent_id,
                network_sim=self.network_simulator,
                local_bcast_interval=0.1,
                global_bcast_interval=1.0,
            )
            if self.network_simulator is not None
            else None
        )

        drone = Drone(
            agent_id=agent_id,
            env=self.environment,
            swarming=swarming,
            network=interface,
            drones_registry=self.drones,
            users_registry=self.users,
            neighbor_provider=self.agents_config.drones_neighbor_provider,
        )

        return drone

    def _create_user(self, global_id: int, type_id: int) -> User:
        interface = SwarmProtocolInterface(
            agent_id=global_id, network_sim=self.network_simulator
        )
        user = User(
            global_id=global_id, type_id=type_id, env=self.environment, net=interface
        )
        return user
