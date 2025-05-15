from dataclasses import dataclass

import numpy as np

from ..environment import Environment
from ..mobility.swarm_position_controller import (
    SwarmPositionController,
    SwarmingType,
    SwarmPositionConfig,
)
from ..mobility.evsm_position_controller import (
    EVSMPositionConfig,
    EVSMPositionController,
)
from ..mobility.sdqn_position_controller import SDQNPositionConfig, SDQNPositionController
from ..network.network_simulator import NetworkSimulator
from ..network.swarm_link import SwarmLink
from .agent import Agent, AgentType
from .agents_registry import AgentsRegistry
from .control_station import ControlStation
from .drone import Drone, NeighborProvider
from .user import User


@dataclass
class AgentsConfig:
    num_gcs: int = 0
    num_drones: int = 0
    num_users: int = 0
    swarming_type: SwarmingType = "evsm"
    neighbor_provider: NeighborProvider = "registry"


class AgentsManager:

    def __init__(
        self,
        agents_config: AgentsConfig,
        swarming_config: SwarmPositionConfig = None,
        environment: Environment = None,
        network_simulator: NetworkSimulator = None,
    ):
        self.agents_config = agents_config
        self.swarming_config = swarming_config
        self.environment = environment if environment is not None else Environment()
        self.network_simulator = network_simulator

        self.agents: list[Agent] = []
        self.control_stations = AgentsRegistry()
        self.drones = AgentsRegistry()
        self.users = AgentsRegistry()
        self._create_agents()

    def get_agent(self, global_id: int) -> Agent:
        return self.agents[global_id]

    def initialize_agent(self, global_id: int, state: np.ndarray) -> None:
        agent = self.agents[global_id]
        agent.initialize(state)

    def initialize_all_agents(
        self, states: np.ndarray, agent_type: AgentType = None
    ) -> None:
        if agent_type is None:
            selected_agents = self.agents
        elif agent_type == "drone":
            selected_agents = self.drones.get_all()
        elif agent_type == "user":
            selected_agents = self.users.get_all()
        elif agent_type == "gcs":
            selected_agents = self.control_stations.get_all()
        else:
            raise ValueError(f"Invalid agent type: {agent_type}")

        for i, agent in enumerate(selected_agents):
            agent.initialize(states[i])

    def update_agents(self, dt: float = 0.01) -> None:
        for agent in self.agents:
            agent.update(dt)

    def _create_agents(self) -> None:
        global_id = 0
        for id in range(self.agents_config.num_gcs):
            gcs = self._create_control_station(global_id=global_id)
            self.agents.append(gcs)
            self.control_stations.register(gcs)
            global_id += 1

        for id in range(self.agents_config.num_drones):
            drone = self._create_drone(agent_id=global_id)
            self.agents.append(drone)
            self.drones.register(drone)
            global_id += 1

        for id in range(self.agents_config.num_users):
            user = self._create_user(global_id=global_id)
            self.agents.append(user)
            self.users.register(user)
            global_id += 1

    def _create_control_station(self, global_id: int) -> ControlStation:
        gcs = ControlStation(
            agent_id=global_id, env=self.environment, network_sim=self.network_simulator
        )
        return gcs

    def _create_drone(self, agent_id: int) -> Drone:
        controller: SwarmPositionController = None
        if self.agents_config.swarming_type == "evsm":
            config = (
                self.swarming_config
                if self.swarming_config is not None
                else EVSMPositionConfig()
            )
            controller = EVSMPositionController(
                config=config, environment=self.environment
            )
        elif self.agents_config.swarming_type == "sdqn":
            config = (
                self.swarming_config
                if self.swarming_config is not None
                else SDQNPositionConfig()
            )
            controller = SDQNPositionController(config=config, environment=self.environment)
        else:
            raise ValueError(
                f"Invalid swarming controller type: {self.agents_config.swarming_type}"
            )

        drone = Drone(
            agent_id=agent_id,
            env=self.environment,
            position_controller=controller,
            network_sim=self.network_simulator,
            drones_registry=self.drones,
            users_registry=self.users,
            neighbor_provider=self.agents_config.neighbor_provider,
        )

        return drone

    def _create_user(self, global_id: int) -> User:
        user = User(
            agent_id=global_id, env=self.environment, network_sim=self.network_simulator
        )
        return user
