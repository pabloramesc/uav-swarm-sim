from abc import ABC, abstractmethod
import numpy as np

from ..network.swarm_link import SwarmLink
from .agents_registry import AgentsRegistry


class NeighborProvider(ABC):
    @abstractmethod
    def get_user_positions(self) -> dict[int, np.ndarray]:
        pass

    @abstractmethod
    def get_drone_positions(self) -> dict[int, np.ndarray]:
        pass


class DummyNeighborProvider(NeighborProvider):
    def get_user_positions(self) -> dict[int, np.ndarray]:
        return {}

    def get_drone_positions(self) -> dict[int, np.ndarray]:
        return {}


class RegistryNeighborProvider(NeighborProvider):
    def __init__(
        self,
        agent_id: int,
        drones_registry: AgentsRegistry,
        users_registry: AgentsRegistry,
    ):
        self.agent_id = int(agent_id)
        self.drones_registry = drones_registry
        self.users_registry = users_registry

    def get_user_positions(self) -> dict[int, np.ndarray]:
        return self.users_registry.get_positions_dict(exclude_id=self.agent_id)

    def get_drone_positions(self) -> dict[int, np.ndarray]:
        return self.drones_registry.get_positions_dict(exclude_id=self.agent_id)


class SwarmLinkNeighborProvider(NeighborProvider):
    def __init__(self, swarm_link: SwarmLink):
        self.swarm_link = swarm_link

    def get_user_positions(self) -> dict[int, np.ndarray]:
        return self.swarm_link.get_positions(node_type="user")

    def get_drone_positions(self) -> dict[int, np.ndarray]:
        return self.swarm_link.get_positions(node_type="drone")
