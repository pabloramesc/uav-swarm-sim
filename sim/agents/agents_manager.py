import numpy as np

from .agent import Agent
from .agents_registry import AgentsRegistry


class AgentsManager:
    """Manages all agent registries."""

    def __init__(self):
        self.all_agents = AgentsRegistry()
        self.registries: dict[str, AgentsRegistry] = {
            "gcs": AgentsRegistry(),
            "user": AgentsRegistry(),
            "drone": AgentsRegistry(),
        }

    @property
    def size(self) -> int:
        return self.all_agents.size

    @property
    def gcs(self) -> AgentsRegistry:
        return self.registries["gcs"]

    @property
    def users(self) -> AgentsRegistry:
        return self.registries["user"]

    @property
    def drones(self) -> AgentsRegistry:
        return self.registries["drone"]

    def clear_registries(self) -> None:
        self.all_agents.clear()
        for reg in self.registries.values():
            reg.clear()

    def register_agent(self, agent: Agent):
        registry = self.registries.get(agent.agent_type)
        if registry is None:
            raise ValueError(f"No registry found for agent type '{agent.agent_type}'")
        registry.register(agent)
        self.all_agents.register(agent)

    def get_registry(self, agent_type: str) -> AgentsRegistry:
        reg = self.registries.get(agent_type)
        if reg is None:
            raise ValueError(f"No register with agent type '{agent_type}'")
        return reg

    def get_states(self) -> dict[str, np.ndarray]:
        return {atype: reg.get_states_array() for atype, reg in self.registries.items()}
