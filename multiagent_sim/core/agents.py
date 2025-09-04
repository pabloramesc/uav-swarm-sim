import numpy as np

from ..agents import Agent, AgentsRegistry


class AgentsManager:
    """Manages all agent registries."""

    def __init__(self):
        self.all_agents = AgentsRegistry()
        self.registries: dict[str, AgentsRegistry] = {
            "gcs": AgentsRegistry(),
            "users": AgentsRegistry(),
            "drones": AgentsRegistry(),
        }

    @property
    def size(self) -> int:
        self.all_agents.num_agents

    def register_agent(self, agent: Agent):
        registry = self.registries.get(agent.agent_type)
        if registry is None:
            raise ValueError(f"No registry found for agent type '{agent.agent_type}'")
        registry.register(agent)
        self.all_agents.register(agent)

    def get_registry(self, agent_type: str) -> AgentsRegistry:
        return self.registries.get(agent_type)

    def get_states(self) -> dict[str, np.ndarray]:
        return {atype: reg for atype, reg in self.registries.items()}