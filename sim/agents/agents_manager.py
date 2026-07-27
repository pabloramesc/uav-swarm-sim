from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from .agent import Agent
from .agents_registry import AgentsRegistry


class AgentsManager:
    """Manages all agent registries."""

    def __init__(self):
        self.all_agents = AgentsRegistry()
        self._registries: dict[str, AgentsRegistry] = {
            "gcs": AgentsRegistry(),
            "drone": AgentsRegistry(),
            "user": AgentsRegistry(),
        }

    @property
    def registries(self) -> Mapping[str, AgentsRegistry]:
        """Read-only mapping of agent types to their registry views."""

        return MappingProxyType(self._registries)

    @property
    def size(self) -> int:
        return self.all_agents.size

    @property
    def gcs(self) -> AgentsRegistry:
        return self._registries["gcs"]

    @property
    def users(self) -> AgentsRegistry:
        return self._registries["user"]

    @property
    def drones(self) -> AgentsRegistry:
        return self._registries["drone"]

    def clear(self) -> None:
        """Remove every agent from the global and typed registry views."""

        self.all_agents._clear()
        for reg in self._registries.values():
            reg._clear()

    def register_agent(self, agent: Agent) -> None:
        registry = self._registries.get(agent.agent_type)
        if registry is None:
            raise ValueError(f"No registry found for agent type '{agent.agent_type}'")
        if agent.agent_id in self.all_agents:
            raise ValueError(f"Agent with ID {agent.agent_id} is already registered.")

        # Validate both destinations before mutating either registry.
        if agent.agent_id in registry:
            raise ValueError(
                f"Agent with ID {agent.agent_id} is already registered "
                f"as '{agent.agent_type}'."
            )
        registry._register(agent)
        self.all_agents._register(agent)

    def unregister_agent(self, agent_id: int) -> Agent:
        """Atomically remove one agent from both registry views."""

        agent = self.all_agents.get_agent(agent_id)
        registry = self._registries[agent.agent_type]
        if agent_id not in registry:
            raise RuntimeError(
                f"Agent {agent_id} is missing from its '{agent.agent_type}' registry."
            )
        registry._unregister(agent_id)
        self.all_agents._unregister(agent_id)
        return agent

    def get_registry(self, agent_type: str) -> AgentsRegistry:
        reg = self._registries.get(agent_type)
        if reg is None:
            raise ValueError(f"No register with agent type '{agent_type}'")
        return reg

    def get_states(self) -> dict[str, np.ndarray]:
        return {
            atype: reg.get_states_array() for atype, reg in self._registries.items()
        }
