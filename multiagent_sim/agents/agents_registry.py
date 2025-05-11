import numpy as np
from abc import ABC
from .agent import Agent


class AgentsRegistry(ABC):
    """
    A lightweight registry for tracking agent instances and their states.
    Useful for querying all or nearby agents in a shared environment.
    """

    def __init__(self):
        self.agents: dict[int, Agent] = {}
    
    @property
    def num_agents(self) -> int:
        """Returns the number of registered agents."""
        return len(self.agents)

    def register(self, agent: Agent) -> None:
        """Adds a new agent to the registry."""
        if agent.agent_id in self.agents:
            raise ValueError(f"Agent with ID {agent.agent_id} is already registered.")
        self.agents[agent.agent_id] = agent

    def get_all(self) -> list[Agent]:
        """Returns all registered agents."""
        return list(self.agents.values())

    def get_agent(self, agent_id: int) -> Agent:
        """Returns the agent with the specified global ID."""
        return self.agents[agent_id]

    def get_state(self, agent_id: int) -> np.ndarray:
        """Returns the state of the agent with the specified global ID."""
        return self.agents[agent_id].state

    def get_states_array(self, exclude_id: int = None) -> np.ndarray:
        """
        Returns an array of all agent states.
        If `exclude_id` is provided, excludes the agent with that ID.
        """
        return np.array([
            agent.state
            for agent_id, agent in self.agents.items()
            if agent_id != exclude_id
        ])

    def get_states_dict(self, exclude_id: int = None) -> dict[int, np.ndarray]:
        """
        Returns a dictionary mapping agent IDs to their states.
        If `exclude_id` is provided, that agent will be excluded.
        """
        return {
            agent_id: agent.state
            for agent_id, agent in self.agents.items()
            if agent_id != exclude_id
        }
        
    def get_positions_dict(self, exclude_id: int = None) -> dict[int, np.ndarray]:
        """
        Returns a dictionary mapping agent IDs to their positions.
        If `exclude_id` is provided, that agent will be excluded.
        """
        return {
            agent_id: agent.position
            for agent_id, agent in self.agents.items()
            if agent_id != exclude_id
        }

    def get_near_positions(
        self, position: np.ndarray, distance: float = 100.0
    ) -> np.ndarray:
        """
        Returns the positions of agents within a given distance from `position`.
        Assumes the position is in the first three elements of the state vector.
        """
        positions = [
            agent.position
            for agent in self.agents.values()
            if 0.0 < np.linalg.norm(agent.position - position) < distance
        ]
        return np.array(positions)
