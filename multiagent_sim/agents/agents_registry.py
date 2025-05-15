import numpy as np
from abc import ABC
from .agent import Agent


class AgentsRegistry(ABC):
    """
    A flexible registry for tracking agent instances and their states.
    Supports lookup by agent ID or index.
    """

    def __init__(self):
        self._agents_dict: dict[int, Agent] = {}
        self._agents_list: list[Agent] = []
        self._id_to_index: dict[int, int] = {}
        self._index_to_id: dict[int, int] = {}

    @property
    def num_agents(self) -> int:
        """Returns the number of registered agents."""
        return len(self._agents_dict)

    def _rebuild_index_mapping(self) -> None:
        """
        Rebuilds the ID-to-index mapping whenever the registry changes.
        """
        self._agents_list = list(self._agents_dict.values())
        self._id_to_index = {
            agent_id: idx for idx, agent_id in enumerate(self._agents_dict.keys())
        }
        self._index_to_id = {
            idx: agent_id for idx, agent_id in enumerate(self._agents_dict.keys())
        }

    def register(self, agent: Agent) -> None:
        """Adds a new agent to the registry."""
        if agent.agent_id in self._agents_dict:
            raise ValueError(f"Agent with ID {agent.agent_id} is already registered.")
        self._agents_dict[agent.agent_id] = agent
        self._rebuild_index_mapping()

    def unregister(self, agent_id: int) -> None:
        """Removes an agent from the registry."""
        if agent_id not in self._agents_dict:
            raise KeyError(f"Agent with ID {agent_id} is not registered.")
        del self._agents_dict[agent_id]
        self._rebuild_index_mapping()

    def get_all(self) -> list[Agent]:
        """Returns all registered agents."""
        return self._agents_list

    def get_agent(self, agent_id: int) -> Agent:
        """Returns the agent with the specified global ID."""
        return self._agents_dict[agent_id]

    def get_state(self, agent_id: int) -> np.ndarray:
        """Returns the state of the agent with the specified global ID."""
        return self._agents_dict[agent_id].state

    def get_states_array(self, exclude_id: int = None) -> np.ndarray:
        """
        Returns an array of all agent states.
        If `exclude_id` is provided, excludes the agent with that ID.
        """
        return np.array(
            [
                agent.state
                for agent_id, agent in self._agents_dict.items()
                if agent_id != exclude_id
            ]
        )

    def get_states_dict(self, exclude_id: int = None) -> dict[int, np.ndarray]:
        """
        Returns a dictionary mapping agent IDs to their states.
        If `exclude_id` is provided, that agent will be excluded.
        """
        return {
            agent_id: agent.state
            for agent_id, agent in self._agents_dict.items()
            if agent_id != exclude_id
        }

    def get_positions_dict(self, exclude_id: int = None) -> dict[int, np.ndarray]:
        """
        Returns a dictionary mapping agent IDs to their positions.
        If `exclude_id` is provided, that agent will be excluded.
        """
        return {
            agent_id: agent.position
            for agent_id, agent in self._agents_dict.items()
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
            for agent in self._agents_dict.values()
            if 0.0 < np.linalg.norm(agent.position - position) < distance
        ]
        return np.array(positions)
    
    def get_id(self, index: int) -> int:
        return self._index_to_id[index]

    def get_index(self, agent_id: int) -> int:
        """
        Returns the contiguous array index for a given global agent ID.
        """
        return self._id_to_index[agent_id]

    def get_indices(self, agent_ids: list[int]) -> np.ndarray:
        """
        Given a list or array of global agent IDs, returns an array of their
        contiguous indices.
        """
        try:
            return np.array([self._id_to_index[agent_id] for agent_id in agent_ids])
        except KeyError as e:
            raise KeyError(f"Agent ID {e.args[0]} is not registered.")

    def initialize(self, states: np.ndarray, time: float = 0.0) -> None:
        for i, agent in enumerate(self._agents_dict.values()):
            agent.initialize(states[i], time)

    def update(self, dt: float = 0.01) -> None:
        for i, agent in enumerate(self._agents_dict.values()):
            agent.update(dt)

    def __iter__(self):
        return iter(self._agent_list)

    def __len__(self):
        return len(self._agent_list)

    def __getitem__(self, index: int) -> Agent:
        return self._agent_list[index]
