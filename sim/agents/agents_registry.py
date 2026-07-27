from __future__ import annotations

import numpy as np

from .agent import Agent


class AgentsRegistry:
    """
    A flexible registry for tracking agent instances and their states.
    Supports lookup by agent ID or index.
    """

    def __init__(self):
        self._agents_dict: dict[int, Agent] = {}
        self._id_to_index: dict[int, int] = {}
        self._index_to_id: dict[int, int] = {}

    @property
    def size(self) -> int:
        """Returns the number of registered agents."""
        return len(self._agents_dict)

    def _clear(self) -> None:
        """Remove all agents.

        Registry structure is mutated only by :class:`AgentsManager`, which
        keeps the typed and global views in sync.
        """

        self._agents_dict.clear()
        self._id_to_index.clear()
        self._index_to_id.clear()

    def _rebuild_index_mapping(self) -> None:
        """Rebuilds the ID-to-index mapping whenever the registry changes."""
        self._id_to_index = {
            agent_id: idx for idx, agent_id in enumerate(self._agents_dict.keys())
        }
        self._index_to_id = {
            idx: agent_id for idx, agent_id in enumerate(self._agents_dict.keys())
        }

    def _register(self, agent: Agent) -> None:
        """Adds a new agent to the registry."""
        if agent.agent_id in self._agents_dict:
            raise ValueError(f"Agent with ID {agent.agent_id} is already registered.")
        self._agents_dict[agent.agent_id] = agent
        self._rebuild_index_mapping()

    def _unregister(self, agent_id: int) -> None:
        """Removes an agent from the registry."""
        if agent_id not in self._agents_dict:
            raise KeyError(f"Agent with ID {agent_id} is not registered.")
        del self._agents_dict[agent_id]
        self._rebuild_index_mapping()

    def get_all(self) -> list[Agent]:
        """Returns all registered agents."""
        return list(self._agents_dict.values())

    def __contains__(self, agent_id: object) -> bool:
        return agent_id in self._agents_dict

    def get_agent(self, agent_id: int) -> Agent:
        """Returns the agent with the specified global ID."""
        return self._agents_dict[agent_id]

    def get_state(self, agent_id: int) -> np.ndarray:
        """Returns the state of the agent with the specified global ID."""
        return self._agents_dict[agent_id].state

    def get_states_array(self, exclude_id: int | None = None) -> np.ndarray:
        """Returns an array of all agent states.
        If `exclude_id` is provided, excludes the agent with that ID.
        """
        states = np.array(
            [
                agent.state
                for agent_id, agent in self._agents_dict.items()
                if agent_id != exclude_id
            ]
        )
        return states if states.shape[0] > 0 else np.zeros((0, 6))

    def get_states_dict(self, exclude_id: int | None = None) -> dict[int, np.ndarray]:
        """Returns a dictionary mapping agent IDs to their states.
        If `exclude_id` is provided, that agent will be excluded.
        """
        return {
            agent_id: agent.state.copy()
            for agent_id, agent in self._agents_dict.items()
            if agent_id != exclude_id
        }

    def get_positions_dict(
        self, exclude_id: int | None = None
    ) -> dict[int, np.ndarray]:
        """Returns a dictionary mapping agent IDs to their positions.
        If `exclude_id` is provided, that agent will be excluded.
        """
        return {
            agent_id: agent.position.copy()
            for agent_id, agent in self._agents_dict.items()
            if agent_id != exclude_id
        }

    def get_near_positions(
        self, position: np.ndarray, distance: float = 100.0
    ) -> np.ndarray:
        """Returns the positions of agents within a given distance from `position`.
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
        """Returns the contiguous array index for a given global agent ID."""
        return self._id_to_index[agent_id]

    def get_indices(self, agent_ids: list[int]) -> np.ndarray:
        """Given a list or array of global agent IDs, returns an array of their
        contiguous indices.
        """
        try:
            return np.array([self._id_to_index[agent_id] for agent_id in agent_ids])
        except KeyError as exc:
            raise KeyError(f"Agent ID {exc.args[0]} is not registered.") from exc

    def __iter__(self):
        return iter(self._agents_dict.values())

    def __len__(self):
        return len(self._agents_dict)

    def __getitem__(self, index: int) -> Agent:
        return tuple(self._agents_dict.values())[index]
