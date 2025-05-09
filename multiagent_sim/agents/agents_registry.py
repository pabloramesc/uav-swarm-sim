import numpy as np
from abc import ABC
from .agent import Agent
from .drone import Drone
from .user import User


class AgentsRegistry(ABC):
    """Tracks agents and provides neighbor queries."""

    def __init__(self):
        self.agents: list[Agent] = []
        self.states: list[np.ndarray] = []

    def register(self, agent: Agent) -> None:
        self.agents.append(agent)
        self.states.append(agent.state)

    def get_all(self) -> list[Agent]:
        return self.agents

    def get_agent(self, idx: int) -> Agent:
        return self.agents[idx]

    def get_state(self, idx: int) -> np.ndarray:
        return self.states[idx]

    def get_states(self, agent_id: int = None) -> np.ndarray:
        agent_states = np.array(self.states)
        if agent_id is not None:
            agent_states = np.delete(agent_states, agent_id, axis=0)
        return agent_states

    def get_near_positions(
        self, position: np.ndarray, distance: float = 100.0
    ) -> np.ndarray:
        agent_positions = np.array(self.states)[:, 0:3]
        distances = np.linalg.norm(agent_positions - position, axis=1)
        in_range = (0.0 < distances) & (distances < distance)
        return agent_positions[in_range]


class DronesRegistry(AgentsRegistry):
    """Tracks drones and provides neighbor queries."""

    def __init__(self):
        super().__init__()
        self.drones: list[Drone] = []

    def register(self, drone: Drone) -> None:
        super().register(drone)
        self.drones.append(drone)

    def get_all(self) -> list[Drone]:
        return self.drones

    def get_drone(self, idx: int) -> Drone:
        return self.drones[idx]


class UsersRegistry(AgentsRegistry):
    """Tracks users and provides spatial queries like proximity search."""

    def __init__(self):
        super().__init__()
        self.users: list[User] = []

    def register(self, user: User) -> None:
        super().register(user)
        self.users.append(user)

    def get_all(self) -> list[User]:
        return self.users

    def get_user(self, idx: int) -> User:
        return self.users[idx]
