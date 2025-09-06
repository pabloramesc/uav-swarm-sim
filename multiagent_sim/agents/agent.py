from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..environment.environment import Environment
from ..utils.logger import create_logger

from .dynamics import Dynamics

AgentType = Literal["drone", "user", "gcs"]


class Agent:
    """Base class for all agents in the simulation."""

    _agent_ids: set[int] = set()

    def __init__(
        self,
        agent_id: int,
        agent_type: AgentType,
        dynamics: Dynamics,
        environment: Environment,
    ):
        """Initializes an agent with a unique ID, type, and environment.

        Args:
            agent_id: Unique identifier of agent.
            agent_type:
            dynamics:
            environment:
        """
        if agent_id in self._agent_ids:
            raise ValueError(f"Agent ID {agent_id} is already taken.")

        if agent_type not in AgentType.__args__:
            raise ValueError(f"Invalid agent type descriptor '{agent_type}'.")

        self.agent_id = int(agent_id)
        self.agent_type = agent_type
        self.dynamics = dynamics
        self.environment = environment

        self.logger = create_logger(f"Agent{agent_id}", level="DEBUG")

        self.time = 0.0

    @classmethod
    def reset_ids(self):
        self._agent_ids.clear()

    def initialize(self, state: np.ndarray, time: float = 0.0) -> None:
        """Initializes the agent's state and simulation time.

        Args:
            state: Initial agent state with dynamics appropriate shape.
            time: Current simulation time in seconds.
        """
        self.dynamics.initialize(state)
        self.time = float(time)

    def update(self, dt: float = 0.01, **kwargs) -> None:
        """
        Updates the simulation time for the agent.

        Args:
            dt: Simulation time step in seconds.
        """
        self.dynamics.step(dt, **kwargs)
        self.time += float(dt)

    def is_collision(self, check_altitude: bool = True) -> bool:
        """Checks if the agent is in collision with any obstacle or the ground.

        Returns:
            True if the agent is in collision, False otherwise.
        """
        return self.environment.is_collision(self.position, check_altitude)

    def is_inside(self) -> bool:
        """Checks if the agent is inside the environment boundary.

        Returns:
            True if the agent is inside the boundary, False otherwise.
        """
        return self.environment.is_inside(self.position)

    def __repr__(self) -> str:
        pos = self.position
        vel = self.velocity
        pos_str = f"[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] m"
        vel_str = f"[{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}] m/s"
        return f"Agent(id={self.agent_id}, type='{self.agent_type}', position={pos_str}, velocity={vel_str})"


@dataclass
class AgentFactory(ABC):
    env: Environment

    @abstractmethod
    def create(self, agent_id: int) -> Agent:
        pass
