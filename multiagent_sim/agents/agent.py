"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

from ..environment.environment import Environment
from ..utils.logger import create_logger

AgentType = Literal["drone", "user", "gcs"]


class Agent(ABC):
    """Represents an agent in the simulation environment."""

    _agent_ids: set[int] = set()

    def __init__(
        self,
        agent_id: int,
        agent_type: AgentType,
        environment: Environment,
    ):
        """Initializes an agent with a unique ID, type, and environment.

        Args:
            agent_id: Unique identifier of agent.
            agent_type:
            environment:
        """
        if agent_id in self._agent_ids:
            raise ValueError(f"Agent ID {agent_id} is already taken.")

        if agent_type not in AgentType.__args__:
            raise ValueError(f"Invalid agent type descriptor '{agent_type}'.")

        self.agent_id = int(agent_id)
        self.agent_type = agent_type
        self.environment = environment

        self.logger = create_logger(f"Agent{agent_id}", level="DEBUG")

        self.time = 0.0
        self.state = np.zeros(6)  # px, py, pz, vx, vy, vz

    @classmethod
    def reset_ids(self):
        self._agent_ids.clear()

    @property
    def position(self) -> np.ndarray:
        """Position of the agent [px, py, pz] in meters."""
        return self.state[0:3]

    @property
    def velocity(self) -> np.ndarray:
        """Velocity of the agent [vx, vy, vz] in m/s."""
        return self.state[3:6]

    @abstractmethod
    def initialize(self, state: np.ndarray, time: float = 0.0) -> None:
        """Initializes the agent's state and simulation time.

        Args:
            state: Initial state [px, py, pz, vx, vy, vz], where
                - px, py, pz: Position in meters.
                - vx, vy, vz: Velocity in m/s.
            time: Current simulation time
        """
        pass

    @abstractmethod
    def update(self, dt: float = 0.01) -> None:
        """
        Updates the simulation time for the agent.

        Args:
            dt: Simulation time step in seconds.
        """

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

    def _initialize_state(self, state: np.ndarray, time: float) -> None:
        self._check_state(state)
        self.state = np.copy(state)
        self.time = float(time)

    def _advance_time(self, dt: float) -> None:
        self.time += float(dt)

    def _check_state(self, state: np.ndarray) -> None:
        if not isinstance(state, np.ndarray):
            raise ValueError("State must be a numpy array")
        if state.shape != (6,):
            raise ValueError("State must be a 1D array of shape (6,)")

    def __repr__(self) -> str:
        pos = self.position
        vel = self.velocity
        pos_str = f"[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] m"
        vel_str = f"[{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}] m/s"
        return f"Agent(id={self.agent_id}, type='{self.agent_type}', position={pos_str}, velocity={vel_str})"
