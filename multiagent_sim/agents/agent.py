"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

from ..environment.environment import Environment
from ..network.swarm_interface import SwarmProtocolInterface

AgentType = Literal["drone", "user", "gcs"]


class Agent(ABC):
    """
    Represents an agent in the simulation environment.
    """

    def __init__(
        self,
        agent_id: int,
        agent_type: AgentType,
        env: Environment,
        net: SwarmProtocolInterface = None,
    ):
        """
        Initializes an agent with a unique ID, type, and environment.
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.environment = env
        self.network = net

        self.time = 0.0
        self.state = np.zeros(6)  # px, py, pz, vx, vy, vz

    @property
    def position(self) -> np.ndarray:
        """
        Position of the agent [px, py, pz] in meters.
        """
        return self.state[0:3]

    @property
    def velocity(self) -> np.ndarray:
        """
        Velocity of the agent [vx, vy, vz] in m/s.
        """
        return self.state[3:6]

    @abstractmethod
    def initialize(self, state: np.ndarray, time: float = 0.0) -> None:
        """
        Initializes the state of the agent.

        Parameters
        ----------
        state : np.ndarray
            Initial state [px, py, pz, vx, vy, vz], where:
            - px, py, pz: Position in meters.
            - vx, vy, vz: Velocity in m/s.
        """
        self._check_state(state)
        self.state = np.copy(state)
        self.time = time

    @abstractmethod
    def update(self, dt: float = 0.01) -> None:
        """
        Updates the simulation time for the agent.

        Parameters
        ----------
        dt : float, optional
            Time step in seconds (default is 0.01).
        """
        self.time += dt

    def is_collision(self, check_altitude: bool = True) -> bool:
        """
        Checks if the agent is in collision with any obstacle or the ground.

        Returns
        -------
        bool
            True if the agent is in collision, False otherwise.
        """
        return self.environment.is_collision(self.position, check_altitude)

    def is_inside(self) -> bool:
        """
        Checks if the agent is inside the environment boundary.

        Returns
        -------
        bool
            True if the agent is inside the boundary, False otherwise.
        """
        return self.environment.is_inside(self.position)

    def _check_state(self, state: np.ndarray) -> None:
        """
        Validates the state array.

        Parameters
        ----------
        state : np.ndarray
            State array to validate.

        Raises
        ------
        ValueError
            If the state is not a numpy array or does not have the correct shape.
        """
        if not isinstance(state, np.ndarray):
            raise ValueError("State must be a numpy array")
        if state.shape != (6,):
            raise ValueError("State must be a 1D array of shape (6,)")

    def __repr__(self) -> str:
        """
        Returns a string representation of the agent.

        Returns
        -------
        str
            String representation of the agent.
        """
        pos = self.position
        vel = self.velocity
        pos_str = f"[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] m"
        vel_str = f"[{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}] m/s"
        return f"Agent(id={self.agent_id}, type='{self.agent_type}', position={pos_str}, velocity={vel_str})"
