"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

from simulator.environment.environment import Environment

AgentType = Literal["drone", "user", "gcs"]


class Agent(ABC):
    """
    Represents an agent in the simulation environment.

    Attributes
    ----------
    id : int
        Unique identifier for the agent.
    type : AgentType
        Type of the agent (e.g., "drone", "user", "gcs").
    time : float
        Simulation time for the agent.
    state : np.ndarray
        State of the agent [px, py, pz, vx, vy, vz], where:
        - px, py, pz: Position in meters.
        - vx, vy, vz: Velocity in m/s.
    environment : Environment
        The simulation environment the agent interacts with.
    """

    def __init__(self, id: int, type: AgentType, env: Environment):
        """
        Initializes an agent with a unique ID, type, and environment.

        Parameters
        ----------
        id : int
            Unique identifier for the agent.
        type : AgentType
            Type of the agent (e.g., "drone", "user", "gcs").
        env : Environment
            The simulation environment the agent interacts with.
        """
        self.id = id
        self.type = type
        self.time = 0.0
        self.state = np.zeros(6)  # px, py, pz, vx, vy, vz
        self.environment = env

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

    def initialize(self, state: np.ndarray) -> None:
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
        self.time = 0.0
        self.state = state

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

    def is_collision(self) -> bool:
        """
        Checks if the agent is in collision with any obstacle or the ground.

        Returns
        -------
        bool
            True if the agent is in collision, False otherwise.
        """
        return self.environment.is_collision(self.position)

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
        return f"Agent(id={self.id}, type='{self.type}', position={pos_str}, velocity={vel_str})"
