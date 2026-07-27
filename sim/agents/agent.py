from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING, Literal

import numpy as np

from .dynamics import Dynamics

if TYPE_CHECKING:
    from ..environment.environment import Environment

AgentType = Literal["drone", "user", "gcs"]
_AGENT_TYPES = frozenset(("drone", "user", "gcs"))


class Agent:
    """Base class for all agents in the simulation."""

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
        if isinstance(agent_id, bool) or not isinstance(agent_id, Integral):
            raise TypeError("agent_id must be an integer.")
        if agent_id < 0:
            raise ValueError("agent_id cannot be negative.")
        if agent_type not in _AGENT_TYPES:
            raise ValueError(f"Invalid agent type descriptor '{agent_type}'.")

        self.agent_id = int(agent_id)
        self.agent_type = agent_type
        self.dynamics = dynamics
        self.environment = environment

        self.time = 0.0

    @property
    def state(self) -> np.ndarray:
        return self.dynamics.state

    @property
    def position(self) -> np.ndarray:
        return self.dynamics.position

    @property
    def velocity(self) -> np.ndarray:
        return self.dynamics.velocity

    def initialize(self, state: np.ndarray, time: float = 0.0) -> None:
        """Initializes the agent's state and simulation time.

        Args:
            state: Initial agent state with dynamics appropriate shape.
            time: Current simulation time in seconds.
        """
        self.dynamics.state = state
        self.time = float(time)

    def prepare_step(self, dt: float) -> None:
        """Capture inputs that must be shared by every agent in this tick."""

    def update(self, dt: float = 0.01) -> None:
        """
        Updates the simulation time for the agent.

        Args:
            dt: Simulation time step in seconds.
        """
        self.dynamics.step(dt, control=np.zeros(self.dynamics.input_shape))
        self.time += float(dt)

    def is_collision(self, check_altitude: bool = True) -> bool:
        """Checks if the agent is in collision with any obstacle or the ground.

        Returns:
            True if the agent is in collision, False otherwise.
        """
        return self.environment.is_collision(
            self.dynamics.position, check_altitude
        ).item()

    def is_inside(self) -> bool:
        """Checks if the agent is inside the environment boundary.

        Returns:
            True if the agent is inside the boundary, False otherwise.
        """
        return self.environment.is_inside(self.dynamics.position).item()

    def __repr__(self) -> str:
        pos = self.dynamics.position
        vel = self.dynamics.velocity
        pos_str = f"[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] m"
        vel_str = f"[{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}] m/s"
        return (
            f"Agent(id={self.agent_id}, type='{self.agent_type}', "
            f"position={pos_str}, velocity={vel_str})"
        )
