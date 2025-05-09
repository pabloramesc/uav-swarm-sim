"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np

from ..environment import Environment
from ..mobility.random_walk import SurfaceRandomWalker
from ..network.swarm_interface import SwarmProtocolInterface
from .agent import Agent, AgentType


class User(Agent):
    """
    Represents a user agent in the simulation environment.

    The user agent performs a random walk within the environment.
    """

    def __init__(
        self,
        global_id: int,
        type_id: int,
        env: Environment,
        network: SwarmProtocolInterface = None,
    ):
        """
        Initializes the user agent with a unique ID, maximum speed, and maximum acceleration.

        """
        super().__init__(
            global_id=global_id, type_id=type_id, agent_type="user", env=env, net=network
        )
        self.random_walk = SurfaceRandomWalker(env)

    def initialize(self, state: np.ndarray, time: float = 0.0) -> None:
        super().initialize(state, time)
        self.random_walk.initialize(self.state)
        if self.network:
            self.network.update(self.time, self.position)

    def update(self, dt: float = 0.01) -> None:
        """
        Updates the state of the user agent by performing a random walk.

        Parameters
        ----------
        dt : float, optional
            The time step in seconds (default is 0.01).
        """
        super().update(dt)
        self.state = self.random_walk.step(dt)
        if self.network:
            self.network.update(self.time, self.position)
