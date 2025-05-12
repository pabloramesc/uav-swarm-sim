"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np

from ..environment import Environment
from ..mobility.random_walker import SurfaceRandomWalker
from ..network.network_simulator import NetworkSimulator
from ..network.swarm_link import SwarmLink
from .agent import Agent, AgentType


class User(Agent):
    """
    Represents a user agent in the simulation environment.

    The user agent performs a random walk within the environment.
    """

    def __init__(self, agent_id: int, env: Environment, network_sim: NetworkSimulator):
        """
        Initializes the user agent with a unique ID, maximum speed, and maximum acceleration.

        """
        super().__init__(agent_id=agent_id, agent_type="user", env=env)

        self.swarm_link = None
        if network_sim is not None:
            self.swarm_link = SwarmLink(
                agent_id=self.agent_id,
                network_sim=network_sim,
                global_bcast_interval=1.0,
                local_bcast_interval=None,
                ack_to_messages=True,
            )

        self.random_walk = SurfaceRandomWalker(env)
        

    def initialize(self, state: np.ndarray, time: float = 0.0) -> None:
        super().initialize(state, time)
        self.random_walk.initialize(self.state)
        self.next_tx_msg: float = 0.0
        self.last_msg_id: int = None

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

        if self.swarm_link is not None:
            self.swarm_link.update(self.time, self.position)
            self._send_random_message()

    def _send_random_message(self) -> None:
        if self.time < self.next_tx_msg:
            return
        
        dst_addr = self.swarm_link.global_bcast_addr
        self.last_msg_id = self.swarm_link.send_message(f"Hello from agent {self.agent_id}!", dst_addr)
        
        self.next_tx_msg = self.time + np.random.uniform(1.0, 10.0)
        
    def _read_responses(self) -> None:
        self.swarm_link.ack_registry[self.last_msg_id]
