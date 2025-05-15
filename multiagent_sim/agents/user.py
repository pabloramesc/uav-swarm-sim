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
from .agent import Agent


class User(Agent):
    """
    Represents a user agent in the simulation environment.

    The user agent performs a random walk within the environment.
    """

    def __init__(
        self, agent_id: int, environment: Environment, network_sim: NetworkSimulator = None
    ):
        """
        Initializes the user agent with a unique ID, maximum speed, and maximum acceleration.

        """
        super().__init__(agent_id=agent_id, agent_type="user", environment=environment)

        self.swarm_link = None
        if network_sim is not None:
            self.swarm_link = SwarmLink(
                agent_id=self.agent_id,
                network_sim=network_sim,
                global_bcast_interval=1.0,
            )

        self.random_walk = SurfaceRandomWalker(environment)

    def initialize(self, state: np.ndarray, time: float = 0.0) -> None:
        super().initialize(state, time)
        self.random_walk.initialize(self.state)
        self.next_tx_msg: float = 0.0

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
            self.print_received_messages(clear=True)

    def _send_random_message(self) -> None:
        if self.time < self.next_tx_msg:
            return

        dst_addr = self.swarm_link.iface.broadcast_address
        msg = f"Hello from agent {self.agent_id}!"
        self.last_msg_id = self.swarm_link.send_message(msg, dst_addr)

        self.logger.debug(f"Sent msg: {msg}")

        self.next_tx_msg = self.time + np.random.uniform(1.0, 10.0)

    def print_received_messages(self, clear: bool = False) -> None:
        for msg in self.swarm_link.get_messages(clear):
            self.logger.debug(f"Received from {msg.source_id} msg: {msg.txt}")
