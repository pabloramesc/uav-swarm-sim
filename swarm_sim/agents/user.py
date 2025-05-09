"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np

from ..environment import Environment
from ..mobility.random_walk import SurfaceRandomWalker
from ..network.network_interface import NetworkInterface, SimPacket
from ..network.network_packets import PositionPacket
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
        net: NetworkInterface = None,
        tx_interval: float = 1.0,
    ):
        """
        Initializes the user agent with a unique ID, maximum speed, and maximum acceleration.

        """
        super().__init__(
            global_id=global_id, type_id=type_id, agent_type="user", env=env, net=net
        )
        self.random_walk = SurfaceRandomWalker(env)
        
        self.tx_interval = tx_interval
        self.last_tx_time = 0.0
        self.next_tx_time = 0.0 + self.tx_interval

    def initialize(self, state: np.ndarray, time: float = 0.0) -> None:
        super().initialize(state, time)
        self.random_walk.initialize(self.state)
        self.last_tx_time = time
        self.next_tx_time = time + self.tx_interval * np.random.normal(1.0, 0.1)

    def update(self, dt: float = 0.01) -> None:
        """
        Updates the state of the user agent by performing a random walk.

        Parameters
        ----------
        dt : float, optional
            The time step in seconds (default is 0.01).
        """
        super().update(dt)
        self.state = self.random_walk(dt)

        if self.time >= self.next_tx_time:
            self.broadcast_position()
            self.last_tx_time = self.time
            self.next_tx_time = self.time + self.tx_interval * np.random.normal(
                1.0, 0.1
            )

    def broadcast_position(self) -> None:
        payload = PositionPacket()
        payload.set_packet_id(self.global_id, self.network.tx_packet_counter)
        payload.set_timestamp(self.time)
        payload.set_position(self.position)
        packet = SimPacket(
            node_id=self.global_id,
            src_addr=self.network.get_ip(),
            dst_addr="10.0.255.255",
            data=payload.serialize(),
        )
        self.network.send(packet)
