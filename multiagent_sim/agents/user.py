from typing import Optional

import numpy as np

from ..environment import Environment
from ..network.swarm_link import SwarmLink
from .agent import Agent
from .dynamics.random_walker import RandomWalkerDynamics


class User(Agent):
    """Represents a user agent in the simulation environment."""

    def __init__(
        self, agent_id: int, env: Environment, swarm_link: Optional[SwarmLink] = None
    ):
        self.dynamics = RandomWalkerDynamics(
            env=env, min_speed=1.0, max_speed=3.0, climb_rate=0.2, turning_rate=0.3
        )
        super().__init__(
            agent_id=agent_id,
            agent_type="user",
            dynamics=self.dynamics,
            environment=env,
        )

        self.swarm_link = swarm_link

    def initialize(self, state: np.ndarray, time: float = 0.0) -> None:
        super().initialize(state, time)
        self.next_tx_msg: float = 0.0

    def update(self, dt: float = 0.01) -> None:
        """Updates the state of the user agent by performing a random walk.

        Args:
            dt: The time step in seconds.
        """
        self._update_swarm_link()

        # super().update(dt)
        self.state = self.dynamics.step(dt, control=np.zeros(3))

    def _update_swarm_link(self) -> None:
        if self.swarm_link is None:
            return
        
        self.swarm_link.update(self.time, position=self.dynamics.position)
        
        # Send broadcast message and schedule next message if needed
        if self.time >= self.next_tx_msg:        
            self.send_broadcast_message()
            self.next_tx_msg = self.time + np.random.uniform(1.0, 10.0)
            
        self.print_received_messages(clear=True)
        
    def send_broadcast_message(self) -> None:
        if self.swarm_link is None:
            raise RuntimeError("No swarm link was provided.")

        dst_addr = self.swarm_link.iface.broadcast_address
        msg = f"Hello from agent {self.agent_id}!"
        self.last_msg_id = self.swarm_link.send_message(msg, dst_addr)

        self.logger.debug(f"Sent msg: {msg}")

    def print_received_messages(self, clear: bool = False) -> None:
        if self.swarm_link is None:
            raise RuntimeError("No swarm link was provided.")

        for msg in self.swarm_link.get_messages(clear):
            self.logger.debug(f"Received from {msg.source_id} msg: {msg.txt}")
