from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .agent import Agent
from .dynamics.random_walker import RandomWalkerDynamics

if TYPE_CHECKING:
    from ..environment import Environment
    from ..network.swarm_link import SwarmLink

logger = logging.getLogger(__name__)


class User(Agent):
    """Represents a user agent in the simulation environment."""

    def __init__(
        self,
        agent_id: int,
        env: Environment,
        swarm_link: SwarmLink | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.dynamics = RandomWalkerDynamics(
            env=env,
            min_speed=1.0,
            max_speed=3.0,
            climb_rate=0.2,
            turning_rate=0.3,
            rng=self.rng,
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
        if self.swarm_link is not None:
            self.swarm_link.reset()
        self.next_tx_msg: float = 0.0
        self.last_msg_id: int | None = None

    def update(self, dt: float = 0.01) -> None:
        """Updates the state of the user agent by performing a random walk.

        Args:
            dt: The time step in seconds.
        """
        super().update(dt)

    def prepare_step(self, dt: float) -> None:
        self._update_swarm_link()

    def _update_swarm_link(self) -> None:
        if self.swarm_link is None:
            return

        self.swarm_link.update(self.time, position=self.dynamics.position)

        # Send broadcast message and schedule next message if needed
        if self.time >= self.next_tx_msg:
            self.send_broadcast_message()
            self.next_tx_msg = self.time + float(self.rng.uniform(1.0, 10.0))

        self.print_received_messages(clear=True)

    def send_broadcast_message(self) -> None:
        if self.swarm_link is None:
            raise RuntimeError("No swarm link was provided.")

        dst_addr = self.swarm_link.iface.broadcast_address
        msg = f"Hello from agent {self.agent_id}!"
        self.last_msg_id = self.swarm_link.send_message(msg, dst_addr)

        logger.debug(f"Sent msg: {msg}")

    def print_received_messages(self, clear: bool = False) -> None:
        if self.swarm_link is None:
            raise RuntimeError("No swarm link was provided.")

        for msg in self.swarm_link.get_messages(clear):
            logger.debug(f"Received from {msg.source_id} msg: {msg.txt}")
