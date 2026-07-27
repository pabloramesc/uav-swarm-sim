from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .agent import Agent
from .dynamics import StaticDynamics

if TYPE_CHECKING:
    from ..environment import Environment
    from ..network.swarm_link import SwarmLink


class ControlStation(Agent):
    """
    Represents a control station (Ground Control Station, GCS) in the simulation environment.

    The control station is responsible for monitoring and managing other agents in the simulation.
    It does not move or perform random walks like other agents but can update its internal state.

    Attributes
    ----------
    id : int
        Unique identifier for the control station.
    """

    def __init__(
        self,
        agent_id: int,
        env: Environment,
        swarm_link: SwarmLink | None = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="gcs",
            dynamics=StaticDynamics(),
            environment=env,
        )

        self.swarm_link = swarm_link

    def initialize(self, state: np.ndarray, time: float = 0.0) -> None:
        super().initialize(state, time)
        if self.swarm_link is not None:
            self.swarm_link.reset()

    def prepare_step(self, dt: float) -> None:
        if self.swarm_link is not None:
            self.swarm_link.update(time=self.time, position=self.dynamics.position)

    def update(self, dt: float = 0.01) -> None:
        """Advance the control station's clock without moving it."""

        super().update(dt)
