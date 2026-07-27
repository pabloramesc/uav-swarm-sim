from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..mobility.position_controller import ControllerContext, PositionController
from .agent import Agent
from .dynamics import PointMassDynamics

if TYPE_CHECKING:
    from ..environment.environment import Environment
    from ..network.swarm_link import SwarmLink
    from .neighbor_provider import NeighborProvider


class Drone(Agent):
    """Represents a Drone (or UAV) in the simulation."""

    def __init__(
        self,
        agent_id: int,
        env: Environment,
        controller: PositionController,
        provider: NeighborProvider | None = None,
        swarm_link: SwarmLink | None = None,
    ):
        self.dynamics = PointMassDynamics(
            mass=1.0,  # 1 kg for simplified equivalence force = acceleration
            max_acc=10.0,  # aprox. 1 G = 9.81 m/s^2
        )
        super().__init__(
            agent_id=agent_id,
            agent_type="drone",
            dynamics=self.dynamics,
            environment=env,
        )

        self.position_controller = controller
        self.neighbor_provider = provider
        self.swarm_link = swarm_link

        self.drone_positions: dict[int, np.ndarray] | None = None
        self.user_positions: dict[int, np.ndarray] | None = None

    def initialize(
        self,
        state: np.ndarray,
        time: float = 0.0,
    ):
        """Initialize drone's state, neighbors info, and position controller.

        Args:
            state: Initial state array as [px, py, pz, vx, vy, vz], where
                - px, py, pz: Position in meters.
                - vx, vy, vz: Velocity in m/s.
            time: Current simulation time.
        """
        super().initialize(state, time)
        if self.swarm_link is not None:
            self.swarm_link.reset()
        self._update_neighbors()

        context = ControllerContext(
            time=self.time,
            agent_state=self.dynamics.state,
            target_position=None,
            drone_positions=self.drone_positions,
            user_positions=self.user_positions,
        )
        self.position_controller.initialize(context)

    def prepare_step(self, dt: float) -> None:
        """Freeze network and neighbor observations before any agent moves."""

        if self.swarm_link is not None:
            self.swarm_link.update(time=self.time, position=self.dynamics.state[0:3])
        self._update_neighbors()

    def update(self, dt: float = 0.01) -> None:
        """Updates the drone's state based on the prepared observations."""

        context = ControllerContext(
            time=self.time,
            agent_state=self.dynamics.state,
            target_position=None,
            drone_positions=self.drone_positions,
            user_positions=self.user_positions,
        )
        force = self.position_controller.update(context)

        self.dynamics.step(dt, control=force)
        self.time += float(dt)

    def _update_neighbors(self) -> None:
        if self.neighbor_provider is None:
            return
        self.drone_positions = self.neighbor_provider.get_drone_positions()
        self.user_positions = self.neighbor_provider.get_user_positions()
