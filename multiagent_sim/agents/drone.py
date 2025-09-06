from dataclasses import dataclass
import numpy as np

from ..environment.environment import Environment
from ..mobility.position_controller import PositionController
from ..network.swarm_link import SwarmLink
from .agent import Agent, AgentFactory
from .neighbor_provider import NeighborProvider
from .dynamics import PointMassDynamics


class Drone(Agent):
    """Represents a Drone (or UAV) in the simulation."""

    def __init__(
        self,
        agent_id: int,
        env: Environment,
        controller: PositionController = None,
        provider: NeighborProvider = None,
        swarm_link: SwarmLink = None,
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

        self.drone_positions: dict[int, np.ndarray] = None
        self.user_positions: dict[int, np.ndarray] = None

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

        self._update_neighbors()
        self.position_controller.initialize(
            time=time,
            state=state,
            drone_positions=self.drone_positions,
            user_positions=self.user_positions,
        )

    def update(self, dt: float = 0.01) -> None:
        """Updates the drone's state based on the control forces and dynamics.

        Args:
            dt: Time step in seconds.
        """
        if self.swarm_link is not None:
            self.swarm_link.update(self.time, self.state[0:3])

        self._update_neighbors()

        force = self.position_controller.update(
            time=self.time,
            state=self.state,
            drone_positions=self.drone_positions,
            user_positions=self.user_positions,
        )

        # super().update(dt, force=force)
        self.dynamics.step(dt, force=force)
        self.time += float(dt)

    def _update_neighbors(self) -> None:
        if self.neighbor_provider is None:
            return
        self.drone_positions = self.neighbor_provider.get_drone_positions()
        self.user_positions = self.neighbor_provider.get_user_positions()


@dataclass
class DroneFactory(AgentFactory):

    def create(self, agent_id: int):
        return Drone(
            agent_id=agent_id,
            env=self.env,
        )
