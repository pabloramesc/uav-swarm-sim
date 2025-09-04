from dataclasses import dataclass
import numpy as np

from ..environment.environment import Environment
from ..mobility.swarm_position_controller import SwarmPositionController
from ..network.swarm_link import SwarmLink
from .agent import Agent, AgentFactory
from .neighbor_provider import NeighborProvider


class Drone(Agent):
    """Represents a Drone (or UAV) in the simulation."""

    def __init__(
        self,
        agent_id: int,
        environment: Environment,
        controller: SwarmPositionController,
        provider: NeighborProvider,
        swarm_link: SwarmLink = None,
        mass: float = 1.0,  # 1 kg for simple equivalence between force and acceleration
        max_acc: float = 10.0,  # aprox. 1 g = 9.81 m/s^2
    ):
        super().__init__(agent_id=agent_id, agent_type="drone", environment=environment)
        self.position_controller = controller
        self.neighbor_provider = provider
        self.swarm_link = swarm_link
        self.mass = float(mass)
        self.max_acc = float(max_acc)

        self.drone_positions: dict[int, np.ndarray] = {}
        self.user_positions: dict[int, np.ndarray] = {}

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
        self._initialize_state(state, time)

        self._update_neighbors()
        self.position_controller.initialize(
            time,
            state,
            drone_positions=self.drone_positions,
            user_positions=self.user_positions,
        )

    def update(self, dt: float = 0.01) -> None:
        """Updates the drone's state based on the control forces and dynamics.

        Args:
            dt: Time step in seconds.
        """
        self._advance_time(dt)

        if self.swarm_link is not None:
            self.swarm_link.update(self.time, self.state[0:3])

        self._update_neighbors()
        acc = self._control_acceleration()
        self._integrate_dynamics(acc, dt)

    def _control_acceleration(self) -> np.ndarray:
        force = self.position_controller.update(
            time=self.time,
            state=self.state,
            drone_positions=self.drone_positions,
            user_positions=self.user_positions,
        )
        return self._limit_acceleration(force / self.mass)

    def _integrate_dynamics(self, acc: np.ndarray, dt: float) -> None:
        x_dot = np.zeros(6)
        x_dot[0:3] = self.state[3:6]  # dx/dt = velocity
        x_dot[3:6] = acc  # dv/dt = acceleration
        self.state += x_dot * dt

    def _limit_acceleration(self, acc: np.ndarray) -> np.ndarray:
        if self.max_acc is None:
            return acc
        mag = np.linalg.norm(acc)
        return (acc / mag) * min(mag, self.max_acc) if mag > 0.0 else acc

    def _update_neighbors(self) -> None:
        self.drone_positions = self.neighbor_provider.get_drone_positions()
        self.user_positions = self.neighbor_provider.get_user_positions()


@dataclass
class DroneFactory(AgentFactory):

    def create(self, agent_id: int):
        return Drone(
            agent_id=agent_id,
            environment=self.env,
        )
