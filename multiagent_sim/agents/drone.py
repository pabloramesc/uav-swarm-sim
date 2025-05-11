"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np

from typing import Literal

from ..environment.environment import Environment
from ..mobility.base_swarming import SwarmingController
from ..network.swarm_link import SwarmLink
from .agent import Agent
from .agents_registry import AgentsRegistry

NeighborProvider = Literal["network", "registry"]


class Drone(Agent):
    """
    Represents a Drone (or UAV) in the simulation.
    """

    def __init__(
        self,
        agent_id: int,
        env: Environment,
        swarming: SwarmingController,
        link: SwarmLink = None,
        drones_registry: AgentsRegistry = None,
        users_registry: AgentsRegistry = None,
        neighbor_provider: NeighborProvider = "registry",
    ):
        super().__init__(agent_id=agent_id, agent_type="drone", env=env)
        self.link = link
        self.swarming = swarming
        self.drones_registry = drones_registry
        self.users_registry = users_registry
        self.neighbor_provider = neighbor_provider

        if self.neighbor_provider == "network" and self.link is None:
            raise ValueError(
                "If neighbor_provider is 'network', a network object must be provided."
            )

        if self.neighbor_provider == "registry" and drones_registry is None:
            raise ValueError(
                "If neighbor_provider is 'registry', drones_registry must be provided."
            )

        self.drone_positions: dict[int, np.ndarray] = None
        self.user_positions: dict[int, np.ndarray] = None

        self.mass = 1.0  # 1 kg for simple equivalence between force and acceleration
        self.max_acc = 10.0  # aprox. 1 g = 9.81 m/s^2

    def initialize(
        self,
        state: np.ndarray,
        time: float = 0.0,
    ):
        super().initialize(state, time)
        self._update_neighbors()
        self.swarming.initialize(time, state, drone_positions=self.drone_positions)

    def update(self, dt: float = 0.01) -> None:
        """
        Updates the drone's state based on the control forces and dynamics.

        Parameters
        ----------
        dt : float, optional
            Time step in seconds (default is 0.01).
        """
        super().update(dt)

        if self.link is not None:
            self.link.update(self.time, self.state[0:3])

        self._update_neighbors()

        # Compute control force using the position controller
        control_force = self.swarming.update(
            time=self.time, state=self.state, drone_positions=self.drone_positions
        )

        # Limit the acceleration to the maximum allowable value
        acc = self._limit_acceleration(control_force / self.mass)

        # Compute the state derivative using the dynamics model
        x_dot = self._compute_dynamics(self.state, acc)

        # Update the drone's state
        self.state += x_dot * dt

    def _compute_dynamics(self, state: np.ndarray, acc: np.ndarray) -> np.ndarray:
        """
        Computes the state derivative based on the drone's dynamics.

        Parameters
        ----------
        state : np.ndarray
            Current state of the drone [px, py, pz, vx, vy, vz].
        acc : np.ndarray
            Acceleration vector [ax, ay, az] in m/s^2.

        Returns
        -------
        np.ndarray
            State derivative [vx, vy, vz, ax, ay, az].
        """
        x_dot = np.zeros(6)
        x_dot[0:3] = state[3:6]  # dx/dt = velocity
        x_dot[3:6] = acc  # dv/dt = acceleration
        return x_dot

    def _limit_acceleration(self, acc: np.ndarray) -> np.ndarray:
        """
        Limits the acceleration to the maximum allowable value.

        Parameters
        ----------
        acc : np.ndarray
            Acceleration vector [ax, ay, az] in m/s^2.

        Returns
        -------
        np.ndarray
            Limited acceleration vector [ax, ay, az].
        """
        if self.max_acc is None:
            return acc

        acc_mag = np.linalg.norm(acc)
        if acc_mag > 0.0:
            acc_dir = acc / acc_mag  # Normalize acceleration direction
        else:
            acc_dir = np.zeros(3)

        return acc_dir * min(acc_mag, self.max_acc)

    def _update_neighbors(self) -> None:
        """
        Updates the neighbor states based on the selected provider.
        """
        if self.neighbor_provider == "network":
            self.drone_positions = self.link.get_drone_positions()
            self.user_positions = self.link.get_user_positions()

        elif self.neighbor_provider == "registry":
            self.drone_positions = self.drones_registry.get_positions_dict(
                exclude_id=self.agent_id
            )
            self.user_positions = self.users_registry.get_positions_dict(
                exclude_id=self.agent_id
            )

        else:
            raise ValueError("Invalid neighbor provider specified.")
