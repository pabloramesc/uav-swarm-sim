"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np

from typing import Literal

from ..environment.environment import Environment
from ..mobility.base_swarming import SwarmingController
from ..network.swarm_interface import SwarmProtocolInterface
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
        network: SwarmProtocolInterface = None,
        drones_registry: AgentsRegistry = None,
        users_registry: AgentsRegistry = None,
        neighbor_provider: NeighborProvider = "registry",
    ):
        super().__init__(
            agent_id=agent_id, agent_type="drone", env=env, net=network
        )
        self.swarming = swarming
        self.drones_registry = drones_registry
        self.users_registry = users_registry
        self.neighbor_provider = neighbor_provider

        if self.neighbor_provider == "network" and network is None:
            raise ValueError(
                "If neighbor_provider is 'network', a network object must be provided."
            )

        if self.neighbor_provider == "registry" and drones_registry is None:
            raise ValueError(
                "If neighbor_provider is 'registry', drones_registry must be provided."
            )

        self.neighbor_drone_ids: np.ndarray = None
        self.neighbor_drone_positions: np.ndarray = None
        self.neighbor_user_positions: np.ndarray = None

        self.mass = 1.0  # 1 kg for simple equivalence between force and acceleration
        self.max_acc = 10.0  # aprox. 1 g = 9.81 m/s^2

    def initialize(
        self,
        state: np.ndarray,
        time: float = 0.0,
    ):
        super().initialize(state, time)
        self._update_neighbors()
        self.swarming.initialize(time, state, neighbor_positions=self.neighbor_drone_positions)

    def update(self, dt: float = 0.01) -> None:
        """
        Updates the drone's state based on the control forces and dynamics.

        Parameters
        ----------
        dt : float, optional
            Time step in seconds (default is 0.01).
        """
        super().update(dt)
        self._update_neighbors()
        
        if self.network is not None:
            self.network.update(self.time, self.state[0:3])

        # Compute control force using the position controller
        control_force = self.swarming.update(
            self.time, self.state, neighbor_positions=self.neighbor_drone_positions,
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
            drone_id_pos = self.network.get_drone_positions()
            self.neighbor_drone_ids = np.array(list(drone_id_pos.keys()))
            self.neighbor_drone_positions = np.array(list(drone_id_pos.values())) if drone_id_pos else np.zeros((0, 3))
            user_positions = self.network.get_user_positions()
            self.neighbor_user_positions = np.array(list(user_positions.values())) if user_positions else np.zeros((0, 3))

        elif self.neighbor_provider == "registry":
            # self.drones_positions = self.drones_registry.get_states(self.agent_id)[:, 0:3]
            drone_id_pos = self.drones_registry.get_states_dict(self.agent_id)
            self.neighbor_drone_ids = np.array(list(drone_id_pos.keys()))
            self.neighbor_drone_positions = np.array(list(drone_id_pos.values()))[
                :, 0:3
            ]
            if self.users_registry.num_agents > 0:
                self.neighbor_user_positions = self.users_registry.get_states_array()[:, 0:3]

        else:
            raise ValueError("Invalid neighbor provider specified.")
