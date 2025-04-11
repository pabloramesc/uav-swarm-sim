"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np
from ..environment import Environment
from ..swarming import PositionController
from .agent import Agent


class Drone(Agent):
    """
    Represents a Drone (or drone) in the simulation.

    This class models the behavior of a drone, including its dynamics, neighbor interactions,
    and position control using a provided position controller.

    Attributes
    ----------
    position_controller : PositionController
        The position controller used to compute control forces for the drone.
    neighbor_ids : np.ndarray
        Array of IDs of neighboring drones.
    neighbor_states : np.ndarray
        Array of states of neighboring drones, where each state is [px, py, pz, vx, vy, vz].
    max_acc : float, optional
        Maximum allowable acceleration for the drone (default is None, meaning no limit).
    """

    def __init__(
        self,
        id: int,
        env: Environment,
        position_controller: PositionController,
        max_acc: float = None,
    ):
        """
        Initializes the drone with a unique ID, environment, and position controller.

        Parameters
        ----------
        id : int
            Unique identifier for the drone.
        env : Environment
            The simulation environment the drone interacts with.
        position_controller : PositionController
            The position controller used to compute control forces for the drone.
        max_acc : float, optional
            Maximum allowable acceleration for the drone in m/s^2 (default is None, meaning no limit).
        """
        super().__init__(id=id, type="drone", env=env)
        self.position_controller = position_controller
        self.neighbor_ids = np.zeros((0,), dtype=int)
        self.neighbor_states = np.zeros((0, 6))
        self.max_acc = max_acc

    def update(self, dt: float = 0.01) -> None:
        """
        Updates the drone's state based on the control forces and dynamics.

        Parameters
        ----------
        dt : float, optional
            Time step in seconds (default is 0.01).
        """
        # Compute control force using the position controller
        control_force = self.position_controller.update(
            self.state, self.neighbor_states, self.time
        )

        # Limit the acceleration to the maximum allowable value
        acc = self._limit_acceleration(control_force / self.mass)

        # Compute the state derivative using the dynamics model
        x_dot = self._compute_dynamics(self.state, acc)

        # Update the drone's state
        self.state += x_dot * dt

    def set_neighbors(self, ids: np.ndarray, states: np.ndarray) -> None:
        """
        Sets the neighboring drones' IDs and states.

        Parameters
        ----------
        ids : np.ndarray
            Array of IDs of neighboring drones.
        states : np.ndarray
            Array of states of neighboring drones, where each state is [px, py, pz, vx, vy, vz].
        """
        self.neighbor_ids = np.copy(ids)
        self.neighbor_states = np.copy(states)

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
