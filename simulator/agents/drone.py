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
    Represents a Drone (or UAV) in the simulation.

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
        """
        super().__init__(id=id, type="drone", env=env)
        self.position_controller = position_controller
        self.neighbor_ids = np.zeros((0,), dtype=int)
        self.neighbor_states = np.zeros((0, 6))

        self.mass = 1.0  # 1 kg for simple equivalence between force and acceleration
        self.max_acc = 10.0  # aprox. 1 g = 9.81 m/s^2
        
        self.path_loss_model = PathLossModel()

    def update(self, dt: float = 0.01) -> None:
        """
        Updates the drone's state based on the control forces and dynamics.

        Parameters
        ----------
        dt : float, optional
            Time step in seconds (default is 0.01).
        """
        super().update(dt)

        # Compute control force using the position controller
        control_force = self.position_controller.update(
            self.state, self.neighbor_states, self.neighbor_ids, self.time
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

    def rx_power(self, pos: np.ndarray) -> float:
        d = np.linalg.norm(pos - self.position)
        return self.path_loss_model.rx_power(d)

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


class PathLossModel:

    def __init__(self, freq: float = 10.0, tx_power: float = 20.0, n: float = 3.0):
        self.freq = freq  # Mhz
        self.tx_power = tx_power  # dBm
        self.n = n

        self.d0 = 1.0
        self.pl0 = 20 * np.log10(self.d0) + 20 * np.log10(self.freq) + 32.44

    def rx_power(self, d: float) -> float:
        pl = self.pl0 + 10 * self.n * np.log10(d / self.d0)
        return self.tx_power - pl
