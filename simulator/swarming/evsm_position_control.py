"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from dataclasses import dataclass

import numpy as np

from simulator.environment import Environment

from .altitude_control import AltitudeController
from .base_position_control import PositionController, PositionControllerConfig
from .evsm_swarming import EVSM


@dataclass
class EVSMConfig(PositionControllerConfig):
    """
    Configuration for the EVSMPositionControl class.

    Attributes
    ----------
    separation_distance : float
        Desired separation distance between agents in meters (default is 50.0).
    obstacle_distance : float
        Minimum distance to obstacles for avoidance in meters (default is 10.0).
    agent_mass : float
        Mass of the agent in kilograms (default is 1.0).
    max_acceleration : float
        Maximum acceleration of the agent in m/s^2 (default is 10.0).
    target_velocity : float
        Target velocity of the agent in m/s (default is 15.0).
    target_altitude : float
        Desired altitude of the agent in meters (default is 100.0).
    """

    separation_distance: float = 50.0  # in meters
    obstacle_distance: float = 10.0  # in meters
    agent_mass: float = 1.0  # simple equivalence between force and acceleration
    max_acceleration: float = 10.0  # 1 g aprox. 9.81 m/s^2
    target_velocity: float = 15.0  # between 5-25 m/S
    target_altitude: float = 100.0  # in meters
    ln_rate: float = 1.0


class EVSMPositionController(PositionController):
    """
    EVSM-based horizontal position control and altitude hold.

    This class combines the EVSM algorithm for horizontal position control
    with a PD controller for altitude hold.

    Attributes
    ----------
    evsm : EVSM
        Instance of the EVSM algorithm for horizontal position control.
    altitude_hold : AltitudeController
        Instance of the AltitudeController for vertical control.
    """

    def __init__(self, config: EVSMConfig, env: Environment):
        """
        Initializes the EVSMPositionControl class.

        Parameters
        ----------
        config : EVSMConfig
            Configuration object containing parameters for the EVSM and altitude controllers.
        env : Environment
            The simulation environment.
        """
        super().__init__(config, env)
        self.config = config

        self.min_ln = 10.0
        self.max_ln = config.separation_distance
        self.ln_rate = config.ln_rate
        self.ln = self.min_ln

        self.evsm = EVSM(
            env=self.env,
            ln=self.min_ln,
            # ln=config.separation_distance,
            ks=config.max_acceleration / config.separation_distance,
            # kd=config.agent_mass / 1.0,
            kd=config.max_acceleration / config.target_velocity,
            d_obs=config.obstacle_distance,
        )

        self.altitude_hold = AltitudeController(
            kp=config.max_acceleration / config.target_altitude,
            # kd=config.agent_mass / 1.0,
            kd=config.max_acceleration / config.target_velocity,
            target_altitude=config.target_altitude,
        )

    def update(
        self,
        state: np.ndarray,
        neighbor_states: np.ndarray,
        neighbor_ids: np.ndarray = None,
        time: float = None,
    ) -> np.ndarray:
        """
        Updates the EVSM controller's state and computes the control output.

        Parameters
        ----------
        state : np.ndarray
            A (6,) array representing the agent's state [px, py, pz, vx, vy, vz].
        neighbor_states : np.ndarray
            A (N, 6) array representing the states [px, py, pz, vx, vy, vz] of N neighbors.
        neighbor_states : np.ndarray
            A (N,) array with the IDs of the N neighbors.
        time : float, optional
            Current simulation time in seconds. Default is None.

        Returns
        -------
        np.ndarray
            Control output [fx, fy, fz].
        """
        force_evsm_update = False
        if not np.array_equal(neighbor_ids, self.neighbor_ids):
            force_evsm_update = True
            
        super().update(state, neighbor_states, neighbor_ids, time)
        
        self._update_natural_length(time)

        control = np.zeros(3)

        # Horizontal control using EVSM (Extended Virtual Spring Mesh)
        control[0:2] = self.evsm.update(
            position=state[0:2],
            velocity=state[3:5],
            neighbors=neighbor_states[:, 0:2],
            time=time,
            force=force_evsm_update,
        )

        # Vertical control by altitude hold
        altitude = state[2] - self.env.get_elevation(state[0:2])
        control[2] = self.altitude_hold.control(
            altitude=altitude, vspeed=state[5]
        )

        return control

    def _update_natural_length(self, time: float) -> None:
        """
        Updates the natural length of the EVSM algorithm based on the rate of change.
        """
        self.ln = min(self.max_ln, self.min_ln + self.ln_rate * time)
        self.evsm.set_natural_length(self.ln)
