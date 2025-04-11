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
from .evsm_algorithm import EVSM


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
    separation_distance: float = 50.0
    obstacle_distance: float = 10.0
    agent_mass: float = 1.0  # simple equivalence between force and acceleration
    max_acceleration: float = 10.0  # 1 g aprox. 9.81 m/s^2
    target_velocity: float = 15.0  # between 5-25 m/S
    target_altitude: float = 100.0


class EVSMPositionControl(PositionController):
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

        self.evsm = EVSM(
            ln=config.separation_distance,
            ks=config.max_acceleration
            / config.separation_distance,  # max acceleration when separation is 0
            # kd=config.agent_mass / 1.0,  # 1 second damping (kd = m / tau)
            kd=config.max_acceleration
            / config.target_velocity,  # max acceleration when speed is target velocity
            d_obs=config.obstacle_distance,
            avoid_regions=env.avoid_regions,
        )

        self.altitude_hold = AltitudeController(
            kp=config.max_acceleration
            / config.target_altitude,  # max acceleration when error is target altitude
            # kd=config.agent_mass / 1.0,  # 1 second damping (kd = m / tau)
            kd=config.max_acceleration
            / config.target_velocity,  # max acceleration when speed is target velocity
            target_altitude=config.target_altitude,
        )

    def update(
        self, agent_state: np.ndarray, neighbot_states: np.ndarray, time: float = None
    ) -> np.ndarray:
        control = np.zeros(3)

        # Horizontal control using EVSM (Extended Virtual Spring Mesh)
        control[0:2] = self.evsm.update(
            position=agent_state[0:2],
            velocity=agent_state[3:5],
            neighbors=neighbot_states[:, 0:2],
        )

        # Vertical control by altitude hold
        control[3] = self.altitude_hold.control(
            altitude=agent_state[2], vspeed=agent_state[5]
        )
        
        return control
