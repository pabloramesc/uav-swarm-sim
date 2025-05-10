"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from dataclasses import dataclass

import numpy as np

from ..environment.environment import Environment
from ..evsm.evsm_algorithm import EVSM
from .altitude_controller import AltitudeController
from .base_swarming import SwarmingController, SwarmingConfig


@dataclass
class EVSMConfig(SwarmingConfig):
    """
    Configuration for the EVSMPositionControl class.

    Attributes
    ----------
    separation_distance : float
        Desired separation distance between agents in meters
        (default is 50.0).
    obstacle_distance : float
        Minimum distance to obstacles for avoidance in meters
        (default is 10.0).
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
    target_height: float = 100.0  # in meters (AGL - Above Ground Level)
    max_height_error: float = 100.0
    ln_rate: float = 1.0


class EVSMController(SwarmingController):
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
            Configuration object containing parameters for the EVSM and
            altitude controllers.
        env : Environment
            The simulation environment.
        """
        super().__init__(config, env)
        self.neighbor_positions: np.ndarray = None

        self.min_ln = 10.0
        self.max_ln = config.separation_distance
        self.ln_rate = config.ln_rate
        self.ln = self.min_ln
        self.target_height = config.target_height

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
            kp=config.max_acceleration / config.max_height_error,
            # kd=config.agent_mass / 1.0,
            kd=config.max_acceleration / config.target_velocity,
        )

    def initialize(self, time: float, state: np.ndarray, **kwargs: dict) -> np.ndarray:
        super().initialize(time, state)

        neighbor_positions: np.ndarray = kwargs.get("neighbor_positions", None)
        if neighbor_positions is None:
            raise ValueError("neighbor_positions must be provided in the kwargs.")

        self.neighbor_positions = neighbor_positions.copy()

    def update(
        self,
        time: float,
        state: np.ndarray,
        **kwargs: dict,
    ) -> np.ndarray:
        """
        Updates the EVSM controller's state and computes the control output.
        """
        super().update(time, state)

        neighbor_positions: np.ndarray = kwargs.get("neighbor_positions", None)
        if neighbor_positions is not None:
            self.neighbor_positions = neighbor_positions.copy()

        self._update_natural_length(time)

        control = np.zeros(3)

        # Horizontal control using EVSM (Extended Virtual Spring Mesh)
        control[0:2] = self.evsm.update(
            position=state[0:2],
            velocity=state[3:5],
            neighbors=neighbor_positions[:, 0:2],
            time=time,
            force_update=False,
        )

        # Vertical control by altitude hold
        target_altitude = self.env.get_elevation(state[0:2]) + self.target_height
        control[2] = self.altitude_hold.control(
            target_altitude=target_altitude, altitude=state[2], vspeed=state[5]
        )

        return control

    def _update_natural_length(self, time: float) -> None:
        """
        Updates the natural length of the EVSM algorithm based on the rate of
        change.
        """
        self.ln = min(self.max_ln, self.min_ln + self.ln_rate * time)
        self.evsm.set_natural_length(self.ln)

    def set_natural_length(self, ln: float) -> None:
        """
        Sets the natural length of the EVSM algorithm.
        """
        self.ln = ln
        self.evsm.set_natural_length(ln)

    def set_target_height(self, target_height: float) -> None:
        """
        Sets the target height for the altitude controller.
        """
        self.target_height = target_height
