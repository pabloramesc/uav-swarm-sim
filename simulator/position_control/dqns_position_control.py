from dataclasses import dataclass

import numpy as np

from ..environment import Environment
from ..swarming.dqns_swarming import DQNS
from .altitude_control import AltitudeController
from .base_position_control import PositionController, PositionControllerConfig
from .horizontal_position_control import HorizontalPositionController


@dataclass
class DQNSConfig(PositionControllerConfig):
    visible_distance: float = 100.0  # in meters
    obstacle_distance: float = 10.0  # in meters
    agent_mass: float = 1.0  # simple equivalence between force and acceleration
    max_acceleration: float = 10.0  # 1 g aprox. 9.81 m/s^2
    target_velocity: float = 15.0  # between 5-25 m/S
    target_height: float = 100.0  # in meters (AGL - Above Ground Level)


class DQNSPostionController(PositionController):
    def __init__(self, config: DQNSConfig, env: Environment) -> None:
        super().__init__(config, env)
        self.config = config

        self.dqns = DQNS(env=self.env)
        self.altitude_hold = AltitudeController(
            kp=config.max_acceleration / config.target_height,
            kd=config.max_acceleration / config.target_velocity,
        )
        self.position_controller = HorizontalPositionController(
            kp=config.max_acceleration / config.visible_distance,
            kd=config.max_acceleration / config.target_velocity,
        )

        self.last_update_time: float = None
        self.min_update_period = 1.0
        self.target_position = np.zeros(2)  # px, py

    def update(
        self,
        state: np.ndarray,
        neighbor_states: np.ndarray,
        neighbor_ids: np.ndarray = None,
        time: float = None,
    ) -> np.ndarray:
        """
        Updates the DQNS controller's state and computes the control output.

        Parameters
        ----------
        state : np.ndarray
            A (6,) array representing the agent's state
            [px, py, pz, vx, vy, vz].
        neighbor_states : np.ndarray
            A (N, 6) array representing the states [px, py, pz, vx, vy, vz]
            of N neighbors.
        neighbor_states : np.ndarray
            A (N,) array with the IDs of the N neighbors.
        time : float, optional
            Current simulation time in seconds. Default is None.

        Returns
        -------
        np.ndarray
            Control output [fx, fy, fz].
        """
        super().update(state, neighbor_states, neighbor_ids, time)

        if self._needs_update(time):
            self.dqns.update(state[0:2], neighbor_states[:, 0:2])

        control = np.zeros(3)

        # Horizontal control using PD (Proportional Derivative)
        control[0:2] = self.position_controller.control(
            target_position=self.target_position,
            position=state[0:2],
            velocity=state[3:5],
        )

        # Vertical control by altitude hold
        target_altitude = self.env.get_elevation(state[0:2]) + self.config.target_height
        control[2] = self.altitude_hold.control(
            target_altitude=target_altitude, altitude=state[2], vspeed=state[5]
        )

        return control

    def get_frame(self) -> np.ndarray:
        return self.dqns.frame

    def set_target_position(self, action: int) -> None:
        self.target_position = self.dqns.target_position(action)

    def _needs_update(self, time: float) -> bool:
        if time is None or self.last_update_time is None:
            return True
        elapsed_time = time - self.last_update_time
        return elapsed_time > self.min_update_period
