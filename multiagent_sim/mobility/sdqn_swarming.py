"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from dataclasses import dataclass

import numpy as np

from ..environment import Environment
from ..sdqn.frame_generator import FrameGenerator
from .altitude_controller import AltitudeController
from .swarm_position_controller import SwarmPositionController, SwarmPositionConfig
from .position_controller import PositionController


@dataclass
class SDQNPositionConfig(SwarmPositionConfig):
    num_cells: int = 64
    num_actions: int = 9
    visible_distance: float = 100.0  # in meters
    obstacle_distance: float = 10.0  # in meters
    agent_mass: float = 1.0  # simple equivalence between force and acceleration
    max_acceleration: float = 10.0  # 1 g aprox. 9.81 m/s^2
    target_velocity: float = 15.0  # between 5-25 m/S
    target_height: float = 100.0  # in meters (AGL - Above Ground Level)


class SDQNPositionController(SwarmPositionController):
    def __init__(self, config: SDQNPositionConfig, env: Environment) -> None:
        super().__init__(config, env)
        self.config = config
        self.update_period = 0.1

        cell_size = 2 * config.visible_distance / config.num_cells

        self.dqns = FrameGenerator(
            env=self.env,
            sense_radius=config.visible_distance,
            num_cells=config.num_cells,
            num_actions=config.num_actions,
        )
        self.altitude_hold = AltitudeController(
            kp=config.max_acceleration / cell_size,
            kd=config.max_acceleration / config.target_velocity,
        )
        self.position_controller = PositionController(
            kp=config.max_acceleration / cell_size,
            kd=config.max_acceleration / config.target_velocity,
        )

        self.last_update_time: float = None
        self.target_position = np.zeros(2)  # px, py

    def initialize(
        self,
        state: np.ndarray,
        neighbor_positions: np.ndarray,
        time: float = None,
    ) -> None:
        super().initialize(state, neighbor_positions, time=time)
        self.dqns.reset(self.state[0:2], self.neighbor_positions[:, 0:2], time)
        self.last_update_time: float = None
        self.target_position = state[0:2]  # px, py

    def update(
        self,
        state: np.ndarray,
        neighbor_positions: np.ndarray,
        user_positions: np.ndarray = None,
        time: float = None,
    ) -> np.ndarray:
        """
        Updates the DQNS controller's state and computes the control output.
        """
        super().update(state, neighbor_positions, user_positions, time)

        if self._needs_update(time):
            self.dqns.update(state[0:2], neighbor_positions[:, 0:2], time)

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
        return self.dqns.compute_state_frame()

    def set_target_position(self, action: int) -> None:
        self.target_position = self.dqns.calculate_target_position(action)

    def _needs_update(self, time: float) -> bool:
        # if time is None or self.last_update_time is None:
        #     return True
        # elapsed_time = time - self.last_update_time
        # return elapsed_time > self.update_period
        return True
