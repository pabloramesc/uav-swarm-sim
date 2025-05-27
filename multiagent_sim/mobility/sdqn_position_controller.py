"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from dataclasses import dataclass

import numpy as np

from ..environment import Environment
from ..sdqn.sdqn_interface import SDQNInterface
from .altitude_controller import AltitudeController
from .swarm_position_controller import SwarmPositionController, SwarmControllerConfig
from .position_controller import PositionController


@dataclass
class SDQNConfig(SwarmControllerConfig):
    displacement: float = 1.0  # in meters
    obstacle_distance: float = 10.0  # in meters
    agent_mass: float = 1.0  # simple equivalence between force and acceleration
    max_acceleration: float = 10.0  # 1 g aprox. 9.81 m/s^2
    max_displacement: float = 100.0
    target_velocity: float = 15.0  # between 5-25 m/S
    target_height: float = 100.0  # in meters (AGL - Above Ground Level)


class SDQNPositionController(SwarmPositionController):
    def __init__(
        self,
        config: SDQNConfig,
        environment: Environment,
        sdqn_iface: SDQNInterface,
    ) -> None:
        super().__init__(config, environment)
        self.config: SDQNConfig = config
        
        self.control_update_period = 0.0
        self.sdqn_update_period = 0.0

        self.sdqn_iface = sdqn_iface

        kp = config.max_acceleration / config.max_displacement
        
        kd = 2 * np.sqrt(kp)  # critical damping
        self.altitude_hold = AltitudeController(kp, kd)
        
        kd = config.max_acceleration / config.target_velocity
        self.position_controller = PositionController(kp, kd)

        self.displacement = config.displacement

        self.target_position = np.zeros(2)  # px, py

        # Neighbor positions cache
        self.control_force = np.zeros(3)
        self.drone_positions: dict[int, np.ndarray] = {}
        self.user_positions: dict[int, np.ndarray] = {}

        self._last_control_update_time: float = None
        self._last_sdqn_update_time: float = None

    def initialize(
        self,
        time: float,
        state: np.ndarray,
        drone_positions: dict[int, np.ndarray] = None,
        user_positions: dict[int, np.ndarray] = None,
    ) -> None:
        super().initialize(time, state)

        self.drone_positions = drone_positions
        self.user_positions = user_positions

        self._update_sdqn_interface()

        self.control_force = np.zeros(3)
        self.target_position = state[0:2]  # px, py
        self._last_control_update_time: float = None
        self._last_sdqn_update_time: float = None

    def update(
        self,
        time: float,
        state: np.ndarray,
        drone_positions: dict[int, np.ndarray] = None,
        user_positions: dict[int, np.ndarray] = None,
    ) -> np.ndarray:
        """
        Updates the DQNS controller's state and computes the control output.
        """
        super().update(time, state)

        if drone_positions is not None:
            self.drone_positions = drone_positions

        if user_positions is not None:
            self.user_positions = user_positions

        # if not self._need_update_control(time):
        #     return self.control_force
        # self._last_control_update_time = time

        # if self._need_update_sdqn(time):
        #     self._update_sdqn_interface()
        #     self.target_position = (
        #         self.state[0:2] + self.displacement * self.sdqn_iface.direction
        #     )
        #     self._last_sdqn_update_time = time
        
        self._update_sdqn_interface()
        
        vel = self.config.target_velocity * self.sdqn_iface.direction
        self.target_position = self.state[0:2] + vel * 0.1
        

        # if self.env.is_collision(pos=self.target_position, check_boundary=True):
        #     self.target_position = self.state[0:2]

        # Horizontal control using PD (Proportional Derivative)
        self.control_force[0:2] = self.position_controller.control(
            target_position=self.target_position,
            position=state[0:2],
            velocity=state[3:5],
        )

        # Vertical control by altitude hold
        target_altitude = self.env.get_elevation(state[0:2]) + self.config.target_height
        self.control_force[2] = self.altitude_hold.control(
            target_altitude=target_altitude, altitude=state[2], vspeed=state[5]
        )

        return self.control_force

    def get_frame(self) -> np.ndarray:
        return self.dqns.compute_state_frame()

    def _need_update_control(self, time: float) -> bool:
        if self._last_control_update_time is None:
            return True
        return (time - self._last_control_update_time) >= self.control_update_period

    def _need_update_sdqn(self, time: float) -> bool:
        if self._last_sdqn_update_time is None:
            return True
        return (time - self._last_sdqn_update_time) >= self.sdqn_update_period

    def _update_sdqn_interface(self) -> None:
        drones_array = self._positions_dict_to_array(self.drone_positions)
        users_array = self._positions_dict_to_array(self.user_positions)
        self.sdqn_iface.update_positions(
            position=self.state[0:2], drones=drones_array, users=users_array
        )

    def _positions_dict_to_array(self, positions: dict[int, np.ndarray]) -> np.ndarray:
        if positions is None or len(positions) == 0:
            return np.zeros((0, 2))
        return np.array([pos[0:2] for pos in positions.values()])
