"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from dataclasses import dataclass

import numpy as np

from ..environment import Environment
from ..sdqn.local_agent import LocalAgent
from .altitude_controller import AltitudeController
from .swarm_position_controller import SwarmPositionController, SwarmPositionConfig
from .position_controller import PositionController


@dataclass
class SDQNPositionConfig(SwarmPositionConfig):
    displacement: float = 10.0  # in meters
    obstacle_distance: float = 10.0  # in meters
    agent_mass: float = 1.0  # simple equivalence between force and acceleration
    max_acceleration: float = 10.0  # 1 g aprox. 9.81 m/s^2
    target_velocity: float = 15.0  # between 5-25 m/S
    target_height: float = 100.0  # in meters (AGL - Above Ground Level)


class SDQNPositionController(SwarmPositionController):
    def __init__(
        self,
        config: SDQNPositionConfig,
        environment: Environment,
        local_agent: LocalAgent,
    ) -> None:
        super().__init__(config, environment)
        self.update_period = 0.1

        self.local_agent = local_agent

        self.altitude_hold = AltitudeController(
            kp=config.max_acceleration / config.displacement,
            kd=config.max_acceleration / config.target_velocity,
        )
        self.position_controller = PositionController(
            kp=config.max_acceleration / config.displacement,
            kd=config.max_acceleration / config.target_velocity,
        )
        
        self._displacement = config.displacement

        self.last_update_time: float = None
        self.target_position = np.zeros(2)  # px, py

        # Neighbor positions cache
        self._drone_positions: dict[int, np.ndarray] = {}
        self._user_positions: dict[int, np.ndarray] = {}

    def initialize(
        self,
        time: float,
        state: np.ndarray,
        drone_positions: dict[int, np.ndarray] = None,
        user_positions: dict[int, np.ndarray] = None,
    ) -> None:
        super().initialize(time, state)

        self._drone_positions = drone_positions
        self._user_positions = user_positions

        self._update_local_agent()

        self.last_update_time: float = None
        self.target_position = state[0:2]  # px, py

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
            self._drone_positions = drone_positions

        if user_positions is not None:
            self._user_positions = user_positions

        if self._needs_update(time):
            self._update_local_agent()

        self.target_position = self._displacement * self.local_agent.direction

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

    def _needs_update(self, time: float) -> bool:
        if time is None or self.last_update_time is None:
            return True
        elapsed_time = time - self.last_update_time
        return elapsed_time > self.update_period

    def _update_local_agent(self) -> None:
        drones_array = self._positions_dict_to_array(self._drone_positions)
        users_array = self._positions_dict_to_array(self._user_positions)
        self.local_agent.update(
            position=self.state[0:2], drones=drones_array, users=users_array
        )

    def _positions_dict_to_array(self, positions: dict[int, np.ndarray]) -> np.ndarray:
        if positions is None or len(positions) == 0:
            return np.zeros((0, 2))
        return np.array([pos[0:2] for pos in positions.values()])
