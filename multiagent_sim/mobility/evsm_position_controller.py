from dataclasses import dataclass

import numpy as np

from ..environment.environment import Environment
from ..evsm.evsm_algorithm import EVSMAlgorithm
from .pid import PIDController
from .position_controller import PositionController, ControllerContext


@dataclass
class EVSMConfig:
    """Configuration for EVSM position controller."""

    separation_distance: float = 100.0
    obstacle_distance: float = 10.0
    target_speed: float = 15.0
    target_altitude: float = 100.0
    initial_natural_length: float = 10.0
    natural_length_rate: float = 1.0
    agent_mass: float = 1.0
    max_acceleration: float = 10.0
    max_position_error: float = 100.0


class EVSMPositionController(PositionController):
    """
    Combines EVSM horizontal control with altitude hold.
    """

    def __init__(
        self,
        config: EVSMConfig,
        environment: Environment,
    ):
        self.config = config
        self.environment = environment

        self.control_update_period = 0.1
        self.springs_update_period = 1.0

        self._initial_nat_length = config.initial_natural_length
        self._max_nat_length = config.separation_distance
        self._nat_length_rate = config.natural_length_rate
        self._current_nat_length = self._initial_nat_length

        self._target_altitude = config.target_altitude

        kp = config.max_acceleration / config.max_position_error
        kd = 2 * np.sqrt(kp)

        self.evsm = EVSMAlgorithm(
            env=self.environment,
            ln=self._initial_nat_length,
            ks=kp,
            kd=kd,
            k_expl=kp,
            d_obs=config.obstacle_distance,
        )

        self.altitude_controller = PIDController(kp, kd)

        self.control_force = np.zeros(3)
        self.drone_positions: dict[int, np.ndarray] = {}

        self._last_control_update_time: float = None
        self._last_springs_update_time: float = None

    def initialize(self, context: ControllerContext) -> None:
        if context.drone_positions is None:
            raise ValueError("Drone positions is required for initialization.")

        self.control_force = np.zeros(3)
        self.drone_positions = context.drone_positions
        self._last_control_update_time: float = None
        self._last_springs_update_time: float = None

    def update(self, context: ControllerContext) -> np.ndarray:
        """Compute control forces: [Fx, Fy, Fz]"""
        if context.time is None:
            raise ValueError("Time is required.")

        if context.drone_positions is not None:
            topology_changed = (
                context.drone_positions.keys() != self.drone_positions.keys()
            )
            self.drone_positions = context.drone_positions

        if not self._need_update_control(context.time):
            return self.control_force
        self._last_control_update_time = context.time

        if topology_changed or self._need_update_springs(context.time):
            update_springs = True
            self._last_springs_update_time = context.time
        else:
            update_springs = False

        if self.drone_positions:
            neighbors = np.stack([pos[0:2] for pos in self.drone_positions.values()])
        else:
            neighbors = np.zeros((0, 2))

        self._update_natural_length(context.time)

        # Horizontal control force (EVSM - Extended Virtual Spring Mesh)
        self.control_force[0:2] = self.evsm.update(
            position=context.state[0:2],
            velocity=context.state[3:5],
            neighbors=neighbors,
            time=context.time,
            update_springs=update_springs,
        )

        # Vertical control force (PD altitude controller)
        ground_elevation = self.environment.get_elevation(context.state[0:2])
        desired_alt = ground_elevation + self._target_altitude
        self.control_force[2] = self.altitude_controller.control(
            target_altitude=desired_alt,
            altitude=context.state[2],
            vspeed=context.state[5],
        )

        return self.control_force

    def _need_update_control(self, time: float) -> bool:
        if self._last_control_update_time is None:
            return True
        return (time - self._last_control_update_time) >= self.control_update_period

    def _need_update_springs(self, time: float) -> bool:
        if self._last_springs_update_time is None:
            return True
        return (time - self._last_springs_update_time) >= self.springs_update_period

    def _update_natural_length(self, time: float) -> None:
        """
        Grow natural length at fixed rate up to maximum.
        """
        new_length = self._initial_nat_length + self._nat_length_rate * time
        self._current_nat_length = min(self._max_nat_length, new_length)
        self.evsm.set_natural_length(self._current_nat_length)

    def set_natural_length(self, length: float) -> None:
        """
        Override the EVSM natural length directly.
        """
        self._current_nat_length = length
        self.evsm.set_natural_length(length)

    def set_target_altitude(self, altitude: float) -> None:
        """
        Update altitude controller setpoint.
        """
        self._target_altitude = altitude
