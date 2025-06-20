from dataclasses import dataclass
import numpy as np

from ..environment.environment import Environment
from ..evsm.evsm_algorithm import EVSMAlgorithm
from .altitude_controller import AltitudeController
from .swarm_position_controller import SwarmPositionController, SwarmControllerConfig


@dataclass
class EVSMConfig(SwarmControllerConfig):
    """
    Configuration for EVSMPositionController.
    """

    separation_distance: float = 100.0
    obstacle_distance: float = 10.0
    target_speed: float = 15.0
    target_altitude: float = 100.0
    initial_natural_length: float = 10.0
    natural_length_rate: float = 1.0
    agent_mass: float = 1.0
    max_acceleration: float = 10.0
    max_position_error: float = 100.0


class EVSMPositionController(SwarmPositionController):
    """
    Combines EVSM horizontal control with altitude hold.
    """

    def __init__(
        self,
        config: EVSMConfig,
        environment: Environment,
    ):
        super().__init__(config, environment)
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
            env=environment,
            ln=self._initial_nat_length,
            ks=kp,
            kd=kd,
            k_expl=kp,
            d_obs=config.obstacle_distance,
        )

        self.vertical_controller = AltitudeController(kp, kd)

        self.control_force = np.zeros(3)
        self.drone_positions: dict[int, np.ndarray] = {}

        self._last_control_update_time: float = None
        self._last_springs_update_time: float = None

    def initialize(
        self,
        time: float,
        state: np.ndarray,
        drone_positions: dict[int, np.ndarray] = None,
        user_positions: dict[int, np.ndarray] = None,
    ) -> None:
        super().initialize(time, state)
        if drone_positions is None:
            raise ValueError("Drone positions is required for initialization")
        self.control_force = np.zeros(3)
        self.drone_positions = drone_positions
        self._last_control_update_time: float = None
        self._last_springs_update_time: float = None

    def update(
        self,
        time: float,
        state: np.ndarray,
        drone_positions: dict[int, np.ndarray] = None,
        user_positions: dict[int, np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute control forces: [Fx, Fy, Fz]
        """
        super().update(time, state)

        if not self._need_update_control(time):
            return self.control_force
        self._last_control_update_time = time

        update_springs = False
        topology_changed = drone_positions.keys() != self.drone_positions.keys()
        if topology_changed or self._need_update_springs(time):
            update_springs = True
            self._last_springs_update_time = time
            self.drone_positions = drone_positions

        if self.drone_positions:
            neighbors = np.stack([pos[0:2] for pos in self.drone_positions.values()])
        else:
            neighbors = np.zeros((0, 2))

        self._update_natural_length(time)

        # Horizontal control force (EVSM - Extended Virtual Spring Mesh)
        self.control_force[0:2] = self.evsm.update(
            position=state[0:2],
            velocity=state[3:5],
            neighbors=neighbors,
            time=time,
            update_springs=update_springs,
        )

        # Vertical control force (PD altitude controller)
        ground_elevation = self.env.get_elevation(state[0:2])
        desired_alt = ground_elevation + self._target_altitude
        self.control_force[2] = self.vertical_controller.control(
            target_altitude=desired_alt,
            altitude=state[2],
            vspeed=state[5],
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
