from dataclasses import dataclass
import numpy as np

from ..environment.environment import Environment
from ..evsm.evsm_algorithm import EVSM
from .altitude_controller import AltitudeController
from .base_swarming import SwarmingController, SwarmingConfig


@dataclass
class EVSMConfig(SwarmingConfig):
    """
    Configuration for EVSMController.

    Attributes
    ----------
    separation_distance : float
        Desired separation between agents (m).
    obstacle_distance : float
        Minimum obstacle avoidance distance (m).
    agent_mass : float
        Agent mass (kg).
    max_acceleration : float
        Maximum acceleration (m/s^2).
    target_speed : float
        Desired horizontal speed (m/s).
    target_altitude : float
        Desired altitude above ground (m).
    max_altitude_error : float
        Maximum altitude error for controller.
    natural_length_rate : float
        Rate of change for EVSM natural length (m/s).
    """

    separation_distance: float = 50.0
    obstacle_distance: float = 10.0
    agent_mass: float = 1.0
    max_acceleration: float = 10.0
    target_speed: float = 15.0
    target_altitude: float = 100.0
    max_altitude_error: float = 100.0
    initial_natural_length: float = 50.0
    natural_length_rate: float = 1.0


class EVSMController(SwarmingController):
    """
    Combines EVSM horizontal control with altitude hold.
    """

    def __init__(
        self,
        config: EVSMConfig,
        environment: Environment,
    ):
        super().__init__(config, environment)

        # Natural length parameters
        self._initial_nat_length = config.initial_natural_length
        self._max_nat_length = config.separation_distance
        self._nat_length_rate = config.natural_length_rate
        self._current_nat_length = self._initial_nat_length

        # Altitude setpoint
        self._target_altitude = config.target_altitude

        # EVSM horizontal control
        self.evsm = EVSM(
            env=environment,
            ln=self._initial_nat_length,
            ks=config.max_acceleration / config.separation_distance,
            kd=config.max_acceleration / config.target_speed,
            d_obs=config.obstacle_distance,
        )

        # Vertical PD control
        self.vertical_controller = AltitudeController(
            kp=config.max_acceleration / config.max_altitude_error,
            kd=config.max_acceleration / config.target_speed,
        )

        # Neighbor positions cache
        self._neighbor_drone_positions: dict[int, np.ndarray] = {}

    def initialize(
        self,
        time: float,
        state: np.ndarray,
        drone_positions: dict[int, np.ndarray] = None,
    ) -> None:
        super().initialize(time, state)
        if drone_positions is None:
            raise ValueError("`drone_positions` is required for initialization")
        self._neighbor_drone_positions = drone_positions.copy()

    def update(
        self,
        time: float,
        state: np.ndarray,
        drone_positions: dict[int, np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute control forces: [Fx, Fy, Fz]
        """
        super().update(time, state)

        # Check for changes in neighbor topology
        force_update = False
        if drone_positions.keys() != self._neighbor_drone_positions.keys():
            force_update = True

        self._neighbor_drone_positions = drone_positions.copy()

        # Prepare neighbor array for EVSM
        neighbors_array = (
            np.array(list(self._neighbor_drone_positions.values()))[:, 0:2]
            if self._neighbor_drone_positions
            else np.zeros((0, 2))
        )

        # Update EVSM natural length
        self._update_natural_length(time)

        # Initialize control vector [Fx, Fy, Fz]
        control_force = np.zeros(3)

        # Horizontal EVSM force
        control_force[:2] = self.evsm.update(
            position=state[0:2],
            velocity=state[3:5],
            neighbors=neighbors_array,
            time=time,
            force_update=force_update,
        )

        # Vertical force via PD altitude controller
        ground_elevation = self.env.get_elevation(state[0:2])
        desired_alt = ground_elevation + self._target_altitude
        control_force[2] = self.vertical_controller.control(
            target_altitude=desired_alt,
            altitude=state[2],
            vspeed=state[5],
        )

        return control_force

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
