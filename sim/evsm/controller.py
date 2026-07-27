"""Position controller for the Extended Virtual Spring Mesh algorithm."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..environment.environment import Environment
from ..mobility.pid import PIDController
from ..mobility.position_controller import ControllerContext, PositionController
from .evsm_algorithm import EVSMAlgorithm


@dataclass(frozen=True)
class EVSMConfig:
    """Configuration shared by every EVSM drone in a simulation.

    Altitudes are expressed above ground level (AGL).  Forces are calculated
    in newtons and the drone dynamics apply the configured mass and acceleration
    limit.
    """

    separation_distance: float = 100.0
    obstacle_distance: float = 10.0
    target_altitude: float = 100.0
    initial_natural_length: float = 10.0
    natural_length_rate: float = 1.0
    agent_mass: float = 1.0
    max_acceleration: float = 10.0
    max_position_error: float = 100.0
    control_update_period: float = 0.1
    springs_update_period: float = 1.0

    def __post_init__(self) -> None:
        values = {field: float(value) for field, value in vars(self).items()}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("EVSM configuration values must be finite.")
        positive = {
            "separation_distance": self.separation_distance,
            "obstacle_distance": self.obstacle_distance,
            "initial_natural_length": self.initial_natural_length,
            "agent_mass": self.agent_mass,
            "max_acceleration": self.max_acceleration,
            "max_position_error": self.max_position_error,
            "control_update_period": self.control_update_period,
            "springs_update_period": self.springs_update_period,
        }
        for name, value in positive.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if self.natural_length_rate < 0.0:
            raise ValueError("natural_length_rate cannot be negative.")
        if self.target_altitude < 0.0:
            raise ValueError("target_altitude cannot be negative.")
        if self.initial_natural_length > self.separation_distance:
            raise ValueError(
                "initial_natural_length cannot exceed separation_distance."
            )


class EVSMPositionController(PositionController):
    """Combine EVSM horizontal forces with PID altitude hold."""

    def __init__(self, config: EVSMConfig, environment: Environment) -> None:
        self.config = config
        self.environment = environment
        self.control_update_period = config.control_update_period
        self.springs_update_period = config.springs_update_period

        self._initial_natural_length = config.initial_natural_length
        self._max_natural_length = config.separation_distance
        self._natural_length_rate = config.natural_length_rate
        self._current_natural_length = self._initial_natural_length
        self._target_altitude = config.target_altitude

        max_force = config.agent_mass * config.max_acceleration
        kp = max_force / config.max_position_error
        kd = 2.0 * np.sqrt(kp * config.agent_mass)

        self.evsm = EVSMAlgorithm(
            env=environment,
            ln=self._initial_natural_length,
            ks=kp,
            kd=kd,
            k_expl=kp,
            d_obs=config.obstacle_distance,
            max_force=max_force,
        )
        self.altitude_controller = PIDController(
            kp=kp,
            kd=kd,
            dt=self.control_update_period,
            limit=max_force,
        )

        self.control_force: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        self.drone_positions: dict[int, NDArray[np.float64]] = {}
        self._last_control_update_time: float | None = None
        self._last_springs_update_time: float | None = None

    @property
    def natural_length(self) -> float:
        return self._current_natural_length

    @property
    def target_altitude(self) -> float:
        return self._target_altitude

    def initialize(self, context: ControllerContext) -> None:
        self.drone_positions = _copy_positions(context.drone_positions)
        self.control_force.fill(0.0)
        self._last_control_update_time = None
        self._last_springs_update_time = None
        self._current_natural_length = self._initial_natural_length
        self.evsm.set_natural_length(self._current_natural_length)
        self.altitude_controller.reset()

    def update(self, context: ControllerContext) -> NDArray[np.float64]:
        """Return the control force ``[Fx, Fy, Fz]`` for one drone."""

        time = float(context.time)
        positions = _copy_positions(context.drone_positions)
        topology_changed = positions.keys() != self.drone_positions.keys()
        self.drone_positions = positions

        control_due = self._needs_update(
            time,
            self._last_control_update_time,
            self.control_update_period,
        )
        if not topology_changed and not control_due:
            return self.control_force.copy()
        self._last_control_update_time = time

        update_springs = topology_changed or self._needs_update(
            time,
            self._last_springs_update_time,
            self.springs_update_period,
        )
        if update_springs:
            self._last_springs_update_time = time

        if self.drone_positions:
            neighbors = np.stack(
                [position[:2] for position in self.drone_positions.values()]
            )
        else:
            neighbors = np.zeros((0, 2), dtype=np.float64)

        state = np.asarray(context.agent_state, dtype=np.float64)
        if state.shape != (6,):
            raise ValueError("EVSM controller requires agent_state with shape (6,).")

        self._update_natural_length(time)
        self.control_force[:2] = self.evsm.update(
            position=state[:2],
            velocity=state[3:5],
            neighbors=neighbors,
            update_springs=update_springs,
        )

        terrain_elevation = float(self.environment.get_elevation(state[:2]).item())
        altitude_error = terrain_elevation + self._target_altitude - state[2]
        vertical_force = self.altitude_controller.control(
            error=altitude_error,
            derivative=-state[5],
        )
        self.control_force[2] = float(np.asarray(vertical_force).item())
        return self.control_force.copy()

    @staticmethod
    def _needs_update(
        time: float,
        previous_time: float | None,
        period: float,
    ) -> bool:
        return previous_time is None or time + 1e-12 >= previous_time + period

    def _update_natural_length(self, time: float) -> None:
        length = self._initial_natural_length + self._natural_length_rate * time
        self.set_natural_length(min(self._max_natural_length, length))

    def set_natural_length(self, length: float) -> None:
        if not 0.0 < length <= self._max_natural_length:
            raise ValueError(
                "Natural length must be positive and no greater than "
                "separation_distance."
            )
        self._current_natural_length = float(length)
        self.evsm.set_natural_length(self._current_natural_length)

    def set_target_altitude(self, altitude: float) -> None:
        if not math.isfinite(altitude) or altitude < 0.0:
            raise ValueError("Target altitude must be non-negative and finite.")
        self._target_altitude = float(altitude)


def _copy_positions(
    positions: dict[int, NDArray[np.float64]] | None,
) -> dict[int, NDArray[np.float64]]:
    if positions is None:
        return {}
    return {
        int(agent_id): np.asarray(position, dtype=np.float64).copy()
        for agent_id, position in positions.items()
    }
