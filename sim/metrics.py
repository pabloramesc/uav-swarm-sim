"""Coverage and connectivity metrics derived from simulation state.

Metrics deliberately live outside the simulation engine.  Callers decide when
to calculate them, so a fast physics or training loop is not forced to run the
Monte Carlo area calculation on every tick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from .environment import Environment
from .math.connectivity import (
    directly_connected,
    globally_connected,
    pairwise_connectivity_matrix,
)
from .math.coverage import covered_positions


@dataclass(frozen=True)
class RadioConfig:
    """Radio assumptions shared by coverage and connectivity metrics."""

    tx_power: float = 20.0
    min_rssi: float = -80.0
    frequency_mhz: float = 2412.0
    path_loss_exponent: float = 2.4

    def __post_init__(self) -> None:
        values = (self.tx_power, self.min_rssi)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Radio power and RSSI values must be finite.")
        if not math.isfinite(self.frequency_mhz) or self.frequency_mhz <= 0.0:
            raise ValueError("frequency_mhz must be positive and finite.")
        if not math.isfinite(self.path_loss_exponent) or self.path_loss_exponent <= 0.0:
            raise ValueError("path_loss_exponent must be positive and finite.")


@dataclass(frozen=True)
class MetricsSnapshot:
    """A point-in-time summary calculated from drone and user positions."""

    area_coverage: float
    users_coverage: float
    direct_connections: float
    global_connections: float
    covered_users: NDArray[np.intp] = field(repr=False)
    directly_connected: NDArray[np.intp] = field(repr=False)
    globally_connected: NDArray[np.intp] = field(repr=False)
    links_matrix: NDArray[np.bool_] = field(repr=False)

    def __post_init__(self) -> None:
        for name in (
            "covered_users",
            "directly_connected",
            "globally_connected",
            "links_matrix",
        ):
            values = np.array(getattr(self, name), copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)


class MetricsCalculator:
    """Calculate radio metrics on demand for one environment."""

    def __init__(
        self,
        environment: Environment,
        *,
        radio: RadioConfig | None = None,
        area_samples: int = 0,
        check_obstacles: bool = True,
        rng: np.random.Generator | None = None,
    ) -> None:
        if isinstance(area_samples, bool) or not isinstance(area_samples, Integral):
            raise TypeError("area_samples must be an integer.")
        if area_samples < 0:
            raise ValueError("area_samples cannot be negative.")

        self.environment = environment
        self.radio = radio or RadioConfig()
        self.area_samples = int(area_samples)
        self.check_obstacles = bool(check_obstacles)
        self.rng = rng or np.random.default_rng()

    def calculate(
        self,
        drone_states: NDArray[np.floating],
        user_states: NDArray[np.floating],
    ) -> MetricsSnapshot:
        drone_positions = _positions(drone_states, name="drone_states")
        user_positions = _positions(user_states, name="user_states")
        cfg = self.radio

        covered_mask = covered_positions(
            tx_positions=drone_positions,
            rx_positions=user_positions,
            tx_power=cfg.tx_power,
            min_rssi=cfg.min_rssi,
            freq_mhz=cfg.frequency_mhz,
            path_loss_exp=cfg.path_loss_exponent,
        )
        covered_users = np.flatnonzero(covered_mask)

        links = pairwise_connectivity_matrix(
            positions=drone_positions,
            tx_power=cfg.tx_power,
            min_rssi=cfg.min_rssi,
            freq_mhz=cfg.frequency_mhz,
            path_loss_exp=cfg.path_loss_exponent,
        )
        direct = directly_connected(
            positions=drone_positions,
            tx_power=cfg.tx_power,
            min_rssi=cfg.min_rssi,
            freq_mhz=cfg.frequency_mhz,
            path_loss_exp=cfg.path_loss_exponent,
        )
        global_ = globally_connected(
            positions=drone_positions,
            tx_power=cfg.tx_power,
            min_rssi=cfg.min_rssi,
            freq_mhz=cfg.frequency_mhz,
            path_loss_exp=cfg.path_loss_exponent,
        )

        num_drones = len(drone_positions)
        num_users = len(user_positions)
        return MetricsSnapshot(
            area_coverage=self._area_coverage(drone_positions),
            users_coverage=len(covered_users) / num_users if num_users else 0.0,
            direct_connections=len(direct) / num_drones if num_drones else 0.0,
            global_connections=len(global_) / num_drones if num_drones else 0.0,
            covered_users=covered_users,
            directly_connected=direct,
            globally_connected=global_,
            links_matrix=links,
        )

    def _area_coverage(self, drone_positions: NDArray[np.float64]) -> float:
        if self.area_samples == 0 or len(drone_positions) == 0:
            return 0.0

        boundary = self.environment.boundary
        if boundary is None:
            raise RuntimeError("Area coverage requires an environment boundary.")

        bounds = boundary.bounds
        receiver_positions = np.zeros((self.area_samples, 3), dtype=np.float64)
        receiver_positions[:, 0] = self.rng.uniform(
            bounds.xmin, bounds.xmax, self.area_samples
        )
        receiver_positions[:, 1] = self.rng.uniform(
            bounds.ymin, bounds.ymax, self.area_samples
        )
        receiver_positions[:, 2] = self.environment.get_elevation(
            receiver_positions[:, :2]
        )

        valid = self.environment.is_inside(receiver_positions)
        if self.check_obstacles:
            valid &= ~self.environment.is_collision(receiver_positions)
        if not valid.any():
            return 0.0

        cfg = self.radio
        covered = covered_positions(
            tx_positions=drone_positions,
            rx_positions=receiver_positions[valid],
            tx_power=cfg.tx_power,
            min_rssi=cfg.min_rssi,
            freq_mhz=cfg.frequency_mhz,
            path_loss_exp=cfg.path_loss_exponent,
        )
        return float(np.mean(covered))


def area_coverage(
    environment: Environment,
    tx_positions: NDArray[np.floating],
    *,
    num_points: int = 1_000,
    check_obstacles: bool = False,
    radio: RadioConfig | None = None,
    rng: np.random.Generator | None = None,
) -> float:
    """Calculate only the environment area covered by transmitters."""

    states = np.zeros((len(tx_positions), 6), dtype=np.float64)
    states[:, :3] = _positions(tx_positions, name="tx_positions")
    calculator = MetricsCalculator(
        environment,
        radio=radio,
        area_samples=num_points,
        check_obstacles=check_obstacles,
        rng=rng,
    )
    return calculator.calculate(states, np.zeros((0, 6))).area_coverage


def _positions(
    states_or_positions: NDArray[np.floating], *, name: str
) -> NDArray[np.float64]:
    values = np.asarray(states_or_positions, dtype=np.float64)
    if values.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] not in (3, 6):
        raise ValueError(f"{name} must have shape (N, 3) or (N, 6).")
    return values[:, :3]
