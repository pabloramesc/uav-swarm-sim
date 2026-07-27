from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .base import Dynamics

if TYPE_CHECKING:
    from sim.environment import Environment


class RandomWalkerDynamics(Dynamics):
    def __init__(
        self,
        env: Environment | None = None,
        min_speed: float = 1.0,
        max_speed: float = 3.0,
        climb_rate: float = 0.2,
        turning_rate: float = 0.3,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()

        self.env = env
        self.min_speed = float(min_speed)
        self.max_speed = float(max_speed)
        self.climb_rate = float(climb_rate)
        self.turning_rate = float(turning_rate)
        self.rng = rng if rng is not None else np.random.default_rng()
        if not all(
            math.isfinite(value)
            for value in (
                self.min_speed,
                self.max_speed,
                self.climb_rate,
                self.turning_rate,
            )
        ):
            raise ValueError("Random-walker parameters must be finite.")
        if self.min_speed < 0.0 or self.max_speed < self.min_speed:
            raise ValueError("Speeds must satisfy 0 <= min_speed <= max_speed.")
        if self.climb_rate < 0.0:
            raise ValueError("climb_rate cannot be negative.")
        if not 0.0 <= self.turning_rate <= 1.0:
            raise ValueError("turning_rate must be between 0 and 1.")

    def step(self, dt: float, control: np.ndarray) -> None:
        self.check_input(control)

        # Random horizontal target velocity
        direction = self.rng.uniform(-1.0, 1.0, size=2)
        direction_norm = np.linalg.norm(direction)
        if direction_norm == 0.0:
            direction = np.array([1.0, 0.0])
        else:
            direction /= direction_norm
        target_vxy = direction * self.rng.uniform(self.min_speed, self.max_speed)

        # Smooth turning
        vxy = self.state[3:5]
        vxy = (1 - self.turning_rate) * vxy + self.turning_rate * target_vxy

        # Speed limit
        speed = np.linalg.norm(vxy)
        if speed > self.max_speed:
            vxy *= self.max_speed / speed

        # Position update with simple collision check
        pxy = self.state[0:2]
        new_pxy = pxy + vxy * dt
        collision = (
            self.env is not None
            and self.env.is_collision(
                new_pxy,
                check_altitude=False,
                check_boundary=True,
            ).item()
        )
        if collision:
            new_pxy = pxy

        # Update horizontal motion
        self.state[0:2] = new_pxy
        self.state[3:5] = vxy

        # Skip altitude tracking if no environment
        if self.env is None:
            return

        # Altitude following
        current_z = self.state[2]
        target_z = self.env.get_elevation(self.state[0:2]).item()
        climb = np.clip(target_z - current_z, -self.climb_rate, +self.climb_rate)

        # Update vertical motion
        self.state[5] = climb
        self.state[2] += climb * dt
