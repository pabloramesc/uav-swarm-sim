import numpy as np
from typing import Optional

from sim.environment import Environment

from .base import Dynamics


class RandomWalkerDynamics(Dynamics):
    def __init__(
        self,
        env: Optional[Environment] = None,
        min_speed: float = 1.0,
        max_speed: float = 3.0,
        climb_rate: float = 0.2,
        turning_rate: float = 0.3,
    ) -> None:
        super().__init__()

        self.env = env
        self.min_speed = float(min_speed)
        self.max_speed = float(max_speed)
        self.climb_rate = float(climb_rate)
        self.turning_rate = float(turning_rate)

    def step(self, dt: float, control: np.ndarray) -> None:
        # Random horizontal target velocity
        direction = np.random.uniform(-1, 1, 2)
        direction /= np.linalg.norm(direction)
        target_vxy = direction * np.random.uniform(self.min_speed, self.max_speed)

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
        if self.env is not None and self.env.is_collision(
            new_pxy, check_altitude=False, check_boundary=True
        ):
            new_pxy = pxy

        # Update horizontal motion
        self.state[0:2] = new_pxy
        self.state[3:5] = vxy

        # Skip altitude tracking if no environment
        if self.env is None:
            return

        # Altitude following
        current_z = self.state[2]
        target_z = self.env.get_elevation(self.state[0:2])
        climb = np.clip(target_z - current_z, -self.climb_rate, +self.climb_rate)

        # Update vertical motion
        self.state[5] = climb
        self.state[2] += climb * dt
