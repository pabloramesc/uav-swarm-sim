from .base import Dynamics
import numpy as np
from typing import Optional


class PointMassDynamics(Dynamics):
    """Point mass 6-DOF dynamics model with acceleration limit."""

    def __init__(self, mass: float = 1.0, max_acc: Optional[float] = None) -> None:
        super().__init__()

        self.mass = float(mass)
        self.max_acc = float(max_acc) if max_acc is not None else None

    def step(self, dt: float, control: np.ndarray) -> None:        
        self.check_input(control)

        acc = control / self.mass
        acc = self._limit_acceleration(acc)

        # Integrate dynamics: x' = x + (dx/dt) · dt
        x_dot = np.zeros(6)
        x_dot[0:3] = self.state[3:6]  # Velocity: dx/dt
        x_dot[3:6] = acc  # Acceleration: dv/dt
        self.state += x_dot * dt

    def _limit_acceleration(self, acc: np.ndarray) -> np.ndarray:
        if self.max_acc is None:
            return acc
        mag = np.linalg.norm(acc).item()
        return (acc / mag) * min(mag, self.max_acc) if mag > 0.0 else acc
