import numpy as np

from .base import Dynamics


class StaticDynamics(Dynamics):
    def step(self, dt: float, control: np.ndarray) -> None:
        """Keep state unchanged while honoring the common dynamics contract."""

        self.check_input(control)
        return None
