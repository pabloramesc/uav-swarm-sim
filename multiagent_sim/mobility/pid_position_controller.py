from dataclasses import dataclass

import numpy as np

from .pid import PIDController
from .position_controller import PositionController, ControllerContext


@dataclass
class PIDConfig:
    """Configuration for PID position controller."""

    kp: float = 0.1  # proportional gain
    ki: float = 0.0  # integral gain
    kd: float = 0.5  # derivative gain
    limit: float = 10.0  # optional control output saturation


class PIDPositionController(PositionController):
    """Simple PID position controller that drives the agent toward a target position."""

    def __init__(self, config: PIDConfig):
        self.config = config

        self.pid_controller = PIDController(
            kp=config.kp, ki=config.ki, kd=config.kd, limit=config.limit
        )
        self.target_position: np.ndarray = None

    def initialize(self, context: ControllerContext):
        """Initialize the controller with optional context."""
        if context.target_position is None:
            raise ValueError("Target position must be provided.")
        self.target_position = np.array(context.target_position)

    def update(self, context: ControllerContext) -> np.ndarray:
        """Compute control force to move toward the target position."""
        if context.target_position is not None:
            self.target_position = context.target_position

        pos = context.state[:3]
        vel = context.state[3:6]

        # PD control: F = kp * error - kd * velocity
        error = self.target_position - pos
        control = self.pid_controller.control(error, derivative=-vel)

        return control
