"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np
from .pd_controller import PDController


class PositionController(PDController):
    """
    A PD controller for 2D horizontal position control.

    Attributes
    ----------
    target_position : float
        Desired horizontal position in meters.
    """

    def __init__(self, kp: float, kd: float):
        """
        Initializes the horizontal position controller with proportional and
        derivative gains.

        Parameters
        ----------
        kp : float
            Proportional gain.
        kd : float
            Derivative gain.
        """
        super().__init__(kp, kd)

    def control(
        self, target_position: np.ndarray, position: np.ndarray, velocity: np.ndarray
    ) -> np.ndarray:
        """
        Computes the control output based on the current position and
        horizontal speed.

        Parameters
        ----------
        target_position: np.ndarray
            A (2,) array with target horizontal position in meters.
        position : np.ndarray
            A (2,) array with current horizontal position in meters.
        velocity : np.ndarray
            A (2,) array with current horizontal speed in m/s.

        Returns
        -------
        np.ndarray
            A (2,) array with output (e.g., thrust or acceleration) to achieve
            the target horizontal position.
        """
        error = target_position - position
        return super().control(error, velocity)
