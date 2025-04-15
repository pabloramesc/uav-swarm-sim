"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from .pd_controller import PDController


class HorizontalPositionController(PDController):
    """
    A PD controller for horizontal position control.

    Attributes
    ----------
    target_position : float
        Desired horizontal position in meters.
    """

    def __init__(self, kp: float, kd: float, target_position: float = 0.0):
        """
        Initializes the horizontal position controller with proportional and derivative gains.

        Parameters
        ----------
        kp : float
            Proportional gain.
        kd : float
            Derivative gain.
        target_position : float, optional
            Desired horizontal position in meters (default is 0.0).
        """
        super().__init__(kp, kd)
        self.target_position = target_position

    def control(self, position: float, hspeed: float) -> float:
        """
        Computes the control output based on the current position and horizontal speed.

        Parameters
        ----------
        position : float
            Current horizontal position in meters.
        hspeed : float
            Current horizontal speed in m/s.

        Returns
        -------
        float
            Control output (e.g., thrust or acceleration) to achieve the target position.
        """
        error = self.target_position - position
        return self.compute(error, hspeed)

    def set_target(self, target_position: float):
        """
        Updates the target horizontal position.

        Parameters
        ----------
        target_position : float
            Desired horizontal position in meters.
        """
        self.target_position = target_position
