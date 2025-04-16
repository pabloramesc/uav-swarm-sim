"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from .pd_controller import PDController


class AltitudeController(PDController):
    """
    A PD controller for altitude control.

    Attributes
    ----------
    target_altitude : float
        Desired altitude in meters.
    """

    def __init__(self, kp: float, kd: float):
        """
        Initializes the altitude controller with proportional and derivative
        gains.

        Parameters
        ----------
        kp : float
            Proportional gain.
        kd : float
            Derivative gain.
        """
        super().__init__(kp, kd)

    def control(self, target_altitude: float, altitude: float, vspeed: float) -> float:
        """
        Computes the control output based on the current altitude and velocity.

        Parameters
        ----------
        target_altitude : float
            Target altitude in meters.
        altitude : float
            Current altitude in meters.
        vspeed : float
            Current vertical speed in m/s.

        Returns
        -------
        float
            Control output (e.g., thrust or acceleration) to achieve the target
            altitude.
        """
        error = target_altitude - altitude
        return super().control(error, vspeed)