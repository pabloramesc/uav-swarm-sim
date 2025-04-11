"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""


class AltitudeController:
    """
    A simple PD controller for altitude control.

    Attributes
    ----------
    kp : float
        Proportional gain.
    kd : float
        Derivative gain.
    target_altitude : float
        Desired altitude in meters.
    """

    def __init__(self, kp: float, kd: float, target_altitude: float = 0.0):
        """
        Initializes the altitude controller with proportional and derivative gains.

        Parameters
        ----------
        kp : float
            Proportional gain.
        kd : float
            Derivative gain.
        target_altitude : float, optional
            Desired altitude in meters (default is 0.0).
        """
        self.kp = kp
        self.kd = kd
        self.target_altitude = target_altitude

    def control(self, altitude: float, vspeed: float) -> float:
        """
        Computes the control output based on the current altitude and velocity.

        Parameters
        ----------
        altitude : float
            Current altitude in meters.
        vspeed : float
            Current vertical speed in m/s.

        Returns
        -------
        float
            Control output (e.g., thrust or acceleration) to achieve the target altitude.
        """
        error = self.target_altitude - altitude
        control_output = self.kp * error - self.kd * vspeed
        return control_output
