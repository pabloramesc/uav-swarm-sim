"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""


class PDController:
    """
    A base PD controller.

    Attributes
    ----------
    kp : float
        Proportional gain.
    kd : float
        Derivative gain.
    """

    def __init__(self, kp: float, kd: float):
        """
        Initializes the PD controller with proportional and derivative gains.

        Parameters
        ----------
        kp : float
            Proportional gain.
        kd : float
            Derivative gain.
        """
        self.kp = kp
        self.kd = kd

    def control(self, error: float, derivative: float) -> float:
        """
        Computes the control output based on the error and its derivative.

        Parameters
        ----------
        error : float
            The error value.
        derivative : float
            The derivative of the error.

        Returns
        -------
        float
            Control output.
        """
        return self.kp * error - self.kd * derivative
