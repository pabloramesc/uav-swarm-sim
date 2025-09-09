import numpy as np


class PIDController:
    """Proportional-Integral-Derivative (PID) controller with optional internal
    derivative and integral calculation using low-pass filter and anti-windup.

    Attributes:
        kp: Proportional gain.
        ki: Integral gain.
        kd: Derivative gain.
    """

    def __init__(
        self,
        kp: float,
        ki: float = 0.0,
        kd: float = 0.0,
        dt: float = 0.01,
        tau: float = 0.1,
        limit: float = None,
    ) -> None:
        """Initializes the PD controller with proportional and derivative gains.

        Args:
            kp: Proportional gain.
            ki: Integral gain.
            kd: Derivative gain.
            dt: Time step for internal derivative/integral calculation.
            tau: Time constant for derivative low-pass filter.
            limit: Max absolute value for control output saturation.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.limit = limit
        self.alpha = (2 * tau - dt) / (2 * tau + dt)

        self.reset()

    def reset(self):
        """Reset internal error states."""
        self.prev_error = 0.0
        self.integral = 0.0
        self.derivative = 0.0

    def control(self, error: np.ndarray, derivative: np.ndarray = None) -> np.ndarray:
        """Computes the PID control output.

        Args:
            error: The error value.
            derivative: The derivative of the error.

        Returns:
            Control output.
        """
        error = np.array(error)

        # If no derivative is provided, calculate from error using LPF
        if derivative is None:
            raw_derivative = (error - self.prev_error) / self.dt
            self.derivative = (
                self.alpha * raw_derivative + (1 - self.alpha) * self.derivative
            )
        else:
            self.derivative = np.array(derivative)

        output_unsat = self.kp * error + self.ki * self.integral + self.kd * self.derivative
        
        # Apply output saturation
        if self.limit is not None:
            output = np.clip(output_unsat, -self.limit, +self.limit)
        else:
            output = output_unsat
            
        self.integral += error * self.dt 
        
        # Anti-windup with back-calculation
        if self.ki != 0:
            self.integral += (self.dt / self.ki) * (output - output_unsat)
        
        self.prev_error = error
        
        return output
