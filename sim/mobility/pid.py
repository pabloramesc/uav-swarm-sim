import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


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
        limit: float | None = None,
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
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.dt = float(dt)
        self.tau = float(tau)
        self.limit = float(limit) if limit is not None else None
        values = (self.kp, self.ki, self.kd, self.dt, self.tau)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("PID parameters must be finite.")
        if min(self.kp, self.ki, self.kd) < 0.0:
            raise ValueError("PID gains cannot be negative.")
        if self.dt <= 0.0 or self.tau <= 0.0:
            raise ValueError("dt and tau must be positive.")
        if self.limit is not None and (
            not math.isfinite(self.limit) or self.limit <= 0.0
        ):
            raise ValueError("limit must be positive and finite.")
        self.alpha = (2 * self.tau - self.dt) / (2 * self.tau + self.dt)

        self.reset()

    def reset(self) -> None:
        """Reset internal error states."""
        self.prev_error: NDArray[np.float64] | None = None
        self.integral: NDArray[np.float64] | None = None
        self.derivative: NDArray[np.float64] | None = None

    def control(
        self,
        error: ArrayLike,
        derivative: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """Computes the PID control output.

        Args:
            error: The error value.
            derivative: The derivative of the error.

        Returns:
            Control output.
        """
        error_array = np.asarray(error, dtype=np.float64)
        if self.prev_error is None:
            self.prev_error = np.zeros_like(error_array)
            self.integral = np.zeros_like(error_array)
            self.derivative = np.zeros_like(error_array)
        elif error_array.shape != self.prev_error.shape:
            raise ValueError(
                f"error shape changed from {self.prev_error.shape} "
                f"to {error_array.shape}; call reset() first."
            )

        assert self.integral is not None
        assert self.derivative is not None

        # If no derivative is provided, calculate from error using LPF
        if derivative is None:
            raw_derivative = (error_array - self.prev_error) / self.dt
            self.derivative = (
                self.alpha * raw_derivative + (1 - self.alpha) * self.derivative
            )
        else:
            derivative_array = np.asarray(derivative, dtype=np.float64)
            if derivative_array.shape != error_array.shape:
                raise ValueError("derivative must have the same shape as error.")
            self.derivative = derivative_array

        output_unsat = (
            self.kp * error_array + self.ki * self.integral + self.kd * self.derivative
        )

        # Apply output saturation
        if self.limit is not None:
            output = np.clip(output_unsat, -self.limit, +self.limit)
        else:
            output = output_unsat

        self.integral += error_array * self.dt

        # Anti-windup with back-calculation
        if self.ki != 0:
            self.integral += (self.dt / self.ki) * (output - output_unsat)

        self.prev_error = error_array

        return np.asarray(output, dtype=np.float64)
