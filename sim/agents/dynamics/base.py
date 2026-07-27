from abc import ABC, abstractmethod

import numpy as np


class Dynamics(ABC):
    """Abstract base class for agent dynamics models."""

    # Defaults represent [px, py, pz, vx, vy, vz] and [fx, fy, fz].
    state_shape: tuple[int, ...] = (6,)
    input_shape: tuple[int, ...] = (3,)

    def __init__(self) -> None:
        self._state: np.ndarray | None = None

    @property
    def state(self) -> np.ndarray:
        """State array of shape (6,) as [px, py, pz, vx, vy, vz] where
        - [px, py, pz] represent absolute positions in m.
        - [vx, vy, vz] represent absolute velocities in m/s.
        """
        if self._state is None:
            raise RuntimeError("State not initialized.")
        return self._state

    @state.setter
    def state(self, value: np.ndarray) -> None:
        try:
            normalized = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("State must be numeric.") from exc
        self.check_state(normalized)
        self._state = normalized.copy()

    @property
    def position(self) -> np.ndarray:
        """Position array of shape (3,) as [px, py, pz] in meters."""
        return self.state[0:3]

    @property
    def velocity(self) -> np.ndarray:
        """Velocity array of shape (3,) as [vx, vy, vz] in m/s."""
        return self.state[3:6]

    @abstractmethod
    def step(self, dt: float, control: np.ndarray) -> None:
        """Advance dynamics by one time step.

        Args:
            dt: Time step in seconds.
            control: Control input force array of shape (3,) as [fx, fy, fz]
                in Newtons (kg·m/s^2).
        """
        raise NotImplementedError

    def check_state(self, state: np.ndarray) -> None:
        if not isinstance(state, np.ndarray):
            raise ValueError("State must be a numpy array.")
        if state.shape != self.state_shape:
            raise ValueError(f"State must have shape {self.state_shape}.")
        if not np.isfinite(state).all():
            raise ValueError("State must contain only finite values.")

    def check_input(self, control: np.ndarray) -> None:
        if not isinstance(control, np.ndarray):
            raise ValueError("Control input must be a numpy array.")
        if control.shape != self.input_shape:
            raise ValueError(f"Control input must have shape {self.input_shape}.")
