from abc import ABC, abstractmethod
import numpy as np

class Dynamics(ABC):
    """Abstrac base class for agent dynamics model."""

    state_shape: tuple[int] = (6,)  # default value for [px, py, pz, vx, vy, vz] state
    input_shape: tuple[int] = (3,)  # default value for [fx, fy, fz] input

    def __init__(self):
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
        self.check_state(value)
        self._state = value

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
        pass

    def check_state(self, state: np.ndarray) -> None:
        if not isinstance(state, np.ndarray):
            raise ValueError("State must be a numpy array.")
        if state.shape != self.state_shape:
            raise ValueError(f"State must have shape {self.state_shape}.")

    def check_input(self, control: np.ndarray) -> None:
        if not isinstance(control, np.ndarray):
            raise ValueError("Control input must be a numpy array.")
        if control.shape != self.input_shape:
            raise ValueError(f"Control input must have shape {self.input_shape}.")
