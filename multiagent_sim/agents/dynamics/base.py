from abc import ABC, abstractmethod
import numpy as np


class Dynamics(ABC):
    """Abstrac base class for agent dynamics model."""

    state_shape = (6,)  # default value, can be overriden in subclasses

    def __init__(self):
        self.state: np.ndarray = None

    @property
    def position(self) -> np.ndarray:
        """Position of the agent [px, py, pz] in meters."""
        return self.state[0:3]

    @property
    def velocity(self) -> np.ndarray:
        """Velocity of the agent [vx, vy, vz] in m/s."""
        return self.state[3:6]

    def initialize(self, state: np.ndarray) -> None:
        self.check_state(state)
        self.state = state.copy()

    @abstractmethod
    def step(self, dt: float, **kwargs) -> None:
        pass

    def check_state(self, state: np.ndarray) -> None:
        if not isinstance(state, np.ndarray):
            raise ValueError("State must be a numpy array.")
        if state.shape != self.state_shape:
            raise ValueError(f"State must have shape {self.state_shape}.")
