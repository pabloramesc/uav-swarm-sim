from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..environment.environment import Environment

SwarmingType = Literal["evsm", "sdqn"]


@dataclass
class SwarmControllerConfig:
    pass


class SwarmPositionController(ABC):
    """
    Base class for position control.
    """

    def __init__(self, config: SwarmControllerConfig, env: Environment) -> None:
        self.config = config
        self.env = env
        self.time = 0.0
        self.state = np.zeros((6,))  # px, py, pz, vx, vy, vz

    @property
    def position(self) -> np.ndarray:
        """
        Current position [px, py, pz] in meters.
        """
        return self.state[0:3]

    @property
    def velocity(self) -> np.ndarray:
        """
        Current velocity [vx, vy, vz] in m/s.
        """
        return self.state[3:6]

    @abstractmethod
    def initialize(
        self,
        time: float,
        state: np.ndarray,
        drone_positions: dict[int, np.ndarray] = None,
        user_positions: dict[int, np.ndarray] = None,
    ) -> None:
        self.time = time
        self.state = state.copy()

    @abstractmethod
    def update(
        self,
        time: float,
        state: np.ndarray,
        drone_positions: dict[int, np.ndarray] = None,
        user_positions: dict[int, np.ndarray] = None,
    ) -> np.ndarray:
        self.time = time
        self.state = state.copy()
