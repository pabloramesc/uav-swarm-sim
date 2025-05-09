from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..environment.environment import Environment

SwarmingType = Literal["evsm", "sdqn"]


@dataclass
class SwarmingConfig:
    pass


class SwarmingController(ABC):
    """
    Base class for position control.
    """

    def __init__(self, config: SwarmingConfig, env: Environment) -> None:
        self.config = config
        self.env = env
        self.time = 0.0
        self.state = np.zeros((6,))  # px, py, pz, vx, vy, vz

    @abstractmethod
    def initialize(self, time: float, state: np.ndarray, **kwargs) -> np.ndarray:
        self.time = time
        self.state = state.copy()

    @abstractmethod
    def update(self, time: float, state: np.ndarray, **kwargs) -> np.ndarray:
        self.time = time
        self.state = state.copy()
