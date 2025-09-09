from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from sim.environment.environment import Environment


@dataclass
class ControllerContext:
    time: float
    agent_state: np.ndarray
    target_position: np.ndarray | None = None
    drone_positions: dict[int, np.ndarray] | None = None
    user_positions: dict[int, np.ndarray] | None = None


class PositionController(ABC):
    """Base class for position control."""

    @abstractmethod
    def initialize(self, context: ControllerContext) -> None:
        pass

    @abstractmethod
    def update(self, context: ControllerContext) -> np.ndarray:
        pass


class DummyPositionController(PositionController):
    """Dummy position controller that does nothing."""

    def initialize(self, context: ControllerContext) -> None:
        return None

    def update(self, context: ControllerContext) -> np.ndarray:
        return np.zeros(3)
