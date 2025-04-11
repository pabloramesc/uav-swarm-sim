from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from simulator.environment import Environment


@dataclass
class PositionControllerConfig:
    pass


class PositionController(ABC):
    """
    Base class for position control.
    """

    def __init__(self, config: PositionControllerConfig, env: Environment) -> None:
        self.config = config
        self.env = env

    @abstractmethod
    def update(
        self, agent_state: np.ndarray, neighbot_states: np.ndarray, time: float = None
    ) -> np.ndarray:
        """
        Updates the controller's state and computes the control output.

        Parameters
        ----------
        agent_state : np.ndarray
            A (6,) array representing the agent's state [px, py, pz, vx, vy, vz].
        neighbor_states : np.ndarray
            A (N, 6) array representing the states [px, py, pz, vx, vy, vz] of N neighbors.
        time : float, optional
            Current simulation time in seconds. Default is None.

        Returns
        -------
        np.ndarray
            Control output [fx, fy, fz].
        """
        pass
