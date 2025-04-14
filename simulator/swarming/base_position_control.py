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
        
        self.time = None
        self.state = np.zeros(6)
        self.neighbor_states = np.zeros((0, 6))
        self.neighbor_ids = np.zeros((0,), dtype=int)

    @abstractmethod
    def update(
        self,
        state: np.ndarray,
        neighbor_states: np.ndarray,
        neighbor_ids: np.ndarray = None,
        time: float = None,
    ) -> np.ndarray:
        """
        Updates the controller's state and computes the control output.

        Parameters
        ----------
        state : np.ndarray
            A (6,) array representing the agent's state [px, py, pz, vx, vy, vz].
        neighbor_states : np.ndarray
            A (N, 6) array representing the states [px, py, pz, vx, vy, vz] of N neighbors.
        neighbor_states : np.ndarray
            A (N,) array with the IDs of the N neighbors.
        time : float, optional
            Current simulation time in seconds. Default is None.

        Returns
        -------
        np.ndarray
            Control output [fx, fy, fz].
        """
        self.state = state
        self.neighbor_states = neighbor_states
        self.neighbor_ids = neighbor_ids
        self.time = time
