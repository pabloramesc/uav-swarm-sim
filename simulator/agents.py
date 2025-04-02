"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from abc import ABC, abstractmethod

import numpy as np


class Agent(ABC):
    pass


class Drone(Agent):
    _counter = 0

    def __init__(self):
        super().__init__()

        self.id = self._counter
        self._counter += 1

        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.links: list[int] = []
        
        self.neighbor_ids: list[int] = []
        self.neighbor_positions: np.ndarray = None
        
    @property
    def state(self) -> np.ndarray:
        "Return a 4-size numpy array with drone state: [px, py, vx, vy]"
        return np.concatenate((self.position, self.velocity))
    
    def set_neighbors(self, positions: np.ndarray, ids: list[int] = None):
        self.neighbor_positions = positions
        if ids is not None:
            self.neighbor_ids = ids
    

class User(Agent):
    pass


class ControlStation(Agent):
    pass
