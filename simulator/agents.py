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

        self.state = np.zeros(6) # px, py, pz, vx, vy, vz
    
    @property
    def position(self) -> np.ndarray:
        return self.state[0:3]
    
    @property
    def velocity(self) -> np.ndarray:
        return self.state[3:6]

class User(Agent):
    pass


class ControlStation(Agent):
    pass
