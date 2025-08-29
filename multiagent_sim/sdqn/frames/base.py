"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class FrameBase(ABC):
    def __init__(self, height: int, width: int, channels: int = 1, label: str = "frame"):
        self.height = height
        self.width = width
        self.channels = channels
        self.shape = (height, width, channels)
        self.label = label
        self.frame = np.zeros(self.shape, dtype=np.float32)

    @abstractmethod
    def set_data(self, *args, **kwargs):
        pass

    @abstractmethod
    def update_frame(self):
        pass

    def generate(self, update: bool = True) -> np.ndarray:
        if update:
            self.update_frame()
        return self.frame.copy()


@dataclass
class FrameFactory:
    label: str = "frame"

    def create(self) -> FrameBase:
        raise NotImplementedError